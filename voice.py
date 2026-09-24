"""
title: Voice
author: local
version: 0.3.0
required_open_webui_version: 0.11.0
description: A switch in the message box. On, every answer is also spoken: the reply goes to the OmniVoice TTS server and comes back as an audio player under the message. Off, nothing runs.
"""

# What this is
# ------------
# A toggle filter that speaks the assistant's replies through the OmniVoice MCP
# server (tts_mcp.py in the OmniVoice checkout), which exposes a plain JSON
# route for exactly this: POST /speak {"text": ...} -> a wav and its URL.
#
# It runs in `outlet`, after the model has finished, so nothing is spoken twice
# and streaming is untouched. The clip starts playing in the browser tab as soon as
# it is ready (AUTOPLAY), and a player is appended to the message, so it is part of
# the saved chat and can be replayed when the conversation is reopened.
#
# Autoplay goes through Open WebUI's `execute` event: a few lines of JavaScript run
# in the tab that sent the message. Browsers only let a page start sound after the
# user has interacted with it; sending the message counts, so this normally plays.
# When the browser refuses anyway, the status line says so and the player is there.
#
# What gets spoken
# ----------------
# The server strips markdown before synthesis - a fenced block becomes the words
# "code block", a link keeps its text - and cuts long answers at a sentence end.
# MAX_CHARS here is the second limit: an answer longer than this is not spoken at
# all rather than spoken in part, because half an answer read aloud is worse than
# none. Short replies under MIN_CHARS ("ok", "done") are skipped too.
#
# Non-verbal cues
# ----------------
# Before synthesis, Kev (System One) reads the reply once and picks one of OmniVoice's
# own non-verbal tags ([laughter], [sigh], [confirmation-en], ...) to prepend, or "none" -
# which is always an option and wins for ordinary informational replies, since a cue is
# for the rare joke, apology, plain yes, or surprised reaction, not routine information.
# It never touches a reply that already starts with one of these tags (the model's own
# call stands) or one that opens with a code block, and any other inline control syntax
# already in the text (phoneme overrides like [B EY1 S], pinyin tone markers) is untouched
# either way - only a recognized, cue-only tag at the very start is ever matched. Fail-open:
# no KEV_URL, no answer, or an error just means the reply is spoken exactly as written.
#
# Cost, and saying so
# -------------------
# Synthesis is roughly 1.5x real time on this hardware: about 3 s for a 2-sentence
# answer, and it happens after the text is already on screen. The first call after
# a restart also pays ~25 s to load the checkpoint. That silence is the thing to
# get right: the filter asks the server whether the model is resident, says which
# of the two waits you are in, and re-emits the status every TICK_SECONDS with the
# elapsed time, so a long wait looks like work rather than a hang.
#
# It is fail-open: if the server is down or slow the message is left exactly as it
# was, with a one-line status saying so.
#
# The alternative, for reference: the same server also implements the OpenAI
# speech API, so Admin Panel -> Settings -> Audio -> TTS Engine = OpenAI with
# base URL http://10.0.0.10:2010/v1 makes the built-in speaker button play
# OmniVoice. That is per-message and manual; this filter is automatic.
#
# Picking a voice
# ---------------
# VOICE (admin) and voice (each user) are dropdowns filled from the server's GET /voices every time the settings
# open: every <name>.pt saved in the OmniVoice demo's Voice Clone tab, and every <name>.wav clip, in the server's
# voices directory. Save a voice in the demo, reopen the settings, and it is there - no restart of either side.
# STYLE is the other way to shape the voice: items from OmniVoice's closed vocabulary ("female, british accent"),
# used only when no voice is picked.
#
# Install: Admin Panel -> Functions -> + -> paste -> enable, then assign it to the
# models you want the switch on. Set TTS_URL in its valves.

import asyncio
import json
import re
import time
import urllib.request
from typing import Any, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field, model_validator

ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjOWNhM2FmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTExIDUgNiA5SDN2NmgzbDUgNHoiLz48cGF0aCBkPSJNMTYgOC41YTQuNSA0LjUgMCAwIDEgMCA3Ii8+PHBhdGggZD0iTTE5IDUuNWE4LjUgOC41IDAgMCAxIDAgMTMiLz48L3N2Zz4="


DEFAULT_TTS_URL = "http://10.0.0.10:2010"
_tts_url = DEFAULT_TTS_URL  # the admin's TTS_URL, remembered whenever the valves are built with it

# OmniVoice's own inline non-verbal vocabulary (see its docs: "Non-Verbal & Pronunciation
# Control"). Kev picks from exactly this set - never a tag OmniVoice does not know - and
# "none" (most replies: plain information, code, instructions) is always an option, so a
# cue is the exception, not the default.
NONVERBAL_TAGS: dict = {
    "laughter": "something genuinely funny, a joke, or gentle teasing",
    "sigh": "resignation, disappointment, or a soft apology for bad news",
    "confirmation-en": "a short, plain spoken acknowledgement: yes, got it, sure",
    "question-en": "a plain, curious follow-up question sound",
    "question-ah": "a light, curious 'ah?' follow-up",
    "question-oh": "a surprised 'oh?' follow-up",
    "question-ei": "a casual, checking-in sound",
    "question-yi": "a puzzled 'hm?' questioning sound",
    "surprise-ah": "a sudden 'ah!' realization",
    "surprise-oh": "a mild 'oh!' surprise",
    "surprise-wa": "an impressed 'wow!' reaction",
    "surprise-yo": "an enthusiastic reaction",
    "dissatisfaction-hnn": "a frustrated or annoyed grunt",
}
# Matches only a leading tag from that exact vocabulary - never a phoneme override like
# "[B EY1 S]", a pinyin tone marker, or anything else OmniVoice's own inline syntax uses -
# so existing inline control syntax is never mistaken for one of ours and never touched.
_LEADING_NONVERBAL_TAG_RE = re.compile(
    r"^\s*\[(" + "|".join(re.escape(t) for t in NONVERBAL_TAGS) + r")\]"
)


def voice_options(__user__=None) -> list:
    """The dropdown entries: the server's presets, read when the settings open. Open WebUI calls this synchronously
    while it renders the settings, so it is a short blocking request, and a server that is down gives a one-entry
    list saying so instead of an error."""
    options = [{"value": "", "label": "Default"}]
    try:
        with urllib.request.urlopen(
            f"{_tts_url.rstrip('/')}/voices", timeout=2
        ) as response:
            voices = json.loads(response.read()).get("voices") or []
    except (
        Exception
    ):  # noqa: BLE001 - a settings page must open even with the server down
        return [
            {
                "value": "",
                "label": f"Default (voice server at {_tts_url} not reachable, no presets listed)",
            }
        ]
    for voice in voices:
        kind = "saved voice" if voice.get("kind") == "saved" else "clip"
        options.append({"value": voice["name"], "label": f"{voice['name']} ({kind})"})
    return options


class Filter:
    class Valves(BaseModel):
        TTS_URL: str = Field(
            default=DEFAULT_TTS_URL,
            description="Base URL of the OmniVoice TTS server (tts_mcp.py).",
        )
        VOICE: str = Field(
            default="",
            description="Voice for everyone who has not picked their own. Default = STYLE, or the model picks.",
            json_schema_extra={"input": {"type": "select", "options": "voice_options"}},
        )
        STYLE: str = Field(
            default="",
            description='Used when no voice is picked: style items from the model\'s vocabulary ("female, british accent").',
        )
        LANGUAGE: str = Field(
            default="",
            description='Language hint for pacing and numbers ("English", "German"). Empty = auto.',
        )
        SPEED: float = Field(
            default=1.0, description="Speaking rate multiplier around 1.0."
        )
        MIN_CHARS: int = Field(
            default=40, description="Replies shorter than this are not worth speaking."
        )
        MAX_CHARS: int = Field(
            default=1200,
            description="Replies longer than this are skipped entirely rather than spoken in part.",
        )
        TIMEOUT: int = Field(
            default=300, description="Seconds to wait for the audio before giving up."
        )
        AUTOPLAY: bool = Field(
            default=True,
            description="Start playing in the browser as soon as the audio is ready.",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Report progress and the duration (or a failure) in the status line.",
        )
        TICK_SECONDS: int = Field(
            default=5,
            description="How often the status line updates while synthesis runs. 0 = only a start and a finish line.",
        )
        KEV_URL: str = Field(
            default="http://10.0.0.10:8009",
            description="Kev System One endpoint. Empty = no non-verbal cue decision; the reply is spoken exactly as written.",
        )
        KEV_API_KEY: str = Field(
            default="",
            description="Bearer token, when the Kev endpoint was started with KEV_API_KEY set.",
        )
        KEV_EXPRESSIVE_CUES: bool = Field(
            default=True,
            description="Let Kev pick one OmniVoice non-verbal cue ([laughter], [sigh], ...) to prepend when a reply calls for it. 'None' is always an option and wins for ordinary informational replies - this is for the rare joke, apology, plain yes, or surprised reaction.",
        )
        KEV_TIMEOUT: float = Field(
            default=3.0,
            description="Seconds to wait for Kev's cue decision before speaking the reply as written.",
        )
        PRIORITY: int = Field(default=10, description="Filter order; lower runs first.")

        voice_options = staticmethod(voice_options)

        @model_validator(mode="after")
        def _remember_url(self):
            # The dropdown's options are fetched by a function Open WebUI calls without these valves, so it reads the
            # URL from here. Open WebUI builds this model from the saved settings before every chat and every save.
            global _tts_url
            _tts_url = self.TTS_URL or DEFAULT_TTS_URL
            return self

    class UserValves(BaseModel):
        voice: str = Field(
            default="",
            description="My own voice. Default = the admin's setting.",
            json_schema_extra={"input": {"type": "select", "options": "voice_options"}},
        )
        voice_options = staticmethod(voice_options)
        autoplay: bool = Field(
            default=True, description="Play the answer for me as soon as it is ready."
        )
        show_status: bool = Field(
            default=True, description="Show the status line for me."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # renders as a switch next to the message box; off = this module never runs
        self.icon = ICON

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __user__: Optional[dict] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        user_valves = (__user__ or {}).get("valves") or self.UserValves()
        messages = body.get("messages") or []
        index = self._last_assistant(messages)
        if index is None:
            return body

        text = self._text(messages[index])
        if "<audio" in text:  # already spoken (a re-run, or a regenerated message)
            return body
        if len(text) < self.valves.MIN_CHARS:
            return body
        if len(text) > self.valves.MAX_CHARS:
            await self._status(
                __event_emitter__,
                user_valves,
                f"not spoken: {len(text)} characters is over the {self.valves.MAX_CHARS} limit",
            )
            return body

        tag = await self._kev_expressive_tag(text)
        if tag:
            text = f"[{tag}] {text}"

        # Synthesis takes seconds, and the first one after a restart takes half a minute while the checkpoint loads.
        # Without a status line that whole time looks like nothing happening, so say what is happening and keep saying
        # it: which phase, and how long it has been running.
        loaded = await self._model_loaded()
        phase = (
            "loading the voice model"
            if loaded is False
            else f"speaking {len(text)} characters"
        )
        if loaded is False:
            phase += " (first request after a restart, ~25 s)"
        await self._status(__event_emitter__, user_valves, f"{phase}...", done=False)

        started = time.perf_counter()
        voice = (
            getattr(user_valves, "voice", "") or self.valves.VOICE or self.valves.STYLE
        )
        task = asyncio.ensure_future(self._speak(text, voice))
        try:
            while True:
                tick = self.valves.TICK_SECONDS
                try:
                    clip = await asyncio.wait_for(
                        asyncio.shield(task), timeout=tick if tick > 0 else None
                    )
                    break
                except asyncio.TimeoutError:
                    await self._status(
                        __event_emitter__,
                        user_valves,
                        f"{phase}... {time.perf_counter() - started:.0f}s",
                        done=False,
                    )
        except (
            Exception
        ) as exception:  # noqa: BLE001 - fail open: the answer matters, the audio does not
            task.cancel()
            await self._status(
                __event_emitter__,
                user_valves,
                f"voice unavailable ({type(exception).__name__}); answer left unspoken",
            )
            return body

        messages[index] = self._with_player(messages[index], text, self._player(clip))
        body["messages"] = messages
        summary = f"{clip['seconds']:.1f} s of audio, {clip['voice']} voice, ready in {time.perf_counter() - started:.0f}s"
        playback = ""
        if self.valves.AUTOPLAY and getattr(user_valves, "autoplay", True):
            playback = await self._autoplay(__event_call__, clip["url"])
        await self._status(
            __event_emitter__, user_valves, f"spoken: {summary}{playback}"
        )
        return body

    # -- pieces

    @staticmethod
    def _last_assistant(messages: list) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "assistant":
                return index
        return None

    @staticmethod
    def _text(message: dict) -> str:
        content = message.get("content") or ""
        if isinstance(content, list):  # multimodal: speak the text parts
            content = "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return content.strip()

    async def _model_loaded(self) -> Optional[bool]:
        """Whether the checkpoint is already resident, so the status line can name the wait. None if the server does
        not say (an older build, or it is not answering yet - the synthesis call will report that properly).
        """
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{self.valves.TTS_URL.rstrip('/')}/health"
                ) as response:
                    if response.status != 200:
                        return None
                    return bool((await response.json()).get("loaded"))
        except (
            Exception
        ):  # noqa: BLE001 - a probe must never be the reason a turn fails
            return None

    async def _kev_expressive_tag(self, text: str) -> Optional[str]:
        """Kev's one-shot judgment on the reply as a whole: does it call for a single
        OmniVoice non-verbal cue right at the start, and if so which? Sound design default
        is silence - "none" is always one of the choices and, being the ordinary case for
        an assistant reply, wins most of the time. Fail-open like every other Kev call
        here: no URL, no answer, or an error simply means the reply is spoken as written.
        """
        if not self.valves.KEV_URL or not self.valves.KEV_EXPRESSIVE_CUES:
            return None
        if _LEADING_NONVERBAL_TAG_RE.match(text):
            return None  # the model already made this call; leave its own tag alone
        if "```" in text[:400]:
            return None  # code up front reads flat; a vocal cue in front of it is noise
        criteria = {"none": "a plain, neutral, technical, or purely informational reply"}
        criteria.update(NONVERBAL_TAGS)
        payload = {
            "state": text[:1000],
            "model": "kev-latest",
            "questions": {
                "cue": {
                    "type": "choice",
                    "instructions": (
                        "Read this as if speaking it aloud to the person who asked. Does "
                        "it call for one short non-verbal vocal cue right at the start, "
                        "and if so which?"
                    ),
                    "criteria": criteria,
                }
            },
        }
        try:
            headers = {"content-type": "application/json"}
            if self.valves.KEV_API_KEY:
                headers["authorization"] = f"Bearer {self.valves.KEV_API_KEY}"
            timeout = aiohttp.ClientTimeout(total=self.valves.KEV_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.valves.KEV_URL.rstrip('/')}/v1/systemone",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        return None
                    data = json.loads(await response.text())
        except Exception:  # noqa: BLE001 - a missed cue is not worth a broken reply
            return None
        choice = ((data.get("answers") or {}).get("cue") or {}).get("choice")
        return choice if choice and choice != "none" else None

    async def _speak(self, text: str, voice: str) -> dict:
        payload = {"text": text, "speed": self.valves.SPEED}
        if voice:
            payload["voice"] = voice
        if self.valves.LANGUAGE:
            payload["language"] = self.valves.LANGUAGE
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.valves.TTS_URL.rstrip('/')}/speak", json=payload
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {body[:200]}")
                return json.loads(body)

    @staticmethod
    def _player(clip: dict) -> str:
        """A player, with a plain link under it: the player is what a browser renders, the link is what survives
        anywhere the HTML is stripped. Open WebUI builds its player from the text between the tags
        (HTMLToken.svelte), not from a src attribute - `<audio src=...></audio>` shows up as raw text.
        """
        # On lines of their own: written inline, markdown splits the element into separate tag tokens and the
        # player shows up as the text "<audio>...</audio>".
        return f'<audio>\n{clip["url"]}\n</audio>\n\n[🔊 {clip["seconds"]:.1f}s audio]({clip["url"]})'

    @staticmethod
    async def _autoplay(event_call, url: str) -> str:
        """Start the clip in the user's tab. Returns what to add to the status line: nothing when it plays, the
        reason when it does not (no tab to play in, or the browser's autoplay policy said no).
        """
        if event_call is None:
            return " (autoplay: no browser tab to play in)"
        code = (
            "const previous = window.__owuiVoice; if (previous) previous.pause();"
            f"const audio = new Audio({json.dumps(url)}); window.__owuiVoice = audio;"
            "try { await audio.play(); return 'playing'; } catch (e) { return 'blocked: ' + e.name; }"
        )
        try:
            result = await asyncio.wait_for(
                event_call({"type": "execute", "data": {"code": code}}), timeout=10
            )
        except (
            Exception
        ) as exception:  # noqa: BLE001 - the clip is in the message either way
            return f" (autoplay failed: {type(exception).__name__}; press play)"
        if result == "playing":
            return ""
        if isinstance(result, str) and result.startswith("blocked"):
            return f" (the browser blocked autoplay, {result[9:]}; press play)"
        return f" (autoplay: {result.get('error') if isinstance(result, dict) else result}; press play)"

    @staticmethod
    def _with_player(message: dict, text: str, player: str) -> dict:
        """The message with the player appended.

        Open WebUI 0.11 keeps a reply twice: `content`, and `output`, a list of Responses-style items that the chat
        view renders instead of `content` whenever it is present. Changing only `content` saves the player to the
        database and never shows it, so it goes at the end of the last text part of the last message item too.
        """
        message = {**message, "content": f"{text}\n\n{player}"}
        output = message.get("output")
        if isinstance(output, list):
            output = json.loads(
                json.dumps(output)
            )  # a copy: the original is Open WebUI's baseline to diff against
            items = [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
            ]
            if items:
                parts = items[-1].setdefault("content", [])
                texts = [
                    part
                    for part in parts
                    if isinstance(part, dict)
                    and part.get("type") in ("output_text", "text")
                ]
                if texts:
                    texts[-1]["text"] = f"{texts[-1].get('text') or ''}\n\n{player}"
                else:
                    parts.append({"type": "output_text", "text": player})
            else:
                output.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": player}],
                    }
                )
            message["output"] = output
        return message

    async def _status(
        self, emitter, user_valves, description: str, done: bool = True
    ) -> None:
        """`done=False` is what makes Open WebUI show it as something in progress rather than a result."""
        if (
            emitter
            and self.valves.SHOW_STATUS
            and getattr(user_valves, "show_status", True)
        ):
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )
