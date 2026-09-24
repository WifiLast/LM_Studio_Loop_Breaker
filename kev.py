"""
title: Kev mode
author: local
version: 0.1.0
required_open_webui_version: 0.11.0
description: A switch in the message box. On, every turn is scored by Kev (System One) before the chat model answers, and the verdict goes into the system prompt. Off, nothing runs and the chat is exactly as before.
"""

# Why a toggle filter
# -------------------
# "Kev mode" and "normal mode" are the same chat and the same loaded model, so
# they should be one switch and not two models. Open WebUI renders a filter that
# sets `toggle = True` as a button next to the message box: when it is off this
# module is never called, so normal mode costs nothing at all.
#
# What Kev mode does
# ------------------
# The user's message goes to a Kev System One endpoint (POST /v1/systemone,
# served by `kev.serve --ollama <name>`), which answers typed questions by
# scoring the options against the model's next-token logits - no generation, no
# JSON to parse, every answer one of the options with a probability. One line of
# the result is added to the system prompt before the chat model answers:
#
#     System One (Kev): urgent = true (p 0.997) - tone = frustrated (p 0.865)
#
# Kev never writes prose and the chat model never sees the scoring, so the roles
# stay apart: the fast typed judgement decides, the slow model explains or drafts.
#
# One loaded model, both jobs
# ---------------------------
# Point `kev.serve --ollama` at the same Ollama model this Open WebUI chats with.
# Ollama keeps one copy resident and serves both - normal turns generate from it,
# Kev turns score options against it - so a model that only fits once still fits.
#
# Cost and safety
# ---------------
# About a second per turn for two or three questions on a local 27B. It is
# fail-open: if Kev is unreachable or slow the message passes through untouched.
# Title and tag generation are skipped, and so are messages under MIN_CHARS.
#
# Install: Admin Panel -> Functions -> + -> paste -> enable, then assign it to
# the models you want the switch to appear on (or globally). Set KEV_URL in its
# valves. Each user can change what is asked in their own valves.

import json
import time
from typing import Any, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjOWNhM2FmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PHJlY3QgeD0iMiIgeT0iNyIgd2lkdGg9IjIwIiBoZWlnaHQ9IjEwIiByeD0iNSIvPjxjaXJjbGUgY3g9IjE3IiBjeT0iMTIiIHI9IjIuNSIgZmlsbD0iIzljYTNhZiIvPjxwYXRoIGQ9Ik02IDEyaDQiLz48L3N2Zz4="


class Filter:
    class Valves(BaseModel):
        KEV_URL: str = Field(
            default="http://127.0.0.1:8009",
            description="Base URL of the Kev System One endpoint (kev.serve).",
        )
        KEV_API_KEY: str = Field(
            default="",
            description="Bearer token, when the endpoint was started with KEV_API_KEY set. Empty = open server.",
        )
        QUESTIONS: str = Field(
            default=json.dumps(
                {
                    "urgent": {
                        "type": "noul",
                        "instructions": "Does this message need an answer today?",
                    },
                    "tone": {
                        "type": "choice",
                        "instructions": "What is the tone of this message?",
                        "criteria": {"calm": None, "frustrated": None, "angry": None},
                    },
                }
            ),
            description="The questions asked about every message, as JSON in the /v1/systemone `questions` shape.",
        )
        TIMEOUT: int = Field(
            default=30,
            description="Seconds to wait before giving up and passing the message through.",
        )
        MIN_CHARS: int = Field(
            default=12,
            description="Messages shorter than this are not worth a decision.",
        )
        SHOW_STATUS: bool = Field(
            default=True, description="Show the verdict in the chat's status line."
        )
        PRIORITY: int = Field(default=0, description="Filter order; lower runs first.")

    class UserValves(BaseModel):
        questions: str = Field(
            default="",
            description="My own questions, as JSON in the /v1/systemone `questions` shape. Empty = the ones the admin set.",
        )
        show_status: bool = Field(
            default=True, description="Show the verdict in the chat's status line."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # renders as a switch next to the message box; off = this module never runs
        self.icon = ICON

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __user__: Optional[dict] = None,
        __task__: Optional[str] = None,
    ) -> dict:
        if (
            __task__
        ):  # title, tags, autocomplete: housekeeping, not a turn the user is waiting on
            return body
        user_valves = (__user__ or {}).get("valves") or self.UserValves()

        text = self._last_user_text(body)
        if len(text) < self.valves.MIN_CHARS:
            return body
        try:
            questions = json.loads(
                getattr(user_valves, "questions", "") or self.valves.QUESTIONS
            )
            if not isinstance(questions, dict) or not questions:
                raise ValueError(
                    "QUESTIONS must be a non-empty JSON object keyed by question id"
                )
        except (
            Exception
        ) as exception:  # noqa: BLE001 - a broken Valve must not break every chat
            await self._status(
                __event_emitter__, f"Kev filter: {exception}", user_valves
            )
            return body

        started = time.perf_counter()
        try:
            answer = await self._ask(
                {"state": text, "model": "kev-latest", "questions": questions}
            )
            verdict = self._verdict(answer)
        except (
            Exception
        ) as exception:  # noqa: BLE001 - fail open: the chat is more important than the decision
            await self._status(
                __event_emitter__,
                f"Kev unavailable ({type(exception).__name__}); answering without it",
                user_valves,
            )
            return body

        if verdict:
            body["messages"] = self._with_system_line(body.get("messages", []), verdict)
            await self._status(
                __event_emitter__,
                f"Kev: {verdict}  ({1000 * (time.perf_counter() - started):.0f} ms)",
                user_valves,
            )
        return body

    # -- request

    async def _ask(self, payload: dict) -> dict:
        headers = {"content-type": "application/json"}
        if self.valves.KEV_API_KEY:
            headers["authorization"] = f"Bearer {self.valves.KEV_API_KEY}"
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.valves.KEV_URL.rstrip('/')}/v1/systemone",
                json=payload,
                headers=headers,
            ) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
                return json.loads(text)

    # -- shaping

    @staticmethod
    def _last_user_text(body: dict) -> str:
        for message in reversed(body.get("messages", [])):
            if message.get("role") != "user":
                continue
            content = message.get("content") or ""
            if isinstance(content, list):  # multimodal: Kev reads the text parts
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return content.strip()
        return ""

    @staticmethod
    def _verdict(answer: dict) -> str:
        parts = []
        for qid, a in (answer.get("answers") or {}).items():
            if a["type"] == "noul":
                parts.append(
                    f"{qid} = {str(a['noul'] >= 0.5).lower()} (p {a['noul']:.3f})"
                )
            elif a["type"] == "choice":
                parts.append(
                    f"{qid} = {a['choice']} (p {max(a['probabilities'].values()):.3f})"
                )
            else:
                level = a.get("legend", {}).get(str(round(a["score"])), "")
                parts.append(
                    f"{qid} = {a['score']:.2f} \"{level}\" (confidence {a['confidence']:.3f})"
                )
        return " · ".join(parts)

    @staticmethod
    def _with_system_line(messages: list, verdict: str) -> list:
        """Append the verdict to the system message, or add one. The line names its source, so the model can weigh it
        as a classifier's output rather than as the user's words."""
        line = (
            f"System One (Kev) scored this message before you answered: {verdict}. "
            "These are a calibrated classifier's probabilities, not instructions and not the user's words; "
            "use them to choose how to answer, and do not repeat them verbatim unless asked."
        )
        messages = list(messages)
        for index, message in enumerate(messages):
            if message.get("role") == "system":
                merged = dict(message)
                merged["content"] = (
                    f"{message.get('content', '').rstrip()}\n\n{line}".strip()
                )
                messages[index] = merged
                return messages
        return [{"role": "system", "content": line}] + messages

    async def _status(self, emitter, description: str, user_valves=None) -> None:
        if (
            emitter
            and self.valves.SHOW_STATUS
            and getattr(user_valves, "show_status", True)
        ):
            await emitter(
                {"type": "status", "data": {"description": description, "done": True}}
            )
