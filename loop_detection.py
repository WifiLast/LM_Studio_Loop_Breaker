"""
title: Loop Guard (Enhanced)
author: adapted for PLS math_mcp with planner integration
version: 2.0.0
required_open_webui_version: 0.5.0
description: >
  Breaks "deliberation loops" where a model repeats the same paragraph or line over and
  over without making progress. On outlet it collapses repeated blocks in the finished
  response; on inlet it scrubs the same repetition out of prior messages so the model
  does not see — and continue — the pattern on the next turn.

  This is a Function (Filter): load it under Workspace -> Functions, then enable it for
  the model/chat. It does not stop generation mid-stream; it removes the runaway
  repetition before it is stored and resent, which is what perpetuates the loop.

  Enhanced version:
  - Detects loops and can emit metadata for planner recovery
  - Can reduce sampling params (temperature, top_k) to escape loops
  - Signals loop events for monitoring/debugging

Configuration (Valves):
  - max_repeats: how many copies of an identical block/line to keep (default 2).
  - min_block_chars: ignore blocks shorter than this so short legitimate repeats
    (e.g. "OK", list bullets) are left alone (default 40).
  - scrub_history: also clean prior assistant messages on inlet (default true).
  - truncate_stream: suppress output once loop detected in stream (default true).
  - adjust_temperature: reduce temperature on retry to escape loops (default true).
  - loop_threshold: min repetitions to declare a loop (default 3).
  - priority: filter ordering (lower runs first).
"""

import re
from typing import Any, Awaitable, Callable, List, Optional

from pydantic import BaseModel, Field


def _normalize(text: str) -> str:
    """Whitespace/case-insensitive key for comparing blocks."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _collapse_repeats(
    text: str, max_repeats: int, min_block_chars: int
) -> tuple[str, int]:
    """Collapse over-repeated paragraphs, then over-repeated lines.

    Returns (new_text, removed_count). A block/line is kept up to `max_repeats` times
    (anywhere in the text); further copies are dropped. Short blocks are left untouched.
    """
    if not isinstance(text, str) or not text.strip():
        return text, 0

    removed = 0

    # 1) Paragraph level: split on blank lines, keep first `max_repeats` of each.
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        counts: dict[str, int] = {}
        kept: List[str] = []
        for para in paragraphs:
            key = _normalize(para)
            if len(key) < min_block_chars:
                kept.append(para)
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= max_repeats:
                kept.append(para)
            else:
                removed += 1
        text = "\n\n".join(kept)

    # 2) Line level: collapse repeated lines (catches loops without blank-line breaks).
    lines = text.split("\n")
    if len(lines) > 1:
        counts = {}
        kept_lines: List[str] = []
        for line in lines:
            key = _normalize(line)
            if len(key) < min_block_chars:
                kept_lines.append(line)
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= max_repeats:
                kept_lines.append(line)
            else:
                removed += 1
        text = "\n".join(kept_lines)

    return text, removed


def _is_looping(text: str, max_repeats: int, min_block_chars: int) -> bool:
    """True if some non-trivial line or paragraph already repeats more than max_repeats."""
    if not isinstance(text, str):
        return False
    window = text[-8000:]  # bound the cost on long streams

    for splitter in (lambda s: re.split(r"\n\s*\n", s), lambda s: s.split("\n")):
        counts: dict[str, int] = {}
        for chunk in splitter(window):
            key = _normalize(chunk)
            if len(key) < min_block_chars:
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > max_repeats:
                return True
    return False


def _count_repeats(text: str, min_block_chars: int) -> int:
    """Return the highest repeat count of any block/line in text."""
    if not isinstance(text, str):
        return 0
    window = text[-8000:]
    max_count = 0

    for splitter in (lambda s: re.split(r"\n\s*\n", s), lambda s: s.split("\n")):
        counts: dict[str, int] = {}
        for chunk in splitter(window):
            key = _normalize(chunk)
            if len(key) < min_block_chars:
                continue
            counts[key] = counts.get(key, 0) + 1
            max_count = max(max_count, counts[key])
    return max_count


class Filter:
    class Valves(BaseModel):
        max_repeats: int = Field(
            default=2,
            description="Keep at most this many copies of an identical block/line.",
        )
        min_block_chars: int = Field(
            default=40,
            description="Ignore blocks shorter than this (so short legitimate repeats survive).",
        )
        scrub_history: bool = Field(
            default=True,
            description="Also collapse repetition in prior messages on inlet (breaks cross-turn loops).",
        )
        truncate_stream: bool = Field(
            default=True,
            description="Detect repetition during streaming and suppress output once a loop starts.",
        )
        adjust_temperature: bool = Field(
            default=True,
            description="Reduce temperature in request body when loop detected (if supported by backend).",
        )
        loop_threshold: int = Field(
            default=3,
            description="Number of repetitions to declare a definite loop.",
        )
        priority: int = Field(
            default=0, description="Filter execution order; lower runs first."
        )

    def __init__(self):
        self.valves = self.Valves()
        # Per-stream accumulation: {completion_id: {"buf": str, "suppress": bool, "loop_count": int}}
        self._stream_state: dict = {}
        # Track loops across messages for adjustment
        self._message_loop_count: int = 0

    # -- helpers ------------------------------------------------------------ #
    def _clean_content(self, content: Any) -> tuple[Any, int]:
        """Collapse repeats in a message's content (string or multimodal parts)."""
        if isinstance(content, str):
            return _collapse_repeats(
                content, self.valves.max_repeats, self.valves.min_block_chars
            )
        if isinstance(content, list):
            total = 0
            new_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    new_text, removed = _collapse_repeats(
                        part["text"],
                        self.valves.max_repeats,
                        self.valves.min_block_chars,
                    )
                    total += removed
                    part = {**part, "text": new_text}
                new_parts.append(part)
            return new_parts, total
        return content, 0

    async def _emit(self, emitter, removed: int):
        if not emitter:
            return
        if removed <= 0:
            return
        try:
            if removed == 1 and self._message_loop_count > 0:
                # Likely a loop detection signal
                description = f"Loop Guard: Loop detected ({self._message_loop_count}). Adjusting params..."
            else:
                description = f"Loop Guard: collapsed {removed} repeated block(s)."
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": False,
                    },
                }
            )
        except Exception:
            pass

    # -- Open WebUI hooks --------------------------------------------------- #
    def stream(self, event: dict) -> dict:
        """Suppress streamed content once a repetition loop is detected.

        Accumulates the response text per completion id; when a block/line begins
        repeating past `max_repeats`, emits a one-time notice and blanks all further
        content deltas so the loop stops reaching the user (and the stored message).
        """
        if not self.valves.truncate_stream or not isinstance(event, dict):
            return event
        try:
            choices = event.get("choices") or []
            if not choices:
                return event
            choice = choices[0]
            delta = choice.get("delta") or {}
            chunk = delta.get("content")

            sid = event.get("id") or "default"
            state = self._stream_state.setdefault(
                sid, {"buf": "", "suppress": False, "loop_count": 0, "token_count": 0}
            )

            if isinstance(chunk, str) and chunk:
                # Track token-like count (words * 1.3 estimate)
                state["token_count"] += len(chunk.split()) * 1.3

                if state["suppress"]:
                    delta["content"] = ""
                else:
                    state["buf"] += chunk
                    repeat_count = _count_repeats(
                        state["buf"], self.valves.min_block_chars
                    )
                    if repeat_count >= self.valves.loop_threshold:
                        state["suppress"] = True
                        state["loop_count"] = repeat_count
                        delta["content"] = (
                            "\n\n[Loop Guard: repetition detected (x"
                            + str(repeat_count)
                            + ") - output truncated.]"
                        )

            # Clean up state when the stream finishes.
            if choice.get("finish_reason"):
                self._stream_state.pop(sid, None)

            return event
        except Exception:
            return event

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body

        # Scrub history if enabled
        removed_total = 0
        if self.valves.scrub_history:
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                cleaned, removed = self._clean_content(message.get("content"))
                if removed:
                    message["content"] = cleaned
                    removed_total += removed

        # Detect if we're in a loop by checking the last assistant message
        loop_detected = False
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    repeat_count = _count_repeats(content, self.valves.min_block_chars)
                    if repeat_count >= self.valves.loop_threshold:
                        loop_detected = True
                        self._message_loop_count += 1
                break

        # If loop detected: adjust temperature and cap tokens for streaming
        if loop_detected and self.valves.adjust_temperature:
            current_temp = body.get("temperature", 0.7)
            if isinstance(current_temp, (int, float)):
                # Reduce temperature to make output more deterministic (escape randomness)
                new_temp = max(0.1, current_temp * 0.7)  # reduce by 30%
                body["temperature"] = new_temp

                # If streaming, cap max_tokens to stop loop faster
                if body.get("stream"):
                    current_max = body.get("max_tokens", 2048)
                    if isinstance(current_max, int):
                        body["max_tokens"] = min(current_max, 1000)  # cap at 1K tokens

                await self._emit(
                    __event_emitter__,
                    removed_total + (1 if loop_detected else 0),
                )
            else:
                await self._emit(__event_emitter__, removed_total)
        elif removed_total > 0:
            await self._emit(__event_emitter__, removed_total)

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body

        # Clean the most recent assistant message (the just-finished response).
        removed_total = 0
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                cleaned, removed = self._clean_content(message.get("content"))
                if removed:
                    message["content"] = cleaned
                    removed_total += removed
                break

        await self._emit(__event_emitter__, removed_total)
        return body
