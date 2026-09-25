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

import asyncio
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjOWNhM2FmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PHJlY3QgeD0iMiIgeT0iNyIgd2lkdGg9IjIwIiBoZWlnaHQ9IjEwIiByeD0iNSIvPjxjaXJjbGUgY3g9IjE3IiBjeT0iMTIiIHI9IjIuNSIgZmlsbD0iIzljYTNhZiIvPjxwYXRoIGQ9Ik02IDEyaDQiLz48L3N2Zz4="

# Added to the system prompt whenever this chat has tools attached (see ENCOURAGE_TOOL_USE).
# Reasoning models talk themselves out of calling a tool as often as they talk themselves
# into one; this leans the other way, since a wrong "let me just work this out" costs a
# stale or invented answer, while a wrong "let me check" costs one extra call.
TOOL_USE_HINT = (
    "While reasoning through this, if any part of it could be resolved with one of the "
    "available tools - a lookup, a calculation, a memory or file operation - do not "
    "hesitate: call it rather than working the answer out unaided or guessing."
)

# Packed into the same Kev call as the admin's/user's own QUESTIONS (one state prefill
# serves every question) when LOGIC_TOOL_DETECT is on and this chat has tools attached.
# A manual chain of thought is where a model's logic errors happen - a solver like Z3
# does not make them - so this is worth detecting before the model starts reasoning by
# hand rather than after.
_LOGIC_TOOL_QUESTION = {
    "type": "noul",
    "instructions": (
        "Is this a formal logic, constraint-satisfaction, satisfiability, or "
        "theorem-proving question - one a symbolic SAT/SMT solver would answer more "
        "reliably than manual step-by-step reasoning?"
    ),
    "criteria": {
        "true": (
            "a logic puzzle, a consistency/contradiction check, proving an "
            "entailment, satisfying a set of constraints, or a case-by-case riddle "
            "that formal search would settle quickly"
        ),
        "false": (
            "ordinary factual, creative, or open-ended reasoning that does not "
            "reduce to formal constraints"
        ),
    },
}

# Packed into the same Kev call as everything else above when PLAN_DETECT is on and no
# plan exists yet for this chat (see Filter._get_plan). Only asked once per chat: once a
# plan exists it is injected every turn without asking again, until PLAN_TTL passes.
_NEEDS_PLAN_QUESTION = {
    "type": "noul",
    "instructions": (
        "Is this a multi-part or complex request that would benefit from being broken "
        "into an explicit plan or checklist of steps, rather than answered directly in "
        "one response?"
    ),
    "criteria": {
        "true": (
            "a multi-step task, a project, or something with several deliverables or "
            "phases that later turns in this chat will keep building on"
        ),
        "false": "a simple question or single request answerable directly",
    },
}

# The structured-output schema for plan generation (LM Studio / OpenAI json_schema mode),
# trimmed to just what's injected as guidance: a short ordered checklist, no dependency
# graph or tool selection - this is a hint for the chat model answering turn by turn, not
# a plan something else executes (contrast planning_standalone.py's fuller Task schema).
_PLAN_JSON_SCHEMA: dict = {
    "name": "plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["task_id", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}

# chat_id -> (created_at, tasks). LRU-evicted (PLAN_MAX_CHATS) and TTL-expired
# (PLAN_TTL), same pattern as the Kev answer cache below.
_CHAT_PLAN_STORE: "OrderedDict[str, tuple]" = OrderedDict()


def _strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?<think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_object_at(text: str, start: int) -> Optional[str]:
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort JSON-object recovery from a completion, the same strategy
    planning_standalone.py/planning_lite.py use: strip thinking/code fences, try a
    direct parse, then a brace-balanced scan preferring an object with `tasks`."""
    cleaned = _strip_code_fences(_strip_think_blocks(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    fallback: Optional[dict] = None
    for m in re.finditer(r"\{", cleaned):
        candidate = _balanced_object_at(cleaned, m.start())
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tasks" in obj:
            return obj
        if fallback is None and isinstance(obj, dict):
            fallback = obj
    return fallback


def _content_from_choices(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return content
            if message.get("reasoning_content"):
                return message["reasoning_content"]
        if choices[0].get("text"):
            return choices[0]["text"]
    if data.get("content"):
        return data["content"]
    return ""


def _owui_extract_content(response: Any) -> str:
    """Pull assistant text out of an Open WebUI chat completion response, whichever
    shape it comes back as (dict, plain string, or a FastAPI Response with `.body`)."""
    if isinstance(response, dict):
        return _content_from_choices(response)
    if isinstance(response, str):
        return response
    body = getattr(response, "body", None)
    if body:
        try:
            return _content_from_choices(json.loads(body))
        except Exception:
            try:
                return (
                    body.decode() if isinstance(body, (bytes, bytearray)) else str(body)
                )
            except Exception:
                return ""
    return ""


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
        ENCOURAGE_TOOL_USE: bool = Field(
            default=True,
            description="When this chat has tools/MCP servers attached, tell the model not to hesitate to call one while reasoning if it would resolve part of the request, instead of working it out unaided.",
        )
        LOGIC_TOOL_DETECT: bool = Field(
            default=True,
            description="Ask Kev whether this message is a formal logic / constraint-satisfaction / theorem-proving question. If so, skip the model's own extended reasoning for this turn and tell it to call a symbolic solver tool (e.g. Z3) instead.",
        )
        LOGIC_TOOL_NAMES: str = Field(
            default=(
                "z3_solve_constraints, z3_prove_theorem, z3_run_script, check_consistency, "
                "check_entailment, solve_equation, solve_matrix_equation"
            ),
            description="Comma-separated candidate tool names for the instruction when LOGIC_TOOL_DETECT fires - covers both math/math_plus_mcp.py (the z3_*/check_* tools) and math/math_solver_mcp.py (solve_equation, solve_matrix_equation). Only the ones actually attached to this chat are named; if neither server's tools can be detected as attached, the full list is named as a fallback.",
        )
        LOGIC_TOOL_THRESHOLD: float = Field(
            default=0.5,
            description="Minimum Kev probability to treat the message as a formal-logic question.",
        )
        LOGIC_FORCE_TOOL_CHOICE: bool = Field(
            default=True,
            description="Also set tool_choice to force a tool call this turn when LOGIC_TOOL_DETECT fires, instead of only instructing the model to use one. Needed in practice - a confident model ignores a plain instruction to use a tool it doesn't feel it needs; only forcing tool_choice reliably gets the call made. Can misfire if no attached tool actually fits the request.",
        )
        PLAN_DETECT: bool = Field(
            default=True,
            description="Ask Kev whether a request is complex/multi-step; if so and no plan exists yet for this chat, generate one (a short checklist, via one completion call to the same chat model) and inject it as guidance every turn thereafter.",
        )
        PLAN_THRESHOLD: float = Field(
            default=0.5,
            description="Minimum Kev probability to treat the request as needing a plan.",
        )
        PLAN_MAX_TASKS: int = Field(
            default=8, description="Max steps in a generated plan."
        )
        PLAN_TEMPERATURE: float = Field(
            default=0.3,
            description="Sampling temperature for plan generation. Kept low for a deterministic, parseable checklist.",
        )
        PLAN_TTL: float = Field(
            default=3600.0,
            description="Seconds a chat's plan stays cached with no new message before it's dropped (a later message then re-triggers detection).",
        )
        PLAN_MAX_CHATS: int = Field(
            default=200, description="Max chats to remember a plan for at once (LRU-evicted)."
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
        __metadata__: Optional[dict] = None,
        __request__: Any = None,
    ) -> dict:
        if (
            __task__
        ):  # title, tags, autocomplete: housekeeping, not a turn the user is waiting on
            return body
        user_valves = (__user__ or {}).get("valves") or self.UserValves()

        chat_id = None
        if isinstance(__metadata__, dict):
            chat_id = __metadata__.get("chat_id") or __metadata__.get("session_id")
        if not chat_id:
            chat_id = body.get("chat_id")
        existing_plan = self._get_plan(chat_id) if chat_id else None

        # Independent of Kev's own scoring below: this chat has tools attached at all, so
        # push back on a reasoning model's habit of working around a tool instead of using
        # one. Added even if the Kev call itself fails or is disabled. An existing plan is
        # injected the same way, every turn, for as long as PLAN_TTL keeps it alive - it
        # doesn't depend on Kev answering this turn either.
        lines = []
        if self.valves.ENCOURAGE_TOOL_USE and self._tools_available(body, __metadata__):
            lines.append(TOOL_USE_HINT)
        if existing_plan:
            lines.append(self._plan_system_line(existing_plan))

        text = self._last_user_text(body)
        if len(text) < self.valves.MIN_CHARS:
            if lines:
                body["messages"] = self._with_system_lines(body.get("messages", []), lines)
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
            if lines:
                body["messages"] = self._with_system_lines(body.get("messages", []), lines)
            await self._status(
                __event_emitter__, f"Kev filter: {exception}", user_valves
            )
            return body

        tools_available = self._tools_available(body, __metadata__)
        if (
            self.valves.LOGIC_TOOL_DETECT
            and tools_available
            and "logic_tool" not in questions
        ):
            questions = {**questions, "logic_tool": _LOGIC_TOOL_QUESTION}
        if (
            self.valves.PLAN_DETECT
            and chat_id
            and existing_plan is None
            and "needs_plan" not in questions
        ):
            questions = {**questions, "needs_plan": _NEEDS_PLAN_QUESTION}

        started = time.perf_counter()
        try:
            answer = await self._ask(
                {"state": text, "model": "kev-latest", "questions": questions}
            )
            verdict = self._verdict(answer)
        except (
            Exception
        ) as exception:  # noqa: BLE001 - fail open: the chat is more important than the decision
            if lines:
                body["messages"] = self._with_system_lines(body.get("messages", []), lines)
            await self._status(
                __event_emitter__,
                f"Kev unavailable ({type(exception).__name__}); answering without it",
                user_valves,
            )
            return body

        logic_answer = (answer.get("answers") or {}).get("logic_tool")
        logic_prob = float(logic_answer["noul"]) if logic_answer else None
        if logic_prob is not None and logic_prob >= self.valves.LOGIC_TOOL_THRESHOLD:
            self._disable_thinking(body)
            candidate_names = [
                n.strip() for n in self.valves.LOGIC_TOOL_NAMES.split(",") if n.strip()
            ]
            attached = self._attached_tool_names(body)
            # Name only the ones actually attached (whichever math server this chat has,
            # math_plus_mcp.py's z3_*/check_* or math_solver_mcp.py's solve_*), falling
            # back to the full candidate list when attachment can't be determined at all.
            tool_names = [n for n in candidate_names if n in attached] or candidate_names
            lines.append(
                f"System One (Kev) flagged this as a formal logic/constraint problem "
                f"(p {logic_prob:.3f}). Skip extended step-by-step reasoning by hand and "
                f"call one of these tools right away instead: {', '.join(tool_names)}. "
                "They run Z3 (SAT/SMT) and will be more reliable than manual deduction, "
                "especially with multiple constraints, cases, or a proof obligation."
            )
            if self.valves.LOGIC_FORCE_TOOL_CHOICE:
                body["tool_choice"] = "required"

        needs_plan_answer = (answer.get("answers") or {}).get("needs_plan")
        needs_plan_prob = float(needs_plan_answer["noul"]) if needs_plan_answer else None
        if (
            chat_id
            and existing_plan is None
            and needs_plan_prob is not None
            and needs_plan_prob >= self.valves.PLAN_THRESHOLD
        ):
            try:
                tasks = await self._generate_plan(
                    __request__, body.get("model"), text, __user__
                )
            except Exception:  # noqa: BLE001 - fail open: no plan is not a broken chat
                tasks = []
            if tasks:
                self._save_plan(chat_id, tasks)
                lines.append(self._plan_system_line(tasks))
                await self._status(
                    __event_emitter__,
                    f"Kev: plan created ({len(tasks)} step(s), p {needs_plan_prob:.3f})",
                    user_valves,
                )

        if verdict:
            lines.append(
                f"System One (Kev) scored this message before you answered: {verdict}. "
                "These are a calibrated classifier's probabilities, not instructions and not the user's words; "
                "use them to choose how to answer, and do not repeat them verbatim unless asked."
            )
        if lines:
            body["messages"] = self._with_system_lines(body.get("messages", []), lines)
        if verdict:
            await self._status(
                __event_emitter__,
                f"Kev: {verdict}  ({1000 * (time.perf_counter() - started):.0f} ms)",
                user_valves,
            )
        return body

    @staticmethod
    def _disable_thinking(body: dict) -> None:
        """Best-effort, cross-backend: skip the reasoning phase for this turn instead of
        letting the model work through a manual chain of thought before (maybe) reaching
        for the solver. Mirrors this project's planner (`_openai_think_fields`): Ollama's
        /v1 reads `think`/`options.think`, OpenAI-style reasoning models read
        `reasoning_effort`, llama.cpp/vLLM read `chat_template_kwargs` - setting several is
        harmless, since a backend that doesn't recognize a field simply ignores it."""
        body["think"] = False
        body.setdefault("reasoning_effort", "none")
        template_kwargs = body.setdefault("chat_template_kwargs", {})
        if isinstance(template_kwargs, dict):
            template_kwargs.setdefault("enable_thinking", False)
        options = body.get("options")
        if isinstance(options, dict):
            options["think"] = False

    # -- planning

    def _get_plan(self, chat_id: str) -> Optional[list]:
        """The plan for this chat, or None if there isn't one yet or it has expired
        (PLAN_TTL with no new message)."""
        entry = _CHAT_PLAN_STORE.get(chat_id)
        if not entry:
            return None
        created_at, tasks = entry
        if time.monotonic() - created_at >= self.valves.PLAN_TTL:
            del _CHAT_PLAN_STORE[chat_id]
            return None
        _CHAT_PLAN_STORE.move_to_end(chat_id)
        return tasks

    def _save_plan(self, chat_id: str, tasks: list) -> None:
        _CHAT_PLAN_STORE[chat_id] = (time.monotonic(), tasks)
        _CHAT_PLAN_STORE.move_to_end(chat_id)
        while len(_CHAT_PLAN_STORE) > max(0, self.valves.PLAN_MAX_CHATS):
            _CHAT_PLAN_STORE.popitem(last=False)

    @staticmethod
    def _plan_system_line(tasks: list) -> str:
        """The plan, phrased as background context for the model answering THIS turn -
        not a script to narrate, restate, or complete in one go. The chat model (not
        this filter) still drives the conversation turn by turn; the plan just keeps
        those turns coherent with the request as a whole."""
        steps = " | ".join(f"{t['task_id']}: {t['description']}" for t in tasks)
        return (
            "This chat is working from a plan drawn up earlier for the user's overall "
            f"request: {steps}. Use it to keep this and later answers coherent with the "
            "whole task, but answer only what the user actually asked this turn - the "
            "plan is background context, not something to narrate, restate, or complete "
            "all at once."
        )

    async def _generate_plan(
        self, request: Any, model_id: str, goal: str, user_dict: Optional[dict]
    ) -> list[dict]:
        """One completion call to the same chat model, decomposing `goal` into a short
        checklist ({"tasks": [{"task_id", "description"}, ...]}) - the same structured-
        output degrade-across-backends strategy planning_standalone.py/planning_lite.py
        use (json_schema -> json_object -> recovered from prose), trimmed to just a flat
        list with no dependency graph or tool selection, since this is guidance for a
        chat model answering turn by turn, not something else executing the plan."""
        if not model_id:
            return []
        from open_webui.utils.chat import generate_chat_completion
        from open_webui.models.users import Users

        user = None
        if user_dict and user_dict.get("id"):
            user = Users.get_user_by_id(user_dict["id"])
            if asyncio.iscoroutine(user):
                user = await user

        system_prompt = (
            "Decompose the user's request into a short ordered checklist of the "
            "concrete steps needed to fulfill it. Return STRICTLY a JSON object: "
            '{"tasks": [{"task_id": "step_1", "description": "..."}, ...]}. Produce '
            f"between 2 and {self.valves.PLAN_MAX_TASKS} steps. No prose, no "
            "explanations, no <think> blocks."
        )
        base_form: dict = {
            "model": model_id,
            "stream": False,
            "temperature": self.valves.PLAN_TEMPERATURE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": goal},
            ],
        }
        attempts = [
            {
                **base_form,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _PLAN_JSON_SCHEMA,
                },
            },
            {**base_form, "response_format": {"type": "json_object"}},
            base_form,
        ]
        for form_data in attempts:
            try:
                response = await generate_chat_completion(request, form_data, user=user)
            except Exception:
                continue
            content = _owui_extract_content(response)
            if not content:
                continue
            parsed = _extract_json_object(content)
            if not (parsed and isinstance(parsed.get("tasks"), list)):
                continue
            tasks: list[dict] = []
            for idx, item in enumerate(parsed["tasks"][: self.valves.PLAN_MAX_TASKS], 1):
                if not isinstance(item, dict):
                    continue
                description = str(item.get("description", "")).strip()
                if not description:
                    continue
                task_id = str(item.get("task_id") or f"step_{idx}").strip()
                tasks.append({"task_id": task_id, "description": description})
            if tasks:
                return tasks
        return []

    # -- tool detection

    @staticmethod
    def _tools_available(body: dict, metadata: Optional[dict]) -> bool:
        """Best-effort: does this request have any tools/MCP servers attached at all?
        Open WebUI's exact shape has moved around across versions, so this checks every
        location seen in the wild rather than one - false negatives just mean the hint is
        skipped, never a broken request."""
        if isinstance(body.get("tools"), list) and body["tools"]:
            return True
        if isinstance(metadata, dict):
            for key in ("tool_ids", "tools", "mcpServers", "mcp_servers"):
                value = metadata.get(key)
                if isinstance(value, (list, dict)) and value:
                    return True
            features = metadata.get("features")
            if isinstance(features, dict) and features.get("tools"):
                return True
        return False

    @staticmethod
    def _attached_tool_names(body: dict) -> set:
        """The actual function names attached to this request (OpenAI tool-schema
        shape: [{"type": "function", "function": {"name": ...}}, ...]), so the logic-tool
        instruction only ever names a tool that is really there - whichever of
        math_plus_mcp.py's z3_*/check_* tools or math_solver_mcp.py's solve_equation/
        solve_matrix_equation happen to be attached to this chat, not both by assumption.
        Empty when `tools` isn't in this shape (older Open WebUI versions attach tools
        later in the pipeline than this filter sees) - callers fall back to the full
        configured list in that case."""
        names = set()
        for entry in body.get("tools") or []:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function")
            name = fn.get("name") if isinstance(fn, dict) else entry.get("name")
            if name:
                names.add(name)
        return names

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
        return Filter._with_system_lines(messages, [line])

    @staticmethod
    def _with_system_lines(messages: list, lines: list) -> list:
        """Append one or more lines to the system message (joined on their own paragraph
        each), or add one. Each line is expected to already name its own source/nature."""
        if not lines:
            return list(messages)
        addition = "\n\n".join(lines)
        messages = list(messages)
        for index, message in enumerate(messages):
            if message.get("role") == "system":
                merged = dict(message)
                merged["content"] = (
                    f"{message.get('content', '').rstrip()}\n\n{addition}".strip()
                )
                messages[index] = merged
                return messages
        return [{"role": "system", "content": addition}] + messages

    async def _status(self, emitter, description: str, user_valves=None) -> None:
        if (
            emitter
            and self.valves.SHOW_STATUS
            and getattr(user_valves, "show_status", True)
        ):
            await emitter(
                {"type": "status", "data": {"description": description, "done": True}}
            )
