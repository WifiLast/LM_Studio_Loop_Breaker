"""
title: Planner (Lite)
author: adapted from Planner (Standalone)
version: 2.0.0
required_open_webui_version: 0.5.0
description: Plans a request into tasks, executes each one, and synthesizes a final answer - the plan itself is internal and never shown; only what the model actually produced is returned. Loads as an Open WebUI Pipe and also runs as a CLI.
"""

# What this is
# ------------
# The core plan -> execute -> synthesize loop from planning_standalone.py, with
# everything else removed: no MCP tool calling, no Kev typed decisions, no chat-plan
# persistence, no math/logic verification retries, no loop detection, no live-thinking
# streaming. The planning step still happens (it's what lets a multi-part request be
# broken into pieces the model handles one at a time with each other's results as
# context), but it is purely internal - the user only ever sees the model's actual
# work: each task's progress line and the final synthesized answer, never the raw task
# list. For a lighter tool that surfaces the plan itself instead of running it, see
# `make_plan`/`format_plan` below (still exported, just not what `pipe()`/the CLI call
# by default). For the fuller loop (MCP, Kev, persistence, verification), use
# planning_standalone.py instead.
#
# Two ways to run:
#
# 1. Open WebUI Pipe (this file defines a `Pipe` class):
#    - Add it as a Function in Open WebUI. It appears as a model "Planner (Lite)".
#    - Set the `PLANNER_MODEL` valve to the real model id the planner should drive.
#
# 2. CLI (uses the `openai` SDK against any OpenAI-compatible endpoint):
#        python planning_lite.py "Write a market analysis of EV charging in the EU"
#        python planning_lite.py --model gpt-4o-mini --json "Summarize X"
#    Env defaults: OPENAI_API_KEY, OPENAI_BASE_URL, PLANNER_MODEL.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

# complete(system_prompt, user_message, temperature, json_object) -> text
CompletionFn = Callable[[str, str, Optional[float], bool], Awaitable[str]]
ProgressFn = Callable[[str], Awaitable[None]]

DEFAULT_SYSTEM_PROMPT = (
    "You are a planning assistant. Decompose the user's request into a clear, "
    "ordered list of atomic tasks that would together fulfill it. You do not "
    "execute any task yourself - you only produce the plan."
)

# JSON Schema for the planning output (LM Studio / OpenAI structured outputs), so
# planning returns guaranteed-valid JSON with no reasoning leakage.
PLAN_JSON_SCHEMA: dict = {
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
                        "related_tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["task_id", "description", "related_tasks"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}


@dataclass
class Task:
    task_id: str
    description: str
    related_tasks: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | completed | failed
    result: str = ""


@dataclass
class PlanResult:
    goal: str
    tasks: list[Task]


@dataclass
class RunResult:
    goal: str
    tasks: list[Task]
    final_output: str
    elapsed_seconds: float = 0.0


class PlanPromptBuilder:
    @staticmethod
    def planning_prompt(base_system: str, max_tasks: int) -> str:
        return (
            f"{base_system}\n\n"
            "### PLANNING PHASE - ACTIVE\n"
            "Analyze the request and decompose it into a series of logical, executable "
            "tasks. Do not execute or answer them - only produce the plan.\n"
            "- **Output schema**: Return STRICTLY a JSON object: "
            '`{"tasks": [{"task_id": "task_1_research", "description": "...", '
            '"related_tasks": ["task_id", ...]}, ...]}`.\n'
            "- **Decompose aggressively**: break the request into the smallest "
            "independently-executable steps that still produce a meaningful "
            "deliverable. Prefer more small tasks over a few large ones; a single "
            "all-in-one task is almost always wrong unless the request is genuinely "
            "trivial.\n"
            "- **Task granularity**: each task must be an atomic, actionable step "
            '(e.g. "Research X", "Draft section Z", "Implement component Q").\n'
            "- **related_tasks**: the raw IDs of earlier tasks this task depends on. "
            "Leave empty only if it truly depends on nothing.\n"
            f"- **Constraint**: produce between 1 and {max_tasks} tasks (1 only for a "
            "genuinely trivial, single-step request). Return ONLY the raw JSON object - "
            "no prose, no explanations, no greetings, no <think> blocks. Do NOT prefix "
            "task_id values with colons (:) or @ symbols.\n"
        )

    @staticmethod
    def execution_prompt(base_system: str) -> str:
        return (
            f"{base_system}\n\n"
            "### EXECUTION PHASE - ACTIVE\n"
            "You are executing ONE task of a larger plan. Produce the complete, "
            "high-quality output for THIS task only.\n"
            "- Use any prerequisite task results provided as context.\n"
            "- Do not restate the whole plan; focus on delivering this task's deliverable.\n"
            "- Be precise and self-contained: downstream tasks may consume your output "
            "verbatim.\n"
        )

    @staticmethod
    def synthesis_prompt(base_system: str) -> str:
        return (
            f"{base_system}\n\n"
            "### SYNTHESIS PHASE - ACTIVE\n"
            "All tasks are finished. Produce a single clean, professional final "
            "response that fulfills the user's original request.\n"
            "- Integrate the task results into a coherent whole.\n"
            "- Do not include planner scaffolding, task IDs as headers, or meta "
            "commentary.\n"
        )


# ---------------------------------------------------------------------------
# JSON extraction (same recovery strategy as planning_standalone.py)
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (closed, or dangling - reasoning that
    never closed before the answer)."""
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
    """Return the brace-balanced {...} substring beginning at `start`, or None."""
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
    """Best-effort extraction of a JSON object in a string: strips thinking-model
    reasoning and code fences, then tries a direct parse, a brace-balanced scan at
    every `{` (preferring an object with `tasks`), and finally a greedy first-to-last
    brace span."""
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
    if fallback is not None:
        return fallback

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Planning core (transport-agnostic; takes an async completion function)
# ---------------------------------------------------------------------------


async def make_plan(
    complete: CompletionFn,
    goal: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tasks: int = 20,
    temperature: float = 0.3,
) -> PlanResult:
    """Ask the model to decompose `goal` into a plan. Never executes anything - the
    returned tasks are a description of what would need to be done, not a result."""
    raw = await complete(
        PlanPromptBuilder.planning_prompt(system_prompt, max_tasks),
        f"User request:\n{goal}",
        temperature,
        True,
    )
    parsed = _extract_json_object(raw)
    tasks: list[Task] = []
    if parsed and isinstance(parsed.get("tasks"), list):
        for idx, item in enumerate(parsed["tasks"][:max_tasks], 1):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            task_id = str(item.get("task_id") or f"task_{idx}").strip()
            related = [
                str(r).strip()
                for r in (item.get("related_tasks") or [])
                if str(r).strip()
            ]
            tasks.append(
                Task(task_id=task_id, description=description, related_tasks=related)
            )
    if not tasks:
        tasks = [Task(task_id="task_1", description=goal)]
    return PlanResult(goal=goal, tasks=tasks)


def format_plan(result: PlanResult) -> str:
    """Render a plan as markdown: one numbered item per task, with its dependencies.
    Not used by `run()`/the Pipe/the CLI by default - the plan itself stays internal
    there. Exported for callers who want the decomposition on its own."""
    lines = [f"**Plan for:** {result.goal}\n"]
    for i, task in enumerate(result.tasks, 1):
        dep = (
            f" _(depends on: {', '.join(task.related_tasks)})_"
            if task.related_tasks
            else ""
        )
        lines.append(f"{i}. **{task.task_id}** — {task.description}{dep}")
    return "\n".join(lines)


async def execute_task(
    complete: CompletionFn,
    goal: str,
    task: Task,
    done: dict[str, Task],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
) -> str:
    """Run one task with a single completion call: no tools, no retries, no loop
    detection - the model's first answer is the result. Prior completed tasks this one
    depends on (or, absent explicit dependencies, every task done so far) are given as
    context."""
    deps = (
        [done[t] for t in task.related_tasks if t in done]
        if task.related_tasks
        else list(done.values())
    )
    context_blocks = [
        f"--- Result of {dep.task_id} ({dep.description}) ---\n{dep.result}"
        for dep in deps
        if dep.status == "completed" and dep.result
    ]
    context = "\n\n".join(context_blocks) if context_blocks else "(no prior results)"
    user_message = (
        f"Original user request:\n{goal}\n\n"
        f"Prerequisite task results:\n{context}\n\n"
        f"YOUR TASK ({task.task_id}):\n{task.description}\n\n"
        "Produce the complete deliverable for this task now."
    )
    return await complete(
        PlanPromptBuilder.execution_prompt(system_prompt),
        user_message,
        temperature,
        False,
    )


async def synthesize(
    complete: CompletionFn,
    goal: str,
    tasks: list[Task],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
) -> str:
    """Merge every task's result into one final answer."""
    results_block = "\n\n".join(
        f"--- {t.task_id} ({t.status}) ---\n{t.result}" for t in tasks
    )
    user_message = (
        f"Original user request:\n{goal}\n\n"
        f"Task results:\n{results_block}\n\n"
        "Write the final response now."
    )
    return await complete(
        PlanPromptBuilder.synthesis_prompt(system_prompt),
        user_message,
        temperature,
        False,
    )


async def run(
    complete: CompletionFn,
    goal: str,
    progress: Optional[ProgressFn] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tasks: int = 20,
    planning_temperature: float = 0.3,
    execution_temperature: float = 0.7,
) -> RunResult:
    """Plan, execute every task, and synthesize a final answer. The plan is only ever
    used internally to structure the work - callers get back the executed result, never
    the raw task list.

    Status is deliberately minimal: one line while it works, one line when it's done -
    like Open WebUI's own collapsed "thought for Ns" summary for a native reasoning
    model, not a play-by-play of every task. Emitting one status per task (what an
    earlier version of this did) just recreates the wall-of-text a reasoning model's
    raw thinking would produce, which is exactly what a clean summary line replaces.
    """
    started = time.monotonic()

    async def emit(message: str) -> None:
        if progress is not None:
            await progress(message)

    await emit("Working...")
    plan = await make_plan(complete, goal, system_prompt, max_tasks, planning_temperature)
    tasks = plan.tasks

    done: dict[str, Task] = {}
    for task in tasks:
        try:
            task.result = await execute_task(
                complete, goal, task, done, system_prompt, execution_temperature
            )
            task.status = "completed"
        except Exception as exc:  # keep going; record the failure
            task.status = "failed"
            task.result = f"[task failed: {exc}]"
        done[task.task_id] = task

    if len(tasks) == 1 and tasks[0].status == "completed" and tasks[0].result.strip():
        return RunResult(
            goal=goal,
            tasks=tasks,
            final_output=tasks[0].result,
            elapsed_seconds=time.monotonic() - started,
        )

    final_output = await synthesize(
        complete, goal, tasks, system_prompt, execution_temperature
    )
    return RunResult(
        goal=goal,
        tasks=tasks,
        final_output=final_output,
        elapsed_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# Open WebUI helpers (response-shape recovery, same as planning_standalone.py)
# ---------------------------------------------------------------------------


def _owui_response_error(response: Any) -> str:
    """Return an error message if the OWUI response is an error payload, else ''."""
    data: Any = None
    if isinstance(response, dict):
        data = response
    else:
        body = getattr(response, "body", None)
        if body is not None:
            try:
                data = json.loads(
                    body.decode("utf-8", "replace")
                    if isinstance(body, (bytes, bytearray))
                    else body
                )
            except Exception:
                return ""
    if isinstance(data, dict):
        err = data.get("error") or data.get("detail")
        if err and not data.get("choices"):
            return err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
    return ""


def _owui_debug_body(response: Any) -> str:
    if isinstance(response, (dict, list)):
        try:
            return json.dumps(response, ensure_ascii=False)
        except Exception:
            return str(response)
    if isinstance(response, str):
        return response
    body = getattr(response, "body", None)
    if body is not None:
        try:
            return (
                body.decode("utf-8", "replace")
                if isinstance(body, (bytes, bytearray))
                else str(body)
            )
        except Exception:
            return repr(body)
    return repr(response)


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
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict) and delta.get("content"):
                return delta["content"]
            if message.get("reasoning_content"):
                return message["reasoning_content"]
        if choices[0].get("text"):
            return choices[0]["text"]
    if data.get("content"):
        return data["content"]
    return ""


def _owui_extract_content(response: Any) -> str:
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


# ---------------------------------------------------------------------------
# Open WebUI Pipe
# ---------------------------------------------------------------------------


class Pipe:
    """Open WebUI Pipe entrypoint. Appears as a selectable model. Only ever produces a
    plan (task list) - it never executes a task or synthesizes an answer."""

    class Valves(BaseModel):
        PLANNER_MODEL: str = Field(
            default="",
            description="The real model id the planner drives (e.g. gpt-4o, llama3.1). Leave blank to auto-select an available model from Open WebUI.",
        )
        TEMPERATURE: float = Field(
            default=0.3,
            description="Sampling temperature for planning. Kept low by default for deterministic, parseable plans.",
        )
        MAX_TASKS: int = Field(default=20, description="Max tasks in the plan.")
        EMIT_STATUS: bool = Field(
            default=True, description="Emit progress as status updates in the UI."
        )
        SYSTEM_PROMPT: str = Field(
            default=DEFAULT_SYSTEM_PROMPT, description="Base system prompt."
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "planner-lite", "name": "Planner (Lite)"}]

    def _resolve_model(self, request: Any, body: dict) -> str:
        """Pick the model the planner should drive: explicit PLANNER_MODEL valve ->
        Open WebUI's configured default model(s) -> the first available real
        (non-pipe/non-arena) model."""
        if self.valves.PLANNER_MODEL.strip():
            return self.valves.PLANNER_MODEL.strip()

        models: dict = {}
        try:
            models = getattr(request.app.state, "MODELS", {}) or {}
        except Exception:
            models = {}

        def _is_usable(mid: str, info: dict) -> bool:
            if not isinstance(info, dict):
                return False
            if info.get("pipe") or info.get("arena"):
                return False
            if "planner-lite" in str(mid):
                return False
            return True

        try:
            defaults = getattr(request.app.state.config, "DEFAULT_MODELS", "") or ""
            for mid in [m.strip() for m in defaults.split(",") if m.strip()]:
                if mid in models and _is_usable(mid, models.get(mid, {})):
                    return mid
        except Exception:
            pass

        for mid, info in models.items():
            if _is_usable(mid, info):
                return mid
        return ""

    @staticmethod
    def _extract_goal(body: dict) -> str:
        for message in reversed(body.get("messages", []) or []):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):  # multimodal content parts
                    content = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if content and content.strip():
                    return content.strip()
        return ""

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Any = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs: Any,
    ) -> str:
        # Lazy OWUI imports so this module also loads outside Open WebUI (CLI).
        from open_webui.utils.chat import generate_chat_completion
        from open_webui.models.users import Users

        valves = self.valves
        model_id = self._resolve_model(__request__, body)
        if not model_id:
            return (
                "⚠️ Planner could not find a model to drive. Set the **PLANNER_MODEL** "
                "valve to a real model id (e.g. `gpt-4o`)."
            )

        goal = self._extract_goal(body)
        if not goal:
            return "⚠️ No user message found to plan from."

        user = None
        if __user__ and __user__.get("id"):
            user = Users.get_user_by_id(__user__["id"])
            if asyncio.iscoroutine(user):
                user = await user

        async def progress(message: str) -> None:
            if __event_emitter__ and valves.EMIT_STATUS:
                await __event_emitter__(
                    {"type": "status", "data": {"description": message, "done": False}}
                )

        async def complete(
            system_prompt: str,
            user_message: str,
            temperature: Optional[float] = None,
            json_object: bool = False,
        ) -> str:
            base_form: dict = {
                "model": model_id,
                "stream": False,
                "temperature": (
                    temperature if temperature is not None else valves.TEMPERATURE
                ),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }
            # Degrade across backends: json_schema (strict, no reasoning leakage) ->
            # json_object (older JSON mode) -> plain (recovered by _extract_json_object).
            attempts: list[dict] = []
            if json_object:
                attempts.append(
                    {
                        **base_form,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": PLAN_JSON_SCHEMA,
                        },
                    }
                )
                attempts.append(
                    {**base_form, "response_format": {"type": "json_object"}}
                )
            attempts.append(base_form)

            last_response: Any = None
            for form_data in attempts:
                response = await generate_chat_completion(
                    __request__, form_data, user=user
                )
                last_response = response
                if _owui_response_error(response):
                    continue
                content = _owui_extract_content(response)
                if content:
                    return content

            raise RuntimeError(
                "Planner model returned an empty response. "
                f"Raw response ({type(last_response).__name__}): "
                f"{_owui_debug_body(last_response)[:1200]}"
            )

        if not valves.PLANNER_MODEL.strip():
            await progress(f"Using model: {model_id}")

        try:
            result = await run(
                complete,
                goal,
                progress=progress,
                system_prompt=valves.SYSTEM_PROMPT,
                max_tasks=valves.MAX_TASKS,
                planning_temperature=min(valves.TEMPERATURE, 0.4),
                execution_temperature=valves.TEMPERATURE,
            )
        except Exception as exc:
            if __event_emitter__ and valves.EMIT_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {exc}", "done": True},
                    }
                )
            return f"❌ Planner error: {exc}"

        if __event_emitter__ and valves.EMIT_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Done ({len(result.tasks)} task(s), {result.elapsed_seconds:.1f}s)",
                        "done": True,
                    },
                }
            )
        return result.final_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lite planner: plans, executes every task, and prints the synthesized answer (the plan itself stays internal)."
    )
    parser.add_argument("goal", nargs="*", help="The request/goal to fulfill.")
    parser.add_argument(
        "--model", default=None, help="Model id (default: $PLANNER_MODEL or gpt-4o-mini)"
    )
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key", default=None, help="API key (default: $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Execution/synthesis temperature"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Max tasks in the (internal) plan"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full result (including each task's own output) as JSON"
    )
    parser.add_argument(
        "--system-prompt", default=None, help="Override the base system prompt"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging"
    )
    return parser


async def _run_cli(args: argparse.Namespace, goal: str) -> int:
    from openai import OpenAI  # lazy import; only needed for the CLI path

    model = args.model or os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    api_url = args.api_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key or "sk-noauth", base_url=api_url)

    async def complete(
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        json_object: bool = False,
    ) -> str:
        def _call() -> str:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature if temperature is not None else 0.3,
            }
            try:
                if json_object:
                    resp = client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs
                    )
                else:
                    resp = client.chat.completions.create(**kwargs)
            except Exception:
                resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned an empty response")
            return content

        return await asyncio.to_thread(_call)

    async def progress(message: str) -> None:
        if not args.quiet:
            print(f"[planner] {message}", file=sys.stderr, flush=True)

    temperature = args.temperature if args.temperature is not None else 0.7
    try:
        result = await run(
            complete,
            goal,
            progress=progress,
            system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
            max_tasks=args.max_tasks or 20,
            planning_temperature=min(temperature, 0.4),
            execution_temperature=temperature,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "goal": result.goal,
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "description": t.description,
                            "related_tasks": t.related_tasks,
                            "status": t.status,
                            "result": t.result,
                        }
                        for t in result.tasks
                    ],
                    "final_output": result.final_output,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(result.final_output)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    goal = " ".join(args.goal).strip() or sys.stdin.read().strip()
    if not goal:
        print(
            "error: no goal provided (pass as arguments or via stdin)", file=sys.stderr
        )
        return 2
    return asyncio.run(_run_cli(args, goal))


if __name__ == "__main__":
    raise SystemExit(main())
