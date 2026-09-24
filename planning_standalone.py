"""
title: Planner (Standalone)
author: adapted from Planner v3 by Haervwe
version: 1.0.0
required_open_webui_version: 0.5.0
description: A standalone, single-agent adaptation of "Planner v3". It plans, executes each task itself (NO subagents), and synthesizes a final answer. Loads as an Open WebUI Pipe and also runs as a CLI.

This strips out the multi-agent / subagent orchestration, UI rendering, MCP,
terminal, skills and file handling. What remains is the core agentic loop:

    plan       -> decompose the request into atomic tasks
    execute    -> run each task with a single LLM (no subagents)
    synthesize -> merge task results into a final answer (with @task_id macros)

Two ways to run:

1. Open WebUI Pipe (this file defines a `Pipe` class):
   - Add it as a Function in Open WebUI. It appears as a model "Planner (Standalone)".
   - Set the `PLANNER_MODEL` valve to the real model id the planner should drive.
   - Uses Open WebUI's native chat completion (no extra API key needed).

2. CLI (uses the `openai` SDK against any OpenAI-compatible endpoint):
       python planning_standalone.py "Write a market analysis of EV charging in the EU"
       python planning_standalone.py --model gpt-4o-mini --no-plan "Summarize X"
   Env defaults: OPENAI_API_KEY, OPENAI_BASE_URL, PLANNER_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

from pydantic import BaseModel, Field

# Type of the injected completion function:
#   complete(system_prompt, user_message, temperature, json_object, params) -> text
# `params` is an optional dict of extra sampling parameters (top_p, max_tokens,
# top_k, min_p, repeat_penalty, frequency_penalty, presence_penalty, seed, ...).
CompletionFn = Callable[
    [str, str, Optional[float], bool, Optional[dict]], Awaitable[str]
]
# Tool-aware chat function for the execution loop:
#   chat(messages, tools, temperature, params) -> assistant message dict
# The returned dict has keys: "content" (str|None) and optional "tool_calls"
# (list of {"id", "type", "function": {"name", "arguments"}}).
ChatFn = Callable[
    [list, Optional[list], Optional[float], Optional[dict]], Awaitable[dict]
]
ProgressFn = Callable[[str], Awaitable[None]]


DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced agentic Planner. You have the ability to formulate a "
    "plan, act on it by executing each task yourself step by step, and track "
    "your progress.\nYour goal is to fulfill the user's request thoroughly and "
    "professionally.\n\n"
    "CRITICAL INSTRUCTIONS FOR ACCURACY:\n"
    "1. ALWAYS use complete, unabbreviated words. Write 'Application Programming Interface' "
    "instead of 'API'. Write 'HyperText Transfer Protocol Secure' instead of 'HTTPS'.\n"
    "2. ALWAYS use precise dates and times. Today is 2026-06-21. Use YYYY-MM-DD format.\n"
    "3. NEVER abbreviate terms like 'etc', 'etc.', 'e.g.', 'i.e.' — write out the full meaning.\n"
    "4. When uncertain about time, ask for clarification rather than guess.\n"
    "5. Expand all technical acronyms on first use: write 'Secure Shell (SSH)' not just 'SSH'."
)


# JSON Schema for the planning output. Used with LM Studio / OpenAI structured
# outputs (`response_format: {"type": "json_schema", ...}`) so planning returns
# guaranteed-valid JSON with no reasoning leakage.
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
                        # Names of tools (from the provided catalog) this task
                        # needs. Empty when the task needs no tools.
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "task_id",
                        "description",
                        "related_tasks",
                        "tools",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Configuration & data containers
# ---------------------------------------------------------------------------


@dataclass
class ToolMetrics:
    """Track performance of individual tools."""

    name: str
    call_count: int = 0
    loop_count: int = 0  # times this tool contributed to loop
    error_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.call_count or 1
        return self.success_count / total


@dataclass
class PlannerConfig:
    """Standalone equivalent of the Planner Valves (only what is relevant)."""

    model: str = field(
        default_factory=lambda: os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    )
    api_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    temperature: float = 0.7
    # Per-phase temperature overrides. None => derive from `temperature`
    # (planning is additionally capped low for deterministic, parseable plans).
    planning_temperature: Optional[float] = None
    execution_temperature: Optional[float] = None
    synthesis_temperature: Optional[float] = None
    # Extra sampling params (top_p, max_tokens, top_k, min_p, repeat_penalty,
    # frequency_penalty, presence_penalty, seed, ...). `sampling_params` applies
    # to every phase; the per-phase dicts merge on top (overriding the base).
    sampling_params: dict = field(default_factory=dict)
    planning_params: dict = field(default_factory=dict)
    execution_params: dict = field(default_factory=dict)
    synthesis_params: dict = field(default_factory=dict)
    plan_mode: bool = True
    # When plan_mode is on, still skip planning and answer in a single pass for
    # short/simple requests (heuristic below) instead of always decomposing.
    auto_skip_plan_for_short_tasks: bool = True
    short_task_max_words: int = 20
    max_tasks: int = 20
    task_result_limit: int = 30000
    enable_review: bool = False
    request_timeout: float = 120.0
    verbose: bool = True
    # MCP tool calling (execution phase only). Each server is a dict:
    #   {"name": "memory", "transport": "streamable-http", "url": "http://host:8082/memory"}
    #   {"name": "x", "transport": "stdio", "command": "python", "args": [...], "env": {...}}
    mcp_servers: list = field(default_factory=list)
    max_tool_iterations: int = 6
    tool_result_limit: int = 4000
    # How long a server's discovered tool list (and reachability) stays cached across
    # requests. Every message would otherwise pay a fresh TCP check + MCP handshake even
    # when no tool ends up being called; 0 disables the cache and re-discovers every time.
    mcp_discovery_cache_ttl: float = 300.0
    # Per-tool description length fed to the model as part of the `tools` schema. A
    # server with many tools (each with a paragraph-long description) otherwise adds
    # thousands of prompt tokens to every request that carries tools, whether or not
    # the request needs any of them.
    mcp_tool_description_limit: int = 200
    # When Kev hasn't classified whether tools are needed (kev_ask_tools off, or no Kev
    # at all), attaching every tool's schema to every request is expensive for a server
    # with many tools. This gates on a cheap heuristic instead of always attaching them.
    mcp_tool_gate_heuristic: bool = True
    # Loop detection thresholds per phase
    plan_loop_threshold: int = 2
    execution_loop_threshold: int = 3
    synthesis_loop_threshold: int = 2
    # Token budget awareness
    max_tokens_per_task: int = 400000
    warn_at_percent: float = 0.8
    # Resume from memory
    enable_memory_resume: bool = True
    # The task organizer (the decomposed plan and which tasks are done) persists across
    # messages in the same chat, so a follow-up message continues the running plan
    # instead of re-planning from scratch; task execution itself stays one-shot per task.
    chat_plan_persist: bool = True
    chat_plan_max_chats: int = 50
    chat_plan_ttl: float = 3600.0
    # Kev (System One) decisions. Empty URL = off, and every call falls back to the
    # behaviour below it. See KevClient for why these three calls and not others.
    kev_url: str = field(
        default_factory=lambda: os.getenv("KEV_URL", "http://10.0.0.10:8009")
    )
    kev_api_key: str = field(default_factory=lambda: os.getenv("KEV_API_KEY", ""))
    kev_timeout: float = 30.0
    kev_state_limit: int = (
        4000  # characters of a task result or draft Kev is asked to judge
    )
    # Kev reads the request once, before anything is generated, and answers three questions in one pass:
    #   1. is it simple?          -> answer in one pass instead of decomposing (replaces the word-count heuristic)
    #   2. does it need thinking? -> no: the model's reasoning is switched off for every phase ("think": false)
    #   3. artistic or scientific -> sets the base temperature (explicit per-phase temperatures still win)
    kev_decide_planning: bool = True
    kev_simple_threshold: float = 0.4
    kev_classify_request: bool = True
    kev_think_threshold: float = 0.5
    kev_artistic_threshold: float = 0.5
    kev_artistic_temperature: float = 1.0
    kev_scientific_temperature: float = 0.3
    # Offer the MCP tools only when the request needs them (a tool-calling loop is a generation per round).
    kev_ask_tools: bool = True
    kev_tools_threshold: float = 0.5
    # Fast paths. With Kev served through Ollama every question is its own ~0.55 s call (they are neither batched
    # nor faster concurrently, measured 2026-09-22), so the profile asks only what can still change the run:
    # greetings skip Kev entirely, a repeated request (regenerate) reuses its answers, the thinking question is not
    # asked of an artistic request (poem 0.004, fantasy 0.028, brainstorm 0.013) or of a model that cannot think,
    # the tools question only when tools exist, the temperature question only when a phase would use it.
    fast_small_talk: bool = True
    small_talk_max_words: int = 6
    kev_cache_size: int = 128
    model_can_think: Optional[bool] = (
        None  # None = unknown (ask); the pipe reads it from Ollama
    )
    # With thinking on, only the execution phase thinks: the plan is JSON and the synthesis merges finished results,
    # so reasoning there is tokens spent before the first useful one.
    think_in_planning: bool = False
    # A plan of one task already produced the answer; the synthesis pass would only rewrite it.
    skip_single_task_synthesis: bool = True
    # Tell the one-pass answer to a simple request to skip preamble and restating the question.
    direct_answer_hint: bool = True
    # After each task, ask whether the result carries out the task; retry if not.
    kev_check_tasks: bool = True
    kev_accept_threshold: float = 0.25
    # Run the review pass only when the draft needs it (overrides enable_review when Kev answers).
    kev_gate_review: bool = True
    kev_review_threshold: float = 0.05
    # The two low thresholds are deliberate, and measured rather than guessed (qwen3.8-27B IQ2_M through Ollama,
    # 2026-09-22). A plain base model's probabilities are uncalibrated - the ordering is reliable, the level moves with
    # the wording - and both errors here are asymmetric: a wrong "retry" or a wrong "review" costs a whole generation,
    # while a wrong "accept" only leaves the behaviour the planner had before Kev. So both act only on a confident no.
    #   task check    good result 0.791 / empty waffle 0.000                  -> 0.25 separates with room either side
    #   review gate   complete 0.998 and 0.316 / vague 0.009 / partial 0.000  -> 0.05 reviews only the clear failures
    #   simple        capital city 0.985, greeting 0.933, sky 0.607 / train sum 0.485, proof 0.093,
    #                 four-part analysis 0.000                                 -> 0.4 answers a one-shot sum in one pass
    #   think         proof 0.906, train sum 0.982, code fix 0.703 / analysis 0.184, poem 0.004, capital 0.012 -> 0.5
    #   artistic      poem, fantasy world, brainstorm 1.000 / greeting 0.100, everything factual <= 0.043     -> 0.5
    #   tools         weather now, exchange rate, web search, "what did I tell you" 0.964-0.998 /
    #                 capital, poem, proof, code fix <= 0.034                                                -> 0.5
    # Re-measure these against any other model before trusting them.


@dataclass
class Task:
    task_id: str
    description: str
    related_tasks: list[str] = field(default_factory=list)
    # Tool names (chosen by the planner) this task is allowed to call.
    tools: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | completed | failed
    result: str = ""


@dataclass
class PlannerResult:
    goal: str
    tasks: list[Task]
    final_output: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Prompt building (adapted from Planner v3 PromptBuilder, subagents removed)
# ---------------------------------------------------------------------------


class PromptBuilder:
    @staticmethod
    def planning_prompt(
        base_system: str, max_tasks: int, tools_catalog: str = ""
    ) -> str:
        if tools_catalog:
            tools_section = (
                "\n### AVAILABLE TOOLS\n"
                "These tools can be called during task execution:\n"
                f"{tools_catalog}\n"
                "- For EACH task, set its `tools` field to the subset of the tool "
                "names above that the task actually needs to call. Use the EXACT "
                "names as written. Leave `tools` empty (`[]`) for tasks that need "
                "no tools (pure reasoning, drafting, analysis).\n"
                "- Be selective: only include a tool when the task genuinely "
                "requires it. Do not attach tools to tasks that just synthesize "
                "or transform prior results.\n"
            )
        else:
            tools_section = (
                "\n### TOOLS\n"
                "No tools are available. Set every task's `tools` field to `[]`.\n"
            )
        return (
            f"{base_system}\n\n"
            "### PLANNING PHASE - ACTIVE\n"
            "Analyze the request and decompose it into a series of logical, "
            "executable tasks that YOU (a single capable agent) will perform "
            "one after another.\n"
            "- **Goal**: Create a step-by-step roadmap to fulfill the user's core objective.\n"
            "- **Output Schema**: Return STRICTLY a JSON object: "
            '`{"tasks": [{"task_id": "task_1_research", "description": "...", '
            '"related_tasks": ["task_id", ...], "tools": ["tool_name", ...]}, ...]}`.\n'
            "- **Decompose aggressively**: Break the request into the SMALLEST "
            "independently-executable steps that still produce a meaningful "
            "deliverable. Prefer MORE small tasks over a few large ones. A good "
            "plan usually has several tasks; a single all-in-one task is almost "
            "always wrong unless the request is genuinely trivial.\n"
            "- **Separate concerns**: Split distinct deliverables, components, "
            "files, sections, or analysis steps into their own tasks (e.g. for a "
            "web app: data model, each component, routing, styling, wiring, README "
            "— as separate tasks).\n"
            "- **Task Granularity**: Each task must be an atomic, actionable step "
            '(e.g. "Research X", "Analyze the results of task_1_research to do Y", '
            '"Draft section Z", "Implement component Q").\n'
            "- **related_tasks**: List the raw IDs of earlier tasks whose output "
            "this task depends on. Leave empty ONLY if it truly depends on nothing "
            "(independent tasks may run in parallel later, so keep dependencies "
            "minimal and accurate).\n"
            f"- **Constraint**: Produce between 3 and {max_tasks} tasks (use fewer "
            "only for genuinely trivial requests). Return ONLY the raw JSON object. "
            "NO prose, NO explanations, NO greetings, NO <think> blocks. Do NOT "
            "prefix task_id values with colons (:) or @ symbols.\n"
            f"{tools_section}"
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
            "- If the task involves code, provide complete, runnable code.\n"
            "- Be precise and self-contained: downstream tasks may consume your output verbatim.\n"
        )

    @staticmethod
    def synthesis_prompt(base_system: str) -> str:
        return (
            f"{base_system}\n\n"
            "### SYNTHESIS PHASE - ACTIVE\n"
            "All tasks are finished. Produce a single clean, professional final "
            "response that fulfills the user's original request.\n"
            "- Integrate the task results into a coherent whole.\n"
            "- You may reference a task's full output by writing `@task_id`; it will be "
            "substituted verbatim. Use this to avoid re-typing large code blocks or reports.\n"
            "- Do not include planner scaffolding, task IDs as headers, or meta commentary.\n"
        )

    @staticmethod
    def review_prompt(base_system: str) -> str:
        return (
            f"{base_system}\n\n"
            "### REVIEW PHASE - ACTIVE\n"
            "Critically review the draft final answer against the user's original "
            "request. Fix gaps, errors, and missing requirements, then return the "
            "IMPROVED final answer only (no critique, no preamble)."
        )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models.

    Handles a closed block and the common case where only an opening <think>
    is present (reasoning never closed before the answer).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Dangling opening tag with no close: drop everything up to it.
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
    """Best-effort extraction of a JSON object in a string.

    Strips thinking-model reasoning and code fences, then tries a direct parse,
    a brace-balanced scan at every `{` (preferring an object with `tasks`), and
    finally a greedy first-to-last brace span.
    """
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
            return obj  # the object we actually want
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
# MCP client (optional; enables tool calling during execution)
# ---------------------------------------------------------------------------


def _mcp_result_to_text(result: Any, limit: int = 4000) -> str:
    """Flatten an MCP CallToolResult into a string for the model."""
    # Prefer textual content parts; fall back to structured content / repr.
    parts: list[str] = []
    content = getattr(result, "content", None)
    if content:
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
                continue
            # Non-text content (image/resource): describe it briefly.
            itype = getattr(item, "type", None) or type(item).__name__
            parts.append(f"[{itype} content omitted]")
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                parts.append(json.dumps(structured, ensure_ascii=False, default=str))
            except Exception:
                parts.append(str(structured))
    text = "\n".join(parts) if parts else "(tool returned no content)"
    if getattr(result, "isError", False):
        text = f"[tool error] {text}"
    if len(text) > limit:
        text = text[:limit] + f"\n...[truncated {len(text) - limit} chars]..."
    return text


# ---------------------------------------------------------------------------
# Kev: typed decisions instead of generated judgement
# ---------------------------------------------------------------------------


# Kev's answers per (endpoint, request), so a regenerate or a resent message pays nothing for its profile.
_KEV_PROFILE_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()


@dataclass
class ChatPlanState:
    """The running plan for one chat: every task decomposed so far across turns, and
    which are done. The organizer (this state) persists across messages in the same
    chat; the workers that execute each task stay one-shot, created fresh per task as
    before - only the plan itself accumulates."""

    turn: int = 0
    tasks: list = field(default_factory=list)  # list[Task], every turn's tasks appended
    done: dict = field(default_factory=dict)  # task_id -> Task, across all turns
    updated_at: float = field(default_factory=time.monotonic)


# chat_id -> ChatPlanState, so a multi-turn chat's plan survives across pipe() calls
# (each of which builds a brand-new StandalonePlanner). LRU-evicted like the Kev cache.
_CHAT_PLAN_STORE: "OrderedDict[str, ChatPlanState]" = OrderedDict()

# Every word of the message has to be one of these for it to count as small talk: "hi, what is 2+2" is a question.
_SMALL_TALK_WORDS = frozenset("""
    hi hey hello hallo servus moin griaß grüß gruess gott di dich good morning afternoon evening night guten morgen
    abend tag thanks thank you thx ty danke dank vielen schön schoen sehr very much ok okay k cool nice great super
    perfect perfekt passt alright fine bye tschüss tschuess ciao baba how are wie geht's gehts es dir there all
    everyone zusammen leute mate
""".split())


class KevClient:
    """Yes/no decisions from a Kev System One endpoint (POST /v1/systemone).

    Three of the planner's judgement calls are not writing tasks at all - is this request worth decomposing, did this
    task actually produce what it was asked for, does the draft need another pass. Asking the planning model costs a
    full generation and comes back as prose to be parsed; Kev scores the two options against the model's next-token
    logits and answers in about 0.2 s with a probability, using the same weights the chat model is already holding.

    Every call is fail-open: a missing URL, an unreachable endpoint or a malformed answer returns None, and the caller
    keeps the behaviour it had before Kev existed. The probabilities are uncalibrated (no fitted temperature exists for
    a plain base model), so they are used as thresholds to choose between two code paths, never reported as a number
    the user should trust.
    """

    def __init__(self, url: str = "", api_key: str = "", timeout: float = 30.0):
        self.url = (url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.calls = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def noul(
        self, state: str, question: str, criteria: Optional[dict] = None
    ) -> Optional[float]:
        """p(true) for one yes/no question about `state`, or None when Kev did not answer."""
        answers = await self.nouls(state, {"q": (question, criteria)})
        return answers.get("q") if answers else None

    async def nouls(self, state: str, questions: dict) -> Optional[dict]:
        """p(true) for several yes/no questions about the same `state` in one request ({id: (question, criteria)}),
        or None when Kev did not answer. One prefill of the state serves every question.
        """
        if not self.enabled or not state.strip() or not questions:
            return None
        payload = {
            "state": state,
            "model": "kev-latest",
            "questions": {
                qid: {
                    "type": "noul",
                    "instructions": question,
                    **({"criteria": criteria} if criteria else {}),
                }
                for qid, (question, criteria) in questions.items()
            },
        }
        try:
            body = await asyncio.to_thread(self._post, payload)
            self.calls += 1
            return {qid: float(body["answers"][qid]["noul"]) for qid in questions}
        except Exception:  # noqa: BLE001 - never let a decision service break the run
            self.failures += 1
            return None

    def _post(self, payload: dict) -> dict:
        import urllib.request

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


def _check_reachable(url: str, timeout: float = 2.0) -> None:
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection(
            (parts.hostname or "localhost", port), timeout=timeout
        ).close()
    except OSError as exc:
        raise ConnectionError(f"{parts.hostname}:{port} not reachable ({exc})") from exc


# Discovered tool list (and reachability) per server config, so a message that never
# ends up calling a tool doesn't pay a fresh TCP check + MCP handshake on every turn.
# name -> (discovered_at, tools_dict | None, note | None); None tools_dict = server was
# unreachable at discovery time, cached so repeat messages don't re-probe it either.
_MCP_DISCOVERY_CACHE: dict[str, tuple[float, Optional[dict], Optional[str]]] = {}


def _cached_mcp_tool_names(servers: list[dict], ttl: float) -> Optional[set[str]]:
    """Tool names already known for `servers` from a previous connection, without opening
    one now - or None if any server's entry is missing or stale, meaning the caller has
    to connect to find out. Lets a decision like "is this worth connecting for" be made
    before paying for a connection at all."""
    now = time.monotonic()
    names: set[str] = set()
    for srv in servers:
        name = srv.get("name") or srv.get("url") or "mcp"
        cached = _MCP_DISCOVERY_CACHE.get(name)
        if not cached or now - cached[0] >= ttl:
            return None
        _, tools, _ = cached
        if tools:
            names.update(tools.keys())
    return names


class MCPClient:
    """Connects to one or more MCP servers and exposes their tools.

    Usage:
        async with MCPClient(servers) as mcp:
            tools = mcp.openai_tools()
            text = await mcp.call_tool(name, args)

    Gracefully degrades: if the `mcp` package is missing or a server cannot be
    reached, the affected server is skipped (with a note) rather than aborting.

    Tool discovery (reachability + list_tools) is cached across requests for
    `discovery_cache_ttl` seconds. A cache hit costs nothing: the actual server
    connection is opened lazily, only if a tool call is made this run.
    """

    def __init__(
        self,
        servers: list[dict],
        result_limit: int = 4000,
        discovery_cache_ttl: float = 300.0,
        tool_description_limit: int = 200,
    ):
        self.servers = servers or []
        self.result_limit = result_limit
        self.discovery_cache_ttl = discovery_cache_ttl
        self.tool_description_limit = tool_description_limit
        self._stack: Any = None
        self.sessions: dict[str, Any] = {}
        # tool_name -> (server_name, openai_tool_schema)
        self.tools: dict[str, tuple] = {}
        # server_name -> config, for servers whose tools came from cache and have not
        # been connected yet in this run (lazily connected on the first call_tool).
        self._unconnected: dict[str, dict] = {}
        self.notes: list[str] = []

    async def __aenter__(self) -> "MCPClient":
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        now = time.monotonic()
        for srv in self.servers:
            name = srv.get("name") or srv.get("url") or "mcp"
            cached = _MCP_DISCOVERY_CACHE.get(name)
            if cached and now - cached[0] < self.discovery_cache_ttl:
                _, cached_tools, cached_note = cached
                if cached_tools:
                    self.tools.update(cached_tools)
                    self._unconnected[name] = srv
                elif cached_note:
                    self.notes.append(cached_note)
                continue
            try:
                session = await self._connect(srv)
                await session.initialize()
                self.sessions[name] = session
                listed = await session.list_tools()
                server_tools: dict[str, tuple] = {}
                for tool in listed.tools:
                    schema = tool.inputSchema or {"type": "object", "properties": {}}
                    server_tools[tool.name] = (
                        name,
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": " ".join(
                                    (tool.description or "").split()
                                )[: self.tool_description_limit],
                                "parameters": schema,
                            },
                        },
                    )
                self.tools.update(server_tools)
                _MCP_DISCOVERY_CACHE[name] = (now, server_tools, None)
            except Exception as exc:  # skip unreachable / misconfigured servers
                note = f"MCP server '{name}' unavailable: {exc}"
                self.notes.append(note)
                _MCP_DISCOVERY_CACHE[name] = (now, None, note)
        return self

    async def _connect(self, srv: dict) -> Any:
        from mcp import ClientSession

        transport = (srv.get("transport") or "streamable-http").lower()
        if transport in ("streamable-http", "http", "streamable_http", "sse"):
            # A refused connection inside the MCP client's task group surfaces as a CancelledError that tears down the
            # whole run instead of skipping this server. Refuse early, as an ordinary error the caller notes.
            await asyncio.to_thread(_check_reachable, srv["url"])
        if transport in ("streamable-http", "http", "streamable_http"):
            from mcp.client.streamable_http import streamablehttp_client

            ctx = await self._stack.enter_async_context(
                streamablehttp_client(srv["url"])
            )
            read, write = ctx[0], ctx[1]  # 3rd item (session-id getter) ignored
        elif transport == "sse":
            from mcp.client.sse import sse_client

            read, write = await self._stack.enter_async_context(sse_client(srv["url"]))
        else:  # stdio
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=srv["command"],
                args=srv.get("args", []),
                env=srv.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))

        session = await self._stack.enter_async_context(ClientSession(read, write))
        return session

    def openai_tools(self) -> list[dict]:
        return [schema for (_srv, schema) in self.tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self.tools.keys())

    async def call_tool(self, name: str, arguments: dict) -> str:
        entry = self.tools.get(name)
        if not entry:
            return f"[unknown tool '{name}']"
        server_name = entry[0]
        session = self.sessions.get(server_name)
        if session is None and server_name in self._unconnected:
            srv = self._unconnected.pop(server_name)
            try:
                session = await self._connect(srv)
                await session.initialize()
                self.sessions[server_name] = session
            except Exception as exc:
                _MCP_DISCOVERY_CACHE.pop(server_name, None)
                return f"[tool '{name}' server '{server_name}' connect failed: {exc}]"
        if session is None:
            return f"[tool '{name}' server '{server_name}' not connected]"
        try:
            result = await session.call_tool(name, arguments or {})
        except Exception as exc:
            return f"[tool '{name}' call failed: {exc}]"
        return _mcp_result_to_text(result, self.result_limit)

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None


class _NullMCP:
    """No-op stand-in when MCP is disabled/unconfigured."""

    notes: list = []
    tools: dict = {}

    async def __aenter__(self) -> "_NullMCP":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def openai_tools(self) -> list:
        return []

    @property
    def tool_names(self) -> list:
        return []

    async def call_tool(self, name: str, arguments: dict) -> str:
        return f"[tool '{name}' unavailable: MCP not configured]"

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Planner core (transport-agnostic; takes an async completion function)
# ---------------------------------------------------------------------------


class StandalonePlanner:
    def __init__(
        self,
        complete: CompletionFn,
        config: Optional[PlannerConfig] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        progress: Optional[ProgressFn] = None,
        chat: Optional[ChatFn] = None,
    ):
        self.complete = complete
        self.config = config or PlannerConfig()
        self.system_prompt = system_prompt
        self._progress = progress
        # Tool-aware chat backend (optional). Required for MCP tool calling.
        self.chat = chat
        # Set during run() when MCP is active.
        self._mcp: Any = None
        # Tool metrics: {tool_name: ToolMetrics}
        self._tool_metrics: dict[str, ToolMetrics] = {}
        # Track previous attempt results for divergence checking
        self._prev_attempt_results: dict[str, str] = {}
        # Typed decisions (off when no URL is configured; every call falls back)
        self.kev = KevClient(config.kev_url, config.kev_api_key, config.kev_timeout)
        # What Kev made of the request (set by _profile_request; None = no answer, keep the configured behaviour)
        self._p_simple: Optional[float] = None
        self._think: Optional[bool] = None
        self._temperature: float = self.config.temperature
        self._tools_wanted: Optional[bool] = (
            None  # False = keep the tools away from the model for this request
        )
        self._direct: bool = False  # simple request: answer without preamble
        # Set from run()'s chat_id argument; None = no chat to persist the plan against.
        self._chat_id: Optional[str] = None

    def _get_chat_plan_state(self) -> Optional[ChatPlanState]:
        """The running plan for this chat, or None if chat-plan persistence doesn't
        apply (no chat_id, or turned off)."""
        if not self.config.chat_plan_persist or not self._chat_id:
            return None
        now = time.monotonic()
        state = _CHAT_PLAN_STORE.get(self._chat_id)
        if state and now - state.updated_at >= self.config.chat_plan_ttl:
            del _CHAT_PLAN_STORE[self._chat_id]
            state = None
        if state is None:
            state = ChatPlanState()
            _CHAT_PLAN_STORE[self._chat_id] = state
        _CHAT_PLAN_STORE.move_to_end(self._chat_id)
        while len(_CHAT_PLAN_STORE) > max(0, self.config.chat_plan_max_chats):
            _CHAT_PLAN_STORE.popitem(last=False)
        return state

    def _namespace_new_tasks(self, tasks: list[Task], turn: int) -> list[Task]:
        """Prefix this turn's task ids so they can't collide with an earlier turn's ids
        in the same chat's accumulated plan, and rewrite related_tasks that pointed at
        another task from this same turn to match."""
        if turn <= 0:
            return tasks
        old_to_new = {t.task_id: f"t{turn}_{t.task_id}" for t in tasks}
        for task in tasks:
            task.task_id = old_to_new[task.task_id]
            task.related_tasks = [old_to_new.get(r, r) for r in task.related_tasks]
        return tasks

    async def _emit(self, message: str) -> None:
        if self._progress is not None:
            await self._progress(message)

    def _truncate(self, text: str) -> str:
        limit = self.config.task_result_limit
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return f"{text[:head]}\n\n...[truncated {len(text) - limit} chars]...\n\n{text[-tail:]}"

    # -- phases ------------------------------------------------------------

    def _planning_temperature(self) -> float:
        if self.config.planning_temperature is not None:
            return self.config.planning_temperature
        # Default: keep planning deterministic regardless of the base temperature.
        return min(self._temperature, 0.4)

    def _execution_temperature(self) -> float:
        if self.config.execution_temperature is not None:
            return self.config.execution_temperature
        return self._temperature

    def _synthesis_temperature(self) -> float:
        if self.config.synthesis_temperature is not None:
            return self.config.synthesis_temperature
        return self._temperature

    def _phase_params(self, phase: str) -> Optional[dict]:
        """Merge base sampling params with this phase's overrides."""
        override = {
            "planning": self.config.planning_params,
            "execution": self.config.execution_params,
            "synthesis": self.config.synthesis_params,
        }.get(phase) or {}
        merged = {**(self.config.sampling_params or {}), **override}
        think_here = self._think
        if think_here and phase != "execution" and not self.config.think_in_planning:
            think_here = False
        if think_here is False and "think" not in merged:
            # Only ever switch reasoning off: "think": true is an error on a model without it, and a thinking model
            # already thinks by default. The backends translate this key for their API.
            merged["think"] = False
        return merged or None

    def _adjusted_temperature(self, attempt: int) -> Optional[float]:
        """Increase temperature on retry to escape loops (exponential backoff)."""
        base = self._execution_temperature()
        if attempt <= 0:
            return base
        # Exponential: 1.1x, 1.3x, 1.6x → smoother curve
        multiplier = 1.0 + (0.1 * (2 ** (attempt - 1)))
        return min(base * multiplier, 2.0)

    def _adjusted_params(self, phase: str, attempt: int) -> Optional[dict]:
        """Adjust sampling params on retry: reduce top_k, increase temperature variation."""
        params = self._phase_params(phase)
        if not params or attempt <= 0:
            return params
        # On retry: increase top_p (more diversity), reduce top_k (fewer candidates)
        adjusted = dict(params) if params else {}
        if "top_k" in adjusted and isinstance(adjusted["top_k"], int):
            adjusted["top_k"] = max(1, adjusted["top_k"] // 2)  # halve top_k
        if "top_p" not in adjusted:
            adjusted["top_p"] = min(1.0, 0.9 + (attempt * 0.05))  # gradually increase
        return adjusted or None

    def _available_tool_names(self) -> list[str]:
        if self._mcp is None or self._tools_wanted is False:
            return []
        try:
            return list(self._mcp.tool_names)
        except Exception:
            return []

    async def _save_to_memory(
        self,
        task_id: str,
        description: str,
        result: str,
        attempt: int = 0,
        had_loop: bool = False,
    ) -> None:
        """Save task result to memory MCP with rich metadata."""
        if self._mcp is None or not hasattr(self._mcp, "call_tool"):
            return
        try:
            # Build labels
            labels = ["planner_task", task_id]
            if attempt > 0:
                labels.append(f"attempt_{attempt}")
            if had_loop:
                labels.append("loop_retry")

            # Try to save via memory MCP
            await self._mcp.call_tool(
                "remember",
                {
                    "text": f"Task: {description}\n\nResult:\n{result[:2000]}",
                    "labels": labels,
                    "metadata": {
                        "task_id": task_id,
                        "timestamp": time.time(),
                        "attempt": attempt,
                        "had_loop": had_loop,
                        "result_length": len(result),
                        "result_chars": len(result),
                    },
                },
            )
        except Exception as exc:
            # Silently fail; memory persistence is nice-to-have, not critical
            if self.config.verbose:
                await self._emit(f"  ⚠ memory save failed: {exc}")

    def _is_looping(self, text: str, phase: str = "execution") -> bool:
        """Detect if output shows repetition pattern (sign of looping)."""
        if not text or len(text) < 100:
            return False
        # Simple heuristic: check if the last 200 chars repeat earlier in text
        window = 200
        tail = text[-window:]
        if len(tail) < window:
            return False
        # Normalize for comparison: lowercase, compress whitespace
        tail_norm = re.sub(r"\s+", " ", tail.lower().strip())
        text_norm = re.sub(r"\s+", " ", text.lower())
        # Count occurrences of the tail pattern
        count = text_norm.count(tail_norm)

        # Use phase-specific threshold
        threshold = {
            "planning": self.config.plan_loop_threshold,
            "execution": self.config.execution_loop_threshold,
            "synthesis": self.config.synthesis_loop_threshold,
        }.get(phase, self.config.execution_loop_threshold)

        return count >= threshold

    def _diverges_from_previous(
        self, new: str, previous: str, threshold: float = 0.7
    ) -> bool:
        """Check if new output is semantically different from previous."""
        if not previous or not new:
            return True
        from difflib import SequenceMatcher

        # Compare first 500 chars (content, not length)
        ratio = SequenceMatcher(None, new[:500], previous[:500]).ratio()
        return ratio < threshold  # <70% similar = different enough

    def _extract_unique_content(self, text: str, min_length: int = 100) -> str:
        """Extract the longest unique subsequence before repetition starts."""
        if not text:
            return text
        lines = text.split("\n")
        seen = set()
        unique = []

        for line in lines:
            key = re.sub(r"\s+", " ", line.strip().lower())
            if key in seen or len(key) < 20:
                break  # stop at first repeat
            seen.add(key)
            unique.append(line)

        result = "\n".join(unique)
        return result if len(result) > min_length else text[:min_length]

    def _normalize_abbreviations(self, text: str) -> str:
        """Expand common abbreviations to full words for clarity."""
        if not text:
            return text

        # Mapping of abbreviations to full forms
        expansions = {
            r"\bAPI\b": "Application Programming Interface",
            r"\bHTTP\b": "HyperText Transfer Protocol",
            r"\bHTTPS\b": "HyperText Transfer Protocol Secure",
            r"\bSSL\b": "Secure Sockets Layer",
            r"\bTLS\b": "Transport Layer Security",
            r"\bSSH\b": "Secure Shell",
            r"\bJSON\b": "JavaScript Object Notation",
            r"\bXML\b": "Extensible Markup Language",
            r"\bSQL\b": "Structured Query Language",
            r"\bDNS\b": "Domain Name System",
            r"\bURL\b": "Uniform Resource Locator",
            r"\bCPU\b": "Central Processing Unit",
            r"\bRAM\b": "Random Access Memory",
            r"\bI/O\b": "Input/Output",
            r"\bUI\b": "User Interface",
            r"\bUX\b": "User Experience",
            r"\bCRUD\b": "Create, Read, Update, Delete",
            r"\bRESTful\b": "Representational State Transfer",
            r"\be\.g\.\b": "for example",
            r"\bi\.e\.\b": "that is",
            r"\betc\.\b": "and so on",
            r"\bvs\.\b": "versus",
            r"\bnum\b": "number",
            r"\bconfig\b": "configuration",
            r"\bdb\b": "database",
            r"\bexec\b": "execute",
            r"\bsync\b": "synchronize",
            r"\basync\b": "asynchronous",
            r"\bpwd\b": "password",
            r"\bvar\b": "variable",
            r"\benv\b": "environment",
            r"\bopt\b": "optional",
            r"\bauth\b": "authentication",
            r"\bencr\b": "encryption",
        }

        result = text
        for abbr, full in expansions.items():
            result = re.sub(abbr, full, result)

        return result

    def _get_current_time_context(self) -> str:
        """Get current date/time context for prompts."""
        from datetime import datetime

        now = datetime.now()
        return (
            f"Current date: {now.strftime('%Y-%m-%d %A')}\n"
            f"Current time: {now.strftime('%H:%M:%S %Z')}"
        )

    async def _resume_from_memory(self, goal: str) -> Optional[dict[str, Task]]:
        """Check memory for previously completed tasks matching this goal."""
        if (
            self._mcp is None
            or not self.config.enable_memory_resume
            or not hasattr(self._mcp, "call_tool")
        ):
            return None
        try:
            # Query memory for tasks with this goal
            cached = await self._mcp.call_tool(
                "retrieve", {"query": f"planner_task {goal[:100]}", "k": 20}
            )
            if not cached or not isinstance(cached, str):
                return None

            # Parse results and reconstruct Task objects
            done: dict[str, Task] = {}
            for line in cached.split("\n"):
                if "task_" in line.lower():
                    # Simple parsing: extract task_id from memory results
                    match = re.search(r"(task_\d+_\w+)", line, re.IGNORECASE)
                    if match:
                        task_id = match.group(1)
                        # Create a stub task marked as completed
                        task = Task(task_id=task_id, description="[resumed]")
                        task.status = "completed"
                        task.result = "[loaded from memory]"
                        done[task_id] = task
            return done if done else None
        except Exception:
            return None

    def _tools_catalog_text(self) -> str:
        """Human-readable catalog of available tools for the planning prompt."""
        if self._mcp is None or self._tools_wanted is False:
            return ""
        try:
            schemas = self._mcp.openai_tools()
        except Exception:
            return ""
        lines = []
        for schema in schemas:
            fn = schema.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            desc = " ".join((fn.get("description") or "").split())[:200]
            lines.append(f"  - {name}: {desc}" if desc else f"  - {name}")
        return "\n".join(lines)

    async def plan(self, goal: str) -> list[Task]:
        await self._emit("Planning...")
        available_tools = set(self._available_tool_names())

        # Inject time context
        time_context = self._get_current_time_context()
        user_msg = f"{time_context}\n\nUser request:\n{goal}"

        raw = await self.complete(
            PromptBuilder.planning_prompt(
                self.system_prompt,
                self.config.max_tasks,
                self._tools_catalog_text(),
            ),
            user_msg,
            self._planning_temperature(),
            True,
            self._phase_params("planning"),
        )
        parsed = _extract_json_object(raw)
        tasks: list[Task] = []
        if parsed and isinstance(parsed.get("tasks"), list):
            for idx, item in enumerate(parsed["tasks"][: self.config.max_tasks], 1):
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
                # Keep only tool names the planner selected that actually exist.
                selected_tools = [
                    str(t).strip()
                    for t in (item.get("tools") or [])
                    if str(t).strip() in available_tools
                ]
                tasks.append(
                    Task(
                        task_id=task_id,
                        description=description,
                        related_tasks=related,
                        tools=selected_tools,
                    )
                )

        if not tasks:
            await self._emit("Planning produced no tasks; using single task")
            # No plan to select tools; allow all available tools as a fallback.
            tasks = [
                Task(
                    task_id="task_1",
                    description=goal,
                    tools=list(available_tools),
                )
            ]

        # Save plan to memory
        plan_summary = f"Plan for: {goal}\n\nTasks:\n" + "\n".join(
            f"  - {t.task_id}: {t.description}" for t in tasks
        )
        await self._save_to_memory("plan", goal, plan_summary)

        selected = sorted({t for task in tasks for t in task.tools})
        if available_tools:
            await self._emit(
                f"Plan ready: {len(tasks)} task(s); tools selected: "
                + (", ".join(selected) if selected else "none")
            )
        else:
            await self._emit(f"Plan ready: {len(tasks)} task(s)")
        return tasks

    async def execute_task(
        self, goal: str, task: Task, done: dict[str, Task], attempt: int = 0
    ) -> str:
        await self._emit(f"Executing {task.task_id}: {task.description[:60]}")

        if task.related_tasks:
            deps = [done[t] for t in task.related_tasks if t in done]
        else:
            deps = [t for t in done.values() if t.status == "completed"]

        context_blocks = [
            f"--- Result of {dep.task_id} ({dep.description}) ---\n{self._truncate(dep.result)}"
            for dep in deps
            if dep.status == "completed" and dep.result
        ]
        context = (
            "\n\n".join(context_blocks) if context_blocks else "(no prior results)"
        )

        user_message = (
            f"Original user request:\n{goal}\n\n"
            f"Prerequisite task results:\n{context}\n\n"
            f"YOUR TASK ({task.task_id}):\n{task.description}\n\n"
            "Produce the complete deliverable for this task now."
        )

        # Inject time context
        time_context = self._get_current_time_context()
        user_message = f"{time_context}\n\n{user_message}"

        # Inject loop-escape guidance on retry
        if attempt > 0:
            user_message += (
                "\n\n⚠️ IMPORTANT: Your previous attempt got stuck in repetition. "
                "Be concise, avoid circular reasoning, and provide a direct answer. "
                "Use complete words, not abbreviations (e.g., 'Application Programming Interface' not 'API')."
            )

        # Check token budget
        estimated_tokens = len(user_message.split()) * 1.3
        if estimated_tokens > self.config.max_tokens_per_task:
            await self._emit(
                f"  ⚠ {task.task_id}: may exceed token budget "
                f"({int(estimated_tokens)}/{self.config.max_tokens_per_task})"
            )

        tools = self._select_tools(task.tools)
        if tools and self.chat is not None:
            await self._emit(
                f"  ↳ tools for {task.task_id}: {', '.join(t['function']['name'] for t in tools)}"
            )
            result = await self._execute_with_tools(user_message, tools, attempt)
        else:
            result = await self.complete(
                PromptBuilder.execution_prompt(self.system_prompt),
                user_message,
                self._adjusted_temperature(attempt),
                False,
                self._adjusted_params("execution", attempt),
            )

        # Normalize abbreviations for clarity (especially important for small models)
        result = self._normalize_abbreviations(result)

        # Save to memory MCP if available
        had_loop = attempt > 0
        await self._save_to_memory(
            task.task_id, task.description, result, attempt, had_loop
        )
        return result

    def _select_tools(self, names: list[str]) -> list[dict]:
        """Return the OpenAI tool schemas matching `names` (planner's selection)."""
        if self._mcp is None or not names or self._tools_wanted is False:
            return []
        try:
            by_name = {
                (t.get("function") or {}).get("name"): t
                for t in self._mcp.openai_tools()
            }
        except Exception:
            return []
        return [by_name[n] for n in names if n in by_name]

    async def _execute_with_tools(
        self, user_message: str, tools: list[dict], attempt: int = 0
    ) -> str:
        """Run one task as a tool-calling loop over the configured MCP tools."""
        messages: list[dict] = [
            {
                "role": "system",
                "content": PromptBuilder.execution_prompt(self.system_prompt)
                + "\n\nYou may call the provided tools to gather information or "
                "perform actions. Call tools only when they materially help; once "
                "you have what you need, write the final deliverable as plain text.",
            },
            {"role": "user", "content": user_message},
        ]
        temperature = self._adjusted_temperature(attempt)
        params = self._adjusted_params("execution", attempt)

        # On retry: disable problematic tools based on metrics, or all tools on 3rd+
        use_tools = tools
        if attempt >= 2:
            await self._emit(
                f"  ↳ attempt {attempt}: tools disabled, forcing reasoning-only"
            )
            use_tools = []
        elif attempt == 1:
            # Filter out tools that caused loops in previous attempts
            problematic = [
                name
                for name, metrics in self._tool_metrics.items()
                if metrics.loop_count > 0
            ]
            if problematic:
                await self._emit(
                    f"  ↳ disabling problematic tools: {', '.join(problematic)}"
                )
                use_tools = [
                    t
                    for t in tools
                    if (t.get("function") or {}).get("name") not in problematic
                ]

        for _ in range(max(1, self.config.max_tool_iterations)):
            message = await self.chat(messages, use_tools, temperature, params)
            tool_calls = message.get("tool_calls") or []
            # Record the assistant turn (content may be empty alongside tool calls).
            assistant_turn: dict = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            messages.append(assistant_turn)

            if not tool_calls:
                return assistant_turn["content"]

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                await self._emit(f"  ↳ tool: {name}({', '.join(args.keys())})")

                # Track tool usage
                if name not in self._tool_metrics:
                    self._tool_metrics[name] = ToolMetrics(name=name)
                self._tool_metrics[name].call_count += 1

                result = await self._mcp.call_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": result,
                    }
                )

                # Mark tool as successful if result is non-error
                if not (isinstance(result, str) and "[tool error]" in result):
                    self._tool_metrics[name].success_count += 1
                else:
                    self._tool_metrics[name].error_count += 1

        # Ran out of iterations: make one final tool-free pass for an answer.
        await self._emit("  ↳ tool budget reached; finalizing")
        final = await self.chat(messages, None, temperature, params)
        return final.get("content") or "[no final answer after tool iterations]"

    async def synthesize(self, goal: str, tasks: list[Task]) -> str:
        await self._emit("Synthesizing final answer...")
        results_block = "\n\n".join(
            f"--- {t.task_id} ({t.status}) ---\n{self._truncate(t.result)}"
            for t in tasks
        )

        # Inject time context
        time_context = self._get_current_time_context()

        user_message = (
            f"{time_context}\n\n"
            f"Original user request:\n{goal}\n\n"
            f"Task results:\n{results_block}\n\n"
            "Write the final response now. Use complete words, not abbreviations."
        )
        draft = await self.complete(
            PromptBuilder.synthesis_prompt(self.system_prompt),
            user_message,
            self._synthesis_temperature(),
            False,
            self._phase_params("synthesis"),
        )
        final = self._resolve_macros(draft, tasks)

        # Normalize abbreviations
        final = self._normalize_abbreviations(final)

        review = self.config.enable_review
        needs_review = await self._draft_needs_review(goal, final)
        if needs_review is not None:  # Kev answered: it decides, in both directions
            if review != needs_review:
                await self._emit(
                    f"Kev: review = {str(needs_review).lower()} (draft judged "
                    f"{'incomplete' if needs_review else 'complete'})"
                )
            review = needs_review

        if review:
            await self._emit("Reviewing final answer...")
            final = await self.complete(
                PromptBuilder.review_prompt(self.system_prompt),
                f"{time_context}\n\n"
                f"Original user request:\n{goal}\n\n"
                f"Draft final answer:\n{final}\n\n"
                "Return the improved final answer. Ensure all abbreviations are expanded to full words.",
                self._synthesis_temperature(),
                False,
                self._phase_params("synthesis"),
            )
            # Normalize again after review
            final = self._normalize_abbreviations(final)

        # Save final result to memory
        await self._save_to_memory("synthesis", goal, final)
        return final

    def _resolve_macros(self, text: str, tasks: list[Task]) -> str:
        by_id = {t.task_id: t for t in tasks}

        def repl(match: "re.Match") -> str:
            task = by_id.get(match.group(1))
            return task.result if task and task.result else match.group(0)

        return re.sub(r"@([A-Za-z0-9_\-]+)", repl, text)

    _MULTI_STEP_MARKERS = re.compile(
        r"\b(then|after that|step\s*\d|also|additionally|as well as|"
        r"followed by|first.*then)\b",
        re.IGNORECASE,
    )
    _LIST_ITEM_RE = re.compile(r"(^|\n)\s*(\d+[.)]|[-*])\s+\S")

    _TOOL_META_QUERY_RE = re.compile(r"\b(mcp|tools?|capabilit\w*)\b", re.IGNORECASE)

    def _looks_like_tool_meta_query(self, goal: str) -> bool:
        """Is the request asking about the toolset itself (what's available, list them,
        check them) rather than asking something that might incidentally need one? This
        is a factual question with a deterministic answer (the discovered tool list), not
        something Kev's "does this need a tool" classifier or the name heuristic should be
        guessing about - both judge whether a tool helps the request's content, and a
        request whose content IS the toolset is exactly the case they misjudge."""
        return bool(self._TOOL_META_QUERY_RE.search(goal))

    def _looks_like_it_needs_tools(
        self, goal: str, names: Optional[Iterable[str]] = None
    ) -> bool:
        """Heuristic fallback for when Kev did not classify (no Kev, or kev_ask_tools off):
        does the request plausibly need one of `names` (or, if omitted, the connected
        tools)? Matches significant words from each tool's own name against the goal, so
        it generalizes to whatever MCP servers happen to be configured instead of a
        hardcoded keyword list."""
        if names is None:
            if self._mcp is None:
                return False
            names = self._mcp.tool_names
        text = goal.lower()
        for name in names:
            for word in re.split(r"[_\-]+", name.lower()):
                if len(word) >= 4 and word in text:
                    return True
        return False

    def _looks_like_short_task(self, goal: str) -> bool:
        """Heuristic: is this simple enough to answer directly, no plan needed?

        Short, single-sentence requests with no multi-step language or list
        structure are treated as trivial. Anything longer or that signals
        multiple deliverables still goes through planning.
        """
        text = goal.strip()
        if not text:
            return True
        if len(text.split()) > self.config.short_task_max_words:
            return False
        if len(re.findall(r"[.!?]+", text)) > 1:
            return False
        if self._MULTI_STEP_MARKERS.search(text):
            return False
        if self._LIST_ITEM_RE.search(text):
            return False
        return True

    # -- decisions ---------------------------------------------------------

    def _kev_state(self, text: str) -> str:
        """A task result or draft, cut to what is worth prefilling for one yes/no question."""
        limit = self.config.kev_state_limit
        if len(text) <= limit:
            return text
        head = limit // 2
        return f"{text[:head]}\n\n...[truncated]...\n\n{text[-(limit - head):]}"

    def _is_small_talk(self, goal: str) -> bool:
        """A greeting, a thanks, an ok: nothing to plan, think about or look up, and not worth a Kev call."""
        words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", goal.lower())
        if not (
            self.config.fast_small_talk
            and words
            and len(words) <= self.config.small_talk_max_words
        ):
            return False
        # Digits, code, or anything but words and light punctuation: not small talk.
        if re.search(r"[0-9=+*/<>{}\[\]`$%]", goal):
            return False
        return all(w in _SMALL_TALK_WORDS for w in words)

    async def _kev_answers(self, goal: str, questions: dict) -> dict:
        """Kev's p(true) for `questions`, from the cache where this request was already asked."""
        key = (self.kev.url, goal)
        cached = _KEV_PROFILE_CACHE.get(key, {})
        missing = {qid: q for qid, q in questions.items() if qid not in cached}
        if missing:
            fresh = await self.kev.nouls(goal, missing)
            if fresh:
                cached = {**cached, **fresh}
                _KEV_PROFILE_CACHE[key] = cached
                while len(_KEV_PROFILE_CACHE) > max(0, self.config.kev_cache_size):
                    _KEV_PROFILE_CACHE.popitem(last=False)
        if key in _KEV_PROFILE_CACHE:
            _KEV_PROFILE_CACHE.move_to_end(key)
        return {qid: cached[qid] for qid in questions if qid in cached}

    async def _profile_request(self, goal: str, tools_available: bool = False) -> None:
        """Before anything is generated: is the request simple, does it need tools, is it artistic or scientific,
        does it need thinking. Asked in two rounds so an answer can make a later question unnecessary; each answer
        that does not come back leaves the configured behaviour in place."""
        cfg = self.config
        if self._is_small_talk(goal):
            self._p_simple, self._think, self._tools_wanted, self._direct = (
                1.0,
                False,
                False,
                True,
            )
            await self._emit(
                "Small talk: answering directly (no Kev, no plan, no tools, no thinking)"
            )
            return
        if not self.kev.enabled:
            return
        began = time.monotonic()

        first: dict = {}
        if (
            cfg.plan_mode
            and cfg.auto_skip_plan_for_short_tasks
            and cfg.kev_decide_planning
        ):
            first["simple"] = (
                "Is this a simple request that one short, direct answer covers?",
                {
                    "true": "a single direct answer covers it",
                    "false": "it needs several steps or a long piece of work",
                },
            )
        temperature_matters = (
            cfg.execution_temperature is None or cfg.synthesis_temperature is None
        )
        if cfg.kev_classify_request and temperature_matters:
            first["artistic"] = (
                "Is this request artistic or creative rather than scientific or factual?",
                {
                    "true": "creative writing, art, storytelling, brainstorming, style",
                    "false": "facts, science, math, code, analysis, precise instructions",
                },
            )
        if cfg.kev_ask_tools and tools_available:
            first["tools"] = (
                "Does answering this need a tool: looking something up online, current or live information, "
                "the user's files, saved memory, or an action outside this conversation?",
                {
                    "true": "it needs information or actions the model does not have",
                    "false": "the model can answer it from what it knows",
                },
            )
        answers = await self._kev_answers(goal, first) if first else {}
        artistic = (
            answers["artistic"] >= cfg.kev_artistic_threshold
            if "artistic" in answers
            else None
        )

        verdicts = []
        if cfg.kev_classify_request and cfg.model_can_think is not False:
            if artistic:
                self._think = False
                verdicts.append("thinking = off (artistic)")
            else:
                answers.update(
                    await self._kev_answers(
                        goal,
                        {
                            "think": (
                                "Does answering this correctly require careful step-by-step reasoning?",
                                {
                                    "true": "it needs working out: logic, math, analysis, planning or code",
                                    "false": "it can be answered from knowledge or by writing directly",
                                },
                            )
                        },
                    )
                )
        elif cfg.model_can_think is False:
            verdicts.append("thinking = n/a (model cannot think)")

        if "simple" in answers:
            self._p_simple = answers["simple"]
            self._direct = self._p_simple >= cfg.kev_simple_threshold
            verdicts.insert(
                0, f"simple = {str(self._direct).lower()} (p {self._p_simple:.3f})"
            )
        if "tools" in answers:
            self._tools_wanted = answers["tools"] >= cfg.kev_tools_threshold
            verdicts.append(
                f"tools = {'on' if self._tools_wanted else 'off'} (p {answers['tools']:.3f})"
            )
        if "think" in answers:
            self._think = answers["think"] >= cfg.kev_think_threshold
            verdicts.append(
                f"thinking = {'on' if self._think else 'off'} (p {answers['think']:.3f})"
            )
        if artistic is not None:
            self._temperature = (
                cfg.kev_artistic_temperature
                if artistic
                else cfg.kev_scientific_temperature
            )
            verdicts.append(
                f"{'artistic' if artistic else 'scientific'} -> temperature {self._temperature:g} "
                f"(p {answers['artistic']:.3f})"
            )
        if verdicts:
            await self._emit(
                f"Kev ({time.monotonic() - began:.1f} s): " + " · ".join(verdicts)
            )

    async def _should_skip_planning(self, goal: str) -> bool:
        """Answer in one pass instead of decomposing? Kev's p(simple) decides when it answered; the word-count
        heuristic decides when it did not. The heuristic mistakes a short hard request ("compare these three vendors on
        price, support and lock-in") for a trivial one, and a long easy one for work."""
        if not (self.config.plan_mode and self.config.auto_skip_plan_for_short_tasks):
            return False
        if self._p_simple is None:
            return self._looks_like_short_task(goal)
        return self._p_simple >= self.config.kev_simple_threshold

    async def _task_result_is_usable(self, task: Task) -> Optional[bool]:
        """Did the task actually produce what it was asked for? None when Kev did not answer, and then nothing
        changes: loop detection stays the only check, as it was before."""
        if not (self.config.kev_check_tasks and task.result.strip()):
            return None
        probability = await self.kev.noul(
            f"TASK:\n{task.description}\n\nRESULT:\n{self._kev_state(task.result)}",
            "Does the result contain the deliverable the task asked for?",
            {
                "true": "the asked-for content is present",
                "false": "it is generic, partial or about something else",
            },
        )
        if probability is None:
            return None
        return probability >= self.config.kev_accept_threshold

    async def _draft_needs_review(self, goal: str, draft: str) -> Optional[bool]:
        """Whether the review pass is worth its generation. None when Kev did not answer."""
        if not self.config.kev_gate_review:
            return None
        probability = await self.kev.noul(
            f"REQUEST:\n{goal}\n\nDRAFT ANSWER:\n{self._kev_state(draft)}",
            "Does the draft address everything the request asked for?",
            {
                "true": "every part of the request is covered",
                "false": "part of the request is not covered",
            },
        )
        if probability is None:
            return None
        return probability < self.config.kev_review_threshold

    # -- orchestration -----------------------------------------------------

    def _make_mcp(self) -> Any:
        """Return an MCP client context manager (real if configured, else null)."""
        if self.config.mcp_servers and self.chat is not None:
            return MCPClient(
                self.config.mcp_servers,
                self.config.tool_result_limit,
                self.config.mcp_discovery_cache_ttl,
                self.config.mcp_tool_description_limit,
            )
        return _NullMCP()

    def _mcp_worth_connecting(self, goal: str, tools_possible: bool) -> bool:
        """Decided after Kev's profile, before paying for a connection: does this request
        plausibly need a tool at all? Kev's own answer wins outright; otherwise this falls
        back to the name heuristic against whatever is already cached (connecting once,
        for an uncached server, to find out - after that its cache carries the answer).
        """
        if not tools_possible:
            return False
        if self._looks_like_tool_meta_query(goal):
            return True
        if self._tools_wanted is True:
            return True
        if self._tools_wanted is False:
            return False
        if not self.config.mcp_tool_gate_heuristic:
            return True
        cached_names = _cached_mcp_tool_names(
            self.config.mcp_servers, self.config.mcp_discovery_cache_ttl
        )
        if cached_names is None:
            return True  # cold cache: connect once so it gets discovered at all
        return self._looks_like_it_needs_tools(goal, cached_names)

    async def run(self, goal: str, chat_id: Optional[str] = None) -> PlannerResult:
        start = time.monotonic()
        self._chat_id = chat_id
        if self._is_small_talk(goal):
            await self._profile_request(goal, tools_available=False)
            async with _NullMCP() as mcp:
                return await self._run_connected(mcp, goal, start)

        # Kev decides first - before any MCP connection or other work - whether this
        # needs planning and whether it needs tools, so a simple, tool-free request never
        # pays for either.
        tools_possible = bool(self.config.mcp_servers) and self.chat is not None
        await self._profile_request(goal, tools_available=tools_possible)

        if self._mcp_worth_connecting(goal, tools_possible):
            mcp_cm = self._make_mcp()
            skipped_by_heuristic = False
        else:
            mcp_cm = _NullMCP()
            skipped_by_heuristic = tools_possible
        async with mcp_cm as mcp:
            return await self._run_connected(
                mcp, goal, start, skipped_by_heuristic=skipped_by_heuristic
            )

    async def _run_connected(
        self,
        mcp: Any,
        goal: str,
        start: float,
        skipped_by_heuristic: bool = False,
    ) -> PlannerResult:
        self._mcp = mcp
        for note in getattr(mcp, "notes", []) or []:
            await self._emit(note)
        tool_names = mcp.tool_names
        configured = bool(self.config.mcp_servers)
        if tool_names:
            # List every discovered tool up front so the user can see the
            # full toolset the planner may draw from.
            await self._emit(
                f"MCP ready: {len(tool_names)} tool(s) found — " + ", ".join(tool_names)
            )
            catalog = self._tools_catalog_text()
            if catalog:
                await self._emit("Available MCP tools:\n" + catalog)
        elif skipped_by_heuristic:
            # We deliberately didn't connect - not a failure, just judged unnecessary.
            await self._emit(
                "MCP configured but not connected: this request didn't look like it "
                "needed a tool."
            )
        elif configured:
            # Servers were configured but no tools were discovered: tell the
            # user explicitly instead of failing silently.
            await self._emit(
                "MCP enabled but no tools were found (check the server URL / "
                "that it is running and exposes tools)."
            )
        elif self.chat is not None:
            # No servers configured at all (e.g. MCP disabled).
            await self._emit("MCP not configured — running without tools.")
        try:
            return await self._run_inner(goal, start)
        finally:
            self._mcp = None

    async def _run_inner(self, goal: str, start: float) -> PlannerResult:
        skip_plan = not self.config.plan_mode
        if not skip_plan and await self._should_skip_planning(goal):
            await self._emit("Short/simple request — skipping planning")
            skip_plan = True

        if skip_plan:
            await self._emit("Single-pass execution (plan mode off)")
            single = Task(task_id="task_1", description=goal)
            gate_open = (
                self._looks_like_tool_meta_query(goal)
                or self._tools_wanted is True
                or (
                    self._tools_wanted is None
                    and (
                        not self.config.mcp_tool_gate_heuristic
                        or self._looks_like_it_needs_tools(goal)
                    )
                )
            )
            tools = (
                self._mcp.openai_tools() if self._mcp is not None and gate_open else []
            )

            # Inject time context
            time_context = self._get_current_time_context()
            user_message = f"{time_context}\n\nUser request:\n{goal}"
            if self._direct and self.config.direct_answer_hint:
                user_message += (
                    "\n\nAnswer directly: no preamble, do not restate the request."
                )

            if tools and self.chat is not None:
                single.result = await self._execute_with_tools(user_message, tools)
            else:
                single.result = await self.complete(
                    PromptBuilder.execution_prompt(self.system_prompt),
                    user_message,
                    self._execution_temperature(),
                    False,
                    self._phase_params("execution"),
                )

            # Normalize abbreviations
            single.result = self._normalize_abbreviations(single.result)

            single.status = "completed"
            return PlannerResult(
                goal, [single], single.result, time.monotonic() - start
            )

        chat_state = self._get_chat_plan_state()
        done: dict[str, Task] = dict(chat_state.done) if chat_state else {}
        if done:
            await self._emit(
                f"Continuing this chat's plan: {len(done)} previously completed "
                "task(s) carried over"
            )

        tasks = await self.plan(goal)
        if chat_state is not None:
            chat_state.turn += 1
            tasks = self._namespace_new_tasks(tasks, chat_state.turn)

        # Try to resume from memory next (only for whatever the chat-plan state didn't
        # already cover - it's the more precise, always-available source).
        if not done and self.config.enable_memory_resume:
            resumed = await self._resume_from_memory(goal)
            if resumed:
                done = resumed
                await self._emit(f"Resumed {len(done)} completed task(s) from memory")
        tasks = [t for t in tasks if t.task_id not in done]

        for task in tasks:
            try:
                # Retry on loop detection, up to 3 attempts
                max_attempts = 3
                prev_result = None
                for attempt in range(max_attempts):
                    task.result = await self.execute_task(goal, task, done, attempt)
                    # Check for repetition loop
                    if self._is_looping(task.result, "execution"):
                        # Verify it's not just noise—output must diverge from previous
                        if prev_result and not self._diverges_from_previous(
                            task.result, prev_result
                        ):
                            await self._emit(
                                f"{task.task_id}: retry produced same output, aborting"
                            )
                            # Use extracted content instead of full looping output
                            task.result = self._extract_unique_content(task.result)
                            break
                        if attempt < max_attempts - 1:
                            await self._emit(
                                f"{task.task_id}: loop detected; retry with adjusted params "
                                f"(attempt {attempt + 1}/{max_attempts - 1})"
                            )
                            prev_result = task.result
                            # Mark loop in metrics for tool tracking
                            for tool in task.tools:
                                if tool not in self._tool_metrics:
                                    self._tool_metrics[tool] = ToolMetrics(name=tool)
                                self._tool_metrics[tool].loop_count += 1
                            continue
                        else:
                            await self._emit(
                                f"{task.task_id}: loop persisted; accepting result"
                            )
                    if (
                        attempt < max_attempts - 1
                        and await self._task_result_is_usable(task) is False
                    ):
                        await self._emit(
                            f"{task.task_id}: Kev says the result does not carry out the task; "
                            f"retrying (attempt {attempt + 1}/{max_attempts - 1})"
                        )
                        prev_result = task.result
                        continue
                    break  # success or final attempt; exit retry loop
                task.status = "completed"
            except Exception as exc:  # keep going; record the failure
                task.status = "failed"
                task.result = f"[task failed: {exc}]"
                await self._emit(f"{task.task_id} failed: {exc}")
            done[task.task_id] = task

        if chat_state is not None:
            chat_state.tasks.extend(tasks)
            chat_state.done.update(done)
            chat_state.updated_at = time.monotonic()

        if (
            self.config.skip_single_task_synthesis
            and len(tasks) == 1
            and tasks[0].status == "completed"
            and tasks[0].result.strip()
        ):
            await self._emit(
                "One-task plan: its result is the answer (synthesis skipped)"
            )
            return PlannerResult(goal, tasks, tasks[0].result, time.monotonic() - start)
        final_output = await self.synthesize(goal, tasks)
        return PlannerResult(goal, tasks, final_output, time.monotonic() - start)


# ---------------------------------------------------------------------------
# Completion backends
# ---------------------------------------------------------------------------


# Chat-completion params the OpenAI SDK accepts as top-level kwargs. Anything
# else (top_k, min_p, repeat_penalty, ...) must be passed via `extra_body` for
# OpenAI-compatible backends like LM Studio.
_OPENAI_STD_PARAMS = {
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "logit_bias",
    "n",
    "logprobs",
    "top_logprobs",
}


def _openai_think_fields(params: Optional[dict]) -> Optional[dict]:
    """Translate the planner's backend-neutral {"think": False} for an OpenAI-compatible API.

    Ollama's /v1 ignores "think" and honours reasoning_effort "none" (measured on 0.34.2); llama.cpp and vLLM read
    chat_template_kwargs. Each server ignores the field meant for the other.
    """
    if not params or "think" not in params:
        return params
    params = dict(params)
    if params.pop("think") is False:
        params.setdefault("reasoning_effort", "none")
        params.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return params or None


def make_openai_completer(config: PlannerConfig) -> CompletionFn:
    """CLI backend: any OpenAI-compatible endpoint via the `openai` SDK."""
    from openai import OpenAI  # lazy import; only needed for the CLI path

    client = OpenAI(
        api_key=config.api_key or "sk-noauth",
        base_url=config.api_url,
        timeout=config.request_timeout,
    )

    async def complete(
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        json_object: bool = False,
        params: Optional[dict] = None,
    ) -> str:
        def _call() -> str:
            kwargs: dict = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": (
                    temperature if temperature is not None else config.temperature
                ),
            }
            # Route standard params as kwargs; everything else via extra_body.
            extra_body: dict = {}
            for key, value in (_openai_think_fields(params) or {}).items():
                if value is None:
                    continue
                if key in _OPENAI_STD_PARAMS:
                    kwargs[key] = value
                else:
                    extra_body[key] = value
            if extra_body:
                kwargs["extra_body"] = extra_body
            try:
                if json_object:
                    resp = client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs
                    )
                else:
                    resp = client.chat.completions.create(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned an empty response")
            return content

        return await asyncio.to_thread(_call)

    return complete


def make_openai_chat(config: PlannerConfig) -> ChatFn:
    """CLI tool-calling backend: OpenAI-compatible chat with `tools` support."""
    from openai import OpenAI  # lazy import; only needed for the CLI path

    client = OpenAI(
        api_key=config.api_key or "sk-noauth",
        base_url=config.api_url,
        timeout=config.request_timeout,
    )

    async def chat(
        messages: list,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> dict:
        def _call() -> dict:
            kwargs: dict = {
                "model": config.model,
                "messages": messages,
                "temperature": (
                    temperature if temperature is not None else config.temperature
                ),
            }
            extra_body: dict = {}
            for key, value in (_openai_think_fields(params) or {}).items():
                if value is None:
                    continue
                if key in _OPENAI_STD_PARAMS:
                    kwargs[key] = value
                else:
                    extra_body[key] = value
            if extra_body:
                kwargs["extra_body"] = extra_body
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            tool_calls = None
            if getattr(msg, "tool_calls", None):
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return {"content": msg.content, "tool_calls": tool_calls}

        return await asyncio.to_thread(_call)

    return chat


def _content_from_choices(data: dict) -> str:
    """Extract assistant text from an OpenAI-style payload.

    Falls back to `reasoning_content` (thinking models like qwen3) and to the
    top-level `content` some backends return when `content` is empty/missing.
    """
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return content
            # Streaming-style chunk shape.
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict) and delta.get("content"):
                return delta["content"]
            # Thinking models may only populate reasoning_content.
            if message.get("reasoning_content"):
                return message["reasoning_content"]
        # Some backends nest plain text under choices[0]["text"].
        if choices[0].get("text"):
            return choices[0]["text"]
    # Top-level fallbacks.
    if data.get("content"):
        return data["content"]
    return ""


def _owui_response_error(response: Any) -> str:
    """Return an error message if the OWUI response is an error payload, else ''.

    Open WebUI surfaces upstream errors as a JSONResponse whose body is
    `{"error": "..."}` (or `{"detail": "..."}`), returned with a 200, so they
    don't raise — we must inspect the body to detect them.
    """
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
    """Return a human-readable view of a response for error diagnostics."""
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


def _owui_extract_content(response: Any) -> str:
    """Pull text content out of an Open WebUI chat completion response."""
    if isinstance(response, dict):
        return _content_from_choices(response)
    # Some OWUI versions return a string directly.
    if isinstance(response, str):
        return response
    # FastAPI Response / StreamingResponse: try the body.
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


def _owui_response_to_data(response: Any) -> Optional[dict]:
    """Best-effort: turn an OWUI response into the parsed JSON dict, or None."""
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if body is not None:
        try:
            return json.loads(
                body.decode("utf-8", "replace")
                if isinstance(body, (bytes, bytearray))
                else body
            )
        except Exception:
            return None
    return None


def _owui_extract_message(response: Any) -> dict:
    """Pull the assistant message (content + tool_calls) from an OWUI response."""
    data = _owui_response_to_data(response)
    if not isinstance(data, dict):
        return {"content": _owui_extract_content(response), "tool_calls": None}
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls") or None
            content = message.get("content")
            if content is None and message.get("reasoning_content"):
                content = message["reasoning_content"]
            return {"content": content, "tool_calls": tool_calls}
    return {"content": _content_from_choices(data), "tool_calls": None}


# ---------------------------------------------------------------------------
# Open WebUI Pipe
# ---------------------------------------------------------------------------


# Ollama model -> can it think (from /api/show); a model's capabilities do not change while it is installed.
_OLLAMA_CAN_THINK: dict = {}


def _ollama_get(
    url: str, path: str, body: Optional[dict] = None, timeout: float = 2.0
) -> Any:
    import urllib.request

    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


async def _ollama_model_state(base_urls: list, name: str) -> tuple:
    """(loaded, can_think) for an Ollama model; either is None when no server answered."""
    loaded: Optional[bool] = None
    can_think: Optional[bool] = _OLLAMA_CAN_THINK.get(name)
    for url in base_urls:
        try:
            running = await asyncio.to_thread(_ollama_get, url, "/api/ps")
            names = {m.get("name") for m in running.get("models", [])} | {
                m.get("model") for m in running.get("models", [])
            }
            loaded = bool(loaded) or name in names
            if can_think is None:
                shown = await asyncio.to_thread(
                    _ollama_get, url, "/api/show", {"model": name}
                )
                can_think = "thinking" in (shown.get("capabilities") or [])
                _OLLAMA_CAN_THINK[name] = can_think
        except Exception:
            continue
        if loaded:
            break
    return loaded, can_think


class Pipe:
    """Open WebUI Pipe entrypoint. Appears as a selectable model."""

    class Valves(BaseModel):
        PLANNER_MODEL: str = Field(
            default="",
            description="The real model id the planner drives (e.g. gpt-4o, llama3.1). Leave blank to auto-select an available model from Open WebUI.",
        )
        TEMPERATURE: float = Field(
            default=0.7,
            description="Base sampling temperature (used by phases without an override).",
        )
        PLANNING_TEMPERATURE: float = Field(
            default=-1.0,
            description="Temperature for the planning phase. -1 = auto (min(base, 0.4); keeps plans deterministic). Set 0.0-2.0 to override.",
        )
        EXECUTION_TEMPERATURE: float = Field(
            default=-1.0,
            description="Temperature for the execution phase. -1 = use base temperature.",
        )
        SYNTHESIS_TEMPERATURE: float = Field(
            default=-1.0,
            description="Temperature for the synthesis/review phase. -1 = use base temperature.",
        )
        # --- Extra sampling params (applied to all phases) --------------------
        # Sentinel -1 means "unset / use the backend default".
        TOP_P: float = Field(
            default=-1.0, description="Nucleus sampling top_p. -1 = unset."
        )
        TOP_K: int = Field(default=-1, description="top_k sampling. -1 = unset.")
        MIN_P: float = Field(default=-1.0, description="min_p sampling. -1 = unset.")
        MAX_TOKENS: int = Field(
            default=-1, description="Max tokens to generate. -1 = unset."
        )
        REPEAT_PENALTY: float = Field(
            default=-1.0, description="repeat_penalty (local backends). -1 = unset."
        )
        FREQUENCY_PENALTY: float = Field(
            default=-999.0, description="frequency_penalty (-2..2). -999 = unset."
        )
        PRESENCE_PENALTY: float = Field(
            default=-999.0, description="presence_penalty (-2..2). -999 = unset."
        )
        SEED: int = Field(
            default=-1, description="Sampling seed for reproducibility. -1 = unset."
        )
        # JSON escape hatches for anything not covered above. Merge order:
        # EXTRA_PARAMS (base) < the dedicated valves above < per-phase JSON.
        EXTRA_PARAMS_JSON: str = Field(
            default="",
            description='Base extra params as a JSON object, e.g. {"tfs_z": 1.0}. Applied to all phases.',
        )
        PLANNING_PARAMS_JSON: str = Field(
            default="",
            description="Per-phase param overrides for PLANNING as a JSON object.",
        )
        EXECUTION_PARAMS_JSON: str = Field(
            default="",
            description="Per-phase param overrides for EXECUTION as a JSON object.",
        )
        SYNTHESIS_PARAMS_JSON: str = Field(
            default="",
            description="Per-phase param overrides for SYNTHESIS/REVIEW as a JSON object.",
        )
        # --- MCP tool calling (execution phase) -------------------------------
        MCP_ENABLED: bool = Field(
            default=True,
            description="Let execution tasks call MCP tools (requires the `mcp` package and a reachable server).",
        )
        MCP_URL: str = Field(
            default="http://localhost:8082/memory",
            description="streamable-http MCP server URL (used when MCP_SERVERS_JSON is empty).",
        )
        MCP_SERVERS_JSON: str = Field(
            default="",
            description='Advanced: JSON array of MCP servers, e.g. [{"name":"memory","transport":"streamable-http","url":"http://host:8082/memory"}]. Overrides MCP_URL.',
        )
        MCP_AUTODISCOVER: bool = Field(
            default=True,
            description="Auto-discover MCP/tool servers from Open WebUI's configuration (used when MCP_SERVERS_JSON is empty). Falls back to MCP_URL.",
        )
        MAX_TOOL_ITERATIONS: int = Field(
            default=6,
            description="Max tool-call rounds per task before forcing a final answer.",
        )
        TOOL_RESULT_LIMIT: int = Field(
            default=4000,
            description="Per tool-result character limit fed back to the model.",
        )
        MCP_DISCOVERY_CACHE_TTL: float = Field(
            default=300.0,
            description="Seconds an MCP server's discovered tool list stays cached across messages, so a message that never calls a tool skips the reachability check + handshake. 0 disables the cache.",
        )
        MCP_TOOL_DESCRIPTION_LIMIT: int = Field(
            default=200,
            description="Max characters of each tool's description sent to the model as part of its tool schema. Lower this if a server exposes many tools with long descriptions and requests feel slow even when no tool ends up being called.",
        )
        MCP_TOOL_GATE_HEURISTIC: bool = Field(
            default=True,
            description="When Kev hasn't classified whether tools are needed, only attach tool schemas to a request when its text plausibly matches a connected tool's name. Disable to always attach every tool (slower with many tools, but never risks missing one on an oddly-worded request).",
        )
        PLAN_MODE: bool = Field(
            default=True,
            description="Decompose into tasks first. Disable for a single-pass answer.",
        )
        AUTO_SKIP_PLAN: bool = Field(
            default=True,
            description="Even with PLAN_MODE on, skip planning and answer directly for short/simple requests.",
        )
        SHORT_TASK_MAX_WORDS: int = Field(
            default=20,
            description="Max word count for a request to be considered 'short' by AUTO_SKIP_PLAN.",
        )
        MAX_TASKS: int = Field(default=20, description="Max tasks in the plan.")
        TASK_RESULT_LIMIT: int = Field(
            default=6000,
            description="Per-task result character soft-limit when feeding results forward.",
        )
        ENABLE_REVIEW: bool = Field(
            default=False, description="Add a final self-review/refine pass."
        )
        CHAT_PLAN_PERSIST: bool = Field(
            default=True,
            description="Keep a chat's decomposed plan (tasks and which are done) alive across messages in the same chat, so a follow-up continues the running plan instead of re-planning from scratch. Task execution itself is unaffected - still one-shot per task.",
        )
        CHAT_PLAN_TTL: float = Field(
            default=3600.0,
            description="Seconds a chat's plan stays cached with no new message before it's dropped.",
        )
        EMIT_STATUS: bool = Field(
            default=True, description="Emit progress as status updates in the UI."
        )
        KEV_URL: str = Field(
            default="http://10.0.0.10:8009",
            description="Kev System One endpoint (e.g. http://127.0.0.1:8009). Empty = no typed decisions, the planner behaves as before.",
        )
        KEV_API_KEY: str = Field(
            default="",
            description="Bearer token, when the Kev endpoint was started with KEV_API_KEY set.",
        )
        KEV_DECIDE_PLANNING: bool = Field(
            default=True,
            description="Let Kev decide whether a request is simple (answered in one pass) or needs decomposing, instead of the word-count heuristic.",
        )
        KEV_CLASSIFY_REQUEST: bool = Field(
            default=True,
            description="Ask Kev whether the request needs thinking (off: reasoning disabled) and whether it is artistic or scientific (sets the base temperature).",
        )
        KEV_ARTISTIC_TEMPERATURE: float = Field(
            default=1.0,
            description="Base temperature when Kev judges the request artistic/creative.",
        )
        KEV_SCIENTIFIC_TEMPERATURE: float = Field(
            default=0.3,
            description="Base temperature when Kev judges the request scientific/factual.",
        )
        KEV_ASK_TOOLS: bool = Field(
            default=True,
            description="Ask Kev whether the request needs tools (lookup, live data, memory); if not, the model is not offered any.",
        )
        FAST_SMALL_TALK: bool = Field(
            default=True,
            description="Greetings, thanks, ok: answer at once, without Kev, planning, tools or thinking.",
        )
        THINK_IN_PLANNING: bool = Field(
            default=False,
            description="When Kev turns thinking on, let the planning and synthesis phases think too (slower). Off = only task execution thinks.",
        )
        SKIP_SINGLE_TASK_SYNTHESIS: bool = Field(
            default=True,
            description="A plan of one task returns that task's result instead of a synthesis pass.",
        )
        SKIP_WARMUP_IF_LOADED: bool = Field(
            default=True,
            description="Skip the warm-up call when Ollama already has the model loaded (checks /api/ps).",
        )
        KEV_CHECK_TASKS: bool = Field(
            default=True,
            description="After each task, ask Kev whether the result carries out the task; retry if it does not.",
        )
        KEV_GATE_REVIEW: bool = Field(
            default=True,
            description="Run the review pass only when Kev judges the draft incomplete. Overrides ENABLE_REVIEW in both directions.",
        )
        SYSTEM_PROMPT: str = Field(
            default=DEFAULT_SYSTEM_PROMPT, description="Base system prompt."
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "planner-standalone", "name": "Planner (Standalone)"}]

    def _resolve_model(self, request: Any, body: dict) -> str:
        """Pick the model the planner should drive.

        Priority: explicit PLANNER_MODEL valve -> Open WebUI's configured default
        model(s) -> the first available real (non-pipe/non-arena) model.
        """
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
            # Skip this planner / any other function-pipe model to avoid recursion.
            if info.get("pipe") or info.get("arena"):
                return False
            if "planner-standalone" in str(mid):
                return False
            return True

        # 1. Open WebUI configured default model(s).
        try:
            defaults = getattr(request.app.state.config, "DEFAULT_MODELS", "") or ""
            for mid in [m.strip() for m in defaults.split(",") if m.strip()]:
                if mid in models and _is_usable(mid, models.get(mid, {})):
                    return mid
        except Exception:
            pass

        # 2. First usable real model in the registry.
        for mid, info in models.items():
            if _is_usable(mid, info):
                return mid

        return ""

    @staticmethod
    def _normalize_mcp_entry(entry: Any) -> Optional[dict]:
        """Coerce an OWUI tool/MCP server entry into our server dict, or None.

        Only entries that look like a native MCP transport (streamable-http /
        sse / stdio) are kept; OpenAPI (`mcpo`) tool servers are skipped because
        the MCPClient speaks the MCP protocol, not OpenAPI.
        """
        if not isinstance(entry, dict):
            return None
        # OWUI nests connection details under "config" in some versions.
        cfg = entry.get("config") if isinstance(entry.get("config"), dict) else {}
        get = lambda k: entry.get(k) or cfg.get(k)

        url = get("url") or get("server_url") or get("base_url")
        command = get("command")
        raw_type = str(
            get("type") or get("transport") or get("spec_type") or ""
        ).lower()

        # Decide transport.
        transport = None
        if command:
            transport = "stdio"
        elif raw_type in ("sse",):
            transport = "sse"
        elif raw_type in ("mcp", "streamable-http", "streamable_http", "http"):
            transport = "streamable-http"
        elif url and ("mcp" in str(url).lower() or "/memory" in str(url).lower()):
            # Heuristic: a URL that looks MCP-ish with no explicit OpenAPI type.
            if raw_type not in ("openapi", "openapi-json", "tool", "tool_server"):
                transport = "streamable-http"

        if transport is None:
            return None
        # Name: explicit name/id, else derive something meaningful from the URL
        # (host + last path segment) so a discovered list isn't all "mcp".
        name = str(get("name") or get("id") or "").strip()
        if not name and url:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(str(url))
                seg = (parsed.path or "").rstrip("/").rsplit("/", 1)[-1]
                host = (parsed.hostname or "").replace(".", "-")
                name = "-".join(p for p in (host, seg) if p) or "mcp"
            except Exception:
                name = "mcp"
        name = name or "mcp"
        if transport == "stdio":
            return {
                "name": name,
                "transport": "stdio",
                "command": command,
                "args": get("args") or [],
                "env": get("env"),
            }
        if not url:
            return None
        return {"name": name, "transport": transport, "url": str(url)}

    def _discover_mcp_servers(self, request: Any) -> list[dict]:
        """Best-effort discovery of MCP servers from Open WebUI's state/config."""
        candidates: list = []
        # Known locations across OWUI versions (all optional / version-dependent).
        try:
            cfg = request.app.state.config
            for attr in (
                "MCP_SERVER_CONNECTIONS",
                "TOOL_SERVER_CONNECTIONS",
            ):
                val = getattr(cfg, attr, None)
                if isinstance(val, list):
                    candidates.extend(val)
        except Exception:
            pass
        try:
            for attr in ("TOOL_SERVERS", "MCP_SERVERS"):
                val = getattr(request.app.state, attr, None)
                if isinstance(val, list):
                    candidates.extend(val)
        except Exception:
            pass

        servers: list[dict] = []
        seen: set = set()
        for entry in candidates:
            srv = self._normalize_mcp_entry(entry)
            if not srv:
                continue
            key = srv.get("url") or srv.get("command")
            if key in seen:
                continue
            seen.add(key)
            servers.append(srv)
        return servers

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
        __metadata__: Optional[dict] = None,
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
            # Some Open WebUI versions expose this as an async method; awaiting a
            # coroutine here prevents `'coroutine' object has no attribute 'role'`
            # when the un-awaited result is later passed to generate_chat_completion.
            if asyncio.iscoroutine(user):
                user = await user

        try:
            owned_by = (
                (getattr(__request__.app.state, "MODELS", {}) or {}).get(model_id) or {}
            ).get("owned_by")
        except Exception:
            owned_by = None

        def _apply_params(form: dict, params: Optional[dict]) -> None:
            """Put the sampling params where the backend reads them. Open WebUI's OpenAI->Ollama conversion keeps
            only `options` (and moves `think` from there to the root); a top-level temperature never reaches Ollama.
            """
            params = {k: v for k, v in (params or {}).items() if v is not None}
            if owned_by == "ollama":
                form["options"] = {"temperature": form["temperature"], **params}
            else:
                form.update(_openai_think_fields(params) or {})

        async def progress(message: str) -> None:
            if __event_emitter__ and valves.EMIT_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": message, "done": False},
                    }
                )

        async def complete(
            system_prompt: str,
            user_message: str,
            temperature: Optional[float] = None,
            json_object: bool = False,
            params: Optional[dict] = None,
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
            # Extra sampling params are forwarded to the backend.
            _apply_params(base_form, params)

            # Request structured JSON when asked, degrading across backends:
            #   1. json_schema  -> LM Studio / OpenAI structured outputs (strict,
            #      no reasoning leakage; the planner's preferred path)
            #   2. json_object  -> older OpenAI-style JSON mode
            #   3. plain        -> last resort; prompt asks for raw JSON and
            #      `_extract_json_object` recovers it from prose/think blocks.
            # Each failed variant returns an error payload, so we try the next.
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
                err = _owui_response_error(response)
                if err:
                    continue  # try the next (less constrained) variant
                content = _owui_extract_content(response)
                if content:
                    return content

            raise RuntimeError(
                "Planner model returned an empty response. "
                f"Raw response ({type(last_response).__name__}): "
                f"{_owui_debug_body(last_response)[:1200]}"
            )

        async def chat(
            messages: list,
            tools: Optional[list] = None,
            temperature: Optional[float] = None,
            params: Optional[dict] = None,
        ) -> dict:
            form_data: dict = {
                "model": model_id,
                "stream": False,
                "temperature": (
                    temperature if temperature is not None else valves.TEMPERATURE
                ),
                "messages": messages,
            }
            _apply_params(form_data, params)
            if tools:
                form_data["tools"] = tools
                form_data["tool_choice"] = "auto"
            response = await generate_chat_completion(__request__, form_data, user=user)
            err = _owui_response_error(response)
            if err:
                # Surface as content so the loop can still terminate gracefully.
                return {"content": f"[model error: {err}]", "tool_calls": None}
            return _owui_extract_message(response)

        def _opt_temp(value: float) -> Optional[float]:
            # A negative valve value means "auto / inherit"; pass None through.
            return None if value is None or value < 0 else value

        def _parse_json_params(raw: str) -> dict:
            raw = (raw or "").strip()
            if not raw:
                return {}
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        # Base sampling params from the dedicated valves (skipping sentinels),
        # then merge the base EXTRA_PARAMS_JSON on top.
        base_params: dict = {}
        if valves.TOP_P >= 0:
            base_params["top_p"] = valves.TOP_P
        if valves.TOP_K >= 0:
            base_params["top_k"] = valves.TOP_K
        if valves.MIN_P >= 0:
            base_params["min_p"] = valves.MIN_P
        if valves.MAX_TOKENS >= 0:
            base_params["max_tokens"] = valves.MAX_TOKENS
        if valves.REPEAT_PENALTY >= 0:
            base_params["repeat_penalty"] = valves.REPEAT_PENALTY
        if valves.FREQUENCY_PENALTY > -900:
            base_params["frequency_penalty"] = valves.FREQUENCY_PENALTY
        if valves.PRESENCE_PENALTY > -900:
            base_params["presence_penalty"] = valves.PRESENCE_PENALTY
        if valves.SEED >= 0:
            base_params["seed"] = valves.SEED
        base_params.update(_parse_json_params(valves.EXTRA_PARAMS_JSON))

        # MCP servers: explicit JSON array wins; else autodiscovery from OWUI;
        # else the single MCP_URL fallback.
        mcp_servers: list = []
        if valves.MCP_ENABLED:
            raw_servers = (valves.MCP_SERVERS_JSON or "").strip()
            if raw_servers:
                try:
                    parsed = json.loads(raw_servers)
                    if isinstance(parsed, list):
                        mcp_servers = [s for s in parsed if isinstance(s, dict)]
                except Exception:
                    await progress("⚠️ MCP_SERVERS_JSON is not valid JSON; ignoring")
            if not mcp_servers and valves.MCP_AUTODISCOVER:
                discovered = self._discover_mcp_servers(__request__)
                if discovered:
                    names = ", ".join(s.get("name", "?") for s in discovered)
                    await progress(
                        f"Discovered {len(discovered)} MCP server(s): {names}"
                    )
                    mcp_servers = discovered
            if not mcp_servers and valves.MCP_URL.strip():
                mcp_servers = [
                    {
                        "name": "memory",
                        "transport": "streamable-http",
                        "url": valves.MCP_URL.strip(),
                    }
                ]

        # Ollama: is the model already loaded (then the warm-up is wasted), and can it think at all (then Kev is
        # not asked whether it should)?
        loaded, can_think = None, None
        if owned_by == "ollama":
            try:
                from open_webui.models.config import Config

                base_urls = list(await Config.get("ollama.base_urls", []) or [])
            except Exception:
                try:
                    from open_webui.config import OLLAMA_BASE_URLS as base_urls
                except Exception:
                    base_urls = []
            try:
                info = (getattr(__request__.app.state, "MODELS", {}) or {}).get(
                    model_id
                ) or {}
                name = (info.get("ollama") or {}).get("model") or model_id
            except Exception:
                name = model_id
            loaded, can_think = await _ollama_model_state(base_urls, name)

        config = PlannerConfig(
            model=model_id,
            temperature=valves.TEMPERATURE,
            planning_temperature=_opt_temp(valves.PLANNING_TEMPERATURE),
            execution_temperature=_opt_temp(valves.EXECUTION_TEMPERATURE),
            synthesis_temperature=_opt_temp(valves.SYNTHESIS_TEMPERATURE),
            sampling_params=base_params,
            planning_params=_parse_json_params(valves.PLANNING_PARAMS_JSON),
            execution_params=_parse_json_params(valves.EXECUTION_PARAMS_JSON),
            synthesis_params=_parse_json_params(valves.SYNTHESIS_PARAMS_JSON),
            plan_mode=valves.PLAN_MODE,
            auto_skip_plan_for_short_tasks=valves.AUTO_SKIP_PLAN,
            short_task_max_words=valves.SHORT_TASK_MAX_WORDS,
            max_tasks=valves.MAX_TASKS,
            task_result_limit=valves.TASK_RESULT_LIMIT,
            enable_review=valves.ENABLE_REVIEW,
            mcp_servers=mcp_servers,
            max_tool_iterations=valves.MAX_TOOL_ITERATIONS,
            tool_result_limit=valves.TOOL_RESULT_LIMIT,
            mcp_discovery_cache_ttl=valves.MCP_DISCOVERY_CACHE_TTL,
            mcp_tool_description_limit=valves.MCP_TOOL_DESCRIPTION_LIMIT,
            mcp_tool_gate_heuristic=valves.MCP_TOOL_GATE_HEURISTIC,
            kev_url=valves.KEV_URL,
            kev_api_key=valves.KEV_API_KEY,
            kev_decide_planning=valves.KEV_DECIDE_PLANNING,
            kev_classify_request=valves.KEV_CLASSIFY_REQUEST,
            kev_artistic_temperature=valves.KEV_ARTISTIC_TEMPERATURE,
            kev_scientific_temperature=valves.KEV_SCIENTIFIC_TEMPERATURE,
            kev_ask_tools=valves.KEV_ASK_TOOLS,
            fast_small_talk=valves.FAST_SMALL_TALK,
            think_in_planning=valves.THINK_IN_PLANNING,
            skip_single_task_synthesis=valves.SKIP_SINGLE_TASK_SYNTHESIS,
            model_can_think=can_think,
            kev_check_tasks=valves.KEV_CHECK_TASKS,
            kev_gate_review=valves.KEV_GATE_REVIEW,
            chat_plan_persist=valves.CHAT_PLAN_PERSIST,
            chat_plan_ttl=valves.CHAT_PLAN_TTL,
            verbose=False,
        )
        planner = StandalonePlanner(
            complete=complete,
            config=config,
            system_prompt=valves.SYSTEM_PROMPT,
            progress=progress,
            chat=chat,
        )

        if not valves.PLANNER_MODEL.strip():
            await progress(f"Using model: {model_id}")

        # Warm up the model before the real work starts. Local backends like
        # Ollama can take tens of seconds to minutes to load a model on its
        # first call; if that cold start happens mid-plan (buried inside one
        # of the Planner's many sequential calls) it's more likely to trip an
        # upstream timeout (reverse proxy, Open WebUI itself) and abort the
        # whole run. Doing one throwaway call up front absorbs that cost
        # before anything depends on it. Best-effort: failures here are
        # ignored and surface naturally on the real calls instead.
        if loaded and valves.SKIP_WARMUP_IF_LOADED:
            await progress(f"Model already loaded: {model_id}")
        else:
            await progress(f"Warming up model: {model_id}...")
            try:
                await complete(
                    "You are a helpful assistant.",
                    "Reply with only the word: ready",
                    0.0,
                    False,
                    {"max_tokens": 5, **({"think": False} if can_think else {})},
                )
            except Exception:
                pass

        chat_id = None
        if isinstance(__metadata__, dict):
            chat_id = __metadata__.get("chat_id") or __metadata__.get("session_id")
        if not chat_id:
            chat_id = body.get("chat_id")

        try:
            result = await planner.run(goal, chat_id=chat_id)
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
        description="Standalone agentic planner (no subagents, no Open WebUI)."
    )
    parser.add_argument("goal", nargs="*", help="The request/goal to fulfill.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default: $PLANNER_MODEL or gpt-4o-mini)",
    )
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key", default=None, help="API key (default: $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Base sampling temperature"
    )
    parser.add_argument(
        "--planning-temperature",
        type=float,
        default=None,
        help="Planning-phase temperature (default: auto = min(base, 0.4))",
    )
    parser.add_argument(
        "--execution-temperature",
        type=float,
        default=None,
        help="Execution-phase temperature (default: base)",
    )
    parser.add_argument(
        "--synthesis-temperature",
        type=float,
        default=None,
        help="Synthesis/review-phase temperature (default: base)",
    )
    # Extra sampling params (applied to all phases).
    parser.add_argument(
        "--top-p", type=float, default=None, help="Nucleus sampling top_p"
    )
    parser.add_argument("--top-k", type=int, default=None, help="top_k sampling")
    parser.add_argument("--min-p", type=float, default=None, help="min_p sampling")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens to generate"
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="repeat_penalty (local backends)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="frequency_penalty (-2..2)",
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=None, help="presence_penalty (-2..2)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    parser.add_argument(
        "--params",
        default=None,
        help="Base extra params as JSON, e.g. '{\"tfs_z\":1.0}' (applied to all phases)",
    )
    parser.add_argument(
        "--planning-params",
        default=None,
        help="Per-phase param overrides for planning as JSON",
    )
    parser.add_argument(
        "--execution-params",
        default=None,
        help="Per-phase param overrides for execution as JSON",
    )
    parser.add_argument(
        "--synthesis-params",
        default=None,
        help="Per-phase param overrides for synthesis/review as JSON",
    )
    # MCP tool calling (execution phase).
    parser.add_argument(
        "--mcp-url",
        default=None,
        help="streamable-http MCP server URL (enables tool calling)",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help="JSON array of MCP servers (overrides --mcp-url)",
    )
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=None,
        help="Max tool-call rounds per task",
    )
    parser.add_argument(
        "--tool-result-limit", type=int, default=None, help="Per tool-result char limit"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Max tasks in the plan"
    )
    parser.add_argument(
        "--no-plan", action="store_true", help="Disable planning; single-pass"
    )
    parser.add_argument(
        "--force-plan",
        action="store_true",
        help="Always plan, even for short/simple requests (disables auto-skip)",
    )
    parser.add_argument(
        "--short-task-max-words",
        type=int,
        default=None,
        help="Max word count for a request to be treated as 'short' (auto-skip planning)",
    )
    parser.add_argument(
        "--review", action="store_true", help="Add a self-review/refine pass"
    )
    parser.add_argument(
        "--kev-url",
        default=None,
        help="Kev System One endpoint for the typed decisions (default: $KEV_URL; empty disables them)",
    )
    parser.add_argument(
        "--no-kev",
        action="store_true",
        help="Ignore $KEV_URL: decide with the word-count heuristic and no task or review checks",
    )
    parser.add_argument(
        "--no-kev-classify",
        action="store_true",
        help="Keep Kev's other decisions but not the thinking switch and the artistic/scientific temperature",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging"
    )
    parser.add_argument("--json", action="store_true", help="Emit full result as JSON")
    parser.add_argument(
        "--system-prompt", default=None, help="Override the base system prompt"
    )
    return parser


async def _run_cli(args: argparse.Namespace, goal: str) -> int:
    config = PlannerConfig()
    if args.model is not None:
        config.model = args.model
    if args.api_url is not None:
        config.api_url = args.api_url
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.planning_temperature is not None:
        config.planning_temperature = args.planning_temperature
    if args.execution_temperature is not None:
        config.execution_temperature = args.execution_temperature
    if args.synthesis_temperature is not None:
        config.synthesis_temperature = args.synthesis_temperature

    def _cli_json(raw: Optional[str]) -> dict:
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: invalid JSON for params: {exc}")

    base_params: dict = _cli_json(args.params)
    for cli_key, param_key in (
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("max_tokens", "max_tokens"),
        ("repeat_penalty", "repeat_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("seed", "seed"),
    ):
        value = getattr(args, cli_key)
        if value is not None:
            base_params[param_key] = value
    config.sampling_params = base_params
    config.planning_params = _cli_json(args.planning_params)
    config.execution_params = _cli_json(args.execution_params)
    config.synthesis_params = _cli_json(args.synthesis_params)

    # MCP servers: --mcp-config (JSON array) wins; else --mcp-url.
    if args.mcp_config:
        try:
            parsed = json.loads(args.mcp_config)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: invalid JSON for --mcp-config: {exc}")
        if isinstance(parsed, list):
            config.mcp_servers = [s for s in parsed if isinstance(s, dict)]
    elif args.mcp_url:
        config.mcp_servers = [
            {"name": "memory", "transport": "streamable-http", "url": args.mcp_url}
        ]
    if args.max_tool_iterations is not None:
        config.max_tool_iterations = args.max_tool_iterations
    if args.tool_result_limit is not None:
        config.tool_result_limit = args.tool_result_limit

    if args.max_tasks is not None:
        config.max_tasks = args.max_tasks
    if args.no_plan:
        config.plan_mode = False
    if args.force_plan:
        config.auto_skip_plan_for_short_tasks = False
    if args.short_task_max_words is not None:
        config.short_task_max_words = args.short_task_max_words
    if args.review:
        config.enable_review = True
    if args.kev_url is not None:
        config.kev_url = args.kev_url
    if args.no_kev:
        config.kev_url = ""
    if args.no_kev_classify:
        config.kev_classify_request = False
    if args.quiet:
        config.verbose = False

    async def progress(message: str) -> None:
        if config.verbose:
            print(f"[planner] {message}", file=sys.stderr, flush=True)

    planner = StandalonePlanner(
        complete=make_openai_completer(config),
        config=config,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        progress=progress,
        chat=make_openai_chat(config) if config.mcp_servers else None,
    )

    try:
        result = await planner.run(goal)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "goal": result.goal,
                    "elapsed_seconds": round(result.elapsed_seconds, 2),
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "description": t.description,
                            "related_tasks": t.related_tasks,
                            "tools": t.tools,
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
        if config.verbose:
            print(
                f"\n[planner] done in {result.elapsed_seconds:.1f}s "
                f"({len(result.tasks)} task(s))",
                file=sys.stderr,
            )
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
