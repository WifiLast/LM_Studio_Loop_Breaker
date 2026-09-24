"""Planner (Standalone) - core module: shared types, prompts, text/JSON helpers.

Part of a 4-file split of the original planning_standalone.py:
    planning_core.py       <- this file: data model, prompts, text/JSON helpers
    planning_kev_mcp.py    <- Kev decision client + MCP client
    planning_engine.py     <- StandalonePlanner (the agentic loop)
    planning_backends.py   <- completion backends, Open WebUI Pipe, CLI

Run `python build_filter.py` to recombine all four into a single-file Open WebUI
Function (planning_standalone.py). This module has no dependency on the other
three: it is the data model (PlannerConfig, Task, PlannerResult, ToolMetrics),
the prompt templates (PromptBuilder), and text/JSON extraction helpers shared by
planning and execution.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

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
    # A task that was handed a math/logic tool (integrate_function, check_equation,
    # z3_solve_constraints, z3_run_script, ...) is retried if it never actually called a
    # verification tool - i.e. it "derived" a formula or strategy by free-form reasoning
    # alone. Measured failure modes on a weak model: math_run.log burned pages of hand
    # algebra re-deriving the same Beta-function identity three times with sign errors;
    # riddle_run.log burned ~700 lines hand-verifying a 3-question strategy for the
    # Three Gods puzzle case by case. A `check_equation` or `z3_run_script` call proves
    # (or disproves) either in one round trip instead.
    kev_check_math: bool = True
    kev_math_verified_threshold: float = 0.5
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
        base_system: str,
        max_tasks: int,
        tools_catalog: str = "",
        has_math_tools: bool = False,
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
        if has_math_tools:
            tools_section += (
                "\n### MATH / LOGIC DERIVATIONS\n"
                "When the request asks you to derive, prove, or solve something "
                "mathematical OR a logic riddle/puzzle (a formula, an integral, an "
                "equation, a logical claim, a 'who is lying' / finite-state puzzle "
                "like the Three Gods or knights-and-knaves), do NOT plan one big task "
                "that derives and trusts its own reasoning end to end - a single long "
                "freeform derivation is where small models silently drop a sign, "
                "mishandle a lying/negation case, or reuse a wrong identity, then "
                "spend many turns re-deriving it by hand. Instead split it into "
                "separate small tasks:\n"
                "  1. **Numeric/case exploration** - a task (with tools such as "
                "`integrate_function`/`evaluate`/`solve_equation`, or by enumerating "
                "the puzzle's finite cases such as every role/assignment combination) "
                "that establishes the ground truth every later step is checked "
                "against.\n"
                "  2. **Derive** - a task (no tools, or the same tools) that proposes "
                "the closed form / proof / strategy / answer from that exploration and "
                "known identities or rules. For a logic riddle, first name the "
                "entities, their possible states, and the clues as plain statements - "
                "skipping that extraction step is the most common cause of a wrong "
                "constraint encoding later.\n"
                "  3. **Verify** - a task that MUST call a verification tool "
                "(`check_equation`, `check_consistency`, `check_entailment`, "
                "`z3_solve_constraints`, `z3_prove_theorem`, `z3_run_script`, or "
                "`verify_claims`) to check the derivation. For a formula: check it "
                "against the numeric results from step 1 at every sample point. For a "
                "logical claim: check satisfiability/entailment with Z3. For a puzzle "
                "with a decision tree or strategy (e.g. 'which questions do I ask'): "
                "use `z3_run_script` to encode every case (each role assignment, and "
                "every coin-flip/branch outcome) as constraints and check the claimed "
                "property (e.g. 'no two different cases produce the same observed "
                "answers') holds for ALL of them at once, rather than checking cases "
                "one at a time by hand. A derivation is only accepted once the tool "
                "confirms it; if it does not confirm, the derivation must be redone, "
                "not patched by more manual reasoning.\n"
                "- Give the verify task `related_tasks` pointing at both the "
                "exploration and derive tasks.\n"
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
    def execution_prompt(base_system: str, math_verification: bool = False) -> str:
        prompt = (
            f"{base_system}\n\n"
            "### EXECUTION PHASE - ACTIVE\n"
            "You are executing ONE task of a larger plan. Produce the complete, "
            "high-quality output for THIS task only.\n"
            "- Use any prerequisite task results provided as context.\n"
            "- Do not restate the whole plan; focus on delivering this task's deliverable.\n"
            "- If the task involves code, provide complete, runnable code.\n"
            "- Be precise and self-contained: downstream tasks may consume your output verbatim.\n"
        )
        if math_verification:
            prompt += (
                "\n### MATH / LOGIC VERIFICATION - REQUIRED\n"
                "This task has a verification tool available (`check_equation`, "
                "`check_consistency`, `check_entailment`, `z3_solve_constraints`, "
                "`z3_prove_theorem`, `z3_run_script`, or `verify_claims`). You MUST "
                "call it before writing your final answer - do not verify by "
                "re-deriving the same reasoning by hand again. Pick the tool that "
                "fits the shape of the claim:\n"
                "  - A numeric formula against a known result -> `check_equation`.\n"
                "  - \"Find an assignment that fits all the clues\" (zebra puzzle, "
                "seating, who-owns-what) -> `z3_solve_constraints`.\n"
                "  - \"Is this single guess/strategy forced, given the clues?\" (e.g. "
                "mislabeled boxes, the three gods) -> `z3_run_script`: enumerate every "
                "valid case with `itertools.permutations`/`itertools.product` "
                "(preloaded, no import needed) and check the property holds for ALL "
                "of them, not just one or two examples you picked by hand.\n"
                "  - \"Does knowing X prove Y?\" (knights and knaves, does a claim "
                "follow from premises) -> `check_entailment` or `z3_prove_theorem`; "
                "for custom categories/roles/permutations prefer `z3_run_script` with "
                "`solver.add(Not(claim))` - `unsat` proves the claim.\n"
                "  - \"Are these facts even possible together?\" -> `check_consistency` "
                "or `z3_solve_constraints`, looking for `unsat`/`contradicts`.\n"
                "  - Needs quantifiers, custom sorts, or a loop over every case -> "
                "`z3_run_script` (the full Z3 Python API plus `itertools`).\n"
                "If the tool's verdict (`sat`/`unsat`/`entailed`/`contradicts`) "
                "disagrees with your derivation, trust the tool and redo the "
                "derivation instead of arguing with it - state the answer from the "
                "solver's actual output, not from a recalled or assumed value.\n"
            )
        return prompt

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


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_DANGLING_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)


def _merge_reasoning(content: Optional[str], reasoning: Optional[str]) -> str:
    """Fold a backend's separate reasoning field into `content` as a
    `<think>...</think>` block, so a model that reports thinking out-of-band isn't
    silently dropped by callers that only look at `content`. A no-op when there is no
    separate reasoning, or it is already embedded (content already starts with a think
    block)."""
    content = content or ""
    if not reasoning or content.lstrip().lower().startswith("<think>"):
        return content
    return f"<think>{reasoning}</think>{content}"


# Field names different backends use for a model's separate (non-inline) reasoning:
# `reasoning_content` (vLLM/SGLang/DeepSeek-style, OpenAI-compatible), `reasoning`, and
# `thinking` (Ollama's native thinking-mode field - NOT reasoning_content, which Ollama
# never sets, so checking only that name silently misses it).
_REASONING_FIELD_NAMES = ("reasoning_content", "reasoning", "thinking")


def _extract_reasoning(message: object) -> Optional[str]:
    """Pull a model's separate reasoning/thinking text off a message, whether it's a
    dict (Open WebUI / raw JSON) or an object with attributes (the `openai` SDK's
    message model)."""
    if isinstance(message, dict):
        for name in _REASONING_FIELD_NAMES:
            value = message.get(name)
            if value:
                return value
        return None
    for name in _REASONING_FIELD_NAMES:
        value = getattr(message, name, None)
        if value:
            return value
    extra = getattr(message, "model_extra", None) or {}
    for name in _REASONING_FIELD_NAMES:
        value = extra.get(name)
        if value:
            return value
    return None


def _extract_think_block(text: str) -> tuple[str, str]:
    """Split a completion into (thought, remainder): the content of a
    `<think>...</think>` block (closed, or dangling - reasoning that ran out of budget
    before answering) and everything else. Empty thought when there is none."""
    if not text or "<think>" not in text.lower():
        return "", text
    match = _THINK_BLOCK_RE.search(text)
    if match:
        thought = match.group(1).strip()
        remainder = (text[: match.start()] + text[match.end() :]).strip()
        return thought, remainder
    match = _THINK_DANGLING_RE.search(text)
    if match:
        return match.group(1).strip(), text[: match.start()].strip()
    return "", text


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


