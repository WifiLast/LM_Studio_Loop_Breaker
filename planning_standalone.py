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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

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
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
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
    max_tasks: int = 20
    task_result_limit: int = 6000
    enable_review: bool = False
    request_timeout: float = 120.0
    verbose: bool = True
    # MCP tool calling (execution phase only). Each server is a dict:
    #   {"name": "memory", "transport": "streamable-http", "url": "http://host:8082/memory"}
    #   {"name": "x", "transport": "stdio", "command": "python", "args": [...], "env": {...}}
    mcp_servers: list = field(default_factory=list)
    max_tool_iterations: int = 6
    tool_result_limit: int = 4000
    # Loop detection thresholds per phase
    plan_loop_threshold: int = 2
    execution_loop_threshold: int = 3
    synthesis_loop_threshold: int = 2
    # Token budget awareness
    max_tokens_per_task: int = 8000
    warn_at_percent: float = 0.8
    # Resume from memory
    enable_memory_resume: bool = True


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


class MCPClient:
    """Connects to one or more MCP servers and exposes their tools.

    Usage:
        async with MCPClient(servers) as mcp:
            tools = mcp.openai_tools()
            text = await mcp.call_tool(name, args)

    Gracefully degrades: if the `mcp` package is missing or a server cannot be
    reached, the affected server is skipped (with a note) rather than aborting.
    """

    def __init__(self, servers: list[dict], result_limit: int = 4000):
        self.servers = servers or []
        self.result_limit = result_limit
        self._stack: Any = None
        self.sessions: dict[str, Any] = {}
        # tool_name -> (server_name, openai_tool_schema)
        self.tools: dict[str, tuple] = {}
        self.notes: list[str] = []

    async def __aenter__(self) -> "MCPClient":
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for srv in self.servers:
            name = srv.get("name") or srv.get("url") or "mcp"
            try:
                session = await self._connect(srv)
                await session.initialize()
                self.sessions[name] = session
                listed = await session.list_tools()
                for tool in listed.tools:
                    schema = tool.inputSchema or {"type": "object", "properties": {}}
                    self.tools[tool.name] = (
                        name,
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": (tool.description or "")[:1024],
                                "parameters": schema,
                            },
                        },
                    )
            except Exception as exc:  # skip unreachable / misconfigured servers
                self.notes.append(f"MCP server '{name}' unavailable: {exc}")
        return self

    async def _connect(self, srv: dict) -> Any:
        from mcp import ClientSession

        transport = (srv.get("transport") or "streamable-http").lower()
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
        return min(self.config.temperature, 0.4)

    def _phase_params(self, phase: str) -> Optional[dict]:
        """Merge base sampling params with this phase's overrides."""
        override = {
            "planning": self.config.planning_params,
            "execution": self.config.execution_params,
            "synthesis": self.config.synthesis_params,
        }.get(phase) or {}
        merged = {**(self.config.sampling_params or {}), **override}
        return merged or None

    def _adjusted_temperature(self, attempt: int) -> Optional[float]:
        """Increase temperature on retry to escape loops (exponential backoff)."""
        base = self.config.execution_temperature or self.config.temperature
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
        if self._mcp is None:
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
        if self._mcp is None:
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

    async def execute_task(self, goal: str, task: Task, done: dict[str, Task], attempt: int = 0) -> str:
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
        context = "\n\n".join(context_blocks) if context_blocks else "(no prior results)"

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
        if self._mcp is None or not names:
            return []
        try:
            by_name = {
                (t.get("function") or {}).get("name"): t
                for t in self._mcp.openai_tools()
            }
        except Exception:
            return []
        return [by_name[n] for n in names if n in by_name]

    async def _execute_with_tools(self, user_message: str, tools: list[dict], attempt: int = 0) -> str:
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
            await self._emit(f"  ↳ attempt {attempt}: tools disabled, forcing reasoning-only")
            use_tools = []
        elif attempt == 1:
            # Filter out tools that caused loops in previous attempts
            problematic = [
                name
                for name, metrics in self._tool_metrics.items()
                if metrics.loop_count > 0
            ]
            if problematic:
                await self._emit(f"  ↳ disabling problematic tools: {', '.join(problematic)}")
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
                fn = (call.get("function") or {})
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
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
            f"--- {t.task_id} ({t.status}) ---\n{self._truncate(t.result)}" for t in tasks
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
            self.config.synthesis_temperature,
            False,
            self._phase_params("synthesis"),
        )
        final = self._resolve_macros(draft, tasks)

        # Normalize abbreviations
        final = self._normalize_abbreviations(final)

        if self.config.enable_review:
            await self._emit("Reviewing final answer...")
            final = await self.complete(
                PromptBuilder.review_prompt(self.system_prompt),
                f"{time_context}\n\n"
                f"Original user request:\n{goal}\n\n"
                f"Draft final answer:\n{final}\n\n"
                "Return the improved final answer. Ensure all abbreviations are expanded to full words.",
                self.config.synthesis_temperature,
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

    # -- orchestration -----------------------------------------------------

    def _make_mcp(self) -> Any:
        """Return an MCP client context manager (real if configured, else null)."""
        if self.config.mcp_servers and self.chat is not None:
            return MCPClient(self.config.mcp_servers, self.config.tool_result_limit)
        return _NullMCP()

    async def run(self, goal: str) -> PlannerResult:
        start = time.monotonic()
        async with self._make_mcp() as mcp:
            self._mcp = mcp
            for note in getattr(mcp, "notes", []) or []:
                await self._emit(note)
            tool_names = mcp.tool_names
            configured = bool(self.config.mcp_servers)
            if tool_names:
                # List every discovered tool up front so the user can see the
                # full toolset the planner may draw from.
                await self._emit(
                    f"MCP ready: {len(tool_names)} tool(s) found — "
                    + ", ".join(tool_names)
                )
                catalog = self._tools_catalog_text()
                if catalog:
                    await self._emit("Available MCP tools:\n" + catalog)
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
        if not self.config.plan_mode:
            await self._emit("Single-pass execution (plan mode off)")
            single = Task(task_id="task_1", description=goal)
            tools = self._mcp.openai_tools() if self._mcp is not None else []

            # Inject time context
            time_context = self._get_current_time_context()
            user_message = f"{time_context}\n\nUser request:\n{goal}"

            if tools and self.chat is not None:
                single.result = await self._execute_with_tools(user_message, tools)
            else:
                single.result = await self.complete(
                    PromptBuilder.execution_prompt(self.system_prompt),
                    user_message,
                    self.config.execution_temperature,
                    False,
                    self._phase_params("execution"),
                )

            # Normalize abbreviations
            single.result = self._normalize_abbreviations(single.result)

            single.status = "completed"
            return PlannerResult(
                goal, [single], single.result, time.monotonic() - start
            )

        tasks = await self.plan(goal)

        # Try to resume from memory first
        done: dict[str, Task] = {}
        if self.config.enable_memory_resume:
            resumed = await self._resume_from_memory(goal)
            if resumed:
                done = resumed
                await self._emit(f"Resumed {len(done)} completed task(s) from memory")
                # Filter out resumed tasks from execution
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
                        if (
                            prev_result
                            and not self._diverges_from_previous(task.result, prev_result)
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
                            await self._emit(f"{task.task_id}: loop persisted; accepting result")
                    break  # success or final attempt; exit retry loop
                task.status = "completed"
            except Exception as exc:  # keep going; record the failure
                task.status = "failed"
                task.result = f"[task failed: {exc}]"
                await self._emit(f"{task.task_id} failed: {exc}")
            done[task.task_id] = task

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
            for key, value in (params or {}).items():
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
            for key, value in (params or {}).items():
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
            return body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)
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
                return body.decode() if isinstance(body, (bytes, bytearray)) else str(body)
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
        TOP_P: float = Field(default=-1.0, description="Nucleus sampling top_p. -1 = unset.")
        TOP_K: int = Field(default=-1, description="top_k sampling. -1 = unset.")
        MIN_P: float = Field(default=-1.0, description="min_p sampling. -1 = unset.")
        MAX_TOKENS: int = Field(default=-1, description="Max tokens to generate. -1 = unset.")
        REPEAT_PENALTY: float = Field(
            default=-1.0, description="repeat_penalty (local backends). -1 = unset."
        )
        FREQUENCY_PENALTY: float = Field(
            default=-999.0, description="frequency_penalty (-2..2). -999 = unset."
        )
        PRESENCE_PENALTY: float = Field(
            default=-999.0, description="presence_penalty (-2..2). -999 = unset."
        )
        SEED: int = Field(default=-1, description="Sampling seed for reproducibility. -1 = unset.")
        # JSON escape hatches for anything not covered above. Merge order:
        # EXTRA_PARAMS (base) < the dedicated valves above < per-phase JSON.
        EXTRA_PARAMS_JSON: str = Field(
            default="",
            description='Base extra params as a JSON object, e.g. {"tfs_z": 1.0}. Applied to all phases.',
        )
        PLANNING_PARAMS_JSON: str = Field(
            default="", description="Per-phase param overrides for PLANNING as a JSON object."
        )
        EXECUTION_PARAMS_JSON: str = Field(
            default="", description="Per-phase param overrides for EXECUTION as a JSON object."
        )
        SYNTHESIS_PARAMS_JSON: str = Field(
            default="", description="Per-phase param overrides for SYNTHESIS/REVIEW as a JSON object."
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
            default=6, description="Max tool-call rounds per task before forcing a final answer."
        )
        TOOL_RESULT_LIMIT: int = Field(
            default=4000, description="Per tool-result character limit fed back to the model."
        )
        PLAN_MODE: bool = Field(
            default=True,
            description="Decompose into tasks first. Disable for a single-pass answer.",
        )
        MAX_TASKS: int = Field(default=20, description="Max tasks in the plan.")
        TASK_RESULT_LIMIT: int = Field(
            default=6000,
            description="Per-task result character soft-limit when feeding results forward.",
        )
        ENABLE_REVIEW: bool = Field(
            default=False, description="Add a final self-review/refine pass."
        )
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
            # Extra sampling params are forwarded verbatim to the backend.
            for key, value in (params or {}).items():
                if value is not None:
                    base_form[key] = value

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
                attempts.append({**base_form, "response_format": {"type": "json_object"}})
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
            for key, value in (params or {}).items():
                if value is not None:
                    form_data[key] = value
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
                    await progress(f"Discovered {len(discovered)} MCP server(s): {names}")
                    mcp_servers = discovered
            if not mcp_servers and valves.MCP_URL.strip():
                mcp_servers = [
                    {
                        "name": "memory",
                        "transport": "streamable-http",
                        "url": valves.MCP_URL.strip(),
                    }
                ]

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
            max_tasks=valves.MAX_TASKS,
            task_result_limit=valves.TASK_RESULT_LIMIT,
            enable_review=valves.ENABLE_REVIEW,
            mcp_servers=mcp_servers,
            max_tool_iterations=valves.MAX_TOOL_ITERATIONS,
            tool_result_limit=valves.TOOL_RESULT_LIMIT,
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

        try:
            result = await planner.run(goal)
        except Exception as exc:
            if __event_emitter__ and valves.EMIT_STATUS:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"Error: {exc}", "done": True}}
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
    parser.add_argument("--model", default=None, help="Model id (default: $PLANNER_MODEL or gpt-4o-mini)")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=None, help="Base sampling temperature")
    parser.add_argument("--planning-temperature", type=float, default=None, help="Planning-phase temperature (default: auto = min(base, 0.4))")
    parser.add_argument("--execution-temperature", type=float, default=None, help="Execution-phase temperature (default: base)")
    parser.add_argument("--synthesis-temperature", type=float, default=None, help="Synthesis/review-phase temperature (default: base)")
    # Extra sampling params (applied to all phases).
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument("--top-k", type=int, default=None, help="top_k sampling")
    parser.add_argument("--min-p", type=float, default=None, help="min_p sampling")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--repeat-penalty", type=float, default=None, help="repeat_penalty (local backends)")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="frequency_penalty (-2..2)")
    parser.add_argument("--presence-penalty", type=float, default=None, help="presence_penalty (-2..2)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    parser.add_argument("--params", default=None, help='Base extra params as JSON, e.g. \'{"tfs_z":1.0}\' (applied to all phases)')
    parser.add_argument("--planning-params", default=None, help="Per-phase param overrides for planning as JSON")
    parser.add_argument("--execution-params", default=None, help="Per-phase param overrides for execution as JSON")
    parser.add_argument("--synthesis-params", default=None, help="Per-phase param overrides for synthesis/review as JSON")
    # MCP tool calling (execution phase).
    parser.add_argument("--mcp-url", default=None, help="streamable-http MCP server URL (enables tool calling)")
    parser.add_argument("--mcp-config", default=None, help="JSON array of MCP servers (overrides --mcp-url)")
    parser.add_argument("--max-tool-iterations", type=int, default=None, help="Max tool-call rounds per task")
    parser.add_argument("--tool-result-limit", type=int, default=None, help="Per tool-result char limit")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks in the plan")
    parser.add_argument("--no-plan", action="store_true", help="Disable planning; single-pass")
    parser.add_argument("--review", action="store_true", help="Add a self-review/refine pass")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    parser.add_argument("--json", action="store_true", help="Emit full result as JSON")
    parser.add_argument("--system-prompt", default=None, help="Override the base system prompt")
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
    if args.review:
        config.enable_review = True
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
        print("error: no goal provided (pass as arguments or via stdin)", file=sys.stderr)
        return 2
    return asyncio.run(_run_cli(args, goal))


if __name__ == "__main__":
    raise SystemExit(main())
