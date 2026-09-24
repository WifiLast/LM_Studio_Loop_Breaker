"""Planner (Standalone) - the core agentic loop (StandalonePlanner).

Part of a 4-file split of planning_standalone.py (see planning_core.py's
docstring, and build_filter.py to recombine). Plan -> execute -> synthesize,
transport-agnostic (takes injected `complete`/`chat` functions from
planning_backends.py), built on planning_core's types/prompts and
planning_kev_mcp's Kev/MCP clients.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Iterable, Optional

from planning_core import (
    ChatFn,
    CompletionFn,
    DEFAULT_SYSTEM_PROMPT,
    PlannerConfig,
    PlannerResult,
    ProgressFn,
    PromptBuilder,
    Task,
    ToolMetrics,
    _extract_json_object,
    _extract_think_block,
)
from planning_kev_mcp import (
    ChatPlanState,
    KevClient,
    MCPClient,
    _CHAT_PLAN_STORE,
    _KEV_PROFILE_CACHE,
    _MATH_TASK_TOOL_NAMES,
    _MATH_VERIFY_TOOL_NAMES,
    _NullMCP,
    _SMALL_TALK_WORDS,
    _cached_mcp_tool_names,
)

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
        # Set True by _execute_with_tools when a verification tool (check_equation,
        # z3_solve_constraints, ...) is actually called during the current task attempt.
        self._verify_tool_called: bool = False

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

    async def _emit_thought(self, label: str, text: str) -> str:
        """Surface a `<think>...</think>` block in `text` as its own progress message
        (so it's visible during Executing, the way math_run.log showed "Gedanke" before
        the tool calls and final answer), then return `text` with the block removed. If
        stripping it would leave nothing (the model ran out of budget mid-thought with no
        final answer), the original text is kept so no content is lost."""
        thought, remainder = _extract_think_block(text)
        if not thought:
            return text
        await self._emit(f"  \U0001f4ad {label} thinking:\n{thought}")
        return remainder if remainder.strip() else text

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
                has_math_tools=bool(available_tools & _MATH_TASK_TOOL_NAMES),
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
        self,
        goal: str,
        task: Task,
        done: dict[str, Task],
        attempt: int = 0,
        retry_hint: str = "",
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

        # Inject loop-escape guidance on retry (or a specific reason, when known)
        if retry_hint:
            user_message += f"\n\n⚠️ IMPORTANT: {retry_hint}"
        elif attempt > 0:
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
        math_verification = bool(set(task.tools) & _MATH_VERIFY_TOOL_NAMES)
        if tools and self.chat is not None:
            await self._emit(
                f"  ↳ tools for {task.task_id}: {', '.join(t['function']['name'] for t in tools)}"
            )
            result = await self._execute_with_tools(
                user_message,
                tools,
                attempt,
                math_verification=math_verification,
                task_id=task.task_id,
            )
        else:
            result = await self.complete(
                PromptBuilder.execution_prompt(
                    self.system_prompt, math_verification=math_verification
                ),
                user_message,
                self._adjusted_temperature(attempt),
                False,
                self._adjusted_params("execution", attempt),
            )
            result = await self._emit_thought(task.task_id, result)

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
        self,
        user_message: str,
        tools: list[dict],
        attempt: int = 0,
        math_verification: bool = False,
        task_id: str = "execution",
    ) -> str:
        """Run one task as a tool-calling loop over the configured MCP tools."""
        messages: list[dict] = [
            {
                "role": "system",
                "content": PromptBuilder.execution_prompt(
                    self.system_prompt, math_verification=math_verification
                )
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
            raw_content = message.get("content") or ""
            # Record the assistant turn (content may be empty alongside tool calls). The
            # raw content (with any <think> block intact) is what the model sees back in
            # its own conversation history; only the RETURNED value has thoughts split out.
            assistant_turn: dict = {
                "role": "assistant",
                "content": raw_content,
            }
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            messages.append(assistant_turn)

            if not tool_calls:
                return await self._emit_thought(task_id, raw_content)

            # This turn thought-and-called-a-tool in the same completion (the common
            # case for a thinking model mid tool loop). Surface that reasoning now
            # instead of only ever showing the final turn's - otherwise every
            # intermediate "Gedanke" before a tool call goes unseen.
            await self._emit_thought(task_id, raw_content)

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
                if name in _MATH_VERIFY_TOOL_NAMES:
                    self._verify_tool_called = True

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
        final_content = final.get("content") or "[no final answer after tool iterations]"
        return await self._emit_thought(task_id, final_content)

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
                single.result = await self._execute_with_tools(
                    user_message, tools, task_id=single.task_id
                )
            else:
                single.result = await self.complete(
                    PromptBuilder.execution_prompt(self.system_prompt),
                    user_message,
                    self._execution_temperature(),
                    False,
                    self._phase_params("execution"),
                )
                single.result = await self._emit_thought(
                    single.task_id, single.result
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
                retry_hint = ""
                needs_verification = bool(set(task.tools) & _MATH_VERIFY_TOOL_NAMES)
                for attempt in range(max_attempts):
                    self._verify_tool_called = False
                    task.result = await self.execute_task(
                        goal, task, done, attempt, retry_hint
                    )
                    retry_hint = ""
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
                    if attempt < max_attempts - 1:
                        # A verification tool (check_equation, z3_solve_constraints, ...) was
                        # attached to this task but never actually called: the model derived a
                        # formula/claim and trusted its own algebra instead of the solver. This
                        # is a deterministic check (did a tool call happen?), not a Kev guess -
                        # it's exactly the failure mode in math_run.log, where a weak model
                        # re-derived the same identity three times by hand instead of asking
                        # `check_equation` once.
                        if (
                            self.config.kev_check_math
                            and needs_verification
                            and not self._verify_tool_called
                        ):
                            await self._emit(
                                f"{task.task_id}: no verification tool was called; retrying "
                                f"(attempt {attempt + 1}/{max_attempts - 1})"
                            )
                            prev_result = task.result
                            retry_hint = (
                                "Your previous attempt stated a formula or claim but never "
                                "called a verification tool (check_equation, check_consistency, "
                                "check_entailment, z3_solve_constraints, z3_prove_theorem, or "
                                "verify_claims). Call one now to check your result before "
                                "writing the final answer."
                            )
                            continue
                        if await self._task_result_is_usable(task) is False:
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


