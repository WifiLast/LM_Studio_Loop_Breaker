"""Planner (Standalone) - Kev decisions + MCP client.

Part of a 4-file split of planning_standalone.py (see planning_core.py's
docstring, and build_filter.py to recombine). Defines the MCP client (tool
discovery/calling over one or more MCP servers) and the Kev System One decision
client, plus the small per-chat/per-request caches they share. Independent of
planning_core.py - it only needs the standard library.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

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

# Tool names from math_plus_mcp.py (or an equivalent math/logic MCP server) that make a
# task a candidate for the "derive, then verify with a solver" flow instead of trusting
# free-form algebra end to end.
_MATH_TASK_TOOL_NAMES = frozenset({
    "evaluate", "check_equation", "check_consistency", "check_entailment",
    "solve_equation", "verify_claims", "z3_solve_constraints", "z3_prove_theorem",
    "z3_run_script", "differentiate", "test_root_stability", "test_derivative_stability",
    "integrate_function", "distribution_pdf", "distribution_cdf",
    "distribution_quantile", "distribution_probability_between",
    "z3_add_constraint", "z3_check_satisfiability", "z3_reset_solver",
    "z3_solver_status", "check_entailment_from_z3",
})
# The subset that actually proves/checks a claim rather than just computing a number -
# these are what a math (or logic-puzzle) task must call at least once before its result
# is trusted. z3_run_script covers what the flat-expression checkers can't: quantifiers,
# custom sorts, or a finite-domain sweep (e.g. every permutation of roles in a "who is
# lying" riddle) built with a loop.
_MATH_VERIFY_TOOL_NAMES = frozenset({
    "check_equation", "check_consistency", "check_entailment", "z3_solve_constraints",
    "z3_prove_theorem", "z3_run_script", "verify_claims", "check_entailment_from_z3",
})

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


