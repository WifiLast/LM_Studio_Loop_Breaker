"""
title: Memory Fact Substitution
author: local
version: 0.2.0
required_open_webui_version: 0.11.0
description: Replaces {{fact: <query>}} placeholders in the user's message with the best-matching, importance-ranked, Kev-verified fact stored in the mcp-memory database, before the chat model ever sees the placeholder syntax.
"""

# What this does
# --------------
# A user (or a saved prompt template) writes something like:
#
#     What's the maintenance interval for {{fact: pump P-101 bearing schedule}}?
#
# Before the message reaches the chat model, this filter finds every
# `{{fact: ...}}` placeholder, looks each query up against the mcp-memory
# database (other/mcp_memory/server.py - the same server kev.py's memory
# retrieval talks to), ranks the candidates it gets back, optionally asks Kev
# (System One, see kev.py) whether the top candidate actually answers the
# query, and only then substitutes it. The model only ever sees the
# substituted sentence, never the placeholder syntax.
#
# Why an embedded MCP client instead of a plain REST lookup
# -----------------------------------------------------------
# mcp-memory (other/mcp_memory/server.py) exposes `remember`/`retrieve` as MCP
# tools over the streamable-HTTP transport, not a plain REST route - there is
# no GET /facts endpoint on that server. Open WebUI loads each Function as an
# isolated module, so this filter can't import kev.py's client either; it
# carries its own copy of the same minimal MCP client kev.py uses
# (`_MCPMemoryClient`: initialize -> notifications/initialized -> tools/call).
#
# Picking the right fact: importance ranking + a Kev verification gate
# ----------------------------------------------------------------------
# `retrieve` can return several candidates for a query. Two independent
# safety nets narrow that down to one trustworthy substitution:
#
#   1. Importance ranking (`_score_snippet`): combines the server's own
#      relevance score with the memory's type, recency, and durability
#      (whether kev.py's write-side importance rule stored it with no TTL -
#      see kev.py's MCP_MEMORY_IMPORTANCE_HIGH_THRESHOLD) into one ranking so
#      the *most important* matching memory is tried first, not just
#      whichever the server lists first.
#   2. Kev verification (`_verify_with_kev`): before substituting, this filter
#      asks the same Kev System One endpoint kev.py talks to (KEV_URL,
#      /v1/systemone) a single fast typed question - does this candidate
#      actually answer this query? - and only substitutes if Kev agrees. If a
#      candidate fails, the next-ranked candidate is tried. This is the
#      hallucination gate: a stale or off-topic memory that merely resembles
#      the query text is caught before ever reaching the chat model as
#      "ground truth."
#
# Fail-open
# ---------
# If the mcp-memory lookup itself fails (network, bad valve, server down), a
# placeholder's original text is left untouched rather than the turn being
# blocked - same philosophy as kev.py: a broken filter must never break the
# chat. If the lookup succeeds but finds nothing, or Kev rejects every
# candidate as unreliable, NOT_FOUND_TEXT is substituted instead - a
# confident "we didn't find anything trustworthy," which is a different
# outcome from a broken lookup. If Kev itself is unreachable, verification is
# skipped for that call only (fail open) and the top-ranked candidate is used
# as-is, rather than rejecting everything just because the extra safety net
# is down.
#
# Install: Admin Panel -> Functions -> + -> paste -> enable, then assign it to
# the models that should get fact substitution (or globally). Set
# MCP_MEMORY_URL in its valves to point at other/mcp_memory/server.py, and
# KEV_URL to point at the same kev.serve endpoint kev.py uses.

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjOWNhM2FmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PHBhdGggZD0iTTQgMTloMTYiLz48cGF0aCBkPSJNNCAxNWgxNiIvPjxwYXRoIGQ9Ik00IDExaDE2Ii8+PHBhdGggZD0iTTQgN2gxNiIvPjwvc3ZnPg=="

_DEFAULT_PATTERN = r"\{\{\s*fact\s*:\s*(.+?)\s*\}\}"

_DEFAULT_TYPE_WEIGHTS = {"directive": 1.0, "task": 0.8, "note": 0.6}

# Sentinel distinguishing "the lookup ran and found nothing trustworthy" from
# a lookup failure (None) - see _lookup's docstring.
_NOT_FOUND = object()

# kev.py's research pipeline saves facts as `[key] fact <<src=URL; date=YYYY-MM-DD;
# volatile=0|1>>`. A copy of its parser lives here (Open WebUI loads each Function in
# isolation) so `{{fact: key}}` can match on the key exactly and the substituted text is
# just the fact, never the key prefix or provenance trailer.
_RESEARCH_MEMORY_RE = re.compile(
    r"^\[(?P<key>[^\]\n]{1,200})\]\s*(?P<fact>.*?)(?:\s*<<(?P<meta>[^<>]*)>>)?\s*$",
    re.DOTALL,
)


def _normalize_key(key: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[{}\[\]<>\r\n]+", " ", str(key or ""))).strip().casefold()


def parse_research_memory(text: Any) -> Optional[dict]:
    """None for any memory not in kev.py's research format (plain notes, triples)."""
    match = _RESEARCH_MEMORY_RE.match(str(text or "").strip())
    if not match or not match.group("fact").strip():
        return None
    return {"key": match.group("key").strip(), "fact": match.group("fact").strip()}


class _MCPMemoryClient:
    """Minimal async client for the mcp-memory FastMCP streamable-HTTP endpoint.

    A copy of kev.py's `_MCPMemoryClient` - implements just enough of the MCP
    Streamable HTTP transport to perform the `initialize` ->
    `notifications/initialized` -> `tools/call` handshake against
    `mcp_memory/server.py`. Duplicated rather than imported because Open WebUI
    loads each Function as an isolated module. Used as an async context
    manager so a single session is reused for the calls made per lookup.
    """

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, base_url: str, security_key: str = "", timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.security_key = security_key or None
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None
        self._request_id = 0

    async def __aenter__(self) -> "_MCPMemoryClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        await self._initialize()
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse_body(content_type: str, text: str) -> Optional[dict]:
        if "text/event-stream" in content_type:
            message: Optional[dict] = None
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    continue
            return message
        if not text.strip():
            return None
        return json.loads(text)

    async def _post(
        self, payload: dict, expect_response: bool = True
    ) -> Optional[dict]:
        assert self._session is not None, "client not initialized"
        async with self._session.post(
            self.base_url, json=payload, headers=self._headers()
        ) as response:
            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id
            text = await response.text()
            response.raise_for_status()
            if not expect_response:
                return None
            return self._parse_body(response.headers.get("content-type", ""), text)

    async def _initialize(self) -> None:
        await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "open-webui-fact-substitution",
                        "version": "1.0.0",
                    },
                },
            }
        )
        # Notifications carry no response body (server replies 202 Accepted).
        await self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )

    async def call_tool(self, name: str, arguments: dict) -> Any:
        args = dict(arguments)
        if self.security_key:
            args.setdefault("security_key", self.security_key)

        message = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )
        if message is None:
            raise RuntimeError(f"empty response from MCP tool '{name}'")
        if "error" in message:
            raise RuntimeError(f"MCP tool '{name}' error: {message['error']}")

        result = message.get("result", {}) or {}
        if result.get("isError"):
            raise RuntimeError(
                f"MCP tool '{name}' reported failure: {self._content_text(result)}"
            )

        structured = result.get("structuredContent")
        if structured is not None:
            # FastMCP wraps scalar/list returns under a 'result' key.
            if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
                return structured["result"]
            return structured

        text = self._content_text(result)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return result

    @staticmethod
    def _content_text(result: dict) -> str:
        parts = []
        for item in result.get("content", []) or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def score_snippet(
    snippet: dict,
    now: datetime,
    weight_score: float,
    weight_type: float,
    weight_recency: float,
    weight_durability: float,
    type_weights: dict,
    recency_half_life_days: float,
) -> float:
    """The importance-ranking rule: combines the server's own relevance score
    (how well this snippet matches the query) with signals about how
    *important* the memory is in general, so that among several matching
    candidates the most important one is tried first for substitution.

    Pure function, no I/O - independently testable. `snippet` is a dict in
    the shape of mcp-memory's `Snippet` (event_id, type, ts, score, ttl_days,
    ...), the same shape `retrieve`'s `snippets` list returns.
    """
    relevance = float(snippet.get("score") or 0.0)

    type_weight = type_weights.get(str(snippet.get("type") or "").lower(), 0.5)

    recency_weight = 0.0
    ts = _parse_timestamp(snippet.get("ts"))
    if ts is not None and recency_half_life_days > 0:
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        recency_weight = 2.0 ** (-age_days / recency_half_life_days)

    # kev.py's write-side importance rule stores durable/high-importance
    # memories with ttl_days=None (never expire) - read that signal back out.
    durability_weight = 1.0 if snippet.get("ttl_days") is None else 0.0

    return (
        weight_score * relevance
        + weight_type * type_weight
        + weight_recency * recency_weight
        + weight_durability * durability_weight
    )


class Filter:
    class Valves(BaseModel):
        MCP_MEMORY_URL: str = Field(
            default="http://10.0.0.10:8082/memory",
            description="Base URL of the mcp-memory FastMCP streamable-HTTP endpoint (other/mcp_memory/server.py) - the same server kev.py's memory retrieval talks to.",
        )
        MCP_MEMORY_SECURITY_KEY: str = Field(
            default="",
            description="Security key for the mcp-memory server, only needed if configured server-side. Empty = disabled.",
        )
        MCP_MEMORY_TIMEOUT: float = Field(
            default=10.0,
            description="Seconds to wait per mcp-memory lookup before giving up and leaving that placeholder untouched.",
        )
        MCP_MEMORY_K: int = Field(
            default=5,
            description="Number of candidate memories to fetch per query from retrieve(), before importance ranking and Kev verification narrow it down to one.",
        )
        MCP_MEMORY_MIN_SCORE: float = Field(
            default=0.5,
            description="Minimum retrieve() relevance score for a memory to be considered a candidate at all.",
        )
        WEIGHT_SCORE: float = Field(
            default=0.5,
            description="Importance-ranking weight for the server's own relevance score.",
        )
        WEIGHT_TYPE: float = Field(
            default=0.15,
            description="Importance-ranking weight for the memory's type (see TYPE_WEIGHTS).",
        )
        WEIGHT_RECENCY: float = Field(
            default=0.15,
            description="Importance-ranking weight for how recently the memory was saved.",
        )
        WEIGHT_DURABILITY: float = Field(
            default=0.2,
            description="Importance-ranking weight for whether the memory was saved as durable/high-importance (no TTL) by kev.py's write-side importance rule.",
        )
        TYPE_WEIGHTS: str = Field(
            default=json.dumps(_DEFAULT_TYPE_WEIGHTS),
            description="JSON object mapping a memory's `type` to an importance weight (0-1). Unknown types default to 0.5.",
        )
        RECENCY_HALF_LIFE_DAYS: float = Field(
            default=30.0,
            description="Days for a memory's recency weight to halve. Larger = recency matters less.",
        )
        KEV_URL: str = Field(
            default="http://10.0.0.10:8009",
            description="Base URL of the Kev System One endpoint (kev.serve) - the same one kev.py's filter uses - for verifying a candidate fact before substitution.",
        )
        KEV_API_KEY: str = Field(
            default="",
            description="Bearer token, when kev.serve was started with KEV_API_KEY set. Empty = open server.",
        )
        KEV_TIMEOUT: float = Field(
            default=10.0,
            description="Seconds to wait for Kev's verdict on a candidate before treating Kev as unreachable for that call.",
        )
        KEV_VERIFY_ENABLED: bool = Field(
            default=True,
            description="Ask Kev whether the top-ranked candidate actually answers the query before substituting it - a hallucination gate independent of the mcp-memory relevance score. Fails open (skips verification for that call) if Kev is unreachable.",
        )
        KEV_VERIFY_THRESHOLD: float = Field(
            default=0.5,
            description="Minimum Kev probability for a candidate to be accepted. Candidates below this are skipped in favor of the next-ranked one.",
        )
        PATTERN: str = Field(
            default=_DEFAULT_PATTERN,
            description="Regex with exactly one capture group (the query) matched against the message text, e.g. {{fact: <query>}}.",
        )
        NOT_FOUND_TEXT: str = Field(
            default="[no matching fact found]",
            description="Text substituted when the lookup succeeds but finds nothing trustworthy (no candidates, or Kev rejected all of them). Leave empty to just delete the placeholder.",
        )
        SCOPE_TO_USER: bool = Field(
            default=False,
            description="Pass the Open WebUI user's id as user_id to the lookup, scoping results to memories saved under that user.",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show how many facts were substituted in the chat's status line.",
        )
        PRIORITY: int = Field(default=0, description="Filter order; lower runs first.")

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # renders as a switch next to the message box
        self.icon = ICON

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __user__: Optional[dict] = None,
        __task__: Optional[str] = None,
    ) -> dict:
        if __task__:  # title, tags, autocomplete: not a turn the user is waiting on
            return body

        try:
            pattern = re.compile(self.valves.PATTERN, re.IGNORECASE | re.DOTALL)
        except re.error:
            return body  # a broken regex must not break the chat

        messages = body.get("messages") or []
        last_user_index = self._last_user_message_index(messages)
        if last_user_index is None:
            return body

        message = messages[last_user_index]
        content = message.get("content")
        queries = self._collect_queries(content, pattern)
        if not queries:
            return body

        user_id = (__user__ or {}).get("id") if self.valves.SCOPE_TO_USER else None
        lookups = await asyncio.gather(
            *(self._lookup(query, user_id) for query in queries),
            return_exceptions=True,
        )
        replacements = {
            query: text
            for query, text in zip(queries, lookups)
            if isinstance(text, str)
        }
        if not replacements:
            return body

        substituted = 0

        def _replace(match: "re.Match[str]") -> str:
            nonlocal substituted
            query = match.group(1).strip()
            if query not in replacements:
                return match.group(0)  # lookup failed/timed out: leave untouched
            substituted += 1
            return replacements[query]

        new_content = self._apply(content, pattern, _replace)
        if substituted:
            messages = list(messages)
            messages[last_user_index] = {**message, "content": new_content}
            body["messages"] = messages
            await self._status(
                __event_emitter__,
                f"Memory facts: substituted {substituted}/{len(queries)} placeholder(s)",
            )
        return body

    # -- lookup: retrieve candidates, rank by importance, verify with Kev

    async def _lookup(self, query: str, user_id: Optional[str]) -> Optional[str]:
        """Returns the substitution text, or None to leave the placeholder
        untouched (a transport/backend failure - not the same as "found
        nothing", which returns NOT_FOUND_TEXT instead)."""
        try:
            snippets = await self._retrieve_candidates(query, user_id)
        except Exception:
            return None  # fail open: mcp-memory unreachable/broken

        if not snippets:
            return self.valves.NOT_FOUND_TEXT

        # A researched memory whose stored key equals the placeholder's query is an exact
        # hit - kev.py already had Kev confirm it against its sources before saving it,
        # so it needs neither ranking nor re-verification here. Newest wins if a refreshed
        # fact exists alongside an older one.
        normalized_query = _normalize_key(query)
        exact = []
        for snippet in snippets:
            parsed = parse_research_memory(snippet.get("text"))
            if parsed and _normalize_key(parsed["key"]) == normalized_query:
                exact.append((snippet, parsed["fact"]))
        if exact:
            exact.sort(
                key=lambda item: _parse_timestamp(item[0].get("ts"))
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            return exact[0][1]

        now = datetime.now(timezone.utc)
        try:
            type_weights = {
                **_DEFAULT_TYPE_WEIGHTS,
                **json.loads(self.valves.TYPE_WEIGHTS),
            }
        except (json.JSONDecodeError, TypeError):
            type_weights = _DEFAULT_TYPE_WEIGHTS

        ranked = sorted(
            snippets,
            key=lambda snippet: score_snippet(
                snippet,
                now,
                self.valves.WEIGHT_SCORE,
                self.valves.WEIGHT_TYPE,
                self.valves.WEIGHT_RECENCY,
                self.valves.WEIGHT_DURABILITY,
                type_weights,
                self.valves.RECENCY_HALF_LIFE_DAYS,
            ),
            reverse=True,
        )

        def _clean(snippet: dict) -> str:
            raw = str(snippet.get("text") or "")
            parsed = parse_research_memory(raw)
            return parsed["fact"] if parsed else raw

        if not self.valves.KEV_VERIFY_ENABLED:
            return _clean(ranked[0]) or self.valves.NOT_FOUND_TEXT

        for snippet in ranked:
            text = _clean(snippet)
            if not text:
                continue
            try:
                probability = await self._verify_with_kev(query, text)
            except Exception:
                # Kev unreachable: fail open for this call by skipping
                # verification rather than rejecting every candidate just
                # because the extra safety net is down.
                return text
            if probability >= self.valves.KEV_VERIFY_THRESHOLD:
                return text
        return self.valves.NOT_FOUND_TEXT

    async def _retrieve_candidates(
        self, query: str, user_id: Optional[str]
    ) -> list[dict]:
        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        async with _MCPMemoryClient(
            self.valves.MCP_MEMORY_URL,
            self.valves.MCP_MEMORY_SECURITY_KEY,
            self.valves.MCP_MEMORY_TIMEOUT,
        ) as mcp:
            result = await mcp.call_tool(
                "retrieve",
                {
                    "query": query,
                    "k": self.valves.MCP_MEMORY_K,
                    "filters": filters,
                    "min_score": self.valves.MCP_MEMORY_MIN_SCORE,
                },
            )
        if isinstance(result, dict):
            return list(result.get("snippets") or [])
        return []

    async def _verify_with_kev(self, query: str, candidate_text: str) -> float:
        """Asks Kev's System One endpoint (same one kev.py talks to) whether
        `candidate_text` actually answers `query`. Returns the `noul`
        probability. Raises on any transport/parsing failure so the caller
        can fail open."""
        payload = {
            "state": f"Query: {query}\nCandidate fact: {candidate_text}",
            "model": "kev-latest",
            "questions": {
                "matches": {
                    "type": "noul",
                    "instructions": (
                        "Does the candidate fact accurately and directly answer the "
                        "query, without contradiction or irrelevance?"
                    ),
                    "criteria": {
                        "true": "the candidate directly and correctly answers the query",
                        "false": (
                            "the candidate is unrelated, contradicts the query, or "
                            "does not actually answer what was asked"
                        ),
                    },
                }
            },
        }
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
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
                answer = json.loads(text)
        return float(answer["answers"]["matches"]["noul"])

    # -- message shaping (mirrors kev.py's multimodal-aware content handling)

    @staticmethod
    def _last_user_message_index(messages: list) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return None

    @staticmethod
    def _collect_queries(content: Any, pattern: "re.Pattern[str]") -> list[str]:
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        queries: list[str] = []
        seen = set()
        for text in texts:
            for match in pattern.finditer(text):
                query = match.group(1).strip()
                if query and query not in seen:
                    seen.add(query)
                    queries.append(query)
        return queries

    @staticmethod
    def _apply(content: Any, pattern: "re.Pattern[str]", replace: Callable) -> Any:
        if isinstance(content, str):
            return pattern.sub(replace, content)
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part = {**part, "text": pattern.sub(replace, part.get("text", ""))}
                new_parts.append(part)
            return new_parts
        return content

    async def _status(self, emitter, description: str) -> None:
        if emitter and self.valves.SHOW_STATUS:
            await emitter(
                {"type": "status", "data": {"description": description, "done": True}}
            )
