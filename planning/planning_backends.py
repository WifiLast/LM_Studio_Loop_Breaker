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

# This is one of a 4-file split, kept for editing convenience:
#   planning_core.py, planning_kev_mcp.py, planning_engine.py, planning_backends.py (this file)
# Run `python build_filter.py` to flatten all four into planning_standalone.py, the
# single file Open WebUI needs and the one the CLI examples above refer to. This file
# also runs standalone as the CLI (`python planning_backends.py "..."`), since it
# imports its sibling modules normally.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

from planning_core import (
    ChatFn,
    CompletionFn,
    DEFAULT_SYSTEM_PROMPT,
    PLAN_JSON_SCHEMA,
    PlannerConfig,
    _extract_reasoning,
    _merge_reasoning,
)
from planning_engine import StandalonePlanner

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
            message = resp.choices[0].message
            content = message.content
            reasoning = _extract_reasoning(message)
            if content is None and not reasoning:
                raise RuntimeError("LLM returned an empty response")
            return _merge_reasoning(content, reasoning)

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
            reasoning = _extract_reasoning(msg)
            return {
                "content": _merge_reasoning(msg.content, reasoning),
                "tool_calls": tool_calls,
            }

        return await asyncio.to_thread(_call)

    return chat


def _content_from_choices(data: dict) -> str:
    """Extract assistant text from an OpenAI-style payload.

    Falls back to a separate reasoning field (`reasoning_content` for vLLM/SGLang/
    DeepSeek-style backends, `thinking` for Ollama) and to the top-level `content`
    some backends return when `content` is empty/missing.
    """
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            reasoning = _extract_reasoning(message)
            if content or reasoning:
                return _merge_reasoning(content, reasoning)
            # Streaming-style chunk shape.
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                delta_reasoning = _extract_reasoning(delta)
                if delta.get("content") or delta_reasoning:
                    return _merge_reasoning(delta.get("content"), delta_reasoning)
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
            content = _merge_reasoning(
                message.get("content"), _extract_reasoning(message)
            )
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

        # Status events are transient: Open WebUI's status widget only ever shows the
        # latest one and doesn't persist them on the message, so a "thinking" line is
        # overwritten by the next status (often within the same tick) before anyone can
        # read it, and it's gone entirely once the response finishes. Collect thought
        # lines separately so they can be embedded in the final message as a `<think>`
        # block - which Open WebUI renders as a persistent, expandable section - instead
        # of relying on the fleeting status trace to carry them.
        collected_thoughts: list[str] = []

        async def progress(message: str) -> None:
            if "\U0001f4ad" in message and " thinking:\n" in message:
                collected_thoughts.append(message.split("\U0001f4ad ", 1)[1])
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
        if collected_thoughts:
            think_block = "\n\n".join(collected_thoughts)
            return f"<think>\n{think_block}\n</think>\n\n{result.final_output}"
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
