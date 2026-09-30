import json
import os
import requests
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

# -- fact recall: lets the model itself look up a stored fact and swap it into
# its own answer, instead of the automatic {{fact: ...}} placeholder substitution
# fact.py's Filter does. Same mcp-memory database and Kev verification gate as
# fact.py/kev.py; duplicated here rather than imported because Open WebUI loads
# each Tools file as an isolated module.

_DEFAULT_TYPE_WEIGHTS = {"directive": 1.0, "task": 0.8, "note": 0.6}


class _MCPMemoryClient:
    """Minimal async client for the mcp-memory FastMCP streamable-HTTP endpoint.

    A copy of fact.py's/kev.py's `_MCPMemoryClient` - implements just enough of
    the MCP Streamable HTTP transport to perform the `initialize` ->
    `notifications/initialized` -> `tools/call` handshake against
    `mcp_memory/server.py`.
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
                        "name": "open-webui-fact-recall-tool",
                        "version": "1.0.0",
                    },
                },
            }
        )
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
    """Same importance-ranking rule as fact.py's `score_snippet` - combines the
    server's own relevance score with type/recency/durability signals so the
    most important matching memory is tried first."""
    relevance = float(snippet.get("score") or 0.0)
    type_weight = type_weights.get(str(snippet.get("type") or "").lower(), 0.5)

    recency_weight = 0.0
    ts = _parse_timestamp(snippet.get("ts"))
    if ts is not None and recency_half_life_days > 0:
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        recency_weight = 2.0 ** (-age_days / recency_half_life_days)

    durability_weight = 1.0 if snippet.get("ttl_days") is None else 0.0

    return (
        weight_score * relevance
        + weight_type * type_weight
        + weight_recency * recency_weight
        + weight_durability * durability_weight
    )


class Tools:
    class Valves(BaseModel):
        MCP_MEMORY_URL: str = Field(
            default="http://10.0.0.10:8082/memory",
            description="Base URL of the mcp-memory FastMCP streamable-HTTP endpoint (other/mcp_memory/server.py) - the same server kev.py and fact.py talk to.",
        )
        MCP_MEMORY_SECURITY_KEY: str = Field(
            default="",
            description="Security key for the mcp-memory server, only needed if configured server-side. Empty = disabled.",
        )
        MCP_MEMORY_TIMEOUT: float = Field(
            default=10.0,
            description="Seconds to wait per mcp-memory lookup before giving up.",
        )
        MCP_MEMORY_K: int = Field(
            default=5,
            description="Number of candidate memories to fetch per query from retrieve(), before ranking and Kev verification narrow it down to one.",
        )
        MCP_MEMORY_MIN_SCORE: float = Field(
            default=0.5,
            description="Minimum retrieve() relevance score for a memory to be considered a candidate at all.",
        )
        WEIGHT_SCORE: float = Field(default=0.5, description="Importance-ranking weight for the server's own relevance score.")
        WEIGHT_TYPE: float = Field(default=0.15, description="Importance-ranking weight for the memory's type (see TYPE_WEIGHTS).")
        WEIGHT_RECENCY: float = Field(default=0.15, description="Importance-ranking weight for how recently the memory was saved.")
        WEIGHT_DURABILITY: float = Field(default=0.2, description="Importance-ranking weight for whether the memory was saved as durable/high-importance (no TTL).")
        TYPE_WEIGHTS: str = Field(
            default=json.dumps(_DEFAULT_TYPE_WEIGHTS),
            description="JSON object mapping a memory's `type` to an importance weight (0-1). Unknown types default to 0.5.",
        )
        RECENCY_HALF_LIFE_DAYS: float = Field(default=30.0, description="Days for a memory's recency weight to halve.")
        KEV_URL: str = Field(
            default="http://10.0.0.10:8009",
            description="Base URL of the Kev System One endpoint (kev.serve) - the same one kev.py/fact.py use - for verifying a candidate fact before it's returned to the model.",
        )
        KEV_API_KEY: str = Field(default="", description="Bearer token, when kev.serve was started with KEV_API_KEY set. Empty = open server.")
        KEV_TIMEOUT: float = Field(default=10.0, description="Seconds to wait for Kev's verdict before treating Kev as unreachable for that call.")
        KEV_VERIFY_ENABLED: bool = Field(
            default=True,
            description="Ask Kev whether the top-ranked candidate actually answers the query before returning it to the model - a hallucination gate. Fails open (skips verification) if Kev is unreachable.",
        )
        KEV_VERIFY_THRESHOLD: float = Field(default=0.5, description="Minimum Kev probability for a candidate to be accepted. Candidates below this are skipped in favor of the next-ranked one.")
        NOT_FOUND_TEXT: str = Field(
            default="No reliable stored fact was found for that - answer without inventing one, or ask the user for it.",
            description="Text returned to the model when the lookup succeeds but finds nothing trustworthy (no candidates, or Kev rejected all of them).",
        )
        SCOPE_TO_USER: bool = Field(
            default=False,
            description="Pass the Open WebUI user's id as user_id to the lookup, scoping results to memories saved under that user.",
        )
        RELATION_EXTRACTOR_URL: str = Field(
            default="http://10.0.0.10:2006/relation",
            description="Base URL of the relation_extractor FastMCP streamable-HTTP endpoint (other/relation_extractor/z3_backend.py's extract_relations_tool).",
        )
        RELATION_EXTRACTOR_SECURITY_KEY: str = Field(
            default="",
            description="Security key for the relation_extractor server, only needed if configured server-side. Empty = disabled.",
        )
        RELATION_EXTRACTOR_TIMEOUT: float = Field(
            default=30.0,
            description="Seconds to wait for an extract_relations_tool call before giving up - the L5 neural pass (ReLiK+GLiREL) is slower than a plain memory lookup.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # Add your custom tools using pure Python code here, make sure to add type hints and descriptions
	
    def get_user_name_and_email_and_id(self, __user__: dict = {}) -> str:
        """
        Get the user name, Email and ID from the user object.
        """

        # Do not include a descrption for __user__ as it should not be shown in the tool's specification
        # The session user object will be passed as a parameter when the function is called

        print(__user__)
        result = ""

        if "name" in __user__:
            result += f"User: {__user__['name']}"
        if "id" in __user__:
            result += f" (ID: {__user__['id']})"
        if "email" in __user__:
            result += f" (Email: {__user__['email']})"

        if result == "":
            result = "User: Unknown"

        return result

    def get_current_time(self) -> str:
        """
        Get the current time in a more human-readable format.
        """

        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")  # Using 12-hour format with AM/PM
        current_date = now.strftime(
            "%A, %B %d, %Y"
        )  # Full weekday, month name, day, and year

        return f"Current Date and Time = {current_date}, {current_time}"

    def calculator(
        self,
        equation: str = Field(
            ..., description="The mathematical equation to calculate."
        ),
    ) -> str:
        """
        Calculate the result of an equation.
        """

        # Avoid using eval in production code
        # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        try:
            result = eval(equation)
            return f"{equation} = {result}"
        except Exception as e:
            print(e)
            return "Invalid equation"

    def get_current_weather(
        self,
        city: str = Field(
            "New York, NY", description="Get the current weather for a given city."
        ),
    ) -> str:
        """
        Get the current weather for a given city.
        """

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return (
                "API key is not set in the environment variable 'OPENWEATHER_API_KEY'."
            )

        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",  # Optional: Use 'imperial' for Fahrenheit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            data = response.json()

            if data.get("cod") != 200:
                return f"Error fetching weather data: {data.get('message')}"

            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            return f"Weather in {city}: {temperature}°C"
        except requests.RequestException as e:
            return f"Error fetching weather data: {str(e)}"

    async def recall_fact(
        self,
        query: str = Field(
            ...,
            description=(
                "The specific word, name, id, date, setting or short phrase you are "
                "unsure about, e.g. 'pump P-101 bearing schedule' or 'Kev's default "
                "API port'."
            ),
        ),
        __user__: dict = {},
    ) -> str:
        """
        Look up a stored, verified fact in long-term memory and use it to replace a
        word or placeholder in your own answer instead of guessing. Call this
        yourself, before you write a sentence containing a specific fact you are not
        certain of (a name, id, date, setting, procedure...), then phrase your reply
        using the returned information in your own words. Returns a short fact, or a
        note that nothing reliable was found - in that case say so instead of
        inventing an answer.
        """
        user_id = (__user__ or {}).get("id") if self.valves.SCOPE_TO_USER else None

        try:
            snippets = await self._retrieve_candidates(query, user_id)
        except Exception as e:
            return f"[memory lookup failed: {e}]"

        if not snippets:
            return self.valves.NOT_FOUND_TEXT

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

        if not self.valves.KEV_VERIFY_ENABLED:
            return str(ranked[0].get("text") or "") or self.valves.NOT_FOUND_TEXT

        for snippet in ranked:
            text = str(snippet.get("text") or "")
            if not text:
                continue
            try:
                probability = await self._verify_with_kev(query, text)
            except Exception:
                # Kev unreachable: fail open for this call rather than rejecting
                # every candidate just because the extra safety net is down.
                return text
            if probability >= self.valves.KEV_VERIFY_THRESHOLD:
                return text
        return self.valves.NOT_FOUND_TEXT

    async def _retrieve_candidates(
        self, query: str, user_id: Optional[str]
    ) -> list:
        filters: dict = {}
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
        """Asks Kev's System One endpoint (same one kev.py/fact.py talk to) whether
        `candidate_text` actually answers `query`. Raises on any transport/parsing
        failure so the caller can fail open."""
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

    async def ask_kev(
        self,
        state: str = Field(
            ...,
            description=(
                "The text to evaluate - a message, a draft answer, a claim, a piece "
                "of dialogue, anything you want Kev's opinion on."
            ),
        ),
        questions: str = Field(
            ...,
            description=(
                "One or more of your own typed questions about `state`, as a JSON "
                "object in the /v1/systemone `questions` shape, e.g.:\n"
                '{"is_angry": {"type": "noul", "instructions": "Is the tone of this '
                'message angry or hostile?", "criteria": {"true": "hostile, angry, or '
                'accusatory wording", "false": "neutral or friendly wording"}}, '
                '"category": {"type": "choice", "instructions": "Which category best '
                'fits this message?", "criteria": {"bug_report": null, "feature_request": '
                'null, "question": null}}}\n'
                "'noul' questions return a true/false probability; 'choice' questions "
                "return the best-matching key from `criteria` with its probability. "
                "Use your own question ids, instructions and criteria - ask only what "
                "you actually need, each question costs one classifier pass."
            ),
        ),
    ) -> str:
        """
        Ask Kev - this project's fast, independently-calibrated System One classifier
        (kev.serve) - one or more of your own yes/no or multiple-choice questions about
        a piece of text, instead of judging it yourself. Useful whenever you want a
        second, calibrated opinion on something like tone, urgency, category,
        contradiction, or duplication rather than reasoning it out unaided. You define
        the question ids, instructions and answer criteria yourself - you are not
        limited to whatever questions the admin configured elsewhere. Returns each
        question's id and verdict with its probability.
        """
        try:
            parsed_questions = json.loads(questions)
        except (json.JSONDecodeError, TypeError) as e:
            return f"[invalid questions JSON: {e}]"
        if not isinstance(parsed_questions, dict) or not parsed_questions:
            return (
                "[questions must be a non-empty JSON object of "
                '{id: {type, instructions, criteria}}]'
            )

        payload = {"state": state, "model": "kev-latest", "questions": parsed_questions}
        headers = {"content-type": "application/json"}
        if self.valves.KEV_API_KEY:
            headers["authorization"] = f"Bearer {self.valves.KEV_API_KEY}"
        timeout = aiohttp.ClientTimeout(total=self.valves.KEV_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.valves.KEV_URL.rstrip('/')}/v1/systemone",
                    json=payload,
                    headers=headers,
                ) as response:
                    text = await response.text()
                    if response.status != 200:
                        return f"[Kev returned HTTP {response.status}: {text[:200]}]"
                    answer = json.loads(text)
        except Exception as e:
            return f"[Kev unreachable: {e}]"

        return self._format_kev_answer(answer)

    @staticmethod
    def _format_kev_answer(answer: dict) -> str:
        """LLM-readable rendering of a /v1/systemone response, tolerant of
        whatever question types/ids the caller asked for (mirrors kev.py's
        `_verdict` but degrades gracefully instead of raising on odd shapes)."""
        parts = []
        for qid, a in (answer.get("answers") or {}).items():
            a_type = a.get("type") if isinstance(a, dict) else None
            if a_type == "noul":
                parts.append(f"{qid}: {a['noul']:.3f} probability true")
            elif a_type == "choice":
                probs = a.get("probabilities") or {}
                best = max(probs.values()) if probs else None
                verdict = f"{qid}: {a.get('choice')}"
                if best is not None:
                    verdict += f" (p {best:.3f})"
                parts.append(verdict)
            elif isinstance(a, dict):
                legend = a.get("legend", {}) or {}
                level = legend.get(str(round(a.get("score", 0))), "")
                verdict = f"{qid}: {a.get('score')}"
                if level:
                    verdict += f' "{level}"'
                if "confidence" in a:
                    verdict += f" (confidence {a['confidence']:.3f})"
                parts.append(verdict)
        return " · ".join(parts) if parts else "[Kev returned no answers]"

    async def extract_relations(
        self,
        sentence: str = Field(
            ...,
            description="The sentence or short passage to pull subject-relation-object facts out of.",
        ),
        relation_labels: str = Field(
            default="",
            description=(
                "Optional comma-separated zero-shot relation labels for GLiREL, e.g. "
                "'controls,part of,located in'. Leave empty to use the server's default label set."
            ),
        ),
    ) -> str:
        """
        Extract structured subject-relation-object facts from text instead of reading them
        out yourself. Calls the relation_extractor MCP server's layered pipeline (L0 structure
        normalization, L3 entities, L4 quantities, and L5 neural relation extraction - ReLiK
        plus zero-shot GLiREL - falling back to a spaCy/NLTK subject-verb-object pass where the
        neural models find nothing). Use this whenever asked to extract, list, or tabulate the
        relationships/facts present in a piece of text, rather than inferring them unaided.
        """
        labels = [label.strip() for label in relation_labels.split(",") if label.strip()] or None

        try:
            async with _MCPMemoryClient(
                self.valves.RELATION_EXTRACTOR_URL,
                self.valves.RELATION_EXTRACTOR_SECURITY_KEY,
                self.valves.RELATION_EXTRACTOR_TIMEOUT,
            ) as mcp:
                result = await mcp.call_tool(
                    "extract_relations_tool",
                    {"sentence": sentence, "relation_labels": labels},
                )
        except Exception as e:
            return f"[relation extraction failed: {e}]"

        if not isinstance(result, dict):
            return str(result)
        if result.get("message") and not result.get("relations") and not result.get("legacy_relations"):
            return str(result["message"])

        relations = result.get("relations") or []
        legacy = result.get("legacy_relations") or []
        if not relations and not legacy:
            return f"No relations found (method: {result.get('method', 'unknown')})."

        lines = [f"Method: {result.get('method', 'unknown')}"]
        for r in relations:
            score = r.get("score")
            lines.append(
                f"- {r.get('subject')} --[{r.get('relation')}]--> {r.get('object')}"
                + (f" (score {score:.2f})" if isinstance(score, (int, float)) else "")
            )
        for r in legacy:
            lines.append(f"- {r.get('subject')} --[{r.get('relation')}]--> {r.get('object')} (legacy fallback)")
        return "\n".join(lines)
