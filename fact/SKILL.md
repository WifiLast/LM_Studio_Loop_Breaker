---
name: memory-mcp
description: Use this skill whenever the user wants to store, search, update, delete, link, or analyze memories through the mcp-memory MCP server (mcp__memory_mcp__* tools), or wants help understanding/extending mcp_memory/server.py itself. Covers remembering notes/code snippets, semantic retrieval, prompt enrichment, code/workspace context tracking, memory graph & links, labels, bulk import/export, and observability. Activates on phrases like "remember this", "save this as a memory", "retrieve memories about X", "search my memories", "link these memories", "enrich this prompt with memory context", "store code context", "what code symbols are tracked", "bulk import/export memories", "show memory graph clusters".
---

# Memory MCP Server

Helps use the `mcp-memory` MCP server (`mcp_memory/server.py` + its registrar modules) correctly:
which tool to call for a given intent, what payload shape it expects, and what workflow
chains multiple tools together.

## Server layout

`mcp_memory/server.py` defines the FastMCP app (`mcp = FastMCP("mcp-memory")`) and the core
memory tools directly. It then calls three registrar functions that add the rest of the
toolset from sibling files:

- `mcp_memory/server.py` — core memory CRUD + images
- `mcp_memory/server_code_workspace.py` (`register_code_workspace_tools`) — code/workspace/rollup tools
- `mcp_memory/server_link_graph_tools.py` (`register_link_graph_tools`) — links, graph traversal, categorization
- `mcp_memory/server_ops_analytics.py` (`register_ops_analytics_tools`) — bulk ops, observability, recommendations, labels

All tools accept an optional `security_key` argument, only enforced if the
`MCP_MEMORY_SECURITY_KEY` env var is set (see `require_security_key` in `server.py`). Most
tools take `user_id` to scope data per user/tenant — default is `"default"`.

When calling these tools live, use the `mcp__memory_mcp__<tool_name>` names (loaded via
ToolSearch if deferred).

## Tool reference by task

### Store a memory
- `remember` — free-text note/directive/task. Give `text`, `user_id`, `type`. Auto-dedupes
  near-identical text (see `deduplicated`/`near_duplicate` in response).
- `remember_code_example` — code snippets with an explanation; skips the SQL-injection
  heuristics `remember` applies to raw text.
- `bulk_remember` — batch version of `remember` for many items at once (rate-limited).
- `upload_memory_image` / `list_memory_images` / `get_memory_image` / `delete_memory_image`
  — attach/inspect/remove images on an existing memory (`memory_id`).

### Retrieve / search
- `retrieve` — semantic search over memories (`query`, `k`, `filters.user_id`, `filters.types`,
  optional `min_score`, `expand_query`, `include_graph`). Returns ranked `snippets` with a
  scoring `trace`.
- `enrich_prompt` — wraps a user prompt with bounded, clearly-separated memory context before
  sending it to an LLM.
- `retrieve_by_symbol` — pull memories + code symbols tied to a `symbol_name` in a workspace.
- `search_by_dependency` — query the code dependency graph (`dependency_kind`, source/target
  symbol ids).
- `find_pattern_matches` — text search over registered reusable code patterns.
- `explain_memory` — provenance/embedding/rollup trace for one `event_id`.

### Update / delete
- `update` — patch any subset of fields on a memory (`text`, `stichwort`, `type`, `ttl_days`,
  `pii_tags`, `consent`, `labels`, `deleted`). Pass `user_id` to enforce ownership.
- `delete` — soft-delete (sets `deleted=true`; restorable via `update`).
- `bulk_delete` — soft-delete a filtered set; requires `confirmation=true`.

### Code & workspace context
- `analyze_code_context` / `store_code_context` — extract or persist symbols/dependencies from
  a code snippet for a `workspace_id`.
- `snapshot_workspace_state` — capture a point-in-time manifest of a workspace root.
- `list_code_symbols` / `list_code_dependencies` / `list_workspace_snapshots` /
  `list_refactoring_opportunities` — list stored records.
- `track_symbol_change` / `query_symbol_history` — record and read per-symbol change history.
- `register_code_pattern` — save a reusable code pattern template.
- `suggest_refactorings` — surface refactor opportunities for a workspace/symbol.
- `get_workspace_evolution` / `get_workspace_evolution_detail` / `compare_snapshots` — diff
  snapshots over time (files added/removed, symbol renames).

### Links & graph
- `link` — create a weighted relationship between two memories (`kind` from `LinkKind`).
- `list_word_links` / `delete_word_link` — word-level link management.
- `get_linked_memories` — memories directly linked to a given one.
- `graph_first_retrieve` — traverse the memory graph outward from seed ids.
- `find_graph_paths` — shortest/all paths between two memories.
- `graph_embedding` — graph-neighbourhood feature vector for a memory.
- `merge_links` — collapse duplicate links between two memories.
- `analyze_memory_graph` / `find_memory_clusters` — clustering + centrality stats for a user's graph.
- `suggest_memory_links` / `recommend_related_memories` — similarity + graph-based suggestions.

### Labels & categorization
- `categorize` — suggest ontology labels for arbitrary text.
- `create_label` / `merge_labels` / `suggest_label_hierarchy` — manage the label taxonomy.

### Summaries & rollups
- `summarize_daily` / `rollup_weekly` — deterministic daily/weekly rollups for a user.
- `summarize_text` — spelling/grammar correction + summarization of arbitrary text.

### Bulk & ops
- `export_memories` / `import_memories` — JSON/CSV/parquet backup and restore.
- `rate_snippet` / `record_retrieval_feedback` — feedback loops for retrieval quality.
- `suggest_next_memory_types` — usage-pattern-based type suggestions.
- `get_observability_snapshot` / `prometheus_metrics` — health/usage metrics.

## Common workflows

**Save then recall:** call `remember` with a clear `stichwort` (title) and specific `type`,
then verify with `retrieve` using a natural-language query and `filters.user_id` matching
what you stored.

**Track code understanding across a session:** `snapshot_workspace_state` once per workspace
root, then `store_code_context` per file/snippet touched, then `get_workspace_evolution`
or `compare_snapshots` to see what changed between snapshots.

**Connect related facts:** after two `remember` calls, `link` them with an appropriate `kind`;
use `graph_first_retrieve` or `find_graph_paths` later to explore the resulting graph instead
of re-retrieving by text.

**Safe deletes:** prefer `update` with `{"deleted": false}` to undo a `delete`/`bulk_delete`
rather than re-creating the memory — soft-delete preserves provenance and links.

## Downstream consumers in this repo

Two Open WebUI plugins in this `fact/` directory sit on top of `retrieve` to turn stored
memories into inline facts, both gated by the same Kev (`kev.py`) System One verification
before anything gets used:

- `fact/fact.py` (Filter) — scans the last user message for `{{fact: <query>}}`
  placeholders, calls `retrieve` per query, ranks candidates by importance (relevance +
  type + recency + durability — see its `score_snippet`), verifies the top-ranked
  candidate with Kev before substituting, and rewrites the message before the chat model
  ever sees the placeholder syntax. Fails open: a lookup error leaves the placeholder
  untouched; a lookup that finds nothing trustworthy substitutes `NOT_FOUND_TEXT` instead.
  Automatic — no model turn required.
- `fact/llm_werkzeug.py` (Tools) — exposes the same retrieve → rank → Kev-verify pipeline
  as an LLM-callable function, `recall_fact(query)`, so the model can look up a fact itself
  mid-generation and phrase it into its own answer, instead of only the regex-driven
  placeholder scan `fact.py` does automatically. The same file also has `ask_kev(state,
  questions)`, letting the model pose its own typed `noul`/`choice` questions straight to
  Kev's `/v1/systemone` endpoint (not memory-backed, but shares the file and the
  `KEV_URL`/`KEV_API_KEY` valves).

Both `fact.py` and `llm_werkzeug.py` duplicate a minimal `_MCPMemoryClient` rather than
importing this server's client code, since Open WebUI loads each Filter/Tools file as an
isolated module.

## Gotchas
- `remember` rejects raw text that looks like it's attempting SQL injection; use
  `remember_code_example` for code snippets that trip this heuristic.
- `retrieve` filters are strict by default (`strict_filters=True`) — an unscoped `user_id`
  can return nothing even if memories exist for a different implicit default.
- `bulk_delete` silently no-ops without `confirmation=true` — it raises instead.
- Ownership checks (`update`, `delete`) are skipped entirely when `user_id` is omitted; only
  omit it for trusted/local callers.
