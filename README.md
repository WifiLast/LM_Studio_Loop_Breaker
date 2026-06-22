# LM Studio Loop Breaker

Two drop-in [Open WebUI](https://github.com/open-webui/open-webui) functions that keep **small / local models** (LM Studio, llama.cpp, Ollama, or any OpenAI-compatible backend) from spiralling into repetition and going off the rails on complex requests.

Small models fail in two characteristic ways:

1. **They loop.** The same paragraph or line is regenerated over and over, the context fills with the runaway text, and the next turn keeps the pattern alive.
2. **They choke on big asks.** Given one large, open-ended request they ramble, lose the thread, and produce shallow output.

This repository ships one function for each failure mode:

| File | Type | What it does |
|------|------|--------------|
| [`loop_detection.py`](loop_detection.py) | Filter | **Detects and aborts deliberation loops** — collapses repeated blocks, scrubs the loop out of history, truncates looping streams, and lowers sampling temperature to escape the loop. |
| [`planning_standalone.py`](planning_standalone.py) | Pipe + CLI | **Splits a task into smaller tasks** — plans → executes each atomic step → synthesizes a final answer, attaching only the MCP tools each step actually needs and enforcing a hard work budget. |

The two are complementary: the planner breaks big work into bounded steps; the loop guard keeps each step from running away.

---

## 1. Loop Detection & Abortion — `loop_detection.py`

An Open WebUI **Filter** (`title: Loop Guard`) that breaks repetition loops at three points in the request lifecycle:

- **`stream`** — accumulates the response as it streams; once any non-trivial block/line repeats past the threshold, it emits a one-time notice and **suppresses all further output**, aborting the loop before it reaches the user or storage.
- **`outlet`** — collapses over-repeated paragraphs and lines in the finished response, keeping at most `max_repeats` copies of each block.
- **`inlet`** — scrubs the same repetition out of prior assistant messages so the model never *sees* (and therefore never *continues*) the pattern on the next turn. When a loop is detected it also **reduces `temperature` (−30%) and caps `max_tokens`** to force a faster, more deterministic escape.

Detection is whitespace- and case-insensitive, ignores short legitimate repeats (bullets, "OK"), and bounds its cost on long streams.

### Configuration (Valves)

| Valve | Default | Purpose |
|-------|---------|---------|
| `max_repeats` | `2` | Copies of an identical block/line to keep. |
| `min_block_chars` | `40` | Ignore blocks shorter than this. |
| `scrub_history` | `true` | Clean prior assistant messages on inlet (breaks cross-turn loops). |
| `truncate_stream` | `true` | Suppress output mid-stream once a loop starts. |
| `adjust_temperature` | `true` | Lower temperature / cap tokens on detection. |
| `loop_threshold` | `3` | Repetitions needed to declare a definite loop. |
| `priority` | `0` | Filter execution order (lower runs first). |

### Install

1. Open WebUI → **Workspace → Functions → +**.
2. Paste the contents of [`loop_detection.py`](loop_detection.py) and save.
3. Enable it for the model or chat you want protected.

---

## 2. Task Splitting with Budgeted Tools — `planning_standalone.py`

A standalone, single-agent adaptation of *Planner v3* (no subagents). It runs the core agentic loop:

```
plan       ->  decompose the request into the smallest independently-executable tasks
execute    ->  run each task with one LLM, calling only the tools that task needs
synthesize ->  merge the task results into one clean final answer (@task_id macros)
```

### Only the required MCP tools

During planning the model is shown a catalog of the connected MCP tools and must set each task's `tools` field to **only the subset that task genuinely requires** — pure reasoning/drafting steps get no tools at all. At execution time each task is handed exactly that subset, so a tool is never offered to a step that has no use for it. Connect MCP servers over `streamable-http`, `sse`, or `stdio`; the planner gracefully degrades (skips unreachable servers, runs tool-free) when none are available.

### Hard work budget

Every axis of effort is capped so a small model can't run forever:

| Budget | Default | Effect |
|--------|---------|--------|
| `max_tasks` | `20` | Maximum tasks in a plan. |
| `max_tool_iterations` | `6` | Tool-call rounds per task before a final tool-free answer is forced. |
| `max_tokens_per_task` | `8000` | Per-task token budget (warns at 80%). |
| `task_result_limit` | `6000` | Chars of each result carried forward to later tasks. |
| `tool_result_limit` | `4000` | Chars of each tool result fed back to the model. |
| retry cap | `3` attempts | On a detected loop, retries with escalating temperature and progressively disabled tools, then accepts the best unique content. |

### Built-in loop resistance

The planner has its own per-phase loop detection (`plan` / `execution` / `synthesis` thresholds). On a repetition it retries with adjusted sampling (higher temperature, halved `top_k`), disables tools that contributed to loops, and finally extracts the longest unique prefix rather than storing the runaway text. It also expands abbreviations to full words — a common small-model accuracy win.

### Run it two ways

**A. Open WebUI Pipe** — appears as a selectable model *"Planner (Standalone)"*:

1. Add [`planning_standalone.py`](planning_standalone.py) as a Function.
2. Set the `PLANNER_MODEL` valve to the real model id it should drive (blank = auto-select).
3. Optionally enable `MCP_ENABLED` and point it at your MCP server(s).

**B. CLI** — against any OpenAI-compatible endpoint (e.g. LM Studio):

```bash
# Env defaults: OPENAI_API_KEY, OPENAI_BASE_URL, PLANNER_MODEL
python planning_standalone.py "Write a market analysis of EV charging in the EU"

# Point at a local LM Studio server, cap the plan, add a memory MCP tool
python planning_standalone.py \
  --api-url http://localhost:1234/v1 \
  --model qwen2.5-7b-instruct \
  --max-tasks 8 \
  --max-tool-iterations 4 \
  --mcp-url http://localhost:8082/memory \
  "Summarize the latest battery-recycling regulations"

# Single-pass (no planning), full result as JSON
python planning_standalone.py --no-plan --json "Summarize X"
```

Key CLI flags: `--model`, `--api-url`, `--api-key`, `--temperature` (plus per-phase variants), sampling controls (`--top-p`, `--top-k`, `--min-p`, `--max-tokens`, `--repeat-penalty`, `--seed`, …), `--mcp-url` / `--mcp-config`, `--max-tool-iterations`, `--tool-result-limit`, `--max-tasks`, `--no-plan`, `--review`, `--quiet`, `--json`.

---

## Requirements

- **Open WebUI** ≥ `0.5.0` for the Filter and Pipe paths.
- **Python 3.10+** with [`pydantic`](https://pypi.org/project/pydantic/) for both files.
- [`openai`](https://pypi.org/project/openai/) for the planner CLI.
- [`mcp`](https://pypi.org/project/mcp/) only if you enable MCP tool calling (optional — degrades gracefully without it).

## Credits

`planning_standalone.py` is adapted from **Planner v3 by Haervwe**, stripped down to a single-agent loop with MCP tool selection and budgeting. `loop_detection.py` is an enhanced loop-guard filter built for the PLS `math_mcp` setup.
