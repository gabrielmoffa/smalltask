# smalltask

Define tools and agents as code. Run them anywhere.

```bash
pip install smalltask
```

---

smalltask is a lightweight framework for building scheduled AI agents. Tools are Python functions. Agents are YAML files. Both live in your git repo — diffable, reviewable, auditable.

Bring your own scheduler (Airflow, cron, GitHub Actions). Bring your own LLM (any OpenAI-compatible endpoint).

---

## Quickstart

```bash
smalltask init                  # scaffold tools/, agents/, and smalltask.yaml
smalltask init --template github   # scaffold GitHub tools + PR digest agent
```

Then run:

```bash
smalltask run agents/example.yaml --var topic="revenue drop" --verbose
```

---

## How it works

**Tools** are `@tool`-decorated Python functions. The function is the security boundary — the agent can only do what you explicitly expose.

```python
# tools/orders.py
from smalltask import tool

@tool
def get_order_summary(days: int) -> dict:
    """Return aggregated order stats for the last N days."""
    ...

@tool
def get_top_customers(days: int, limit: int) -> list:
    """Return the top customers by spend in the last N days."""
    ...
```

**Agents** are YAML files. They declare the prompt, which tools to use, and which LLM to call.

```yaml
# agents/weekly_review.yaml
name: weekly_review
description: Weekly order digest with anomaly detection.

llm:
  connection: openrouter
  model: anthropic/claude-sonnet-4-6-20250514

prompt: |
  You are a data analyst reviewing the last 7 days of orders.
  Summarise volume, revenue, refund rate, and top customers.
  Flag anything unusual. Be direct. Use numbers.

tools:
  - orders.get_order_summary
  - orders.get_top_customers
```

Reference tools as `file.function` to be explicit and avoid name collisions.

Use `$varname` in prompts for runtime variables:

```yaml
prompt: |
  Review orders for the week of $week.
  ...
```

```bash
smalltask run agents/weekly_review.yaml --var week=2024-W01
```

---

## Connections

Define LLM provider connections once in a project-level `smalltask.yaml`, then reference them by name in any agent YAML.

```yaml
# smalltask.yaml
connections:
  openrouter:
    url: https://openrouter.ai/api/v1/chat/completions
    api_key_env: OPENROUTER_API_KEY

  ollama:
    url: http://localhost:11434/v1/chat/completions

  groq:
    url: https://api.groq.com/openai/v1/chat/completions
    api_key_env: GROQ_API_KEY

  together:
    url: https://api.together.xyz/v1/chat/completions
    api_key_env: TOGETHER_API_KEY

  bedrock:
    url: https://bedrock-runtime.us-east-1.amazonaws.com/v1/chat/completions
    api_key_env: AWS_SECRET_ACCESS_KEY
```

Then agent YAMLs stay clean:

```yaml
llm:
  connection: openrouter
  model: anthropic/claude-sonnet-4-6-20250514
  max_tokens: 2048
```

The connection provides the URL, auth, and headers. The agent provides (or overrides) the model and other settings. `smalltask init` scaffolds a `smalltask.yaml` with commented-out presets for common providers.

You can still use inline `llm.url` directly if you prefer — connections are optional.

---

## Project structure

```
your-repo/
├── smalltask.yaml          # connection presets (one per project)
├── tools/
│   ├── orders.py           # get_order_summary, get_top_customers, ...
│   ├── github.py           # list_open_prs, get_workflow_runs, ...
│   └── slack.py            # post_message, ...
├── agents/
│   ├── weekly_review.yaml
│   └── github_pr_digest.yaml
└── dags/
    └── weekly_review_dag.py   # optional: Airflow integration
```

Tools are discovered from the `tools/` directory. Agent YAMLs reference them by name.

---

## Schedulers

smalltask doesn't own scheduling — it drops into whatever you already have.

### GitHub Actions

The fastest way to get a scheduled agent running. No infrastructure required.

```yaml
# .github/workflows/weekly_review.yml
name: Weekly order review

on:
  schedule:
    - cron: '0 9 * * 1'   # every Monday at 9am UTC
  workflow_dispatch:        # also allow manual runs from the GitHub UI

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install smalltask

      - run: smalltask run agents/weekly_review.yaml --var week=$(date +%Y-W%V)
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Store your API key under **Settings → Secrets → Actions** in the GitHub repo.

### Cron

```bash
# crontab -e
0 9 * * 1 cd /path/to/repo && smalltask run agents/weekly_review.yaml --var week=$(date +\%Y-W\%V) >> /var/log/smalltask.log 2>&1
```

### Airflow

```python
from airflow.operators.python import PythonOperator
from smalltask.runner import run_agent
from pathlib import Path

PythonOperator(
    task_id="weekly_review",
    python_callable=run_agent,
    op_kwargs={
        "agent_path": Path("agents/weekly_review.yaml"),
        "input_vars": {"week": "{{ ds }}"},
    },
)
```

### Python

```python
from smalltask.runner import run_agent
from pathlib import Path

result = run_agent(
    agent_path=Path("agents/weekly_review.yaml"),
    input_vars={"week": "2024-W01"},
)
```

---

## Agent YAML reference

| Field | Required | Description |
|---|---|---|
| `name` | yes | Agent identifier |
| `description` | no | Human-readable description |
| `prompt` | yes | System prompt. Supports `$var` interpolation. |
| `tools` | yes | List of tool names (`file.function` or bare `function`) |
| `llm.connection` | no | Named connection from `smalltask.yaml` |
| `llm.url` | no | OpenAI-compatible endpoint URL (alternative to `connection`) |
| `llm.model` | yes | Model identifier |
| `llm.api_key_env` | no | Name of env var holding the API key (set in connection or here) |
| `llm.max_tokens` | no | Max tokens per LLM call (default: 4096) |
| `llm.timeout` | no | HTTP timeout in seconds (default: 120) |
| `llm.extra_headers` | no | Additional HTTP headers (e.g. `HTTP-Referer`) |
| `max_iterations` | no | Max agentic loop iterations (default: 20) |
| `max_total_tokens` | no | Token budget across all iterations — stops early if exceeded (default: no limit) |
| `pre_hook` | no | List of tool calls to run before the LLM loop (see [Hooks](#hooks)) |
| `post_hook` | no | List of tool calls to run after the LLM loop (see [Hooks](#hooks)) |

---

## Hooks

Hooks let you run deterministic tool calls before and after the LLM loop. They use the same tools you already have — no new concepts.

```yaml
name: metrics_alert
prompt: |
  Analyze the attached metrics. Flag anomalies. Be direct.

llm:
  connection: openrouter
  model: anthropic/claude-sonnet-4-6-20250514

tools:
  - analysis.plot_revenue
  - analysis.get_summary

pre_hook:
  - analysis.snapshot_metrics:
      days: 7
  - analysis.check_threshold:
      metric: error_rate
      max: 0.05

post_hook:
  - reporting.upload_charts
  - reporting.send_slack_report:
      channel: "#alerts"
```

### Pre-hooks

Pre-hooks run sequentially before the LLM. Their results are injected into the prompt so the LLM can see the data.

Each entry is a tool name with optional args:

```yaml
pre_hook:
  - orders.get_summary:
      days: 7
  - orders.check_threshold:
      metric: refund_rate
      max: 0.05
```

**Skip gate** — if a pre-hook returns `{"skip": True}`, the agent stops immediately without calling the LLM. Use this to avoid wasting tokens when there's nothing to act on:

```python
@tool
def check_threshold(metric: str, max: float) -> dict:
    """Only run the agent if a metric exceeds a threshold."""
    value = get_current_value(metric)
    if value <= max:
        return {"skip": True, "reason": f"{metric} is {value}, below {max}"}
    return {"value": value}
```

### Post-hooks

Post-hooks run after the LLM finishes. The framework auto-injects two special parameters if your tool accepts them:

- **`output`** (`str`) — the LLM's final response text.
- **`tool_results`** (`list`) — every tool call made during the agent loop. Each entry is `{"tool": name, "args": {...}, "result": ...}`.

Just declare the parameters you need — the framework fills them in:

```python
@tool
def send_slack_report(output: str, tool_results: list, channel: str) -> str:
    """Post the LLM report and any chart images to Slack."""
    charts = [r["result"] for r in tool_results if r["result"].endswith(".png")]
    post_to_slack(channel=channel, text=output, attachments=charts)
    return f"sent to {channel} with {len(charts)} charts"
```

```yaml
post_hook:
  - slack.send_slack_report:
      channel: "#alerts"
```

The `channel` comes from the YAML. The `output` and `tool_results` are injected by the framework.

You can filter `tool_results` however you want — by tool name, by result content, by args:

```python
# Get all chart paths
charts = [r["result"] for r in tool_results if r["tool"].startswith("plot_")]

# Get results from a specific tool
summaries = [r["result"] for r in tool_results if r["tool"] == "analysis.get_summary"]

# Get all tool calls that used a specific argument
weekly = [r for r in tool_results if r["args"].get("days") == 7]
```

---

## Multi-agent

Sub-agents can be called as tools. The parent agent passes a task string; the sub-agent runs its full loop and returns a string result.

```python
from smalltask.runner import agent_tool, run_agent
from pathlib import Path

run_agent(
    Path("agents/orchestrator.yaml"),
    extra_tools={
        "summarize": agent_tool(
            name="summarize",
            agent_path=Path("agents/summarize.yaml"),
            description="Summarise a block of text. Pass it as 'task'.",
        )
    },
)
```

The orchestrator YAML lists `summarize` in its `tools:` section like any other tool.

---

## Examples

### Daily improvement PRs

A fully working example that runs as a daily GitHub Action: reads your codebase, picks one improvement, opens a PR, and notifies you on Telegram.

See [`examples/daily_improvements/`](examples/daily_improvements/) for the tools and agent YAML, and [`.github/workflows/daily_improvements.yml`](.github/workflows/daily_improvements.yml) for the workflow.

Features demonstrated:
- **Pre-hook** — checks for pending bot PRs (skips if one exists; closes and retries if you commented `/reject`)
- **Agentic loop** — LLM reads files, decides on an improvement, writes the change, creates a PR
- **Post-hook** — sends a Telegram notification with the PR link

---

## Templates

`smalltask init --list` shows available starter templates:

| Template | Scaffolds |
|---|---|
| `default` | Generic stub tools + example agent + `smalltask.yaml` |
| `github` | GitHub REST API tools + PR digest agent + `smalltask.yaml` |

```bash
smalltask init --template github
```

---

## LLM compatibility

smalltask uses native OpenAI-compatible tool calling over raw HTTP — no SDK, no provider lock-in. It works with any endpoint that supports the OpenAI tool-calling format:

- [OpenRouter](https://openrouter.ai) — access any model via one API key
- [Ollama](https://ollama.com) — local models
- [Groq](https://groq.com)
- [Together AI](https://www.together.ai)
- Anthropic, OpenAI, Gemini via their OpenAI-compatible layers
- Any Bedrock / Azure endpoint with an OpenAI-compatible adapter

---

## Contributing

```bash
git clone https://github.com/gabrielmoffa/smalltask
cd smalltask
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest tests/
```

The tests cover core logic (schema generation, tool loading, prompt parsing) without requiring a real LLM or API key. If you change `loader.py` or `prompt_tools.py`, run them before pushing.

---

## License

MIT
