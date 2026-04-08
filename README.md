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
smalltask init                  # scaffold tools/ and agents/
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

**Agents** are YAML files. They declare the prompt, which tools to use, and which LLM endpoint to call.

```yaml
# agents/weekly_review.yaml
name: weekly_review
description: Weekly order digest with anomaly detection.

llm:
  url: https://openrouter.ai/api/v1/chat/completions
  model: anthropic/claude-3.5-sonnet
  api_key_env: OPENROUTER_API_KEY

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

## Project structure

```
your-repo/
├── tools/
│   ├── orders.py       # get_order_summary, get_top_customers, ...
│   ├── github.py       # list_open_prs, get_workflow_runs, ...
│   └── slack.py        # post_message, ...
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
| `llm.url` | yes | OpenAI-compatible endpoint URL |
| `llm.model` | yes | Model identifier |
| `llm.api_key_env` | no | Name of env var holding the API key |
| `llm.max_tokens` | no | Max tokens per LLM call (default: 4096) |
| `llm.timeout` | no | HTTP timeout in seconds (default: 120) |
| `llm.extra_headers` | no | Additional HTTP headers (e.g. `HTTP-Referer`) |
| `max_iterations` | no | Max agentic loop iterations (default: 20) |
| `max_total_tokens` | no | Token budget across all iterations — stops early if exceeded (default: no limit) |

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

## Templates

`smalltask init --list` shows available starter templates:

| Template | Scaffolds |
|---|---|
| `default` | Generic stub tools + example agent |
| `github` | GitHub REST API tools + PR digest agent |

```bash
smalltask init --template github
```

---

## LLM compatibility

smalltask uses prompt-based tool calling over raw HTTP — no SDK, no provider lock-in. It works with any OpenAI-compatible endpoint:

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
