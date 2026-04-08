"""CLI entrypoint: smalltask run / smalltask init."""

from pathlib import Path

import click

from smalltask.runner import run_agent

# ---------------------------------------------------------------------------
# Default (blank) template
# ---------------------------------------------------------------------------

_EXAMPLE_TOOL = '''\
"""
Example tools for smalltask.

These functions are the security boundary — the agent can only do what these
functions allow. Replace the stub implementations with real logic.
"""

from smalltask import tool


@tool
def search_records(query: str, limit: int = 10) -> list:
    """Search records matching a query string.

    Args:
        query: Search term to match against record names and descriptions.
        limit: Maximum number of results to return.
    """
    # TODO: replace with a real data source (DB, API, etc.)
    return [
        {"id": f"REC-{i:04d}", "name": f"Record {i}", "description": f"Matches \'{query}\'"}
        for i in range(1, limit + 1)
    ]


@tool
def get_summary_stats() -> dict:
    """Return a high-level summary of the current dataset."""
    # TODO: replace with a real query
    return {
        "total_records": 1042,
        "active": 891,
        "pending": 151,
    }
'''

_EXAMPLE_AGENT = '''\
name: example_agent
description: Searches records and summarises the dataset.

llm:
  url: https://openrouter.ai/api/v1/chat/completions
  model: anthropic/claude-3.5-sonnet
  api_key_env: OPENROUTER_API_KEY
  max_tokens: 2048

prompt: |
  You are a helpful data analyst.

  The user wants to understand the current state of the dataset and find
  records related to "$topic".

  Using the available tools:
  1. Retrieve summary statistics for the dataset.
  2. Search for records related to the topic.
  3. Write a concise report covering what you found.

  Be direct. Use numbers. No fluff.

tools:
  - example.get_summary_stats
  - example.search_records
'''

# ---------------------------------------------------------------------------
# GitHub template
# ---------------------------------------------------------------------------

_GITHUB_TOOL = '''\
"""
GitHub tools for smalltask.

Read-only tools against the GitHub REST API. Requires a GITHUB_TOKEN
environment variable with at least `repo:read` scope.

Set it in your shell or scheduler:
    export GITHUB_TOKEN=ghp_...

These tools are the security boundary — the agent cannot call arbitrary
GitHub endpoints, only what is exposed here.
"""

import os
from datetime import datetime, timedelta, timezone

import httpx

from smalltask import tool

_BASE = "https://api.github.com"


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


@tool
def list_open_prs(repo: str, include_drafts: bool = False) -> list:
    """List open pull requests for a GitHub repository.

    Args:
        repo: Repository in owner/name format, e.g. acme/backend.
        include_drafts: Include draft PRs. Defaults to False.
    """
    resp = httpx.get(
        f"{_BASE}/repos/{repo}/pulls",
        headers=_headers(),
        params={"state": "open", "per_page": 50},
    )
    resp.raise_for_status()
    prs = resp.json()
    if not include_drafts:
        prs = [p for p in prs if not p.get("draft")]
    return [
        {
            "number": p["number"],
            "title": p["title"],
            "author": p["user"]["login"],
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
            "url": p["html_url"],
            "reviewers_requested": [r["login"] for r in p.get("requested_reviewers", [])],
        }
        for p in prs
    ]


@tool
def list_recent_merged_prs(repo: str, days: int = 7) -> list:
    """List pull requests merged in the last N days.

    Args:
        repo: Repository in owner/name format.
        days: Number of days to look back.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    resp = httpx.get(
        f"{_BASE}/repos/{repo}/pulls",
        headers=_headers(),
        params={"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100},
    )
    resp.raise_for_status()
    return [
        {
            "number": p["number"],
            "title": p["title"],
            "author": p["user"]["login"],
            "merged_at": p["merged_at"],
            "url": p["html_url"],
        }
        for p in resp.json()
        if p.get("merged_at") and p["merged_at"] >= since
    ]


@tool
def get_pr_review_status(repo: str, pr_number: int) -> dict:
    """Get the current review status of a pull request.

    Returns who has approved, requested changes, or not yet reviewed.

    Args:
        repo: Repository in owner/name format.
        pr_number: The PR number.
    """
    resp = httpx.get(
        f"{_BASE}/repos/{repo}/pulls/{pr_number}/reviews",
        headers=_headers(),
    )
    resp.raise_for_status()

    # Keep only the latest review state per reviewer
    latest: dict[str, str] = {}
    for r in resp.json():
        latest[r["user"]["login"]] = r["state"]

    return {
        "pr_number": pr_number,
        "approved_by": [u for u, s in latest.items() if s == "APPROVED"],
        "changes_requested_by": [u for u, s in latest.items() if s == "CHANGES_REQUESTED"],
        "commented_by": [u for u, s in latest.items() if s == "COMMENTED"],
    }


@tool
def list_issues(repo: str, state: str = "open", label: str = "") -> list:
    """List issues for a GitHub repository.

    Args:
        repo: Repository in owner/name format.
        state: Issue state — open, closed, or all.
        label: Filter by label name. Leave empty to skip filter.
    """
    params: dict = {"state": state, "per_page": 50}
    if label:
        params["labels"] = label
    resp = httpx.get(f"{_BASE}/repos/{repo}/issues", headers=_headers(), params=params)
    resp.raise_for_status()
    return [
        {
            "number": i["number"],
            "title": i["title"],
            "author": i["user"]["login"],
            "labels": [la["name"] for la in i.get("labels", [])],
            "created_at": i["created_at"],
            "url": i["html_url"],
        }
        for i in resp.json()
        if "pull_request" not in i  # issues endpoint returns PRs too
    ]


@tool
def get_workflow_runs(repo: str, workflow: str, conclusion: str = "failure") -> list:
    """Get recent runs for a GitHub Actions workflow.

    Args:
        repo: Repository in owner/name format.
        workflow: Workflow filename, e.g. ci.yml or publish.yml.
        conclusion: Filter by conclusion — failure, success, cancelled, or all.
    """
    params: dict = {"per_page": 10}
    if conclusion != "all":
        params["status"] = conclusion
    resp = httpx.get(
        f"{_BASE}/repos/{repo}/actions/workflows/{workflow}/runs",
        headers=_headers(),
        params=params,
    )
    resp.raise_for_status()
    return [
        {
            "id": r["id"],
            "conclusion": r["conclusion"],
            "branch": r["head_branch"],
            "commit": r["head_sha"][:7],
            "started_at": r["run_started_at"],
            "url": r["html_url"],
        }
        for r in resp.json().get("workflow_runs", [])
    ]
'''

_GITHUB_AGENT = '''\
name: github_pr_digest
description: Weekly digest of open PRs, recent merges, and CI health.

llm:
  url: https://openrouter.ai/api/v1/chat/completions
  model: anthropic/claude-3.5-sonnet
  api_key_env: OPENROUTER_API_KEY
  max_tokens: 2048

prompt: |
  You are an engineering lead reviewing the state of the $repo GitHub repository.

  Produce a concise weekly digest covering:
  1. Open PRs — who is waiting on review, who is blocked, how long PRs have been open
  2. Merged this week — what shipped
  3. CI health — any workflow failures on main
  4. Action items — specific people who should review specific PRs

  Be direct. Use names and numbers. No fluff.

tools:
  - github.list_open_prs
  - github.list_recent_merged_prs
  - github.get_pr_review_status
  - github.get_workflow_runs
  - github.list_issues
'''

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, tuple[str, str, str, str, str]] = {
    # name: (tool_filename, tool_content, agent_filename, agent_content, hint)
    "default": (
        "example.py",
        _EXAMPLE_TOOL,
        "example.yaml",
        _EXAMPLE_AGENT,
        "smalltask run agents/example.yaml --var topic=<your topic> --verbose",
    ),
    "github": (
        "github.py",
        _GITHUB_TOOL,
        "github_pr_digest.yaml",
        _GITHUB_AGENT,
        "export GITHUB_TOKEN=ghp_...\nsmalltask run agents/github_pr_digest.yaml --var repo=owner/name --verbose",
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """smalltask: define tools and agents as code, run them anywhere."""


@cli.command()
@click.argument("agent", type=click.Path(exists=True, path_type=Path))
@click.option("--tools-dir", "-t", type=click.Path(path_type=Path), default=None,
              help="Directory containing tool Python files. Auto-detected if not set.")
@click.option("--var", "-v", multiple=True, metavar="KEY=VALUE",
              help="Input variables to interpolate into the prompt. Repeatable.")
@click.option("--verbose", is_flag=True, help="Print tool calls and results.")
def run(agent: Path, tools_dir: Path | None, var: tuple[str, ...], verbose: bool):
    """Run an agent defined by a YAML file.

    Example:

        smalltask run agents/weekly_review.yaml --var week=2024-W01
    """
    input_vars = {}
    for v in var:
        if "=" not in v:
            raise click.BadParameter(f"Expected KEY=VALUE, got: {v}", param_hint="--var")
        k, val = v.split("=", 1)
        input_vars[k] = val

    result = run_agent(agent, tools_dir=tools_dir, input_vars=input_vars or None, verbose=verbose)
    click.echo(result)


@cli.command()
@click.argument("directory", type=click.Path(path_type=Path), default=".")
@click.option(
    "--template", "-t",
    type=click.Choice(list(_TEMPLATES.keys())),
    default="default",
    show_default=True,
    help="Starter template to scaffold.",
)
@click.option("--list", "list_templates", is_flag=True, help="List available templates and exit.")
def init(directory: Path, template: str, list_templates: bool):
    """Scaffold a new smalltask project in DIRECTORY (default: current directory).

    \b
    Examples:
        smalltask init
        smalltask init --template github
        smalltask init my-project/ --template github
        smalltask init --list
    """
    if list_templates:
        click.echo("Available templates:\n")
        for name in _TEMPLATES:
            tool_file, _, agent_file, _, _ = _TEMPLATES[name]
            click.echo(f"  {name:<12}  tools/{tool_file}  +  agents/{agent_file}")
        return

    directory = directory.resolve()
    tool_filename, tool_content, agent_filename, agent_content, hint = _TEMPLATES[template]

    tools_dir = directory / "tools"
    agents_dir = directory / "agents"
    tools_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    tool_file = tools_dir / tool_filename
    agent_file = agents_dir / agent_filename
    created = []

    if not tool_file.exists():
        tool_file.write_text(tool_content)
        created.append(str(tool_file.relative_to(directory)))
    else:
        click.echo(f"  skip    {tool_file.relative_to(directory)} (already exists)")

    if not agent_file.exists():
        agent_file.write_text(agent_content)
        created.append(str(agent_file.relative_to(directory)))
    else:
        click.echo(f"  skip    {agent_file.relative_to(directory)} (already exists)")

    for path in created:
        click.echo(f"  create  {path}")

    if created:
        click.echo(f"\nDone. Next steps:\n\n  {hint}\n")
