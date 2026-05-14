"""Track which AI-news URLs have been sent. Pruned to a 7-day window."""

import json
import os
import subprocess
from datetime import date, timedelta

from smalltask import tool

_STATE_PATH = "examples/ai_news/sent_news.json"
_WINDOW_DAYS = 7


def _load() -> dict:
    if not os.path.exists(_STATE_PATH):
        return {"sent": []}
    with open(_STATE_PATH) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {"sent": []}
    if "sent" not in data or not isinstance(data["sent"], list):
        return {"sent": []}
    return data


def _save(data: dict) -> None:
    with open(_STATE_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _prune(entries: list[dict]) -> list[dict]:
    cutoff = (date.today() - timedelta(days=_WINDOW_DAYS)).isoformat()
    return [e for e in entries if e.get("sent_at", "") >= cutoff]


def _git_commit_state(message: str) -> str:
    """Stage, commit, and push sent_news.json. Returns status string."""
    try:
        subprocess.run(["git", "add", _STATE_PATH], check=True, capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            return "no state change to commit"
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
        subprocess.run(["git", "push"], check=True, capture_output=True)
        return "state committed and pushed"
    except subprocess.CalledProcessError as e:
        return f"git commit/push failed: {e}"


def _append_urls(urls: list[str]) -> None:
    data = _load()
    today = date.today().isoformat()
    existing = {e["url"] for e in data["sent"]}
    for u in urls:
        if u and u not in existing:
            data["sent"].append({"url": u, "sent_at": today})
            existing.add(u)
    data["sent"] = _prune(data["sent"])
    _save(data)


@tool
def load_recent_urls() -> list[str]:
    """Return URLs sent in the last 7 days."""
    data = _load()
    pruned = _prune(data["sent"])
    return [e["url"] for e in pruned]


@tool
def append_sent(output: str, tool_results: list) -> str:
    """Post-hook: extract URLs from successful send_story calls and
    append them to sent_news.json, then commit + push.

    Auto-receives output and tool_results from the post-hook framework.
    """
    urls: list[str] = []
    for r in tool_results:
        name = r.get("tool") or ""
        if "send_story" not in name:
            continue
        args = r.get("args") or {}
        url = args.get("url") if isinstance(args, dict) else None
        if not url:
            continue
        result = str(r.get("result", ""))
        if result.startswith("Error") or "failed" in result.lower():
            continue
        urls.append(url)

    if not urls:
        return "no URLs to record"

    _append_urls(urls)
    return _git_commit_state(f"chore(ai_news): record {len(urls)} sent stories")
