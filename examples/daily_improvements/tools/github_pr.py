"""Tools for managing GitHub pull requests."""

import json
import os
import subprocess
from datetime import datetime, timezone

from smalltask import tool


def _github_api(method: str, path: str, data: dict | list | None = None) -> dict | list:
    """Call the GitHub REST API."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    import urllib.request
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


_BRANCH_PREFIX = "smalltask/improvement-"


@tool
def check_pending_pr() -> dict:
    """Check for open improvement PRs from the bot.

    Returns skip=True if a PR is still pending review.
    If the latest PR was rejected (has a /reject comment), closes it
    and allows a new one to be created.
    """
    try:
        prs = _github_api("GET", "/pulls?state=open")
    except Exception:
        return {"status": "ready", "message": "Could not fetch PRs, proceeding anyway"}

    # Filter to PRs created by the bot (branch prefix match)
    bot_prs = [p for p in prs if p.get("head", {}).get("ref", "").startswith(_BRANCH_PREFIX)]

    if not bot_prs:
        return {"status": "ready", "message": "No pending PR — good to go"}

    for pr in bot_prs:
        number = pr["number"]
        try:
            comments = _github_api("GET", f"/issues/{number}/comments")
        except Exception:
            comments = []

        rejected = any("/reject" in c.get("body", "") for c in comments)

        if rejected:
            try:
                _github_api("PATCH", f"/pulls/{number}", {"state": "closed"})
            except Exception:
                pass
            return {
                "status": "rejected_closed",
                "message": f"Closed rejected PR #{number}, creating a new one",
            }
        else:
            return {
                "skip": True,
                "reason": f"PR #{number} is still pending review — skipping",
            }

    return {"status": "ready"}


@tool
def create_pr(title: str, description: str) -> str:
    """Create a branch, commit all staged changes, push, and open a pull request.

    Args:
        title: Short PR title describing the improvement.
        description: Markdown body explaining what changed and why.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"{_BRANCH_PREFIX}{ts}"

    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", title], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)

    pr = _github_api("POST", "/pulls", {
        "title": title,
        "body": description + "\n\n---\n_Created by [smalltask](https://github.com/gabrielmoffa/smalltask) bot_",
        "head": branch,
        "base": "main",
    })

    return pr["html_url"]
