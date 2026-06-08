"""Tools for managing GitHub pull requests."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from smalltask import tool
from smalltask.runner import run_agent


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


def _human_comments(number: int) -> list[dict]:
    """Return non-bot comments on a PR as [{"body", "created_at"}, ...]."""
    comments = _github_api("GET", f"/issues/{number}/comments")
    out = []
    for c in comments:
        user = c.get("user") or {}
        if user.get("type") == "Bot":
            continue
        out.append({"body": c.get("body", ""), "created_at": c.get("created_at", "")})
    return out


def _last_commit_date(branch: str) -> str:
    """Return the ISO timestamp of the latest commit on a branch."""
    commit = _github_api("GET", f"/commits/{branch}")
    return commit["commit"]["committer"]["date"]


def _classify_pr_action(human_comments: list[dict], last_commit: str) -> tuple[str, str | None]:
    """Decide what to do with an open bot PR based on human comments.

    Returns one of:
      ("reject", None)         — a human comment contains '/reject'
      ("address", feedback)    — there are human comments newer than the last
                                 commit; feedback is their combined text
      ("pending", None)        — nothing new to act on
    """
    for c in human_comments:
        if "/reject" in c["body"]:
            return ("reject", None)

    unaddressed = [c for c in human_comments if c["created_at"] > last_commit]
    if unaddressed:
        return ("address", "\n\n".join(c["body"] for c in unaddressed))

    return ("pending", None)


def _checkout_branch(branch: str) -> None:
    subprocess.run(["git", "fetch", "origin", branch], check=True)
    subprocess.run(["git", "checkout", branch], check=True)


@tool
def check_pending_pr() -> dict:
    """Gate + feedback router for the daily improvement agent.

    When an open bot PR exists, the PR comment thread is the control channel:
    - a human comment containing '/reject' -> close it, allow a fresh PR
    - any other human comment newer than the last commit -> address it on the
      same branch (commit + push), then skip (no new PR, no second review)
    - no new human comment -> skip (still pending your review)
    No open bot PR -> ready to create a new improvement PR.
    """
    try:
        prs = _github_api("GET", "/pulls?state=open")
    except Exception:
        return {"status": "ready", "message": "Could not fetch PRs, proceeding anyway"}

    # Filter to PRs created by the bot (branch prefix match)
    bot_prs = [p for p in prs if p.get("head", {}).get("ref", "").startswith(_BRANCH_PREFIX)]

    if not bot_prs:
        return {"status": "ready", "message": "No pending PR — good to go"}

    pr = max(bot_prs, key=lambda p: p["number"])
    number = pr["number"]
    branch = pr["head"]["ref"]

    try:
        human_comments = _human_comments(number)
        last_commit = _last_commit_date(branch)
    except Exception as e:
        return {"skip": True, "reason": f"Could not read PR #{number} state: {e}"}

    action, feedback = _classify_pr_action(human_comments, last_commit)

    if action == "reject":
        try:
            _github_api("PATCH", f"/pulls/{number}", {"state": "closed"})
        except Exception:
            pass
        return {
            "status": "rejected_closed",
            "message": f"Closed rejected PR #{number}, creating a new one",
        }

    if action == "address":
        # Check out the PR branch first so the agent's edits land on it.
        _checkout_branch(branch)
        agent_path = Path(__file__).resolve().parent.parent / "agents" / "address_feedback.yaml"
        result = run_agent(
            agent_path=agent_path,
            input_vars={"pr_number": str(number), "feedback": feedback},
            verbose=True,
        )
        return {"skip": True, "reason": f"Addressed feedback on PR #{number}: {result}"}

    return {"skip": True, "reason": f"PR #{number} is still pending review — skipping"}


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


@tool
def get_pr_diff(pr_number: int) -> str:
    """Fetch the diff for a pull request.

    Args:
        pr_number: The PR number to fetch the diff for.
    """
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff",
    }

    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


@tool
def push_to_pr_branch(message: str) -> str:
    """Commit all current changes and push them to the current PR branch.

    The repo must already be checked out on the PR's branch. Use this to push
    a follow-up commit that addresses review feedback — it does NOT create a
    new branch or a new PR.

    Args:
        message: Commit message describing what changed.
    """
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)
    return "Pushed follow-up commit to PR branch"


@tool
def comment_pr(pr_number: int, body: str) -> str:
    """Post a comment on a pull request.

    Args:
        pr_number: The PR number to comment on.
        body: The markdown comment body.
    """
    _github_api("POST", f"/issues/{pr_number}/comments", {"body": body})
    return f"Commented on PR #{pr_number}"
