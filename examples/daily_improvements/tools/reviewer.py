"""Post-hook tool that delegates PR review to an independent reviewer agent."""

import re
from pathlib import Path

from smalltask import tool
from smalltask.runner import run_agent


@tool
def review_pr(tool_results: list) -> str:
    """Review the PR that was just created by delegating to an independent reviewer agent.

    Extracts the PR number from the create_pr tool result, then spawns
    the pr_reviewer agent which reads the diff, evaluates the change,
    and comments on the PR.

    Args:
        tool_results: Automatically injected list of tool calls from the agent run.
    """
    # Find the PR URL from the create_pr tool result
    pr_url = None
    for entry in tool_results:
        if entry.get("tool") == "github_pr.create_pr":
            pr_url = entry.get("result", "")
            break

    if not pr_url:
        return "No PR was created — skipping review"

    # Extract PR number from URL like https://github.com/owner/repo/pull/123
    match = re.search(r"/pull/(\d+)", pr_url)
    if not match:
        return f"Could not parse PR number from: {pr_url}"

    pr_number = match.group(1)

    agent_path = Path(__file__).resolve().parent.parent / "agents" / "pr_reviewer.yaml"

    result = run_agent(
        agent_path=agent_path,
        input_vars={"pr_number": pr_number},
        verbose=True,
    )

    return result
