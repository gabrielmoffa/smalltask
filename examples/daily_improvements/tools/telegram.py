"""Post-hook tool for sending Telegram notifications."""

import json
import os
import urllib.request

from smalltask import tool


@tool
def notify_telegram(output: str, tool_results: list) -> str:
    """Send a Telegram message with the new PR link.

    Auto-receives 'output' and 'tool_results' from the post-hook framework.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        return "Telegram not configured — skipping notification"

    # Extract PR URL from tool results
    pr_url = ""
    for r in tool_results:
        result = str(r.get("result", ""))
        if "github.com" in result and "/pull/" in result:
            pr_url = result
            break

    if pr_url:
        message = f"New improvement PR:\n{pr_url}"
    else:
        message = f"Daily improvement agent finished:\n{output[:500]}"

    data = json.dumps({
        "chat_id": chat_id,
        "text": message,
    }).encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)
    return "Telegram notification sent"
