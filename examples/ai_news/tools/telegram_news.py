"""Send one Telegram message per AI-news story."""

import json
import os
import urllib.request

from smalltask import tool


def _build_message(title: str, url: str, source: str,
                   summary: str, questions: list[str]) -> str:
    parts = []
    header = f"📰 *{title}*"
    if source:
        header += f" — {source}"
    parts.append(header)
    if url:
        parts.append(url)
    parts.append("")
    parts.append(summary)
    if questions:
        parts.append("")
        parts.append("💭 *Questions for you:*")
        for q in questions:
            parts.append(f"• {q}")
    return "\n".join(parts)


@tool
def send_story(title: str, url: str, source: str,
               summary: str, questions: list[str]) -> str:
    """Send one Telegram message describing an AI-news story.

    Args:
        title: Headline.
        url: Article URL (may be empty for the 'no fresh news' fallback).
        source: Publisher name (e.g., 'TechCrunch'). Empty string OK.
        summary: 2-3 sentence summary.
        questions: 2-3 social-media-angled questions.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return "Error: Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)"

    message = _build_message(title, url, source, summary, questions)
    data = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }).encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:
        return f"Error: Telegram send failed: {e}"
    return f"sent: {url or '(no url)'}"
