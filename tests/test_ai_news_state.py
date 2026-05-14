"""Tests for ai_news news_state tool."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ai_news" / "tools"))

import news_state  # noqa: E402


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def test_load_recent_urls_reads_within_window(tmp_path, monkeypatch):
    state_file = tmp_path / "sent_news.json"
    state_file.write_text(json.dumps({
        "sent": [
            {"url": "fresh", "sent_at": _days_ago(1)},
            {"url": "stale", "sent_at": _days_ago(10)},
        ]
    }))
    monkeypatch.setattr(news_state, "_STATE_PATH", str(state_file))
    urls = news_state.load_recent_urls()
    assert urls == ["fresh"]


def test_append_sent_prunes_old_entries(tmp_path, monkeypatch):
    state_file = tmp_path / "sent_news.json"
    state_file.write_text(json.dumps({
        "sent": [
            {"url": "stale", "sent_at": _days_ago(10)},
            {"url": "kept", "sent_at": _days_ago(2)},
        ]
    }))
    monkeypatch.setattr(news_state, "_STATE_PATH", str(state_file))
    monkeypatch.setattr(news_state, "_git_commit_state", lambda msg: "skipped")
    news_state._append_urls(["new1", "new2"])
    data = json.loads(state_file.read_text())
    urls = sorted(e["url"] for e in data["sent"])
    assert urls == ["kept", "new1", "new2"]


def test_append_sent_post_hook_extracts_urls(tmp_path, monkeypatch):
    state_file = tmp_path / "sent_news.json"
    state_file.write_text(json.dumps({"sent": []}))
    monkeypatch.setattr(news_state, "_STATE_PATH", str(state_file))
    monkeypatch.setattr(news_state, "_git_commit_state", lambda msg: f"committed: {msg}")

    tool_results = [
        {"tool": "news_feed.fetch_headlines", "args": {}, "result": "[...]"},
        {"tool": "telegram_news.send_story", "args": {"url": "https://a"}, "result": "sent: https://a"},
        {"tool": "telegram_news.send_story", "args": {"url": "https://b"}, "result": "Error: Telegram not configured"},
        {"tool": "telegram_news.send_story", "args": {"url": "https://c"}, "result": "sent: https://c"},
    ]
    out = news_state.append_sent(output="done", tool_results=tool_results)
    assert "committed" in out
    data = json.loads(state_file.read_text())
    urls = sorted(e["url"] for e in data["sent"])
    assert urls == ["https://a", "https://c"]
