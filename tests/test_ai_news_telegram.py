"""Tests for ai_news telegram_news tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ai_news" / "tools"))

import telegram_news  # noqa: E402


def test_build_message_includes_title_url_summary_and_questions():
    msg = telegram_news._build_message(
        title="OpenAI launches X",
        url="https://example.com/x",
        source="TechCrunch",
        summary="OpenAI announced X. It does Y.",
        questions=["Is X overhyped?", "Would you ship with X?"],
    )
    assert "OpenAI launches X" in msg
    assert "TechCrunch" in msg
    assert "https://example.com/x" in msg
    assert "OpenAI announced X" in msg
    assert "Is X overhyped?" in msg
    assert "Would you ship with X?" in msg


def test_build_message_handles_no_url_for_empty_day():
    msg = telegram_news._build_message(
        title="No fresh AI news today",
        url="",
        source="",
        summary="Nothing new in the feed.",
        questions=[],
    )
    assert "No fresh AI news today" in msg
    assert "https://" not in msg
