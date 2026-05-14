"""Tests for ai_news news_feed tool."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ai_news" / "tools"))

import news_feed  # noqa: E402


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Google News</title>
  <item>
    <title>OpenAI launches new model - TechCrunch</title>
    <link>https://news.google.com/rss/articles/CBMabc?oc=5</link>
    <pubDate>Wed, 14 May 2026 10:00:00 GMT</pubDate>
    <source url="https://techcrunch.com">TechCrunch</source>
  </item>
  <item>
    <title>Anthropic releases Claude update - The Verge</title>
    <link>https://news.google.com/rss/articles/CBMxyz?oc=5</link>
    <pubDate>Wed, 14 May 2026 09:00:00 GMT</pubDate>
    <source url="https://www.theverge.com">The Verge</source>
  </item>
</channel></rss>"""


def test_parse_rss_returns_items():
    items = news_feed._parse_rss(SAMPLE_RSS)
    assert len(items) == 2
    assert items[0]["title"] == "OpenAI launches new model - TechCrunch"
    assert items[0]["url"] == "https://news.google.com/rss/articles/CBMabc?oc=5"
    assert items[0]["source"] == "TechCrunch"
    assert items[0]["published"] == "2026-05-14"


def test_fetch_headlines_dedupes_across_queries():
    fake_items = [
        {"title": "A", "url": "u1", "source": "s", "published": "2026-05-14"},
        {"title": "B", "url": "u2", "source": "s", "published": "2026-05-14"},
    ]
    fake_items_dup = [
        {"title": "A", "url": "u1", "source": "s", "published": "2026-05-14"},
        {"title": "C", "url": "u3", "source": "s", "published": "2026-05-14"},
    ]
    call_outputs = iter([fake_items, fake_items_dup, []])
    with patch.object(news_feed, "_fetch_one", lambda q: next(call_outputs)):
        result = news_feed.fetch_headlines()
    urls = [r["url"] for r in result]
    assert urls == ["u1", "u2", "u3"]
