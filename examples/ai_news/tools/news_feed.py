"""Fetch headlines from Google News RSS for AI-related keywords."""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from smalltask import tool

_QUERIES = ["AI", "LLM", "artificial intelligence"]
_RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
_USER_AGENT = "Mozilla/5.0 (compatible; smalltask-ai-news/1.0)"
_TIMEOUT = 15


def _parse_pub_date(raw: str) -> str:
    """Convert RFC-822 'Wed, 14 May 2026 10:00:00 GMT' to 'YYYY-MM-DD'."""
    try:
        dt = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_rss(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None and source_el.text else ""
        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "source": source,
            "published": _parse_pub_date(pub),
        })
    return items


def _fetch_one(query: str) -> list[dict]:
    url = _RSS_URL.format(q=urllib.parse.quote_plus(query))
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return _parse_rss(body)


@tool
def fetch_headlines() -> list[dict]:
    """Fetch latest AI-related headlines from Google News RSS.

    Queries Google News for 'AI', 'LLM', and 'artificial intelligence',
    merges and dedupes by URL, returns up to ~30 items as
    [{title, url, source, published}].
    """
    seen = set()
    merged: list[dict] = []
    errors: list[str] = []
    for q in _QUERIES:
        try:
            for item in _fetch_one(q):
                if item["url"] in seen:
                    continue
                seen.add(item["url"])
                merged.append(item)
        except Exception as e:
            errors.append(f"{q}: {e}")
    if not merged and errors:
        raise RuntimeError("All RSS queries failed: " + "; ".join(errors))
    return merged[:30]
