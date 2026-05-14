# AI News Digest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Daily GitHub-Actions-driven smalltask that fetches Google News RSS for AI keywords, picks 3 unseen stories, and posts each to Telegram with a summary + social-media-angled questions.

**Architecture:** New example under `examples/ai_news/` mirroring the layout of `examples/daily_improvements/`. Four small Python tool modules (RSS, state, article fetch, Telegram) + one agent YAML + one workflow YAML. State (`sent_news.json`) is committed to the repo and pruned to a 7-day rolling window. No new Python dependencies — stdlib only.

**Tech Stack:** Python 3.11 stdlib (`urllib`, `xml.etree.ElementTree`, `html.parser`, `json`, `datetime`, `subprocess`), pytest, smalltask framework, GitHub Actions, Telegram Bot API, OpenRouter (Claude Sonnet 4.6).

**Spec:** `docs/superpowers/specs/2026-05-14-ai-news-digest-design.md`

---

## File Structure

**Create:**
- `examples/ai_news/agents/ai_news.yaml` — agent definition
- `examples/ai_news/tools/__init__.py` — empty (the dir is auto-discovered by smalltask's tool loader; an empty file makes intent clear)
- `examples/ai_news/tools/news_feed.py` — `fetch_headlines()` over Google News RSS
- `examples/ai_news/tools/news_state.py` — `load_recent_urls()`, `append_sent()` with git commit/push
- `examples/ai_news/tools/article.py` — `fetch_text(url)` HTML → text
- `examples/ai_news/tools/telegram_news.py` — `send_story(...)` Telegram message per story
- `examples/ai_news/sent_news.json` — initial empty state file `{"sent": []}`
- `.github/workflows/ai_news.yml` — daily cron workflow
- `tests/test_ai_news_feed.py` — RSS parsing tests
- `tests/test_ai_news_state.py` — state load/append/prune tests
- `tests/test_ai_news_article.py` — HTML → text tests
- `tests/test_ai_news_telegram.py` — Telegram message-building tests

**Modify:** none.

---

## Task 1: news_feed.fetch_headlines — RSS parsing core

**Files:**
- Create: `examples/ai_news/tools/news_feed.py`
- Test: `tests/test_ai_news_feed.py`

- [ ] **Step 1: Write the failing test for RSS parsing**

Create `tests/test_ai_news_feed.py`:

```python
"""Tests for ai_news news_feed tool."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Make the example tools importable
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_news_feed.py::test_parse_rss_returns_items -v`
Expected: FAIL — module `news_feed` not found.

- [ ] **Step 3: Implement `_parse_rss` and the public `fetch_headlines`**

Create `examples/ai_news/tools/news_feed.py`:

```python
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
        source = (source_el.text or "").strip() if source_el is not None else ""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ai_news_feed.py -v`
Expected: PASS

- [ ] **Step 5: Add a dedup test**

Append to `tests/test_ai_news_feed.py`:

```python
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
        result = news_feed.fetch_headlines.fn()
    urls = [r["url"] for r in result]
    assert urls == ["u1", "u2", "u3"]
```

Note: `fetch_headlines.fn` is the underlying function — smalltask's `@tool` wraps callables but exposes the raw function as `.fn` (see existing tests that import tools). If `.fn` is not the right attribute, call `news_feed.fetch_headlines.__wrapped__` or use `news_feed.fetch_headlines.func` — verify by inspecting `smalltask/loader.py` once at the start; adjust the test accordingly.

- [ ] **Step 6: Verify @tool attribute name then run the dedup test**

Run: `python -c "from smalltask import tool; @tool\ndef f(): return 1\nprint(dir(f))"` and locate the underlying-function attribute. Update the test in Step 5 to use the correct attribute, then:

Run: `pytest tests/test_ai_news_feed.py -v`
Expected: PASS (both tests)

- [ ] **Step 7: Commit**

```bash
git add examples/ai_news/tools/news_feed.py tests/test_ai_news_feed.py
git commit -m "feat(ai_news): add Google News RSS fetcher"
```

---

## Task 2: news_state.load_recent_urls + append_sent

**Files:**
- Create: `examples/ai_news/tools/news_state.py`
- Create: `examples/ai_news/sent_news.json`
- Test: `tests/test_ai_news_state.py`

- [ ] **Step 1: Write failing tests for load and prune**

Create `tests/test_ai_news_state.py`:

```python
"""Tests for ai_news news_state tool."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ai_news" / "tools"))

import news_state  # noqa: E402


def _today() -> str:
    return date.today().isoformat()


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
    urls = news_state.load_recent_urls.fn()  # adjust attr if needed (see Task 1 Step 6)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ai_news_state.py -v`
Expected: FAIL — module `news_state` not found.

- [ ] **Step 3: Implement news_state.py**

Create `examples/ai_news/tools/news_state.py`:

```python
"""Track which AI-news URLs have been sent. Pruned to a 7-day window."""

import json
import os
import re
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
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], capture_output=False
        )
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
    """Post-hook: extract URLs from successful send_story tool calls
    and append them to sent_news.json, then commit + push.

    Auto-receives output and tool_results from the post-hook framework.
    """
    urls: list[str] = []
    for r in tool_results:
        name = r.get("tool") or r.get("name") or ""
        if "send_story" not in name:
            continue
        args = r.get("args") or r.get("input") or {}
        url = args.get("url") if isinstance(args, dict) else None
        if not url:
            continue
        result = str(r.get("result", ""))
        # Only record URLs whose send_story call did not return an error
        if result.startswith("Error") or "failed" in result.lower():
            continue
        urls.append(url)

    if not urls:
        return "no URLs to record"

    _append_urls(urls)
    return _git_commit_state(f"chore(ai_news): record {len(urls)} sent stories")
```

- [ ] **Step 4: Create initial state file**

Create `examples/ai_news/sent_news.json`:

```json
{
  "sent": []
}
```

- [ ] **Step 5: Run state tests**

Run: `pytest tests/test_ai_news_state.py -v`
Expected: PASS

- [ ] **Step 6: Verify the tool_results shape matches the runner**

Open `smalltask/runner.py` and search for where `tool_results` is built and passed to post-hooks. Confirm the field names used in `append_sent` (`tool` / `name`, `args` / `input`, `result`). If field names differ, adjust `append_sent` to read whatever the runner actually emits. Re-run the tests if `append_sent` was changed.

- [ ] **Step 7: Commit**

```bash
git add examples/ai_news/tools/news_state.py examples/ai_news/sent_news.json tests/test_ai_news_state.py
git commit -m "feat(ai_news): add sent-URL state with 7-day prune and git commit"
```

---

## Task 3: article.fetch_text — HTML → plain text

**Files:**
- Create: `examples/ai_news/tools/article.py`
- Test: `tests/test_ai_news_article.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ai_news_article.py`:

```python
"""Tests for ai_news article fetcher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ai_news" / "tools"))

import article  # noqa: E402


SAMPLE_HTML = """
<html><head><title>x</title>
<script>var x = 1;</script>
<style>.a{color:red}</style>
</head>
<body>
<h1>Big news</h1>
<p>This is a   paragraph with <a href="#">a link</a>.</p>
<script>more();</script>
<p>Second &amp; final.</p>
</body></html>
"""


def test_html_to_text_strips_scripts_and_collapses_whitespace():
    text = article._html_to_text(SAMPLE_HTML)
    assert "var x" not in text
    assert ".a{color" not in text
    assert "Big news" in text
    assert "This is a paragraph with a link." in text
    assert "Second & final." in text


def test_html_to_text_truncates():
    big = "<p>" + ("word " * 10000) + "</p>"
    text = article._html_to_text(big, max_chars=500)
    assert len(text) <= 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_news_article.py -v`
Expected: FAIL — module `article` not found.

- [ ] **Step 3: Implement article.py**

Create `examples/ai_news/tools/article.py`:

```python
"""Fetch an article URL and return readable plain text."""

import html
import re
import urllib.request

from smalltask import tool

_USER_AGENT = "Mozilla/5.0 (compatible; smalltask-ai-news/1.0)"
_TIMEOUT = 20
_MAX_CHARS = 4000

_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(html_text: str, max_chars: int = _MAX_CHARS) -> str:
    text = _SCRIPT_RE.sub(" ", html_text)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


@tool
def fetch_text(url: str) -> str:
    """Fetch a URL and return a plain-text excerpt (~4000 chars max).

    Returns a short 'Error: ...' string on failure rather than raising,
    so the agent can decide whether to skip or retry.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
            # Best-effort decode
            encoding = resp.headers.get_content_charset() or "utf-8"
            body = raw.decode(encoding, errors="replace")
    except Exception as e:
        return f"Error: could not fetch {url}: {e}"
    return _html_to_text(body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ai_news_article.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/ai_news/tools/article.py tests/test_ai_news_article.py
git commit -m "feat(ai_news): add article text fetcher"
```

---

## Task 4: telegram_news.send_story — one message per story

**Files:**
- Create: `examples/ai_news/tools/telegram_news.py`
- Test: `tests/test_ai_news_telegram.py`

- [ ] **Step 1: Write failing tests for message formatting**

Create `tests/test_ai_news_telegram.py`:

```python
"""Tests for ai_news telegram_news tool."""

import sys
from pathlib import Path
from unittest.mock import patch

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ai_news_telegram.py -v`
Expected: FAIL — module `telegram_news` not found.

- [ ] **Step 3: Implement telegram_news.py**

Create `examples/ai_news/tools/telegram_news.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ai_news_telegram.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/ai_news/tools/telegram_news.py tests/test_ai_news_telegram.py
git commit -m "feat(ai_news): add per-story Telegram sender"
```

---

## Task 5: Wire up the agent YAML

**Files:**
- Create: `examples/ai_news/agents/ai_news.yaml`
- Create: `examples/ai_news/tools/__init__.py` (empty file)

- [ ] **Step 1: Create empty `__init__.py`**

Create `examples/ai_news/tools/__init__.py` with content: (empty file — zero bytes).

- [ ] **Step 2: Create the agent YAML**

Create `examples/ai_news/agents/ai_news.yaml`:

```yaml
name: ai_news
description: >
  Daily AI news digest. Pulls Google News RSS for AI keywords,
  picks 3 unseen stories, sends each to Telegram with a summary
  and social-media-angled questions for the user.

llm:
  connection: openrouter
  model: anthropic/claude-sonnet-4.6
  max_tokens: 4096

prompt: |
  You are an AI-news curator and content-prompt writer for a single user
  who creates social media content about AI. You run once per day.

  Steps:

  1. Call fetch_headlines() to get the latest AI-related headlines from
     Google News (queries: AI, LLM, artificial intelligence). You'll get
     up to ~30 items with {title, url, source, published}.

  2. Call load_recent_urls() to get URLs sent in the last 7 days.
     Filter the headline list to drop any URL already in that set.

  3. From the remaining headlines, pick THREE that are most interesting
     for an AI practitioner who creates social-media content. Prefer
     VARIETY — don't pick three stories about the same product launch.
     Prefer concrete news (releases, research, incidents, policy) over
     opinion pieces.

  4. For each of the 3 picks, call fetch_text(url) to get the article
     body. If a fetch returns an "Error:" string, replace that pick with
     the next-best unseen headline and try again. Do this at most once
     per pick — if a replacement also fails, proceed using just the
     title.

  5. For each of the 3 stories, draft:
     - A 2-3 sentence summary capturing what's actually new / notable.
     - 2-3 questions ANGLED FOR SOCIAL-MEDIA CONTENT. Not generic "what
       do you think?". Good examples:
       * "Hot take angle: this kills RAG for most teams — defend or
         attack that framing?"
       * "Would you ship something on top of this today, or wait?"
       * "Contrarian read: the real story isn't the model, it's the
         eval methodology. Buyable?"
     Tailor questions to the actual story — they should give the user
     raw material for a post.

  6. For each story, call send_story(title=..., url=..., source=...,
     summary=..., questions=[...]). One call per story. Three calls
     total.

  7. If load_recent_urls filtered out EVERYTHING, call send_story once
     with title="No fresh AI news today", url="", source="",
     summary="The feed had nothing new since the last digest.",
     questions=[] — then stop.

  Final message: one line listing the 3 URLs you sent (or "no fresh news").

  Hard rules:
  - Never invent URLs, titles, or facts. Only use values returned by tools.
  - Exactly three send_story calls per run (or one for the empty-day case).
  - If a tool call fails, retry at most once.
  - Do not call any tool not listed below.

tools:
  - news_feed.fetch_headlines
  - news_state.load_recent_urls
  - article.fetch_text
  - telegram_news.send_story

post_hook:
  - news_state.append_sent
```

- [ ] **Step 3: Lint the agent YAML by loading it**

Run:
```bash
python -c "from smalltask.loader import load_agent_config; print(load_agent_config('examples/ai_news/agents/ai_news.yaml')['name'])"
```
Expected: prints `ai_news`. If the loader raises, fix the YAML.

- [ ] **Step 4: Verify tool resolution**

Run:
```bash
smalltask run examples/ai_news/agents/ai_news.yaml --dry-run --verbose 2>&1 | head -40 || true
```

If `--dry-run` is not supported, instead inspect that the tool loader can find the tool modules:

```bash
python -c "
from smalltask.loader import load_agent_config, load_tools_for_agent
cfg = load_agent_config('examples/ai_news/agents/ai_news.yaml')
print('tools resolved:', list(load_tools_for_agent(cfg).keys()))
" 2>&1 | head
```

Adjust to whatever helper smalltask exposes — read `smalltask/loader.py` for the right entry point. The goal is to confirm all four tools (`news_feed.fetch_headlines`, `news_state.load_recent_urls`, `article.fetch_text`, `telegram_news.send_story`) plus the post-hook (`news_state.append_sent`) resolve. If anything is missing, the tool discovery is probably looking in a different default directory — check `smalltask.yaml` for a `tools_dir` setting or look at how `daily_improvements` is invoked (the workflow passes only the agent path; smalltask must auto-discover sibling tool dirs).

- [ ] **Step 5: Commit**

```bash
git add examples/ai_news/agents/ai_news.yaml examples/ai_news/tools/__init__.py
git commit -m "feat(ai_news): add agent definition"
```

---

## Task 6: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ai_news.yml`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ai_news.yml`:

```yaml
name: AI news digest

on:
  schedule:
    - cron: '0 8 * * *'   # every day at 08:00 UTC
  workflow_dispatch:

jobs:
  ai-news:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install .

      - name: Configure git identity
        run: |
          git config user.name "smalltask-bot"
          git config user.email "smalltask-bot@users.noreply.github.com"

      - name: Run ai_news agent
        run: smalltask run examples/ai_news/agents/ai_news.yaml --verbose
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
```

- [ ] **Step 2: Validate YAML parses**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ai_news.yml'))"
```
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ai_news.yml
git commit -m "ci(ai_news): add daily workflow"
```

---

## Task 7: End-to-end smoke test (manual)

**Files:** none

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (existing + the four new files).

- [ ] **Step 2: Local dry run with real RSS, mocked Telegram**

Temporarily unset Telegram env vars so `send_story` short-circuits with an "Error: Telegram not configured" string (the agent will still complete the flow; only state will not be updated because the post-hook treats `Error:` as failure).

Run:
```bash
unset TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
OPENROUTER_API_KEY=$OPENROUTER_API_KEY smalltask run examples/ai_news/agents/ai_news.yaml --verbose
```
Expected: agent fetches headlines, picks 3, fetches their text, attempts to send (gets "Error: Telegram not configured"), exits cleanly. `sent_news.json` should NOT be modified.

If the agent loops or fails to pick stories, refine the prompt in `examples/ai_news/agents/ai_news.yaml` and re-test.

- [ ] **Step 3: Manual workflow_dispatch on a feature branch (optional pre-merge)**

Push a feature branch, then in the GitHub UI run the workflow via "Run workflow" against that branch. Verify:
- Telegram receives 3 messages.
- `sent_news.json` gets a commit with 3 new URLs.
- Re-running immediately should produce a "no fresh news" message (or pick different stories, depending on feed velocity).

- [ ] **Step 4: Final commit (if prompt was tuned)**

```bash
git add examples/ai_news/agents/ai_news.yaml
git commit -m "tune(ai_news): refine agent prompt after smoke test"
```

(Skip if no changes.)

- [ ] **Step 5: Open PR**

```bash
git push -u origin <branch>
gh pr create --title "Add ai_news daily digest smalltask" --body "$(cat <<'EOF'
## Summary
- New `examples/ai_news/` smalltask that posts 3 fresh AI-news stories per day to Telegram with social-media-angled questions.
- Google News RSS for headlines; 7-day rolling dedup committed to repo as `sent_news.json`.
- Daily GitHub Actions cron at 08:00 UTC.

## Test plan
- [x] Unit tests for RSS parsing, state prune, HTML→text, Telegram message format
- [ ] Manual `workflow_dispatch` run on this branch; verify 3 Telegram messages and state commit
- [ ] Re-run on same day; verify "no fresh news" path
EOF
)"
```

---

## Self-review notes (resolved)

- **Spec coverage:** Each spec section maps to a task: news_feed → T1, news_state → T2, article → T3, telegram_news → T4, agent + post_hook → T5, workflow → T6, error-handling matrix → exercised in T7.
- **Tool attribute (`fn`) uncertainty:** Task 1 Step 6 explicitly asks the engineer to verify the correct attribute name on smalltask's `@tool` decorator before relying on it in tests. This is intentional — not a placeholder — because the test-import pattern needs to match the framework.
- **tool_results shape:** Task 2 Step 6 asks the engineer to confirm field names against `smalltask/runner.py` before considering `append_sent` final.
- **No placeholders, TBDs, or "implement later" steps.** Each code step contains complete code.
