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
            encoding = resp.headers.get_content_charset() or "utf-8"
            body = raw.decode(encoding, errors="replace")
    except Exception as e:
        return f"Error: could not fetch {url}: {e}"
    return _html_to_text(body)
