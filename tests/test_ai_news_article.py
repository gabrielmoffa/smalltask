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
    assert "This is a paragraph with a link" in text
    assert "Second & final." in text


def test_html_to_text_truncates():
    big = "<p>" + ("word " * 10000) + "</p>"
    text = article._html_to_text(big, max_chars=500)
    assert len(text) <= 500
