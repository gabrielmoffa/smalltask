"""Tests for the raw HTTP LLM client."""

from unittest.mock import Mock, patch

from smalltask.llm import complete


def _response() -> Mock:
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {},
    }
    return resp


def test_complete_uses_max_tokens_by_default():
    with patch("smalltask.llm.httpx.post", return_value=_response()) as post:
        complete(
            [{"role": "user", "content": "hello"}],
            {"url": "https://example.test/chat", "model": "test-model"},
        )

    payload = post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload


def test_complete_uses_max_completion_tokens_when_configured():
    with patch("smalltask.llm.httpx.post", return_value=_response()) as post:
        complete(
            [{"role": "user", "content": "hello"}],
            {
                "url": "https://example.test/chat",
                "model": "test-model",
                "max_tokens": 100,
                "max_completion_tokens": 200,
            },
        )

    payload = post.call_args.kwargs["json"]
    assert payload["max_completion_tokens"] == 200
    assert "max_tokens" not in payload
