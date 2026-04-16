"""
Raw HTTP LLM client — no SDK, no provider code.

Expects an OpenAI-compatible chat completions endpoint.
Works with OpenRouter, Ollama, Groq, Together, Bedrock (via their compat layer),
Gemini (via their compat layer), or anything else that speaks the format.

Agent YAML config:
    llm:
      url: https://openrouter.ai/api/v1/chat/completions
      model: anthropic/claude-opus-4-6
      api_key_env: OPENROUTER_API_KEY   # env var name, not the key itself
      max_tokens: 4096                  # optional
      extra_headers:                    # optional
        HTTP-Referer: https://yoursite.com
"""

import json
import os
from typing import Any

import httpx


def complete(
    messages: list[dict],
    llm_config: dict,
    tools: list[dict] | None = None,
) -> tuple[dict, dict]:
    """
    Send messages to the configured LLM endpoint and return (message, usage).

    message: the assistant message dict from the API response.  Always contains
        ``"role"`` and ``"content"`` (which may be ``None``).  When the model
        invokes tools it also contains ``"tool_calls"``.
    usage: dict with prompt_tokens, completion_tokens, total_tokens (0 if absent).

    messages: list of {"role": "user"|"assistant"|"system"|"tool", "content": "..."}
    llm_config: dict from agent YAML's `llm` key
    tools: optional list of tool definitions in OpenAI function-calling format
    """
    url = llm_config.get("url")
    if not url:
        raise ValueError("llm.url is required in agent config")

    model = llm_config.get("model")
    if not model:
        raise ValueError("llm.model is required in agent config")

    api_key_env = llm_config.get("api_key_env")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    extra_headers = llm_config.get("extra_headers", {})
    headers.update(extra_headers)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": llm_config.get("max_tokens", 4096),
    }

    if tools:
        payload["tools"] = tools

    timeout = llm_config.get("timeout", 120)

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM request failed ({response.status_code}): {response.text[:500]}"
        )

    data = response.json()

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM response shape: {e}\n{json.dumps(data)[:500]}")

    raw_usage = data.get("usage", {})
    usage = {
        "prompt_tokens": raw_usage.get("prompt_tokens", 0),
        "completion_tokens": raw_usage.get("completion_tokens", 0),
        "total_tokens": raw_usage.get("total_tokens", 0),
    }
    return message, usage
