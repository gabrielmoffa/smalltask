"""Prompt-based tool calling: build system prompt, parse tool calls from text."""

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def build_tool_system_prompt(tools: dict) -> str:
    """
    Generate the tool-calling section of the system prompt.

    tools: {name: {"definition": {...}, "fn": fn}}
    """
    lines = ["## Available Tools\n"]

    for name, tool in tools.items():
        defn = tool["definition"]
        lines.append(f"### {name}")
        lines.append(defn.get("description", name))

        schema = defn.get("input_schema", {})
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        if props:
            lines.append("Parameters:")
            for param, info in props.items():
                req = "required" if param in required else "optional"
                desc = info.get("description", "")
                type_ = info.get("type", "string")
                suffix = f" — {desc}" if desc else ""
                lines.append(f"  - {param} ({type_}, {req}){suffix}")

        lines.append("")

    lines += [
        "## Tool Calling Format",
        "",
        "When you need to call a tool, output EXACTLY this — nothing else on that turn:",
        "",
        '<tool_call>{"name": "tool_name", "args": {"param": "value"}}</tool_call>',
        "",
        "IMPORTANT RULES:",
        "- STOP your response immediately after the closing </tool_call> tag(s).",
        "- NEVER generate <tool_result> tags yourself — the system provides those.",
        "- NEVER simulate, predict, or hallucinate tool results.",
        "- Wait for the real tool result before continuing.",
        "",
        "The system will execute your tool call and return the result as:",
        "",
        '<tool_result name="tool_name">result here</tool_result>',
        "",
        "You may call multiple tools in one turn by outputting multiple <tool_call> tags.",
        "When you have enough information, respond normally without any <tool_call> tags.",
    ]

    return "\n".join(lines)


def parse_tool_calls(text: str) -> list[dict]:
    """
    Extract all tool calls from model output.

    Returns a list of {"name": str, "args": dict}. Empty list if none found.

    If the model hallucinates ``<tool_result>`` tags (self-simulation), we
    truncate the text at the first occurrence so that only tool calls issued
    *before* the hallucinated result are executed.
    """
    # Guard against self-simulation: ignore everything after a hallucinated
    # <tool_result> tag, since any subsequent <tool_call> is based on a
    # result the model fabricated, not a real execution.
    result_tag_pos = text.find("<tool_result")
    if result_tag_pos != -1:
        text = text[:result_tag_pos]

    calls = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(raw.strip())
            if "name" not in payload:
                continue
            payload.setdefault("args", {})
            calls.append(payload)
        except json.JSONDecodeError:
            continue
    return calls


def format_tool_result(name: str, result: str) -> str:
    return f'<tool_result name="{name}">\n{result}\n</tool_result>'


# ---------------------------------------------------------------------------
# Native (OpenAI-compatible) tool calling
# ---------------------------------------------------------------------------

def tools_to_openai_format(tools: dict) -> list[dict]:
    """Convert smalltask tool definitions to OpenAI function-calling format.

    This format is supported by OpenRouter, OpenAI, Groq, Together, Ollama,
    and most OpenAI-compatible providers.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["definition"]["name"],
                "description": tool["definition"].get("description", name),
                "parameters": tool["definition"].get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            },
        }
        for name, tool in tools.items()
    ]


def parse_native_tool_calls(message: dict) -> list[dict]:
    """Extract tool calls from an OpenAI-format assistant message.

    Returns list of {"name": str, "args": dict, "id": str}.
    Empty list if no tool_calls in the message.
    """
    calls = []
    for tc in message.get("tool_calls", []):
        if tc.get("type", "function") != "function":
            continue
        fn = tc["function"]
        try:
            args = (
                json.loads(fn["arguments"])
                if isinstance(fn["arguments"], str)
                else fn["arguments"]
            )
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append({"name": fn["name"], "args": args, "id": tc.get("id", "")})
    return calls
