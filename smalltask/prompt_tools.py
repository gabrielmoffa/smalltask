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
        "The result will come back as:",
        "",
        '<tool_result name="tool_name">result here</tool_result>',
        "",
        "You may call tools multiple times. When you have enough information, "
        "respond normally without any <tool_call> tags.",
    ]

    return "\n".join(lines)


def parse_tool_calls(text: str) -> list[dict]:
    """
    Extract all tool calls from model output.

    Returns a list of {"name": str, "args": dict}. Empty list if none found.
    """
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
