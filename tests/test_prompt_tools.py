"""Tests for prompt-based tool calling utilities."""

import pytest

from smalltask.prompt_tools import (
    build_tool_system_prompt,
    format_tool_result,
    parse_tool_calls,
)


def _make_tool(name: str, description: str, params: dict | None = None) -> dict:
    return {
        "definition": {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": params or {},
                "required": list(params.keys()) if params else [],
            },
        },
        "fn": lambda **kwargs: None,
    }


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------

def test_parse_single_call():
    text = '<tool_call>{"name": "foo", "args": {"x": 1}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "foo", "args": {"x": 1}}]


def test_parse_multiple_calls():
    text = (
        '<tool_call>{"name": "a"}</tool_call>'
        " some text "
        '<tool_call>{"name": "b", "args": {"y": 2}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0] == {"name": "a", "args": {}}  # args defaulted
    assert calls[1] == {"name": "b", "args": {"y": 2}}


def test_parse_no_calls():
    assert parse_tool_calls("no tool calls here") == []
    assert parse_tool_calls("") == []


def test_parse_malformed_json_skipped():
    text = "<tool_call>not valid json</tool_call>"
    assert parse_tool_calls(text) == []


def test_parse_missing_name_skipped():
    text = '<tool_call>{"args": {"x": 1}}</tool_call>'
    assert parse_tool_calls(text) == []


def test_parse_args_defaults_to_empty_dict():
    text = '<tool_call>{"name": "noop"}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0]["args"] == {}


def test_parse_call_with_surrounding_text():
    text = "I'll call the tool now.\n<tool_call>{\"name\": \"greet\", \"args\": {}}</tool_call>\nDone."
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "greet"


# ---------------------------------------------------------------------------
# format_tool_result
# ---------------------------------------------------------------------------

def test_format_tool_result():
    result = format_tool_result("my_tool", "hello")
    assert result == '<tool_result name="my_tool">\nhello\n</tool_result>'


def test_format_tool_result_multiline():
    result = format_tool_result("t", "line1\nline2")
    assert "line1\nline2" in result


# ---------------------------------------------------------------------------
# build_tool_system_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_tool_name_and_description():
    tools = {"do_thing": _make_tool("do_thing", "Does a thing")}
    prompt = build_tool_system_prompt(tools)
    assert "do_thing" in prompt
    assert "Does a thing" in prompt


def test_build_prompt_lists_parameters():
    tools = {
        "search": _make_tool("search", "Search records", {"query": {"type": "string"}})
    }
    prompt = build_tool_system_prompt(tools)
    assert "query" in prompt
    assert "string" in prompt


def test_build_prompt_includes_calling_format():
    prompt = build_tool_system_prompt({})
    assert "<tool_call>" in prompt
    assert "<tool_result" in prompt


def test_build_prompt_multiple_tools():
    tools = {
        "a": _make_tool("a", "Tool A"),
        "b": _make_tool("b", "Tool B"),
    }
    prompt = build_tool_system_prompt(tools)
    assert "Tool A" in prompt
    assert "Tool B" in prompt
