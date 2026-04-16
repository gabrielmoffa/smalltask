"""Tests for prompt-based tool calling utilities."""

import pytest

from smalltask.prompt_tools import (
    build_tool_system_prompt,
    format_tool_result,
    parse_native_tool_calls,
    parse_tool_calls,
    tools_to_openai_format,
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


# ---------------------------------------------------------------------------
# tools_to_openai_format
# ---------------------------------------------------------------------------

def test_tools_to_openai_format_structure():
    tools = {"search": _make_tool("search", "Search records", {"query": {"type": "string"}})}
    result = tools_to_openai_format(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search"
    assert result[0]["function"]["description"] == "Search records"
    assert result[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_tools_to_openai_format_multiple():
    tools = {
        "a": _make_tool("a", "Tool A"),
        "b": _make_tool("b", "Tool B"),
    }
    result = tools_to_openai_format(tools)
    assert len(result) == 2
    names = {r["function"]["name"] for r in result}
    assert names == {"a", "b"}


def test_tools_to_openai_format_empty():
    assert tools_to_openai_format({}) == []


# ---------------------------------------------------------------------------
# parse_native_tool_calls
# ---------------------------------------------------------------------------

def test_parse_native_single_call():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "hello"}'},
            }
        ],
    }
    calls = parse_native_tool_calls(message)
    assert len(calls) == 1
    assert calls[0] == {"name": "search", "args": {"query": "hello"}, "id": "call_abc"}


def test_parse_native_multiple_calls():
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": '{"x": 1}'}},
        ],
    }
    calls = parse_native_tool_calls(message)
    assert len(calls) == 2
    assert calls[0]["name"] == "a"
    assert calls[1]["args"] == {"x": 1}


def test_parse_native_no_tool_calls():
    message = {"role": "assistant", "content": "Just text."}
    assert parse_native_tool_calls(message) == []


def test_parse_native_dict_arguments():
    """Some providers return arguments as a dict instead of a JSON string."""
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "foo", "arguments": {"bar": 42}}},
        ],
    }
    calls = parse_native_tool_calls(message)
    assert calls[0]["args"] == {"bar": 42}


def test_parse_native_malformed_arguments():
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "foo", "arguments": "not json"}},
        ],
    }
    calls = parse_native_tool_calls(message)
    assert calls[0]["args"] == {}


def test_parse_native_missing_id():
    message = {
        "role": "assistant",
        "tool_calls": [
            {"type": "function", "function": {"name": "foo", "arguments": "{}"}},
        ],
    }
    calls = parse_native_tool_calls(message)
    assert calls[0]["id"] == ""
