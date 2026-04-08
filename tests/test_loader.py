"""Tests for tool loading and schema generation."""

import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from smalltask.loader import _build_schema, load_agent_config, load_tool, load_tools_from_file


# ---------------------------------------------------------------------------
# _build_schema — type mapping
# ---------------------------------------------------------------------------

def test_basic_types():
    def fn(s: str, i: int, f: float, b: bool) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["s"]["type"] == "string"
    assert schema["properties"]["i"]["type"] == "integer"
    assert schema["properties"]["f"]["type"] == "number"
    assert schema["properties"]["b"]["type"] == "boolean"


def test_plain_list_and_dict():
    def fn(items: list, meta: dict) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["meta"]["type"] == "object"


def test_generic_list():
    def fn(tags: List[str]) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["tags"]["type"] == "array"


def test_generic_dict():
    def fn(data: Dict[str, int]) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["data"]["type"] == "object"


def test_optional_unwraps_to_inner_type():
    def fn(name: Optional[str] = None) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["name"]["type"] == "string"


def test_unknown_type_falls_back_to_string():
    class Custom:
        pass

    def fn(x: Custom) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["properties"]["x"]["type"] == "string"


# ---------------------------------------------------------------------------
# _build_schema — required vs optional
# ---------------------------------------------------------------------------

def test_required_and_optional_params():
    def fn(required: str, optional: int = 0) -> None:
        pass

    schema = _build_schema(fn)
    assert "required" in schema["required"]
    assert "optional" not in schema["required"]


def test_no_required_params():
    def fn(a: str = "x", b: int = 1) -> None:
        pass

    schema = _build_schema(fn)
    assert schema["required"] == []


# ---------------------------------------------------------------------------
# _build_schema — docstring parsing
# ---------------------------------------------------------------------------

def test_docstring_description_extracted():
    def fn(query: str) -> None:
        """Search records.

        query: The search string to match against.
        """

    schema = _build_schema(fn)
    assert schema["properties"]["query"]["description"] == "The search string to match against."


def test_missing_docstring_param_has_no_description():
    def fn(x: str) -> None:
        """Do something."""

    schema = _build_schema(fn)
    assert "description" not in schema["properties"]["x"]


# ---------------------------------------------------------------------------
# load_tool
# ---------------------------------------------------------------------------

def test_load_tool_name_and_description():
    def my_func(x: str) -> str:
        """Short description.

        Longer details.
        """

    tool = load_tool(my_func)
    assert tool["definition"]["name"] == "my_func"
    assert tool["definition"]["description"] == "Short description."


def test_load_tool_no_docstring_uses_name():
    def unnamed() -> None:
        pass

    tool = load_tool(unnamed)
    assert tool["definition"]["description"] == "unnamed"


def test_load_tool_has_fn_callable():
    def fn() -> str:
        return "hi"

    tool = load_tool(fn)
    assert callable(tool["fn"])
    assert tool["fn"]() == "hi"


# ---------------------------------------------------------------------------
# load_tools_from_file
# ---------------------------------------------------------------------------

def test_load_tools_from_file_only_decorated(tmp_path):
    code = textwrap.dedent("""\
        import smalltask

        @smalltask.tool
        def exposed(x: str) -> str:
            \"\"\"Exposed tool.\"\"\"
            return x

        def hidden(x: str) -> str:
            \"\"\"Not decorated, should be excluded.\"\"\"
            return x
    """)
    f = tmp_path / "tools.py"
    f.write_text(code)
    tools = load_tools_from_file(f)
    assert "exposed" in tools
    assert "hidden" not in tools


def test_load_tools_from_file_fallback_all_public(tmp_path):
    code = textwrap.dedent("""\
        def public_fn(x: str) -> str:
            \"\"\"A public function.\"\"\"
            return x

        def _private_fn() -> None:
            pass
    """)
    f = tmp_path / "tools.py"
    f.write_text(code)
    tools = load_tools_from_file(f)
    assert "public_fn" in tools
    assert "_private_fn" not in tools


# ---------------------------------------------------------------------------
# load_agent_config
# ---------------------------------------------------------------------------

def test_load_agent_config_valid(tmp_path):
    f = tmp_path / "agent.yaml"
    f.write_text("name: test\nprompt: hello\ntools:\n  - foo\n")
    config = load_agent_config(f)
    assert config["name"] == "test"
    assert config["max_tokens"] == 4096
    assert config["max_iterations"] == 20
    assert config["max_total_tokens"] is None


def test_load_agent_config_missing_keys(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("name: test\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_agent_config(f)


def test_load_agent_config_custom_max_iterations(tmp_path):
    f = tmp_path / "agent.yaml"
    f.write_text("name: t\nprompt: p\ntools: []\nmax_iterations: 5\n")
    config = load_agent_config(f)
    assert config["max_iterations"] == 5


def test_load_agent_config_custom_token_budget(tmp_path):
    f = tmp_path / "agent.yaml"
    f.write_text("name: t\nprompt: p\ntools: []\nmax_total_tokens: 10000\n")
    config = load_agent_config(f)
    assert config["max_total_tokens"] == 10000
