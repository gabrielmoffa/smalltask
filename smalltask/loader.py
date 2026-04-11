"""Load tools from Python files and agent configs from YAML."""

import importlib.util
import inspect
import sys
import types
import typing
from pathlib import Path
from typing import Any, Callable, Union

import yaml


# Map Python types to JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    tuple: "array",
}

_DEFAULT_MAX_ITERATIONS = 20


def _resolve_json_type(python_type: Any) -> str:
    """Map a Python type annotation (including generics) to a JSON Schema type string."""
    if python_type in _TYPE_MAP:
        return _TYPE_MAP[python_type]

    origin = typing.get_origin(python_type)
    args = typing.get_args(python_type)

    if origin is Union:
        # Optional[X] is Union[X, None] — use the first non-None type
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _resolve_json_type(non_none[0])

    if origin is list:
        return "array"

    if origin is dict:
        return "object"

    if origin is tuple:
        return "array"

    return "string"


def _build_schema(fn: Callable) -> dict:
    """Generate a JSON Schema input_schema from a function's type hints and docstring."""
    sig = inspect.signature(fn)
    hints = {}
    try:
        hints = fn.__annotations__
    except AttributeError:
        pass

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue

        python_type = hints.get(name, str)
        json_type = _resolve_json_type(python_type)

        prop: dict[str, Any] = {"type": json_type}

        # Pull per-param description from docstring (Google style: "param: description")
        doc = inspect.getdoc(fn) or ""
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name}:") or stripped.startswith(f"{name} ("):
                desc = stripped.split(":", 1)[-1].strip()
                if desc:
                    prop["description"] = desc
                break

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def load_tool(fn: Callable) -> dict:
    """
    Convert a Python function into an Anthropic tool definition dict.

    Returns {"definition": {...}, "fn": fn}
    """
    name = fn.__name__
    doc = inspect.getdoc(fn) or ""
    # First line of docstring = tool description
    description = doc.splitlines()[0] if doc else name

    return {
        "definition": {
            "name": name,
            "description": description,
            "input_schema": _build_schema(fn),
        },
        "fn": fn,
    }


def load_tools_from_file(path: Path) -> dict[str, dict]:
    """
    Import a Python file and return tool definitions.

    If any functions in the file are decorated with @smalltask.tool, only those are
    returned. Otherwise all public functions are returned (backward compat).

    Returns {function_name: {"definition": {...}, "fn": fn}}
    """
    module_name = f"smalltask_tools.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    all_public = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if obj.__module__ != module_name:
            continue
        all_public[name] = load_tool(obj)

    decorated = {k: v for k, v in all_public.items() if getattr(v["fn"], "_smalltask_tool", False)}
    return decorated if decorated else all_public


def load_tools_from_dir(tools_dir: Path, names: list[str]) -> dict[str, dict]:
    """
    Load specific tools by name from a tools directory.

    Supports two name formats:
    - Short name: ``get_orders`` — scans all .py files for a matching function name.
    - Namespaced: ``orders.get_orders`` — loads ``get_orders`` specifically from
      ``orders.py``. Preferred: unambiguous and collision-free.

    Returns {name: {"definition": {...}, "fn": fn}}
    """
    import warnings

    found: dict[str, dict] = {}
    remaining = set(names)

    # Resolve namespaced names first (file.function_name)
    for name in list(remaining):
        if "." not in name:
            continue
        file_stem, func_name = name.split(".", 1)
        py_file = tools_dir / f"{file_stem}.py"
        if not py_file.exists():
            continue  # fall through to the ValueError below
        file_tools = load_tools_from_file(py_file)
        if func_name not in file_tools:
            continue
        tool_entry = dict(file_tools[func_name])
        tool_entry["definition"] = dict(tool_entry["definition"])
        tool_entry["definition"]["name"] = name  # LLM sees the namespaced name
        found[name] = tool_entry
        remaining.discard(name)

    # Resolve short names by scanning all .py files
    for py_file in sorted(tools_dir.glob("*.py")):
        short_remaining = {n for n in remaining if "." not in n}
        if not short_remaining:
            break
        file_tools = load_tools_from_file(py_file)
        for name in list(short_remaining):
            if name in file_tools:
                if name in found:
                    warnings.warn(
                        f"Tool '{name}' defined in multiple files; "
                        f"using first match, ignoring {py_file.name}. "
                        f"Use namespaced names (e.g. {py_file.stem}.{name}) to be explicit.",
                        stacklevel=2,
                    )
                else:
                    found[name] = file_tools[name]
                    remaining.discard(name)

    if remaining:
        raise ValueError(f"Tools not found in {tools_dir}: {sorted(remaining)}")

    return found


def load_agent_config(agent_path: Path) -> dict:
    """Load and validate an agent YAML file."""
    with open(agent_path) as f:
        config = yaml.safe_load(f)

    required_keys = {"name", "prompt", "tools"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Agent config missing required keys: {missing}")

    config.setdefault("model", "claude-opus-4-6")
    config.setdefault("max_tokens", 4096)
    config.setdefault("max_iterations", _DEFAULT_MAX_ITERATIONS)
    config.setdefault("max_total_tokens", None)
    config.setdefault("pre_hook", [])
    config.setdefault("post_hook", [])

    # Normalize hooks: each entry is {tool_name: {args}} or just a string (no args)
    for key in ("pre_hook", "post_hook"):
        raw = config[key]
        if not isinstance(raw, list):
            raw = [raw]
        normalized = []
        for entry in raw:
            if isinstance(entry, str):
                normalized.append({"tool": entry, "args": {}})
            elif isinstance(entry, dict):
                # e.g. {"orders.get_order_summary": {"days": 7}}
                tool_name = next(iter(entry))
                args = entry[tool_name]
                if args is None:
                    args = {}
                normalized.append({"tool": tool_name, "args": args})
            else:
                raise ValueError(f"Invalid {key} entry: {entry}")
        config[key] = normalized

    return config
