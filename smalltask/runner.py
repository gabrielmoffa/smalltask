"""Core agent runner: load config + tools, run prompt-based agentic loop."""

import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

from smalltask.llm import complete
from smalltask.loader import load_agent_config, load_tools_from_dir, resolve_llm_config
from smalltask.prompt_tools import build_tool_system_prompt, format_tool_result, parse_tool_calls


def agent_tool(
    name: str,
    agent_path: Path,
    description: str | None = None,
    tools_dir: Path | None = None,
) -> dict:
    """
    Wrap a sub-agent as a tool callable by a parent agent.

    The sub-agent receives a single ``task`` string argument and returns its
    final response as a string. This lets one agent delegate to another without
    any special multi-agent infrastructure — the parent just calls it like any
    other tool.

    Usage in Python::

        from smalltask.runner import agent_tool, run_agent
        from pathlib import Path

        extra_tools = {
            "summarize": agent_tool(
                name="summarize",
                agent_path=Path("agents/summarize.yaml"),
                description="Summarise a block of text. Pass the text as 'task'.",
            )
        }
        run_agent(Path("agents/orchestrator.yaml"), extra_tools=extra_tools)

    Or reference it in the orchestrator YAML (no extra Python required) by placing
    the sub-agent path under a ``agents:`` key — see docs for that form.
    """
    def _call(task: str) -> str:
        return run_agent(
            agent_path=agent_path,
            tools_dir=tools_dir,
            input_vars={"task": task},
        )

    _call.__name__ = name
    _call.__doc__ = description or f"Sub-agent: {name}. Pass the task as a string."
    _call._smalltask_tool = True  # expose via @tool machinery

    from smalltask.loader import load_tool
    return load_tool(_call)


def _resolve_tools_dir(agent_path: Path, tools_dir: Path | None) -> Path:
    if tools_dir:
        return tools_dir
    candidate = agent_path.parent
    while candidate != candidate.parent:
        tools_candidate = candidate / "tools"
        if tools_candidate.is_dir():
            return tools_candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        "Could not find a tools/ directory. Pass --tools-dir explicitly."
    )


def _collect_hook_tools(hooks: list[dict]) -> set[str]:
    """Return the set of tool names referenced by hook entries."""
    return {entry["tool"] for entry in hooks}


def _run_hooks(
    hooks: list[dict],
    tools: dict,
    verbose: bool,
    output: str | None = None,
    tool_results: list[dict] | None = None,
) -> list[dict]:
    """
    Execute a list of hook entries sequentially.

    Each entry is {"tool": "name", "args": {...}}.
    For post-hooks, the framework auto-injects special parameters if the tool
    function accepts them and they weren't already provided in the YAML args:

    - ``output`` — the LLM's final response text.
    - ``tool_results`` — the full list of tool calls made during the agent loop,
      each as ``{"tool": name, "args": {...}, "result": ...}``.

    Returns a list of {"tool": name, "result": result_value} dicts.
    If any tool returns a dict containing ``"skip": True``, the list is
    terminated early with that entry included.
    """
    auto_inject = {}
    if output is not None:
        auto_inject["output"] = output
    if tool_results is not None:
        auto_inject["tool_results"] = tool_results

    results = []
    for entry in hooks:
        name = entry["tool"]
        args = dict(entry["args"])

        fn = tools.get(name, {}).get("fn")
        if fn is None:
            raise ValueError(f"Hook tool '{name}' not found")

        # Auto-inject available context into parameters the tool accepts
        if auto_inject:
            sig = inspect.signature(fn)
            for param_name, value in auto_inject.items():
                if param_name in sig.parameters and param_name not in args:
                    args[param_name] = value

        if verbose:
            print(f"[smalltask] Hook: {name}({json.dumps(args, default=str)})")

        try:
            raw = fn(**args)
        except Exception as e:
            raise RuntimeError(f"Hook tool '{name}' failed: {e}") from e

        if verbose:
            preview = json.dumps(raw, default=str) if not isinstance(raw, str) else raw
            print(f"[smalltask] Hook result ({name}): {preview[:300]}\n")

        results.append({"tool": name, "result": raw})

        # Check for skip gate
        if isinstance(raw, dict) and raw.get("skip") is True:
            break

    return results


def run_agent(
    agent_path: Path,
    tools_dir: Path | None = None,
    input_vars: dict[str, str] | None = None,
    verbose: bool = False,
    extra_tools: dict[str, dict] | None = None,
    max_iterations: int | None = None,
    max_total_tokens: int | None = None,
) -> str:
    """
    Run an agent defined by a YAML file.

    Tool calling is prompt-based — no provider SDK required.
    The LLM endpoint is configured in the agent YAML under `llm`.

    extra_tools: optional dict of {name: tool_entry} to inject alongside the
    tools listed in the YAML. Used for multi-agent setups where a sub-agent is
    passed in as a callable tool via agent_tool().

    max_iterations: override the iteration cap from the agent YAML (default 20).
    max_total_tokens: override the token budget from the agent YAML (default: no limit).
    """
    config = load_agent_config(agent_path)
    raw_llm = config.get("llm")
    if not raw_llm:
        raise ValueError("Agent config must have an `llm` section. See examples/.")
    llm_config = resolve_llm_config(raw_llm, agent_path)

    _max_iterations = max_iterations if max_iterations is not None else config["max_iterations"]
    _max_total_tokens = max_total_tokens if max_total_tokens is not None else config["max_total_tokens"]

    resolved_tools_dir = _resolve_tools_dir(agent_path, tools_dir)

    # Collect all tool names needed: agent tools + hook tools
    hook_tool_names = (
        _collect_hook_tools(config["pre_hook"])
        | _collect_hook_tools(config["post_hook"])
    )
    all_tool_names = list(config["tools"]) + [n for n in hook_tool_names if n not in config["tools"]]

    tools = load_tools_from_dir(resolved_tools_dir, all_tool_names) if all_tool_names else {}
    if extra_tools:
        tools = {**tools, **extra_tools}

    prompt = config["prompt"]
    if input_vars:
        prompt = Template(prompt).safe_substitute(input_vars)

    if verbose:
        print(f"\n[smalltask] Agent: {config['name']}")
        print(f"[smalltask] Model: {llm_config.get('model')}")
        print(f"[smalltask] Tools: {list(tools.keys())}\n")

    # --- Pre-hooks ---
    if config["pre_hook"]:
        if verbose:
            print("[smalltask] Running pre-hooks...")
        pre_results = _run_hooks(config["pre_hook"], tools, verbose)

        # Check if any pre-hook signalled skip
        for entry in pre_results:
            if isinstance(entry["result"], dict) and entry["result"].get("skip") is True:
                reason = entry["result"].get("reason", "pre-hook returned skip")
                if verbose:
                    print(f"[smalltask] Skipped: {reason}")
                return f"[skipped: {reason}]"

        # Inject pre-hook results into the prompt
        hook_context_parts = []
        for entry in pre_results:
            result = entry["result"]
            formatted = json.dumps(result) if not isinstance(result, str) else result
            hook_context_parts.append(f"### {entry['tool']}\n{formatted}")
        hook_context = "\n\n".join(hook_context_parts)
        prompt = f"## Pre-hook data\n\n{hook_context}\n\n## Task\n\n{prompt}"
    else:
        prompt = f"## Task\n\n{prompt}"

    tool_system_prompt = build_tool_system_prompt(tools)
    system_content = f"{tool_system_prompt}\n\n{prompt}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Please complete the task described above."},
    ]

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_tool_results: list[dict] = []
    final_output = None

    for iteration in range(_max_iterations):
        response_text, usage = complete(messages, llm_config)

        for key in total_usage:
            total_usage[key] += usage[key]

        if _max_total_tokens is not None and total_usage["total_tokens"] >= _max_total_tokens:
            if verbose:
                print(
                    f"[smalltask] Token budget reached: {total_usage['total_tokens']} / {_max_total_tokens}"
                )
            final_output = "[token budget exceeded]"
            break

        if verbose:
            print(f"[smalltask] Response (iter {iteration + 1}):\n{response_text}\n")
            print(f"[smalltask] Tokens this call: prompt={usage['prompt_tokens']} completion={usage['completion_tokens']}")

        tool_calls = parse_tool_calls(response_text)

        if not tool_calls:
            # No tool calls — this is the final answer
            if verbose:
                print(
                    f"[smalltask] Total tokens used: prompt={total_usage['prompt_tokens']} "
                    f"completion={total_usage['completion_tokens']} "
                    f"total={total_usage['total_tokens']}"
                )
            final_output = response_text.strip()
            break

        # Append assistant message with all tool calls
        messages.append({"role": "assistant", "content": response_text})

        # Execute all tool calls in parallel
        def _execute(call: dict) -> tuple[str, dict, str]:
            name = call["name"]
            args = call.get("args", {})
            fn = tools.get(name, {}).get("fn")
            if fn is None:
                return name, args, f"Error: tool '{name}' not found"
            if verbose:
                print(f"[smalltask] Tool call: {name}({json.dumps(args)})")
            try:
                raw = fn(**args)
                result = json.dumps(raw) if not isinstance(raw, str) else raw
            except Exception as e:
                result = f"Error: {e}"
            if verbose:
                print(f"[smalltask] Tool result ({name}): {result[:300]}\n")
            return name, args, result

        results: list[tuple[str, dict, str]] = [None] * len(tool_calls)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {pool.submit(_execute, call): i for i, call in enumerate(tool_calls)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        # Record tool results for post-hooks
        for name, args, result in results:
            all_tool_results.append({"tool": name, "args": args, "result": result})

        # Inject all results as a single user message to preserve ordering
        combined = "\n\n".join(format_tool_result(name, result) for name, _, result in results)
        messages.append({"role": "user", "content": combined})

    if final_output is None:
        if verbose:
            print(
                f"[smalltask] Total tokens used: prompt={total_usage['prompt_tokens']} "
                f"completion={total_usage['completion_tokens']} "
                f"total={total_usage['total_tokens']}"
            )
        final_output = "[max iterations reached]"

    # --- Post-hooks ---
    if config["post_hook"]:
        if verbose:
            print("[smalltask] Running post-hooks...")
        _run_hooks(config["post_hook"], tools, verbose, output=final_output, tool_results=all_tool_results)

    return final_output
