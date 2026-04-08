"""Core agent runner: load config + tools, run prompt-based agentic loop."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

from smalltask.llm import complete
from smalltask.loader import _DEFAULT_MAX_ITERATIONS, load_agent_config, load_tools_from_dir
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
    llm_config = config.get("llm")
    if not llm_config:
        raise ValueError("Agent config must have an `llm` section. See examples/.")

    _max_iterations = max_iterations if max_iterations is not None else config["max_iterations"]
    _max_total_tokens = max_total_tokens if max_total_tokens is not None else config["max_total_tokens"]

    resolved_tools_dir = _resolve_tools_dir(agent_path, tools_dir)
    tools = load_tools_from_dir(resolved_tools_dir, config["tools"])
    if extra_tools:
        tools = {**tools, **extra_tools}

    prompt = config["prompt"]
    if input_vars:
        prompt = Template(prompt).safe_substitute(input_vars)

    tool_system_prompt = build_tool_system_prompt(tools)
    system_content = f"{tool_system_prompt}\n\n## Task\n\n{prompt}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Please complete the task described above."},
    ]

    if verbose:
        print(f"\n[smalltask] Agent: {config['name']}")
        print(f"[smalltask] Model: {llm_config.get('model')}")
        print(f"[smalltask] Tools: {list(tools.keys())}\n")

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for iteration in range(_max_iterations):
        response_text, usage = complete(messages, llm_config)

        for key in total_usage:
            total_usage[key] += usage[key]

        if _max_total_tokens is not None and total_usage["total_tokens"] >= _max_total_tokens:
            if verbose:
                print(
                    f"[smalltask] Token budget reached: {total_usage['total_tokens']} / {_max_total_tokens}"
                )
            return "[token budget exceeded]"

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
            return response_text.strip()

        # Append assistant message with all tool calls
        messages.append({"role": "assistant", "content": response_text})

        # Execute all tool calls in parallel
        def _execute(call: dict) -> tuple[str, str]:
            name = call["name"]
            args = call.get("args", {})
            fn = tools.get(name, {}).get("fn")
            if fn is None:
                return name, f"Error: tool '{name}' not found"
            if verbose:
                print(f"[smalltask] Tool call: {name}({json.dumps(args)})")
            try:
                raw = fn(**args)
                result = json.dumps(raw) if not isinstance(raw, str) else raw
            except Exception as e:
                result = f"Error: {e}"
            if verbose:
                print(f"[smalltask] Tool result ({name}): {result[:300]}\n")
            return name, result

        results: list[tuple[str, str]] = [None] * len(tool_calls)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {pool.submit(_execute, call): i for i, call in enumerate(tool_calls)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        # Inject all results as a single user message to preserve ordering
        combined = "\n\n".join(format_tool_result(name, result) for name, result in results)
        messages.append({"role": "user", "content": combined})

    if verbose:
        print(
            f"[smalltask] Total tokens used: prompt={total_usage['prompt_tokens']} "
            f"completion={total_usage['completion_tokens']} "
            f"total={total_usage['total_tokens']}"
        )
    return "[max iterations reached]"
