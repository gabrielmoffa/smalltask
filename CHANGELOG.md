# Changelog

All notable changes to this project will be documented here. Newest first.

---

## [0.1.0] — 2025-04-09

Initial release.

### Added
- `@tool` decorator to mark functions as agent-callable tools
- YAML-based agent definitions (`name`, `prompt`, `tools`, `llm`, `max_iterations`, `max_total_tokens`)
- Prompt-based tool calling — works with any OpenAI-compatible endpoint (OpenRouter, Ollama, Groq, Bedrock, etc.)
- `smalltask run` CLI command with `--var` interpolation and `--verbose` output
- `smalltask init` with `default` and `github` starter templates
- `run_agent()` Python API with `extra_tools` for multi-agent composition
- `agent_tool()` helper to wrap a sub-agent as a callable tool
- Namespaced tool names (`file.function_name`) for unambiguous, collision-free references
- Parallel tool execution via `ThreadPoolExecutor`
- Token usage tracking per iteration and totals
- Configurable `max_iterations` (default 20) and `max_total_tokens` (default: no limit)
- Auto-detection of `tools/` directory by walking up from the agent YAML
- JSON Schema generation from Python type hints and Google-style docstrings
