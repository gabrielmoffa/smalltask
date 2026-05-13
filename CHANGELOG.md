# Changelog

All notable changes to this project will be documented here. Newest first.

---

## [0.3.6] — 2026-05-13

### Added
- LLM configs can pass a `reasoning` block through to compatible chat endpoints.

### Fixed
- Verbose output now displays reasoning text returned as either `reasoning_content` or `reasoning`.
- In-memory `extra_tools` are no longer loaded from disk before being injected.

---

## [0.3.5] — 2026-05-13

### Changed
- Release the current `main` package state to PyPI.

---

## [0.3.4] — 2026-04-28

### Fixed
- Array parameters in generated tool schemas now include `items`, matching OpenAI's schema requirements.

---

## [0.3.3] — 2026-04-28

### Fixed
- Native tool-call parsing now safely handles `tool_calls: null`.
- Added regression coverage for Python 3.10+ union type schema generation.

---

## [0.3.2] — 2026-04-28

### Fixed
- Native tool-call parsing now safely handles a missing assistant message.
- LLM requests can use `max_completion_tokens` without also sending `max_tokens`.

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
