"""Tests for pre-hook and post-hook functionality."""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smalltask.loader import load_agent_config
from smalltask.runner import _run_hooks, run_agent


# ---------------------------------------------------------------------------
# Helper: build a tool entry from a plain function
# ---------------------------------------------------------------------------

def _tool_entry(fn):
    from smalltask.loader import load_tool
    return load_tool(fn)


# ---------------------------------------------------------------------------
# load_agent_config — hook parsing
# ---------------------------------------------------------------------------

class TestHookParsing:
    def test_no_hooks_defaults_to_empty(self, tmp_path):
        f = tmp_path / "agent.yaml"
        f.write_text("name: t\nprompt: p\ntools: []\n")
        config = load_agent_config(f)
        assert config["pre_hook"] == []
        assert config["post_hook"] == []

    def test_hook_with_args(self, tmp_path):
        f = tmp_path / "agent.yaml"
        f.write_text(textwrap.dedent("""\
            name: t
            prompt: p
            tools: []
            pre_hook:
              - orders.get_summary:
                  days: 7
        """))
        config = load_agent_config(f)
        assert config["pre_hook"] == [{"tool": "orders.get_summary", "args": {"days": 7}}]

    def test_hook_string_shorthand(self, tmp_path):
        f = tmp_path / "agent.yaml"
        f.write_text(textwrap.dedent("""\
            name: t
            prompt: p
            tools: []
            post_hook:
              - slack.post_message
        """))
        config = load_agent_config(f)
        assert config["post_hook"] == [{"tool": "slack.post_message", "args": {}}]

    def test_hook_with_null_args(self, tmp_path):
        f = tmp_path / "agent.yaml"
        f.write_text(textwrap.dedent("""\
            name: t
            prompt: p
            tools: []
            pre_hook:
              - my_tool:
        """))
        config = load_agent_config(f)
        assert config["pre_hook"] == [{"tool": "my_tool", "args": {}}]

    def test_multiple_hooks(self, tmp_path):
        f = tmp_path / "agent.yaml"
        f.write_text(textwrap.dedent("""\
            name: t
            prompt: p
            tools: []
            pre_hook:
              - step_one:
                  x: 1
              - step_two:
                  y: 2
        """))
        config = load_agent_config(f)
        assert len(config["pre_hook"]) == 2
        assert config["pre_hook"][0]["tool"] == "step_one"
        assert config["pre_hook"][1]["tool"] == "step_two"


# ---------------------------------------------------------------------------
# _run_hooks — execution
# ---------------------------------------------------------------------------

class TestRunHooks:
    def test_basic_execution(self):
        def snapshot(days: int) -> dict:
            """Take a snapshot."""
            return {"revenue": 1000, "days": days}

        tools = {"snapshot": _tool_entry(snapshot)}
        hooks = [{"tool": "snapshot", "args": {"days": 7}}]

        results = _run_hooks(hooks, tools, verbose=False)
        assert len(results) == 1
        assert results[0]["tool"] == "snapshot"
        assert results[0]["result"] == {"revenue": 1000, "days": 7}

    def test_skip_gate_stops_execution(self):
        call_log = []

        def check_threshold(metric: str, max_val: float) -> dict:
            """Check if metric is within threshold."""
            call_log.append("check")
            return {"skip": True, "reason": f"{metric} is fine"}

        def should_not_run() -> str:
            """This should not be called."""
            call_log.append("bad")
            return "oops"

        tools = {
            "check_threshold": _tool_entry(check_threshold),
            "should_not_run": _tool_entry(should_not_run),
        }
        hooks = [
            {"tool": "check_threshold", "args": {"metric": "refund_rate", "max_val": 0.05}},
            {"tool": "should_not_run", "args": {}},
        ]

        results = _run_hooks(hooks, tools, verbose=False)
        assert len(results) == 1  # stopped after skip
        assert results[0]["result"]["skip"] is True
        assert call_log == ["check"]

    def test_post_hook_receives_output(self):
        captured = {}

        def save_report(output: str, path: str) -> str:
            """Save report to file."""
            captured["output"] = output
            captured["path"] = path
            return "saved"

        tools = {"save_report": _tool_entry(save_report)}
        hooks = [{"tool": "save_report", "args": {"path": "/tmp/report.txt"}}]

        results = _run_hooks(hooks, tools, verbose=False, output="LLM said hello")
        assert captured["output"] == "LLM said hello"
        assert captured["path"] == "/tmp/report.txt"
        assert results[0]["result"] == "saved"

    def test_output_not_injected_when_not_accepted(self):
        def simple_tool(x: int) -> int:
            """No output param."""
            return x * 2

        tools = {"simple_tool": _tool_entry(simple_tool)}
        hooks = [{"tool": "simple_tool", "args": {"x": 5}}]

        results = _run_hooks(hooks, tools, verbose=False, output="ignored")
        assert results[0]["result"] == 10

    def test_missing_tool_raises(self):
        hooks = [{"tool": "nonexistent", "args": {}}]
        with pytest.raises(ValueError, match="Hook tool 'nonexistent' not found"):
            _run_hooks(hooks, {}, verbose=False)

    def test_tool_exception_raises_runtime_error(self):
        def bad_tool() -> None:
            """Breaks."""
            raise ValueError("boom")

        tools = {"bad_tool": _tool_entry(bad_tool)}
        hooks = [{"tool": "bad_tool", "args": {}}]

        with pytest.raises(RuntimeError, match="Hook tool 'bad_tool' failed"):
            _run_hooks(hooks, tools, verbose=False)

    def test_sequential_execution_order(self):
        log = []

        def step_a() -> str:
            """Step A."""
            log.append("a")
            return "a done"

        def step_b() -> str:
            """Step B."""
            log.append("b")
            return "b done"

        tools = {"step_a": _tool_entry(step_a), "step_b": _tool_entry(step_b)}
        hooks = [
            {"tool": "step_a", "args": {}},
            {"tool": "step_b", "args": {}},
        ]

        results = _run_hooks(hooks, tools, verbose=False)
        assert log == ["a", "b"]
        assert len(results) == 2

    def test_tool_results_injected(self):
        captured = {}

        def send_report(output: str, tool_results: list) -> str:
            """Send report with artifacts."""
            captured["output"] = output
            captured["tool_results"] = tool_results
            return "sent"

        tools = {"send_report": _tool_entry(send_report)}
        hooks = [{"tool": "send_report", "args": {}}]
        fake_results = [
            {"tool": "plot_revenue", "args": {"days": 7}, "result": "/tmp/revenue.png"},
            {"tool": "plot_refunds", "args": {"days": 7}, "result": "/tmp/refunds.png"},
            {"tool": "get_summary", "args": {"days": 7}, "result": '{"revenue": 42000}'},
        ]

        _run_hooks(hooks, tools, verbose=False, output="Analysis done.", tool_results=fake_results)
        assert captured["output"] == "Analysis done."
        assert captured["tool_results"] == fake_results
        assert len(captured["tool_results"]) == 3

    def test_tool_results_not_injected_when_not_accepted(self):
        def simple_hook(output: str) -> str:
            """Only takes output."""
            return f"got: {output}"

        tools = {"simple_hook": _tool_entry(simple_hook)}
        hooks = [{"tool": "simple_hook", "args": {}}]

        results = _run_hooks(
            hooks, tools, verbose=False,
            output="hello",
            tool_results=[{"tool": "x", "args": {}, "result": "y"}],
        )
        assert results[0]["result"] == "got: hello"

    def test_tool_results_filter_by_name(self):
        """Verify a post-hook can filter tool_results by tool name."""
        captured = {}

        def upload_charts(tool_results: list) -> str:
            """Upload only chart artifacts."""
            charts = [r["result"] for r in tool_results if r["tool"].startswith("plot_")]
            captured["charts"] = charts
            return f"uploaded {len(charts)} charts"

        tools = {"upload_charts": _tool_entry(upload_charts)}
        hooks = [{"tool": "upload_charts", "args": {}}]
        fake_results = [
            {"tool": "plot_revenue", "args": {"days": 7}, "result": "/tmp/revenue.png"},
            {"tool": "get_summary", "args": {"days": 7}, "result": '{"revenue": 42000}'},
            {"tool": "plot_refunds", "args": {"days": 7}, "result": "/tmp/refunds.png"},
        ]

        results = _run_hooks(hooks, tools, verbose=False, tool_results=fake_results)
        assert captured["charts"] == ["/tmp/revenue.png", "/tmp/refunds.png"]
        assert results[0]["result"] == "uploaded 2 charts"


# ---------------------------------------------------------------------------
# run_agent integration — pre-hook skip
# ---------------------------------------------------------------------------

class TestRunAgentWithHooks:
    def _setup_agent(self, tmp_path, pre_hook=None, post_hook=None):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "hooks.py").write_text(textwrap.dedent("""\
            def check_metric(threshold: float) -> dict:
                \"\"\"Check if metric exceeds threshold.\"\"\"
                return {"skip": True, "reason": "metric below threshold"}

            def snapshot(days: int) -> dict:
                \"\"\"Take a data snapshot.\"\"\"
                return {"revenue": 42000, "orders": 150}

            def save_report(output: str) -> str:
                \"\"\"Save the report.\"\"\"
                return f"saved: {len(output)} chars"

            def noop() -> str:
                \"\"\"A no-op tool for the agent to have.\"\"\"
                return "ok"
        """))

        import yaml
        config = {
            "name": "test_agent",
            "prompt": "Analyze the data.",
            "llm": {"url": "http://fake", "model": "fake-model"},
            "tools": ["hooks.noop"],
        }
        if pre_hook is not None:
            config["pre_hook"] = pre_hook
        if post_hook is not None:
            config["post_hook"] = post_hook

        agent_file = tmp_path / "agents" / "test.yaml"
        agent_file.parent.mkdir()
        agent_file.write_text(yaml.dump(config))
        return agent_file

    def test_pre_hook_skip_prevents_llm_call(self, tmp_path):
        agent_file = self._setup_agent(
            tmp_path,
            pre_hook=[{"hooks.check_metric": {"threshold": 0.05}}],
        )

        # LLM should never be called — no need to mock httpx
        result = run_agent(agent_file)
        assert "[skipped:" in result
        assert "metric below threshold" in result

    def test_pre_hook_data_injected_into_prompt(self, tmp_path):
        agent_file = self._setup_agent(
            tmp_path,
            pre_hook=[{"hooks.snapshot": {"days": 7}}],
        )

        captured_messages = []

        def fake_complete(messages, llm_config, tools=None):
            captured_messages.extend(messages)
            return (
                {"role": "assistant", "content": "Final analysis."},
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        with patch("smalltask.runner.complete", side_effect=fake_complete):
            result = run_agent(agent_file)

        system_msg = captured_messages[0]["content"]
        assert "Pre-hook data" in system_msg
        assert "42000" in system_msg
        assert "150" in system_msg

    def test_post_hook_receives_llm_output(self, tmp_path):
        agent_file = self._setup_agent(
            tmp_path,
            post_hook=["hooks.save_report"],
        )

        def fake_complete(messages, llm_config, tools=None):
            return (
                {"role": "assistant", "content": "The analysis is complete."},
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        with patch("smalltask.runner.complete", side_effect=fake_complete):
            result = run_agent(agent_file)

        assert result == "The analysis is complete."

    def test_both_hooks(self, tmp_path):
        agent_file = self._setup_agent(
            tmp_path,
            pre_hook=[{"hooks.snapshot": {"days": 7}}],
            post_hook=["hooks.save_report"],
        )

        def fake_complete(messages, llm_config, tools=None):
            return (
                {"role": "assistant", "content": "Done."},
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        with patch("smalltask.runner.complete", side_effect=fake_complete):
            result = run_agent(agent_file)

        assert result == "Done."

    def test_post_hook_receives_tool_results_from_loop(self, tmp_path):
        """End-to-end: LLM calls tools, post-hook gets the full tool_results list."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "analysis.py").write_text(textwrap.dedent("""\
            import json as _json

            def plot_revenue(days: int) -> str:
                \"\"\"Plot revenue chart.\"\"\"
                return f"/tmp/charts/revenue_{days}d.png"

            def get_summary(days: int) -> dict:
                \"\"\"Get order summary.\"\"\"
                return {"revenue": 42000, "orders": 150}

            def collect_artifacts(output: str, tool_results: list) -> str:
                \"\"\"Collect artifacts from the run.\"\"\"
                # Write to a file so the test can read it back
                with open("/tmp/_smalltask_test_tool_results.json", "w") as f:
                    _json.dump(tool_results, f)
                charts = [r["result"] for r in tool_results if r["result"].endswith(".png")]
                return f"collected {len(charts)} charts"
        """))

        import yaml
        config = {
            "name": "test_tool_results",
            "prompt": "Analyze revenue.",
            "llm": {"url": "http://fake", "model": "fake-model"},
            "tools": ["analysis.plot_revenue", "analysis.get_summary"],
            "post_hook": ["analysis.collect_artifacts"],
        }
        agent_file = tmp_path / "agents" / "test.yaml"
        agent_file.parent.mkdir()
        agent_file.write_text(yaml.dump(config))

        call_count = [0]

        def fake_complete(messages, llm_config, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "analysis.plot_revenue",
                                    "arguments": '{"days": 7}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "analysis.get_summary",
                                    "arguments": '{"days": 7}',
                                },
                            },
                        ],
                    },
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                )
            return (
                {"role": "assistant", "content": "Revenue looks good."},
                {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )

        with patch("smalltask.runner.complete", side_effect=fake_complete):
            result = run_agent(agent_file)

        assert result == "Revenue looks good."

        # Read back what the post-hook captured
        import json as json_mod
        captured_path = Path("/tmp/_smalltask_test_tool_results.json")
        assert captured_path.exists(), "Post-hook did not write tool_results"
        captured = json_mod.loads(captured_path.read_text())
        captured_path.unlink()

        assert len(captured) == 2
        tool_names = [r["tool"] for r in captured]
        assert "analysis.plot_revenue" in tool_names
        assert "analysis.get_summary" in tool_names
        plot_entry = next(r for r in captured if r["tool"] == "analysis.plot_revenue")
        assert plot_entry["result"] == "/tmp/charts/revenue_7d.png"
