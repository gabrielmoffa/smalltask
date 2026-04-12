"""Tests for hook execution in the agent runner."""

from unittest.mock import MagicMock, patch

import pytest

from smalltask.runner import _collect_hook_tools, _run_hooks


def test_collect_hook_tools_basic():
    hooks = [
        {"tool": "github_pr.check_pending_pr", "args": {}},
        {"tool": "telegram.notify_telegram", "args": {}},
    ]
    result = _collect_hook_tools(hooks)
    assert result == {"github_pr.check_pending_pr", "telegram.notify_telegram"}


def test_collect_hook_tools_empty():
    assert _collect_hook_tools([]) == set()


def test_run_hooks_basic():
    mock_fn = MagicMock(return_value="hook_output")
    tools = {"my_hook": {"fn": mock_fn}}
    hooks = [{"tool": "my_hook", "args": {"x": 1}}]

    results = _run_hooks(hooks, tools, verbose=False)

    mock_fn.assert_called_once_with(x=1)
    assert results == [{"tool": "my_hook", "result": "hook_output"}]


def test_run_hooks_skip_gate():
    mock_fn = MagicMock(return_value={"skip": True, "reason": "nothing to do"})
    second_fn = MagicMock(return_value="should_not_run")
    tools = {
        "gate_hook": {"fn": mock_fn},
        "second_hook": {"fn": second_fn},
    }
    hooks = [
        {"tool": "gate_hook", "args": {}},
        {"tool": "second_hook", "args": {}},
    ]

    results = _run_hooks(hooks, tools, verbose=False)

    assert len(results) == 1
    assert results[0]["result"]["skip"] is True
    second_fn.assert_not_called()


def test_run_hooks_missing_tool():
    hooks = [{"tool": "nonexistent", "args": {}}]
    with pytest.raises(ValueError, match="Hook tool 'nonexistent' not found"):
        _run_hooks(hooks, {}, verbose=False)
