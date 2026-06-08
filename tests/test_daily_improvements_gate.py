"""Tests for the daily_improvements PR-comment gate decision logic."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "examples" / "daily_improvements" / "tools")
)

import github_pr  # noqa: E402

LAST_COMMIT = "2026-06-08T09:00:00Z"
BEFORE = "2026-06-08T08:00:00Z"
AFTER = "2026-06-08T10:00:00Z"


def _c(body: str, created_at: str) -> dict:
    return {"body": body, "created_at": created_at}


def test_no_human_comments_is_pending():
    action, feedback = github_pr._classify_pr_action([], LAST_COMMIT)
    assert action == "pending"
    assert feedback is None


def test_comment_older_than_last_commit_is_pending():
    comments = [_c("already handled this", BEFORE)]
    action, feedback = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "pending"


def test_reject_comment_rejects():
    comments = [_c("/reject", AFTER)]
    action, feedback = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "reject"
    assert feedback is None


def test_reject_takes_precedence_over_feedback():
    comments = [
        _c("make it clearer", AFTER),
        _c("actually /reject this", AFTER),
    ]
    action, _ = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "reject"


def test_unaddressed_comment_is_addressed():
    comments = [_c("do what the review says but make it clearer", AFTER)]
    action, feedback = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "address"
    assert "make it clearer" in feedback


def test_multiple_unaddressed_comments_are_combined():
    comments = [
        _c("rename the variable", AFTER),
        _c("and add a guard clause", AFTER),
    ]
    action, feedback = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "address"
    assert "rename the variable" in feedback
    assert "add a guard clause" in feedback


def test_only_comments_after_last_commit_count_as_feedback():
    comments = [
        _c("old note already handled", BEFORE),
        _c("new note to handle", AFTER),
    ]
    action, feedback = github_pr._classify_pr_action(comments, LAST_COMMIT)
    assert action == "address"
    assert "new note to handle" in feedback
    assert "old note already handled" not in feedback
