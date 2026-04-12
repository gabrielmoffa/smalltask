"""Tools for reading and modifying repository files."""

import subprocess

from smalltask import tool


@tool
def list_repo_files(pattern: str = "*.py") -> list:
    """List tracked files in the repository matching a glob pattern.

    Args:
        pattern: Glob pattern to filter files (e.g. '*.py', 'smalltask/*.py').
    """
    result = subprocess.run(
        ["git", "ls-files", pattern],
        capture_output=True, text=True, check=True,
    )
    files = [f for f in result.stdout.strip().split("\n") if f]
    return files


@tool
def read_file(path: str) -> str:
    """Read the full contents of a file.

    Args:
        path: Relative path to the file from the repo root.
    """
    with open(path) as f:
        return f.read()


@tool
def replace_in_file(path: str, old_string: str, new_string: str) -> str:
    """Replace a specific string in a file. Use this to make targeted edits.

    You must provide enough surrounding context in old_string to match
    exactly one location in the file. The match must be unique — if
    old_string appears more than once, the call will fail.

    Args:
        path: Relative path to the file from the repo root.
        old_string: The exact text to find (include surrounding lines for uniqueness).
        new_string: The text to replace it with.
    """
    with open(path) as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1:
        return f"Error: old_string matches {count} locations in {path} — include more context to make it unique"

    new_content = content.replace(old_string, new_string, 1)
    with open(path, "w") as f:
        f.write(new_content)

    return f"Replaced in {path}"
