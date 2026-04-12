"""Tools for reading and modifying repository files."""

import os
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
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Relative path to the file from the repo root.
        content: The full file content to write.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"
