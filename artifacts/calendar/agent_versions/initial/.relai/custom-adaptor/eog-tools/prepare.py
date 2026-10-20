"""Prepare the current optimizer worktree's uv environment and authenticated SDK."""
from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    if not Path(".venv").exists():
        subprocess.run(["uv", "sync", "--locked"], check=True)
    probe = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c",
         "from eog_tools.bootstrap import require_sdk_version; require_sdk_version()"],
        check=False,
        capture_output=True,
    )
    if probe.returncode:
        subprocess.run(["uv", "run", "--no-sync", "eog-tools", "bootstrap-sdk"], check=True)


if __name__ == "__main__":
    main()
