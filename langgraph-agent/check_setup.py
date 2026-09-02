"""Setup checker: verifies your keys load and this project's packages installed.

The implementation lives in `env_utils.py` at the repo root, next to the `.env`
that both projects share. This file runs it with the repo root still as the
anchor for `.env`, while the interpreter stays this project's `.venv`, so the
package check reports on the project you ran it from.

    uv run python check_setup.py
"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "env_utils.py"),
    run_name="__main__",
)
