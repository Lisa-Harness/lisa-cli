"""Tests for the CLI ``lisa update`` command.

``lisa update`` is a thin pass-through to ``uv tool upgrade lisa-cli``. The
command exists (rather than telling operators to run uv themselves) because
it knows which uv invocation is safe: ``uv tool upgrade`` replays the install
receipt, while ``uv tool install --upgrade`` would rewrite it and drop the
``--with`` framework requirements, extras, and ``no-sources`` the installer
recorded.

These tests stub ``is_tool_install``/``shutil.which``/``subprocess.run`` so no
real uv process runs; they assert the emitted argv, the refusal paths (repo
checkout, missing uv, Windows), and the packs-venv reminder.
"""

from __future__ import annotations

# pylint: disable=protected-access
from pathlib import Path
from typing import Any
from unittest.mock import patch

import typer

from hosts.cli import main as cli_main

# ── Helpers ───────────────────────────────────────────────────────────


class _CapturingConsole:
    """Records the plain text of every ``console.print`` positional arg."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            if isinstance(arg, str):
                self.lines.append(arg)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _run_update(
    *,
    force: bool = False,
    dry_run: bool = False,
    tool_install: bool = True,
    uv_path: str | None = "/usr/local/bin/uv",
    returncode: int = 0,
    packs_venv: Path | None = None,
    os_name: str = "posix",
) -> tuple[_CapturingConsole, list[list[str]], int | None]:
    """Invoke ``_update_command`` with stubbed environment seams.

    Returns ``(console, recorded_argv_list, exit_code)`` where exit code is
    ``None`` when the callback returned normally.
    """
    console = _CapturingConsole()
    invocations: list[list[str]] = []
    venv = packs_venv if packs_venv is not None else Path("/nonexistent/packs-venv")

    class _Completed:
        pass

    def _fake_run(argv, **_kwargs):
        invocations.append(list(argv))
        completed = _Completed()
        completed.returncode = returncode
        return completed

    exit_code: int | None = None
    with (
        patch("runtime.paths.is_tool_install", return_value=tool_install),
        patch("runtime.paths.default_packs_venv", return_value=venv),
        patch.object(cli_main.shutil, "which", return_value=uv_path),
        patch.object(cli_main.subprocess, "run", new=_fake_run),
        patch.object(cli_main.os, "name", new=os_name),
        patch.object(cli_main, "console", new=console),
    ):
        try:
            cli_main._update_command(force=force, dry_run=dry_run)
        except typer.Exit as exc:
            exit_code = exc.exit_code
    return console, invocations, exit_code


# ── Happy path ────────────────────────────────────────────────────────


def test_updates_with_uv_tool_upgrade() -> None:
    """The command upgrades via `uv tool upgrade`, never `install --upgrade`."""
    console, invocations, exit_code = _run_update()

    assert invocations == [["/usr/local/bin/uv", "tool", "upgrade", "lisa-cli"]]
    assert exit_code is None
    assert "Restart lisa" in console.text
    assert "tool install" not in " ".join(" ".join(a) for a in invocations)


def test_force_adds_reinstall_flag() -> None:
    """--force maps to `--reinstall` so same-version local edits are picked up."""
    _, invocations, exit_code = _run_update(force=True)

    assert invocations == [["/usr/local/bin/uv", "tool", "upgrade", "--reinstall", "lisa-cli"]]
    assert exit_code is None


# ── Refusal paths ────────────────────────────────────────────────────


def test_repo_checkout_points_at_git_pull() -> None:
    """Repo mode has no tool to upgrade; the fix is git pull + uv sync."""
    console, invocations, exit_code = _run_update(tool_install=False)

    assert invocations == []
    assert exit_code == 2
    assert "git pull" in console.text


def test_missing_uv_prints_command_and_fails() -> None:
    """Without uv on PATH the operator gets the exact command, exit 1."""
    console, invocations, exit_code = _run_update(uv_path=None)

    assert invocations == []
    assert exit_code == 1
    assert "uv tool upgrade lisa-cli" in console.text
    assert "not on PATH" in console.text


def test_windows_prints_command_instead_of_upgrading() -> None:
    """Windows locks the running .exe trampoline, so defer to the operator."""
    console, invocations, exit_code = _run_update(os_name="nt")

    assert invocations == []
    assert exit_code == 1
    assert "uv tool upgrade lisa-cli" in console.text
    assert "Windows" in console.text


def test_windows_dry_run_still_reports_success() -> None:
    """--dry-run is always a successful report, even where upgrade is blocked."""
    console, invocations, exit_code = _run_update(dry_run=True, os_name="nt")

    assert invocations == []
    assert exit_code == 0
    assert "Run:" in console.text


# ── --dry-run ────────────────────────────────────────────────────────


def test_dry_run_does_not_execute() -> None:
    """--dry-run prints the argv and exits 0 without shelling out."""
    console, invocations, exit_code = _run_update(dry_run=True, force=True)

    assert invocations == []
    assert exit_code == 0
    assert "uv tool upgrade --reinstall lisa-cli" in console.text
    assert "Restart lisa" not in console.text


# ── Failure propagation ──────────────────────────────────────────────


def test_propagates_uv_failure_exit_code() -> None:
    """A failing uv run surfaces its exit code and no success message."""
    console, _, exit_code = _run_update(returncode=7)

    assert exit_code == 7
    assert "failed" in console.text
    assert "Restart lisa" not in console.text


# ── Packs venv reminder ──────────────────────────────────────────────


def test_warns_when_packs_venv_exists(tmp_path: Path) -> None:
    """An existing packs venv is flagged as stale after a successful upgrade."""
    venv = tmp_path / "venv"
    venv.mkdir()

    console, _, exit_code = _run_update(packs_venv=venv)

    assert exit_code is None
    assert "pack update --all" in console.text


def test_no_packs_venv_warning_without_a_venv() -> None:
    """A machine with no packs venv needs no reminder."""
    console, _, _ = _run_update()

    assert "pack update --all" not in console.text
