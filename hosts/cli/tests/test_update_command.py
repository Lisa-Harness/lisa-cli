"""Tests for the CLI ``lisa update`` command and its uv receipt handling."""

from __future__ import annotations

# pylint: disable=protected-access
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from hosts.cli import main as cli_main
from hosts.cli import update as cli_update

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
    windows: bool = False,
    tool_dir: Path | None = None,
    with_receipt: bool = True,
    version: str = "",
    pin: str = "",
    channel: str = "",
    ref: str = "",
    latest: bool = False,
    framework_version: str = "",
    framework_pin: str = "",
    framework_channel: str = "",
    framework_ref: str = "",
    framework_latest: bool = False,
    receipt_text: str = '[tool]\nrequirements = [{ name = "lisa-cli", extras = ["prompt"] }]\n',
) -> tuple[_CapturingConsole, list[list[str]], int | None, dict[str, str]]:
    """Invoke ``_update_command`` with stubbed environment seams.

    ``tool_dir`` stands in for ``sys.prefix``; a ``uv-receipt.toml`` is created
    there unless ``with_receipt`` is False (the "not a uv tool install" case).
    Returns ``(console, recorded_argv_list, exit_code, child_env)`` where exit
    code is ``None`` when the callback returned normally.
    """
    console = _CapturingConsole()
    invocations: list[list[str]] = []
    child_env: dict[str, str] = {}
    venv = packs_venv if packs_venv is not None else Path("/nonexistent/packs-venv")

    scratch: Path | None = None
    if tool_dir is None:
        scratch = Path(tempfile.mkdtemp())
        tool_dir = scratch / "tools" / "lisa-cli"

    class _Completed:
        pass

    def _fake_run(argv, **kwargs):
        invocations.append(list(argv))
        child_env.update(kwargs.get("env") or {})
        completed = _Completed()
        completed.returncode = returncode
        return completed

    receipt: Path | None = None
    if with_receipt:
        tool_dir.mkdir(parents=True, exist_ok=True)
        receipt = tool_dir / "uv-receipt.toml"
        receipt.write_text(receipt_text, encoding="utf-8")

    exit_code: int | None = None
    try:
        with (
            patch("runtime.paths.is_tool_install", return_value=tool_install),
            patch("runtime.paths.default_packs_venv", return_value=venv),
            patch.object(cli_main.shutil, "which", return_value=uv_path),
            patch.object(cli_update.subprocess, "run", new=_fake_run),
            patch.object(cli_main, "_UPDATE_IN_PLACE_BLOCKED", new=windows),
            patch.object(cli_main.sys, "prefix", new=str(tool_dir)),
            patch.object(cli_main, "console", new=console),
        ):
            try:
                cli_main._update_command(
                    force=force,
                    dry_run=dry_run,
                    version=version,
                    pin=pin,
                    channel=channel,
                    ref=ref,
                    latest=latest,
                    framework_version=framework_version,
                    framework_pin=framework_pin,
                    framework_channel=framework_channel,
                    framework_ref=framework_ref,
                    framework_latest=framework_latest,
                )
            except typer.Exit as exc:
                exit_code = exc.exit_code
    finally:
        if receipt is not None and scratch is not None:
            receipt.unlink(missing_ok=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    return console, invocations, exit_code, child_env


# ── Happy path ────────────────────────────────────────────────────────


def test_updates_with_uv_tool_upgrade() -> None:
    """The command upgrades via `uv tool upgrade`, never `install --upgrade`."""
    console, invocations, exit_code, _env = _run_update()

    assert invocations == [["/usr/local/bin/uv", "tool", "upgrade", "lisa-cli"]]
    assert exit_code is None
    assert "Restart lisa" in console.text
    assert "tool install" not in " ".join(" ".join(a) for a in invocations)


def test_force_adds_reinstall_flag() -> None:
    """--force maps to `--reinstall` so same-version local edits are picked up."""
    _, invocations, exit_code, _env = _run_update(force=True)

    assert invocations == [["/usr/local/bin/uv", "tool", "upgrade", "--reinstall", "lisa-cli"]]
    assert exit_code is None


# ── Refusal paths ────────────────────────────────────────────────────


def test_repo_checkout_points_at_git_pull() -> None:
    """Repo mode has no tool to upgrade; the fix is git pull + uv sync."""
    console, invocations, exit_code, _env = _run_update(tool_install=False, with_receipt=False)

    assert invocations == []
    assert exit_code == 2
    assert "git pull" in console.text


def test_missing_uv_prints_command_and_fails() -> None:
    """Without uv on PATH the operator gets the exact command, exit 1."""
    console, invocations, exit_code, _env = _run_update(uv_path=None)

    assert invocations == []
    assert exit_code == 1
    assert "tool" in console.text and "upgrade" in console.text and "lisa-cli" in console.text
    assert "not on PATH" in console.text


def test_windows_prints_command_instead_of_upgrading() -> None:
    """Windows locks the running .exe trampoline, so defer to the operator."""
    console, invocations, exit_code, _env = _run_update(windows=True)

    assert invocations == []
    assert exit_code == 1
    assert "tool" in console.text and "upgrade" in console.text and "lisa-cli" in console.text
    assert "Windows" in console.text


def test_windows_dry_run_still_reports_success() -> None:
    """--dry-run is always a successful report, even where upgrade is blocked."""
    console, invocations, exit_code, _env = _run_update(dry_run=True, windows=True)

    assert invocations == []
    assert exit_code == 0
    assert "Run:" in console.text


# ── --dry-run ────────────────────────────────────────────────────────


def test_dry_run_does_not_execute() -> None:
    """--dry-run prints the argv and exits 0 without shelling out."""
    console, invocations, exit_code, _env = _run_update(dry_run=True, force=True)

    assert invocations == []
    assert exit_code == 0
    assert "uv tool upgrade --reinstall lisa-cli" in console.text
    assert "Restart lisa" not in console.text


# ── Failure propagation ──────────────────────────────────────────────


def test_propagates_uv_failure_exit_code() -> None:
    """A failing uv run surfaces its exit code and no success message."""
    console, _, exit_code, _env = _run_update(returncode=7)

    assert exit_code == 7
    assert "failed" in console.text
    assert "Restart lisa" not in console.text


# ── Packs venv reminder ──────────────────────────────────────────────


def test_warns_when_packs_venv_exists(tmp_path: Path) -> None:
    """An existing packs venv is flagged as stale after a successful upgrade."""
    venv = tmp_path / "venv"
    venv.mkdir()

    console, _, exit_code, _env = _run_update(packs_venv=venv)

    assert exit_code is None
    assert "pack update --all" in console.text


def test_no_packs_venv_warning_without_a_venv() -> None:
    """A machine with no packs venv needs no reminder."""
    console, _, _, _env = _run_update()

    assert "pack update --all" not in console.text


# ── Locating the real install ────────────────────────────────────────


def test_points_uv_at_the_running_tool_dir(tmp_path: Path) -> None:
    """A custom-UV_TOOL_DIR install is upgraded where it actually lives.

    uv otherwise consults only UV_TOOL_DIR/default, misses the install, and
    answers with `uv tool install lisa-cli` — the receipt-clobbering path.
    """
    tool_dir = tmp_path / "custom-tools" / "lisa-cli"

    _, invocations, exit_code, child_env = _run_update(tool_dir=tool_dir)

    assert exit_code is None
    assert invocations  # uv actually ran
    assert child_env["UV_TOOL_DIR"] == str(tool_dir.parent)


def test_refuses_when_not_a_uv_tool_install(tmp_path: Path) -> None:
    """No receipt means uv does not own this interpreter, so do not guess."""
    console, invocations, exit_code, _ = _run_update(
        tool_dir=tmp_path / "plain-venv", with_receipt=False
    )

    assert invocations == []
    assert exit_code == 2
    assert "uv-receipt.toml" in console.text


def test_uv_tool_with_editable_framework_can_update(tmp_path: Path) -> None:
    console, invocations, exit_code, _ = _run_update(
        tool_dir=tmp_path / "tools" / "lisa-cli", tool_install=False
    )
    assert exit_code is None
    assert invocations == [["/usr/local/bin/uv", "tool", "upgrade", "lisa-cli"]]
    assert "Restart lisa" in console.text


def test_pin_persists_in_receipt_and_reports_recovery(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    console, invocations, exit_code, _ = _run_update(pin="0.3.0", tool_dir=tool_dir)
    receipt = tomllib.loads((tool_dir / "uv-receipt.toml").read_text(encoding="utf-8"))
    assert receipt["tool"]["requirements"][0]["specifier"] == "==0.3.0"
    assert invocations and exit_code is None
    assert "tracking pin 0.3.0" in console.text
    assert "installer with --latest" in console.text


def test_temporary_version_restores_unpinned_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    console, invocations, exit_code, _ = _run_update(version="0.3.0", tool_dir=tool_dir)
    receipt = tomllib.loads((tool_dir / "uv-receipt.toml").read_text(encoding="utf-8"))
    assert receipt["tool"]["requirements"] == [{"name": "lisa-cli", "extras": ["prompt"]}]
    assert invocations and exit_code is None
    assert "temporary" in console.text
    assert "next normal update" in console.text


def test_conflicting_versions_fail_before_invoking_uv(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    console, invocations, exit_code, _ = _run_update(
        version="0.3.0", pin="0.3.1", tool_dir=tool_dir
    )
    assert exit_code == 2
    assert not invocations
    assert "only one" in console.text


def test_invalid_receipt_encoding_fails_cleanly(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_bytes(b"\xff")
    console = _CapturingConsole()
    with (
        patch.object(cli_main, "console", console),
        pytest.raises(typer.Exit) as exc,
    ):
        cli_main._update_receipt_status(tmp_path, cli_update.Selection())
    assert exc.value.exit_code == 1
    assert "utf-8" in console.text


def test_dry_run_is_copyable_and_does_not_change_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    console, invocations, exit_code, _ = _run_update(channel="0.3", dry_run=True, tool_dir=tool_dir)
    receipt = tomllib.loads((tool_dir / "uv-receipt.toml").read_text(encoding="utf-8"))
    assert receipt["tool"]["requirements"] == [{"name": "lisa-cli", "extras": ["prompt"]}]
    assert exit_code == 0 and not invocations
    assert "Would select CLI channel 0.3" in console.text
    assert "--cli-mode channel" in console.text
    assert str(tool_dir) in console.text


def test_windows_selected_update_uses_external_helper(tmp_path: Path) -> None:
    tool_dir = tmp_path / "custom tools" / "lisa-cli"
    console, invocations, exit_code, _ = _run_update(
        windows=True, framework_ref="release/0.5", tool_dir=tool_dir
    )
    assert exit_code == 1 and not invocations
    assert "'--framework-mode' 'ref'" in console.text
    assert "'release/0.5'" in console.text
    assert str(tool_dir) in console.text


def test_plain_update_displays_existing_pin(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    receipt = '[tool]\nrequirements = [{name="lisa-cli", specifier="==0.3.0"}]\n'
    console, _, exit_code, _ = _run_update(tool_dir=tool_dir, receipt_text=receipt)
    assert exit_code is None
    assert "CLI tracking: ==0.3.0" in console.text
    assert "lisa update --latest" in console.text
    assert "--framework-latest" in console.text


def test_typer_parses_update_version_without_global_version_conflict(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    tool_dir.mkdir(parents=True)
    (tool_dir / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{name="lisa-cli"}]\n', encoding="utf-8"
    )
    console = _CapturingConsole()
    with (
        patch.object(cli_main.sys, "prefix", str(tool_dir)),
        patch.object(cli_main.shutil, "which", return_value="/usr/local/bin/uv"),
        patch.object(cli_main, "console", console),
    ):
        result = CliRunner().invoke(cli_main.app, ["update", "--version", "0.3.0", "--dry-run"])
    assert result.exit_code == 0
    assert "Would select CLI version 0.3.0" in console.text
