"""Tests for the CLI ``pack update`` action.

The CLI ``lisa pack update`` command mirrors the in-REPL ``/pack update``
command by delegating to :func:`runtime.commands.run_pack_update` (single
pack) or :func:`runtime.commands.run_pack_update_all` (``--all``). These
tests verify the dispatch wiring, argument validation, exit codes, and
the restart-flag tolerance (success exits 0 regardless) without paying
the full ``_initialize_runtime`` cost: we stub the runtime builder and
the core framework functions, then invoke the Typer callback directly.
"""

from __future__ import annotations

# pylint: disable=protected-access
from pathlib import Path
from typing import Any
from unittest.mock import patch

import typer
from runtime.runtime import HostRuntime

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


class _CapturingConsole:
    """Records the plain text of every ``console.print`` positional arg."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            if isinstance(arg, str):
                self.lines.append(arg)


def _invoke_pack_update(
    tmp_path: Path,
    *,
    action: str = "update",
    target: str = "",
    all_packs: bool = False,
    runtime: HostRuntime | None = None,
    update_result: tuple[bool, str] | None = None,
    update_all_result: tuple[bool, str] | None = None,
) -> tuple[_CapturingConsole, dict[str, Any], int | None]:
    """Call the CLI _pack_command with a stubbed runtime + core handlers.

    Success/failure ``typer.Exit`` signals are caught and returned as the
    third tuple element (``None`` when the callback returned normally).
    Returns ``(capturing_console, captured, exit_code)`` where ``captured``
    records which framework function ran and with what arguments.
    """
    from hosts.cli import main as cli_main

    if runtime is None:
        runtime = _make_runtime(tmp_path)

    console = _CapturingConsole()
    captured: dict[str, Any] = {}
    exit_code: int | None = None

    async def _fake_update(_runtime, package_name: str):
        captured["update"] = package_name
        return update_result or (True, f"updated {package_name}")

    async def _fake_update_all(_runtime):
        captured["update_all"] = True
        return update_all_result or (True, "updated all")

    async def _fake_install(_runtime, spec: str):
        captured["install"] = spec
        return (True, f"installed {spec}")

    with (
        patch.object(
            cli_main,
            "_initialize_runtime_or_exit",
            return_value=runtime,
        ),
        patch(
            "runtime.commands.run_pack_update",
            new=_fake_update,
        ),
        patch(
            "runtime.commands.run_pack_update_all",
            new=_fake_update_all,
        ),
        patch(
            "runtime.commands.run_pack_install",
            new=_fake_install,
        ),
        patch.object(cli_main, "console", new=console),
    ):
        try:
            cli_main._pack_command(
                action=action,
                target=target,
                all_packs=all_packs,
                log_level="",
                log="",
                no_log=False,
                data_dir=str(tmp_path),
                thread_dir="",
                no_thread=False,
                session_context="",
                timezone="",
                state_snapshot="",
                streaming="none",
                host_log="",
            )
        except typer.Exit as exc:
            exit_code = exc.exit_code
    return console, captured, exit_code


# ── dispatch wiring ───────────────────────────────────────────────────


def test_pack_update_single_delegates_to_core(tmp_path: Path) -> None:
    """'update <pack>' delegates to run_pack_update with the target name."""
    console, captured, exit_code = _invoke_pack_update(
        tmp_path,
        target="lisa-pack-cloudflare",
        update_result=(True, "Updated 'lisa-pack-cloudflare'."),
    )
    assert captured["update"] == "lisa-pack-cloudflare"
    assert "update_all" not in captured
    assert any("Updated 'lisa-pack-cloudflare'." in line for line in console.lines)
    assert exit_code == 0


def test_pack_update_all_delegates_to_core(tmp_path: Path) -> None:
    """'update --all' delegates to run_pack_update_all, ignoring any target."""
    console, captured, exit_code = _invoke_pack_update(
        tmp_path,
        target="",
        all_packs=True,
        update_all_result=(True, "Updated 2 pack(s)."),
    )
    assert captured.get("update_all") is True
    assert "update" not in captured
    assert any("Updated 2 pack(s)." in line for line in console.lines)
    assert exit_code == 0


def test_pack_update_target_takes_precedence_over_all_flag(
    tmp_path: Path,
) -> None:
    """With both a target and --all, --all wins (batch update)."""
    _console, captured, exit_code = _invoke_pack_update(
        tmp_path,
        target="lisa-pack-cloudflare",
        all_packs=True,
        update_all_result=(True, "Updated all."),
    )
    assert captured.get("update_all") is True
    assert "update" not in captured
    assert exit_code == 0


# ── argument validation ───────────────────────────────────────────────


def test_pack_update_without_target_or_all_shows_usage(tmp_path: Path) -> None:
    """'update' with neither a target nor --all prints usage and exits 2."""
    console, _captured, exit_code = _invoke_pack_update(tmp_path, target="", all_packs=False)
    assert exit_code == 2
    assert any("Usage: lisa pack update" in line for line in console.lines)


def test_pack_update_whitespace_target_shows_usage(tmp_path: Path) -> None:
    """A whitespace-only target is treated like a missing target."""
    _console, _captured, exit_code = _invoke_pack_update(tmp_path, target="   ", all_packs=False)
    assert exit_code == 2


# ── failure paths ─────────────────────────────────────────────────────


def test_pack_update_failure_prints_details_and_exits_1(
    tmp_path: Path,
) -> None:
    """A failed single-pack update prints details in red and exits 1."""
    console, _captured, exit_code = _invoke_pack_update(
        tmp_path,
        target="lisa-pack-cloudflare",
        update_result=(False, "No lock entry for 'lisa-pack-cloudflare'."),
    )
    assert exit_code == 1
    assert any("No lock entry" in line for line in console.lines)


def test_pack_update_all_failure_prints_details_and_exits_1(
    tmp_path: Path,
) -> None:
    """A failed update --all prints details in red and exits 1."""
    _console, _captured, exit_code = _invoke_pack_update(
        tmp_path,
        all_packs=True,
        update_all_result=(False, "boom"),
    )
    assert exit_code == 1


def test_pack_update_unknown_action_exits_2(tmp_path: Path) -> None:
    """An unrecognized action name exits 2 with a supported-actions hint."""
    console, _captured, exit_code = _invoke_pack_update(tmp_path, action="upgrade", target="x")
    assert exit_code == 2
    assert any("Unknown action" in line for line in console.lines)


# ── restart-flag tolerance ────────────────────────────────────────────


def test_pack_update_restart_flag_exits_zero(tmp_path: Path) -> None:
    """A successful update still exits 0 even when restart_required is set.

    The framework reloads pack state on its next boot, so the CLI does not
    demand a manual restart.
    """
    runtime = _make_runtime(tmp_path)
    runtime.session.request_restart("pack update: lisa-pack-cloudflare")

    _console, captured, exit_code = _invoke_pack_update(
        tmp_path,
        target="lisa-pack-cloudflare",
        runtime=runtime,
        update_result=(True, "Updated 'lisa-pack-cloudflare'."),
    )
    assert exit_code == 0
    assert captured["update"] == "lisa-pack-cloudflare"


def test_pack_update_all_restart_flag_exits_zero(tmp_path: Path) -> None:
    """After update --all with restart_required set, the CLI exits 0."""
    runtime = _make_runtime(tmp_path)
    runtime.session.request_restart("pack update-all: 2 pack(s)")

    _console, _captured, exit_code = _invoke_pack_update(
        tmp_path,
        all_packs=True,
        runtime=runtime,
        update_all_result=(True, "Updated 2 pack(s)."),
    )
    assert exit_code == 0


def test_pack_update_all_noop_exits_zero(
    tmp_path: Path,
) -> None:
    """run_pack_update_all's nothing-to-update result exits 0 cleanly.

    The framework returns (True, ...) with restart_required unset when
    every pack is bundled/placeholder; the CLI exits cleanly.
    """
    runtime = _make_runtime(tmp_path)
    _console, _captured, exit_code = _invoke_pack_update(
        tmp_path,
        all_packs=True,
        runtime=runtime,
        update_all_result=(
            True,
            "No runtime-installed packs with recorded sources to update.",
        ),
    )
    assert exit_code == 0
    assert runtime.session.restart_required is False


# ── no cross-action leakage ───────────────────────────────────────────


def test_pack_update_does_not_call_install(tmp_path: Path) -> None:
    """The update action must never fall through to run_pack_install."""
    _console, captured, _exit_code = _invoke_pack_update(
        tmp_path,
        target="lisa-pack-cloudflare",
        update_result=(True, "Updated."),
    )
    assert "install" not in captured
