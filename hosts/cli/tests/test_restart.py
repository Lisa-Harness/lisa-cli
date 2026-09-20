"""Tests for the CLI host restart contract after pack lifecycle commands.

One-shot paths (``prompt``, ``pack``) exit with ``_RESTART_REQUIRED_EXIT``
on all platforms instead of re-execing — a fresh process re-run loads the
reloaded entry points, pack catalogs, and MCP manifests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from host.runtime import HostRuntime

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from host.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── _exit_restart_required ─────────────────────────────────────────────


def test_exit_restart_required_raises_typer_exit() -> None:
    """_exit_restart_required raises typer.Exit with the restart code."""
    import typer
    from hosts.cli import main as cli_main

    with pytest.raises(typer.Exit) as exc_info:
        cli_main._exit_restart_required()

    assert exc_info.value.exit_code == cli_main._RESTART_REQUIRED_EXIT


# ── prompt command applies the exit contract ───────────────────────────


def test_prompt_pack_mutation_exit_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt turn that flags restart_required exits with the restart code."""
    from unittest.mock import patch

    import typer
    from hosts.cli import main as cli_main

    runtime = _make_runtime(tmp_path)

    class _FakeTransport:
        approval_unresolved = False
        yield_unresolved = False

    async def _fake_run_prompt_with_teardown(_runtime, _message, **_kwargs):
        runtime.session.request_restart("pack enable: lisa-pack-cloudflare")
        return runtime.session.restart_required, _FakeTransport()

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)
    monkeypatch.setattr(cli_main, "_run_prompt_with_teardown", _fake_run_prompt_with_teardown)

    with patch.object(cli_main, "console"):
        with pytest.raises(typer.Exit) as exc_info:
            cli_main._prompt_command(
                message="continue",
                stdin=False,
                raw=False,
                log_level="",
                log="",
                no_log=False,
                data_dir=str(tmp_path),
                profile="",
                overlays_dir="",
                thread_dir="",
                no_thread=False,
                session_context="",
                timezone="",
                state_snapshot="",
                streaming="none",
                host_log="",
                quiet=False,
                approve="",
                deny="",
                remember="",
            )
    assert exc_info.value.exit_code == cli_main._RESTART_REQUIRED_EXIT


def test_runtime_restart_flag_roundtrip(tmp_path: Path) -> None:
    """SessionState.request_restart sets the flag + reason the exit contract reads."""
    runtime = _make_runtime(tmp_path)
    assert runtime.session.restart_required is False
    runtime.session.request_restart("pack install: lisa-pack-brave")
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack install: lisa-pack-brave"