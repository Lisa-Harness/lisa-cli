"""Tests for the post-mutation exit contract after pack lifecycle commands.

One-shot paths (``prompt``, ``pack``) exit 0 on success: the runtime
reloads pack state (entry points, catalogs, MCP manifests) on its next
boot, so a manual restart demand is pointless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.runtime import HostRuntime

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── prompt command ignores the restart flag ────────────────────────────


def test_prompt_pack_mutation_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt turn that flags restart_required still exits 0."""
    from unittest.mock import patch

    import typer

    from hosts.cli import main as cli_main

    runtime = _make_runtime(tmp_path)

    class _FakeTransport:
        approval_unresolved = False
        yield_unresolved = False

    async def _fake_run_prompt_with_teardown(_runtime, _message, **_kwargs):
        runtime.session.request_restart("pack enable: lisa-pack-cloudflare")
        return _FakeTransport()

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)
    monkeypatch.setattr(cli_main, "_run_prompt_with_teardown", _fake_run_prompt_with_teardown)

    with (
        patch.object(cli_main, "console"),
        pytest.raises(typer.Exit) as exc_info,
    ):
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
    assert exc_info.value.exit_code == 0


# ── SessionState restart flag round-trip ──────────────────────────────


def test_runtime_restart_flag_roundtrip(tmp_path: Path) -> None:
    """SessionState.request_restart sets the flag + reason that runtime code reads."""
    runtime = _make_runtime(tmp_path)
    assert runtime.session.restart_required is False
    runtime.session.request_restart("pack install: lisa-pack-brave")
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack install: lisa-pack-brave"
