"""Tests for the CLI ``--profile`` option and ``_resolve_profiled_data_dir``.

``--profile <name>`` (and ``LISA_PROFILE``) resolve the data dir to
``~/.lisa/profiles/<name>`` before the runtime is built, sharing the base
packs venv. ``--data-dir`` still wins. The tests stub the runtime builder
so no HostRuntime is actually constructed.
"""

from __future__ import annotations

# pylint: disable=protected-access
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

# ── _resolve_profiled_data_dir ────────────────────────────────────────


def test_explicit_data_dir_wins_over_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """--data-dir takes precedence; the profile is ignored."""
    from hosts.cli import main as cli_main

    monkeypatch.setenv("LISA_PROFILE", "taxes")
    assert cli_main._resolve_profiled_data_dir("/tmp/explicit", "survey") == "/tmp/explicit"


def test_profile_resolves_to_profiles_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--profile <name> → <base>/profiles/<name> via host.config."""
    from hosts.cli import main as cli_main

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PROFILE", raising=False)
    base = tmp_path / ".lisa"
    with patch("host.config.default_data_dir", return_value=base):
        got = cli_main._resolve_profiled_data_dir("", "taxes")
    assert Path(got) == (base / "profiles" / "taxes").resolve()


def test_no_args_no_profile_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither input, return '' so the framework env chain decides."""
    from hosts.cli import main as cli_main

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PROFILE", raising=False)
    assert cli_main._resolve_profiled_data_dir("", "") == ""


def test_invalid_profile_name_raises() -> None:
    """An unsafe profile name surfaces as ValueError (rendered as a panel)."""
    from hosts.cli import main as cli_main

    with pytest.raises(ValueError, match=r"[Ii]nvalid profile name"):
        cli_main._resolve_profiled_data_dir("", "../escape")


# ── command wiring: --profile reaches the runtime builder ─────────────


def _capture_runtime_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Patch _initialize_runtime_or_exit to capture its kwargs."""

    from host.testing import build_test_runtime

    from hosts.cli import main as cli_main

    runtime = build_test_runtime(tmp_path, models_yaml="")
    captured: dict = {}

    def _fake_initialize(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_initialize)
    monkeypatch.setattr(cli_main, "console", Console(quiet=True))
    return captured


def test_models_command_passes_profile_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The --profile kwarg is forwarded to the runtime builder."""
    from hosts.cli import main as cli_main

    captured = _capture_runtime_kwargs(monkeypatch, tmp_path)
    fake_handler = _fake_handler()
    with (
        patch.object(cli_main, "render_models", side_effect=RuntimeError("no manifest")),
        patch("host.commands.handle_models_command", new=fake_handler),
    ):
        cli_main._models_command(
            action="discover",
            force=False,
            log_level="",
            log="",
            no_log=False,
            data_dir="",
            profile="taxes",
            thread_dir="",
            no_thread=False,
            session_context="",
            session_context_refresh=False,
            timezone="",
            state_snapshot="",
            streaming="none",
            host_log="",
        )
    assert captured.get("profile") == "taxes"


def _fake_handler():
    async def _handler(_runtime, stripped, message):
        return "ok"

    return _handler


def test_intake_command_resolves_profiled_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """intake resolves --profile to <base>/profiles/<name> before running."""
    import sys
    import types

    from hosts.cli import main as cli_main

    base = tmp_path / ".lisa"
    captured: dict = {}

    async def _fake_intake(*, data_dir):
        captured["data_dir"] = data_dir

        class _State:
            # RUF012: simple mutable defaults on a throwaway stub class
            signal_snapshots: list = []  # noqa: RUF012
            active_alerts: list = []  # noqa: RUF012

        return _State()

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PROFILE", raising=False)
    mod = types.ModuleType("host.intake")
    mod.run_intake = _fake_intake
    monkeypatch.setitem(sys.modules, "host.intake", mod)
    with (
        patch("host.config.default_data_dir", return_value=base),
        patch.object(cli_main, "console", Console(quiet=True)),
    ):
        cli_main._intake_command(
            data_dir="",
            profile="taxes",
            log_level="",
            host_log="",
        )
    assert Path(captured["data_dir"]) == (base / "profiles" / "taxes").resolve()


# ── once-unknown pre-parse: --profile value is skipped ────────────────


def test_profile_value_not_mistaken_for_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--profile taxes → 'taxes' is a value, not a rerouted positional."""
    from hosts.cli import main as cli_main

    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_once(["--profile", "taxes", "verify"])
    assert out == ["--profile", "taxes", "verify"]