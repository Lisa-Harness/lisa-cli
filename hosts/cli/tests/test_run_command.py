"""Tests for the generic ``run`` slash-command escape hatch.

``lisa run '/slash args'`` normalizes whatever the operator typed to a
single slash-command string and forwards it to ``_run_prompt_host_command`` —
the same one-shot host dispatch path used by the curated ``approve`` /
``deny`` / ``policy`` shims — so runtime slash handlers remain the single
source of truth.

These tests stub ``_run_prompt_host_command`` and record the dispatched string;
no runtime is built, no MCP subprocess is spawned, and no model is
called.
"""

from __future__ import annotations

# pylint: disable=protected-access
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from hosts.cli import main as cli_main

runner = CliRunner()


@pytest.fixture
def _captured(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace the host dispatch with a recorder and return its record."""
    captured: dict[str, Any] = {"commands": []}

    def _fake_run(command: str, **_kwargs: Any) -> None:
        captured["commands"].append(command)

    monkeypatch.setattr(cli_main, "_run_prompt_host_command", _fake_run)
    return captured


def test_run_requires_an_argument(_captured: Any) -> None:
    """No argument is a usage error, with no dispatch."""
    result = runner.invoke(cli_main.app, ["run"])
    assert result.exit_code == 2
    assert "No slash command given" in result.output
    assert not _captured["commands"]


def test_run_quoted_slash_command_verbatim(_captured: Any) -> None:
    """A quoted slash string is preserved exactly as typed."""
    runner.invoke(cli_main.app, ["run", "/policy mode trust"])
    assert _captured["commands"] == ["/policy mode trust"]


def test_run_prepends_leading_slash(_captured: Any) -> None:
    """Bare slash-command words get a leading '/'; plain words re-quote to themselves."""
    runner.invoke(cli_main.app, ["run", "models", "show"])
    # shlex.quote('show') == 'show' (no shell-special chars), so the
    # re-joined form is unchanged.
    assert _captured["commands"] == ["/models show"]


def test_run_multi_token_requotes(_captured: Any) -> None:
    """Tokens with shell-special chars are quoted; `--` separates args from Click options."""
    runner.invoke(
        cli_main.app,
        ["run", "policy", "rule", "add", "allow", "--", "--prefix", "git status"],
    )
    assert _captured["commands"] == ["/policy rule add allow --prefix 'git status'"]


def test_run_single_token_with_leading_slash_unchanged(_captured: Any) -> None:
    """An already-prefixed string is passed through verbatim."""
    runner.invoke(cli_main.app, ["run", "/mode chat"])
    assert _captured["commands"] == ["/mode chat"]


def test_run_forwards_shared_runtime_options(_captured: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared runtime flags are forwarded alongside the command string."""
    kwargs_seen: dict[str, Any] = {}

    def _fake_run(command: str, **kwargs: Any) -> None:
        kwargs_seen["cmd"] = command
        kwargs_seen.update(kwargs)

    monkeypatch.setattr(cli_main, "_run_prompt_host_command", _fake_run)
    runner.invoke(
        cli_main.app,
        [
            "run",
            "--data-dir",
            "/tmp/lisa-data",
            "--profile",
            "work",
            "--log-level",
            "debug",
            "/settings",
        ],
    )
    assert kwargs_seen["cmd"] == "/settings"
    assert kwargs_seen["data_dir"] == "/tmp/lisa-data"
    assert kwargs_seen["profile"] == "work"
    assert kwargs_seen["log_level"] == "debug"


def test_run_embedded_quote_in_single_token_preserved(_captured: Any) -> None:
    """Quoted single-token user input is kept exactly as typed."""
    runner.invoke(cli_main.app, ["run", "/models discover --force"])
    assert _captured["commands"] == ["/models discover --force"]


def test_run_with_double_dash_separator(_captured: Any) -> None:
    """A `--` separator lets arguments begin with a dash and still route."""
    runner.invoke(cli_main.app, ["run", "--", "--help"])
    assert _captured["commands"] == ["/--help"]


def test_run_silently_handles_restart_required(_captured: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart-required response surfaces the console notice and exit 69."""
    monkeypatch.setattr(
        cli_main,
        "_run_prompt_host_command",
        lambda _cmd, **_kw: cli_main._exit_restart_required(),
    )
    with patch.object(cli_main, "console", new=Console(quiet=True, record=True)):
        result = runner.invoke(cli_main.app, ["run", "/pack install foo"])
    assert result.exit_code == cli_main._RESTART_REQUIRED_EXIT
