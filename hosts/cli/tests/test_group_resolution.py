"""Tests for the custom Typer group's command resolution.

Click owns the argv grammar via ``_LisaGroup``: no pre-Typer argv rewrites
exist. These tests exercise the group's ``get_command`` /
``resolve_command`` contract through the real Typer app with CliRunner:

- A first positional starting with ``/`` is a usage error pointing at the
  named command — the slash pre-parse routing to ``run`` is gone.
- With ``LISA_PROMPT_UNKNOWN`` on, an unknown first positional resolves to
  the ``prompt`` command (tokens become the one-shot message).
- Group-level hoist flags (``-r``, ``-q``) parse and ride onto the injected
  prompt turn.
- Pack-registered host commands (``stalls``, ``retry``, ...) resolve as
  dynamically generated commands that dispatch to the pack handler.
"""

from __future__ import annotations

# pylint: disable=protected-access
from typing import Any

import pytest
from typer.testing import CliRunner

from hosts.cli import main as cli_main

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_prompt_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the prompt-unknown env var so tests are deterministic."""
    monkeypatch.delenv(cli_main._PROMPT_UNKNOWN_ENV, raising=False)


@pytest.fixture
def _no_pack_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the pack-command registry so routing is deterministic."""
    monkeypatch.setattr(cli_main, "_pack_command_specs", {})


# ── Slash-prefix usage errors ─────────────────────────────────────────


def test_slash_command_is_usage_error_pointing_at_named_command() -> None:
    """`lisa /settings` no longer routes to run; it points at `lisa settings`."""
    result = runner.invoke(cli_main.app, ["/settings"])
    assert result.exit_code == 2
    assert "Slash commands are not subcommands" in result.output
    assert "lisa settings" in result.output


def test_slash_command_unknown_still_names_the_bare_form() -> None:
    result = runner.invoke(cli_main.app, ["/nosuchthing"])
    assert result.exit_code == 2
    assert "Slash commands are not subcommands" in result.output


def test_bare_slash_token_is_usage_error() -> None:
    """The bare token "/" is malformed input, not a command."""
    result = runner.invoke(cli_main.app, ["/"])
    assert result.exit_code == 2


# ── prompt-unknown lazy routing ───────────────────────────────────────


def test_unknown_command_errors_without_env() -> None:
    """Without LISA_PROMPT_UNKNOWN, an unknown command is a normal error."""
    result = runner.invoke(cli_main.app, ["continue"])
    assert result.exit_code == 2


def test_known_subcommand_never_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registered subcommands resolve normally even with routing on."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    for known in ("verify", "pack", "prompt", "models", "policy", "approve", "help"):
        result = runner.invoke(cli_main.app, [known, "--help"])
        assert result.exit_code == 0, known


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_falsy_env_values_disable_routing(
    val: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if val:
        monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, val)
    result = runner.invoke(cli_main.app, ["continue"])
    assert result.exit_code == 2


def test_unknown_routes_to_prompt_with_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """With routing on, an unknown positional becomes the prompt message."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    import io
    import sys

    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    import runtime.handle_message as hm

    seen: dict[str, Any] = {}

    async def _fake(_rt, message, _transport):
        seen["message"] = message
        return "ok"

    monkeypatch.setattr(hm, "handle_message", _fake)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    result = runner.invoke(cli_main.app, ["continue the task"])
    assert result.exit_code == 0
    assert seen["message"] == "continue the task"


def test_hoisted_raw_flag_parses_at_group_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """`lisa -r "fix it"` parses -r at group level and rides onto the turn."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    import io
    import sys

    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    import runtime.handle_message as hm

    seen: dict[str, Any] = {}

    async def _fake(_rt, message, _transport):
        seen["message"] = message
        return "ok"

    monkeypatch.setattr(hm, "handle_message", _fake)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    result = runner.invoke(cli_main.app, ["-r", "fix it"])
    assert result.exit_code == 0
    # The raw output contract is prepended: -r was honored on the turn.
    assert seen["message"].startswith("OUTPUT CONTRACT (raw mode)")
    assert "fix it" in seen["message"]


# ── Dynamic pack subcommands ──────────────────────────────────────────


class _FakeSpec:
    def __init__(self, prefix: str, help_text: str = "") -> None:
        self.prefix = prefix
        self.help_text = help_text


class _FakeEntry:
    def __init__(self, specs: list[_FakeSpec]) -> None:
        self.specs = specs
        self.pack_name = "fake-pack"


@pytest.fixture
def _recovery_like_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry shaped like the recovery pack (shared leading verbs)."""
    entry = _FakeEntry(
        [_FakeSpec("/retry", "Retry one item"), _FakeSpec("/retry-safe", "Retry safe items")]
    )
    monkeypatch.setattr(
        cli_main,
        "_pack_command_specs",
        {
            "stalls": _FakeEntry([_FakeSpec("/stalls", "List stalled items")]),
            "retry": entry,
            "retry-safe": entry,
        },
    )


def test_pack_command_dispatches_to_host_handler(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    """`lisa stalls` resolves as a real command and dispatches /stalls."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli_main,
        "_run_host_command",
        lambda command, **_kw: captured.setdefault("command", command),
    )
    result = runner.invoke(cli_main.app, ["stalls"])
    assert result.exit_code == 0
    assert captured["command"] == "/stalls"


def test_pack_command_with_args_dispatches_verbatim(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli_main,
        "_run_host_command",
        lambda command, **_kw: captured.setdefault("command", command),
    )
    result = runner.invoke(cli_main.app, ["retry", "abc123"])
    assert result.exit_code == 0
    assert captured["command"] == "/retry abc123"


def test_pack_command_longest_prefix_not_shadowed(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    """`lisa retry-safe` dispatches /retry-safe, not the shorter /retry."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli_main,
        "_run_host_command",
        lambda command, **_kw: captured.setdefault("command", command),
    )
    result = runner.invoke(cli_main.app, ["retry-safe"])
    assert result.exit_code == 0
    assert captured["command"] == "/retry-safe"


def test_pack_command_shows_in_help(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    result = runner.invoke(cli_main.app, ["--help"])
    assert result.exit_code == 0
    assert "stalls" in result.output


def test_pack_command_help_text_from_spec(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    result = runner.invoke(cli_main.app, ["stalls", "--help"])
    # Rich help output for the generated command carries the spec help text.
    assert "List stalled items" in result.output


def test_pack_command_slash_form_points_at_bare(
    monkeypatch: pytest.MonkeyPatch, _recovery_like_pack: None
) -> None:
    """`lisa /stalls` points at `lisa stalls` (not at run)."""
    result = runner.invoke(cli_main.app, ["/stalls"])
    assert result.exit_code == 2
    assert "lisa stalls" in result.output