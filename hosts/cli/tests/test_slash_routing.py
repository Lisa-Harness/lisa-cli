"""Tests for the ``/``-prefix slash-command pre-parse routing.

A first positional starting with "/" is a runtime slash command, not a
prompt message. That is unconditional — it applies regardless of whether
LISA_PROMPT_UNKNOWN routing is on — and it routes to ``run``, not
``prompt``. Prompt-only flags that appear before the positional have no
meaning on ``run`` and are dropped.

Mirrors test_prompt_unknown_routing.py style: pure argv-list assertions,
no runtime, no Typer callback invocation.
"""

from __future__ import annotations

# pylint: disable=protected-access
import pytest

from hosts.cli import main as cli_main


@pytest.fixture(autouse=True)
def _no_prompt_unknown(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Remove the prompt-unknown env var so tests are deterministic."""
    monkeypatch.delenv(cli_main._PROMPT_UNKNOWN_ENV, raising=False)


def test_slash_command_routed_to_run_without_prompt_unknown() -> None:
    """`/policy mode trust` becomes `run /policy mode trust`."""
    out = cli_main._maybe_route_to_prompt(["/policy", "mode", "trust"])
    assert out == ["run", "/policy", "mode", "trust"]


def test_slash_command_routed_when_prompt_unknown_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The / hook fires before the prompt-unknown gate; both paths agree."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["/models", "show"])
    assert out == ["run", "/models", "show"]


def test_slash_command_first_after_flags() -> None:
    """Flags before the slash positional stay; the slash routes to run.

    The CLI's standalone-flag contract treats value-less flags and
    `--flag=value` inline-value flags as single tokens. `--log-level=debug`
    keeps the option pair atomic so the next token is the first real
    positional; split-form `--log-level debug` would see `debug` as the
    (only) positional and route *it* — that behavior is exercised in
    test_prompt_unknown_routing.py.
    """
    out = cli_main._maybe_route_to_prompt(["--log-level=debug", "/modes"])
    assert out == ["--log-level=debug", "run", "/modes"]


def test_prompt_only_flags_dropped_before_run() -> None:
    """Prompt-only flags ahead of a slash command are dropped (no hoist)."""
    out = cli_main._maybe_route_to_prompt(["-r", "-q", "/settings"])
    assert out == ["run", "/settings"]


def test_bare_slash_token_passes_through() -> None:
    """The bare token "/" is not routed (guards malformed input)."""
    out = cli_main._maybe_route_to_prompt(["/"])
    assert out == ["/"]


def test_bare_slash_then_command_passes_through() -> None:
    """'/' alone is still not a command; nothing routes."""
    out = cli_main._maybe_route_to_prompt(["/", "policy"])
    assert out == ["/", "policy"]


def test_double_dash_separator_keeps_run_routing() -> None:
    """After `--`, the first positional starts with '/'; it still routes."""
    out = cli_main._maybe_route_to_prompt(["--", "/settings"])
    assert out == ["--", "run", "/settings"]


def test_known_subcommand_not_routed_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered subcommand still wins — never rewritten to run."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    for known in (
        "verify",
        "pack",
        "prompt",
        "models",
        "intake",
        "modes",
        "last",
        "tools",
        "operations",
        "help",
        "run",
    ):
        assert cli_main._maybe_route_to_prompt([known]) == [known]
