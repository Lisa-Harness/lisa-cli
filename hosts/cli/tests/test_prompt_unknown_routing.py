"""Tests for the prompt-unknown lazy routing pre-parse.

When ``LISA_PROMPT_UNKNOWN`` is set, an unrecognized first positional is
rerouted to the ``prompt`` subcommand so users can run `lisa continue` or
`lisa "do something else"` without typing ``prompt``. Known subcommands
are never rerouted. The group callback takes no value-consuming options
(runtime options live on each subcommand), so every ``--flag`` token is
standalone when locating the subcommand position.
"""

from __future__ import annotations

# pylint: disable=protected-access
import pytest

from hosts.cli import main as cli_main

# ── _maybe_route_to_prompt ──────────────────────────────────────────────


def test_no_env_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env var, argv is returned unchanged."""
    monkeypatch.delenv(cli_main._PROMPT_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_prompt(["verify", "--full"]) == ["verify", "--full"]


def test_env_unknown_positional_routed_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized first positional becomes the `prompt` argument."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt(["continue"]) == ["prompt", "continue"]


def test_known_subcommand_not_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered subcommand is left in place even when routing is on."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    for known in (
        "verify",
        "pack",
        "prompt",
        "models",
        "intake",
        "policy",
        "approve",
        "deny",
        "help",
    ):
        assert cli_main._maybe_route_to_prompt([known]) == [known]


def test_unknown_flag_token_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no group value-options, a stray unknown flag is skipped as one
    token: the first positional after it is what routes (and `debug` in
    `--log-level debug` now reads as the positional it is — runtime
    options belong after the subcommand)."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["--log-level", "debug"])
    assert out == ["--log-level", "prompt", "debug"]


def test_quoted_multiword_message_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quoted message stays a single token after routing."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["do something else"])
    assert out == ["prompt", "do something else"]


def test_equals_form_flag_token_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--data-dir=foo` is one token; the next positional is still found."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["--data-dir=/tmp", "continue"])
    assert out == ["--data-dir=/tmp", "prompt", "continue"]


def test_double_dash_terminates_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `--`, the next token is treated as the positional."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["--", "continue"])
    assert out == ["--", "prompt", "continue"]


def test_no_positional_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare `lisa` or only-options: no positional, no routing."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt([]) == []
    assert cli_main._maybe_route_to_prompt(["--version"]) == ["--version"]


# ── Prompt-only flag hoisting ──────────────────────────────────────────


def test_raw_flag_hoisted_after_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lisa -r "fix"` parses as `lisa prompt -r "fix"`."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt(["-r", "fix it"]) == [
        "prompt",
        "-r",
        "fix it",
    ]


def test_raw_and_quiet_hoisted_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple prompt-only flags keep their order after the injected prompt."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt(["-q", "--raw", "summarize"]) == [
        "prompt",
        "-q",
        "--raw",
        "summarize",
    ]


def test_long_raw_flag_hoisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The long form --raw hoists the same as -r."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt(["--raw", "fix"]) == ["prompt", "--raw", "fix"]


def test_prompt_flag_after_positional_not_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flags after the positional are already prompt options; head is empty."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_prompt(["fix it", "-r"]) == ["prompt", "fix it", "-r"]


def test_prompt_only_flag_without_routing_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without routing active, argv is untouched even with -r present."""
    monkeypatch.delenv(cli_main._PROMPT_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_prompt(["-r", "fix it"]) == ["-r", "fix it"]


def test_stray_flag_and_prompt_flags_coexist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown leading flag stays put; only prompt-only flags hoist after
    `prompt` (group value-options no longer exist, so the flag cannot swallow
    the next token)."""
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_prompt(["--log-level", "-r", "fix it"])
    assert out == ["--log-level", "prompt", "-r", "fix it"]


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_falsy_env_values_disable_routing(
    val: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsy env values do not enable routing."""
    if val:
        monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, val)
    else:
        monkeypatch.delenv(cli_main._PROMPT_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_prompt(["continue"]) == ["continue"]


# ── _prompt_unknown_active / _first_positional_index unit checks ────────


def test_prompt_unknown_active_env_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cli_main._PROMPT_UNKNOWN_ENV, "yes")
    assert cli_main._prompt_unknown_active() is True


def test_first_positional_index_basic() -> None:
    assert cli_main._first_positional_index(["verify"]) == 0
    assert cli_main._first_positional_index(["--log-level", "verify"]) == 1
    assert (
        cli_main._first_positional_index(["--data-dir=/tmp", "verify"]) == 1
    )
    assert cli_main._first_positional_index(["--no-log", "verify"]) == 1
    assert cli_main._first_positional_index([]) == -1
    assert cli_main._first_positional_index(["--", "continue"]) == 1
    assert cli_main._first_positional_index(["--"]) == -1