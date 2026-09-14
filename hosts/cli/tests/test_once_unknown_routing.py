"""Tests for the once-unknown lazy routing pre-parse.

When ``LISA_ONCE_UNKNOWN`` is set (or ``--once-unknown`` is passed), an
unrecognized first positional is rerouted to the ``once`` subcommand so
users can run ``lisa continue`` or ``lisa "do something else"`` without
typing ``once``. Known subcommands are never rerouted, and group-level
options (and their values) are skipped when locating the subcommand
position so e.g. ``--log-level debug`` does not mistake ``debug`` for
the subcommand.
"""

from __future__ import annotations

# pylint: disable=protected-access
import pytest

from hosts.cli import main as cli_main

# ── _maybe_route_to_once ──────────────────────────────────────────────


def test_no_env_no_flag_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without env or flag, argv is returned unchanged."""
    monkeypatch.delenv(cli_main._ONCE_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_once(["verify", "--full"]) == ["verify", "--full"]


def test_env_unknown_positional_routed_to_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized first positional becomes the `once` argument."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(["continue"]) == ["once", "continue"]


def test_flag_unknown_positional_routed_to_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The --once-unknown flag enables routing and is stripped from argv."""
    monkeypatch.delenv(cli_main._ONCE_UNKNOWN_ENV, raising=False)
    out = cli_main._maybe_route_to_once(["--once-unknown", "continue"])
    assert out == ["once", "continue"]


def test_known_subcommand_not_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered subcommand is left in place even when routing is on."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    for known in ("verify", "pack", "once", "cli", "models", "intake", "help"):
        assert cli_main._maybe_route_to_once([known]) == [known]


def test_group_option_value_not_mistaken_for_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group-level option value (e.g. `debug`) is skipped, not rerouted."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    # --log-level debug is a group option + value; `verify` is the real
    # positional and is a known command, so no routing happens.
    assert cli_main._maybe_route_to_once(["--log-level", "debug", "verify"]) == [
        "--log-level",
        "debug",
        "verify",
    ]


def test_group_option_then_unknown_positional_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group options before an unknown positional are preserved, then routed."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_once(["--log-level", "debug", "continue"])
    assert out == ["--log-level", "debug", "once", "continue"]


def test_quoted_multiword_message_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quoted message stays a single token after routing."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_once(["do something else"])
    assert out == ["once", "do something else"]


def test_equals_form_group_option_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--data-dir=foo` is one token; the next positional is still found."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_once(["--data-dir=/tmp", "continue"])
    assert out == ["--data-dir=/tmp", "once", "continue"]


def test_double_dash_terminates_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `--`, the next token is treated as the positional."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    out = cli_main._maybe_route_to_once(["--", "continue"])
    assert out == ["--", "once", "continue"]


def test_no_positional_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare `lisa` or only-options: no positional, no routing."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once([]) == []
    assert cli_main._maybe_route_to_once(["--version"]) == ["--version"]


# ── Once-only flag hoisting ──────────────────────────────────────────


def test_raw_flag_hoisted_after_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lisa -r "fix"` parses as `lisa once -r "fix"`."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(["-r", "fix it"]) == [
        "once",
        "-r",
        "fix it",
    ]


def test_raw_and_quiet_hoisted_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple once-only flags keep their order after the injected once."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(["-q", "--raw", "summarize"]) == [
        "once",
        "-q",
        "--raw",
        "summarize",
    ]


def test_long_raw_flag_hoisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The long form --raw hoists the same as -r."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(["--raw", "fix"]) == ["once", "--raw", "fix"]


def test_once_flag_after_positional_not_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flags after the positional are already once options; head is empty."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(["fix it", "-r"]) == ["once", "fix it", "-r"]


def test_once_only_flag_without_routing_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without routing active, argv is untouched even with -r present."""
    monkeypatch.delenv(cli_main._ONCE_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_once(["-r", "fix it"]) == ["-r", "fix it"]


def test_group_options_and_once_flags_coexist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group options stay put; only once-only flags hoist after `once`."""
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "1")
    assert cli_main._maybe_route_to_once(
        ["--log-level", "debug", "-r", "fix it"]
    ) == ["--log-level", "debug", "once", "-r", "fix it"]


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_falsy_env_values_disable_routing(
    val: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsy env values do not enable routing."""
    if val:
        monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, val)
    else:
        monkeypatch.delenv(cli_main._ONCE_UNKNOWN_ENV, raising=False)
    assert cli_main._maybe_route_to_once(["continue"]) == ["continue"]


# ── _once_unknown_active / _first_positional_index unit checks ─────────


def test_once_unknown_active_env_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cli_main._ONCE_UNKNOWN_ENV, "yes")
    assert cli_main._once_unknown_active([]) is True


def test_once_unknown_active_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_main._ONCE_UNKNOWN_ENV, raising=False)
    assert cli_main._once_unknown_active(["--once-unknown"]) is True


def test_first_positional_index_basic() -> None:
    assert cli_main._first_positional_index(["verify"]) == 0
    assert cli_main._first_positional_index(["--log-level", "debug", "verify"]) == 2
    assert (
        cli_main._first_positional_index(["--data-dir=/tmp", "verify"]) == 1
    )
    assert cli_main._first_positional_index(["--no-log", "verify"]) == 1
    assert cli_main._first_positional_index([]) == -1
    assert cli_main._first_positional_index(["--", "continue"]) == 1
    assert cli_main._first_positional_index(["--"]) == -1