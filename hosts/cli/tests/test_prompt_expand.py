"""Tests for editor-text ``$()`` shell substitution (hosts.cli.expand).

The expander is a pure function over text with a runner seam — no
subprocess runs in these tests:

- ``$(cmd)`` expands with nesting; ``\\$(`` stays literal ``$(``.
- Single-quoted spans do not expand (POSIX single-quote semantics);
  double-quoted spans do.
- Unterminated ``$(`` stays literal (prose, not a typo).
- A failed command keeps the literal span and is reported; the call
  site turns a non-empty failure list into exit 2 (tested via
  ``_collect_interactive_or_error``).
- Backticks and ``$VAR`` stay literal — out of scope by design.
"""

from __future__ import annotations

# pylint: disable=protected-access
from typing import Any

import pytest
from rich.console import Console

from hosts.cli import expand as expand_mod
from hosts.cli import main as cli_main
from hosts.cli.expand import expand_shell_substitutions

BS = chr(92)  # a literal backslash, unhassled by shell quoting in sources


@pytest.fixture(autouse=True)
def _fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute a deterministic runner: cmd -> ``<cmd>``."""
    expand_mod._runner = lambda cmd: (f"<{cmd}>", True)
    yield
    expand_mod._runner = None  # type: ignore[assignment]


# ── Expansion core ────────────────────────────────────────────────────


def test_simple_expansion() -> None:
    got, failures = expand_shell_substitutions("run $(date) now")
    assert got == "run <date> now"
    assert failures == []


def test_nested_expansion() -> None:
    got, _ = expand_shell_substitutions("a $(echo $(echo inner)) b")
    assert got == "a <echo $(echo inner)> b"


def test_escape_keeps_literal() -> None:
    got, _ = expand_shell_substitutions(f"use {BS}$(cmd) verbatim")
    assert got == "use $(cmd) verbatim"


def test_single_quotes_inhibit_expansion() -> None:
    got, _ = expand_shell_substitutions("keep 'the $(cmd) hidden' here")
    assert got == "keep 'the $(cmd) hidden' here"


def test_double_quotes_still_expand() -> None:
    got, _ = expand_shell_substitutions('note "the $(cmd) works" yes')
    assert got == 'note "the <cmd> works" yes'


def test_unterminated_stays_literal() -> None:
    text = "maybe prose $(not a command"
    got, failures = expand_shell_substitutions(text)
    assert got == text
    assert failures == []


def test_backticks_and_vars_stay_literal() -> None:
    text = "git `status` and $HOME untouched"
    got, _ = expand_shell_substitutions(text)
    assert got == text


def test_failed_command_reports_and_keeps_literal() -> None:
    expand_mod._runner = lambda cmd: ("", False)
    got, failures = expand_shell_substitutions("bad $(boom) tail")
    assert got == "bad $(boom) tail"
    assert failures == ["boom"]


def test_empty_output_expands_to_empty() -> None:
    expand_mod._runner = lambda cmd: ("", True)
    got, failures = expand_shell_substitutions("a $(nothing) b")
    assert got == "a  b"
    assert failures == []


def test_multiline_text_expands_per_line() -> None:
    got, _ = expand_shell_substitutions("line1 $(a)\nline2 $(b)")
    assert got == "line1 <a>\nline2 <b>"


# ── Call-site wiring ──────────────────────────────────────────────────


def _invoke_collect(monkeypatch: pytest.MonkeyPatch, reader_result: Any) -> Any:
    """Invoke the no-input collector with a scripted reader + tty stdio."""
    import io
    import sys

    monkeypatch.setattr(cli_main, "_prompt_reader", lambda console: reader_result)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    stdout_obj: Any = io.StringIO()
    stdout_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    monkeypatch.setattr(sys, "stdout", stdout_obj)
    import typer

    try:
        return cli_main._collect_interactive_or_error("", "", "/tmp/d", "")
    except typer.Exit as exc:
        return exc.exit_code


def test_wiring_expands_editor_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collector runs the expansion over reader output."""
    result = _invoke_collect(monkeypatch, "summarize $(whoami)")
    assert result == "summarize <whoami>"


def test_wiring_failed_substitution_exits_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed expansion is a loud usage error, not a mutated message."""
    monkeypatch.setattr(expand_mod, "_runner", lambda cmd: ("", False))
    result = _invoke_collect(monkeypatch, "try $(broken-command)")
    assert result == 2


def test_wiring_escapes_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An escaped span survives the collector untouched."""
    result = _invoke_collect(monkeypatch, f"want {BS}$(this) literal")
    assert result == "want $(this) literal"


def test_console_error_printed_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure notice names the failing command."""
    import io
    import sys
    from unittest import mock

    monkeypatch.setattr(expand_mod, "_runner", lambda cmd: ("", False))
    monkeypatch.setattr(cli_main, "_prompt_reader", lambda console: "x $(boom)")
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    stdout_obj: Any = io.StringIO()
    stdout_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    monkeypatch.setattr(sys, "stdout", stdout_obj)

    console_file = io.StringIO()
    console = Console(file=console_file)
    with mock.patch.object(cli_main, "console", new=console):
        import typer

        with pytest.raises(typer.Exit) as exc_info:
            cli_main._collect_interactive_or_error("", "", "/tmp/d", "")
    assert exc_info.value.exit_code == 2
    assert "boom" in console_file.getvalue()
