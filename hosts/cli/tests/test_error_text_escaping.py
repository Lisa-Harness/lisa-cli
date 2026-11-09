"""Dynamic error/detail text is content, never markup.

Rich's console.print parses ``[tags]`` in f-string interpolation silently:
an unknown or unclosed tag in a pack failure detail (raw uv output), an
exception message, or a path is swallowed — output like
``error: bad link at [build]`` loses the bracket. These tests lock the
escape/markup=False contract on the paths that interpolate runtime text.
"""

from __future__ import annotations

# pylint: disable=protected-access
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
import typer
from rich.console import Console

from hosts.cli import main as cli_main

#: A token that the markup parser would eat if the text went through it.
_BRACKET = "[build]"


class _CapturingConsole:
    """Records (text, kwargs) of every console.print call."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            # Render exactly like a real console so assertions see what
            # the operator would see (markup processed, escapes resolved).
            buf = StringIO()
            _render_console = Console(file=buf, width=200)
            _render_console.print(arg, **kwargs)
            self.lines.append(buf.getvalue())
            self.kwargs.append(kwargs)


# ── _finish_pack_mutation ─────────────────────────────────────────────


def test_pack_failure_detail_keeps_brackets() -> None:
    """A failing pack update prints raw uv output verbatim (markup off)."""
    detail = f"Install failed for 'p'.\n{_BRACKET} was not found"
    console = _CapturingConsole()
    with (
        patch.object(cli_main, "console", new=console),
        pytest.raises(typer.Exit) as excinfo,
    ):
        cli_main._finish_pack_mutation(None, (False, detail))
    assert excinfo.value.exit_code == 1
    assert _BRACKET in console.lines[0]
    assert console.kwargs[0].get("markup") is False


def test_pack_success_detail_keeps_brackets() -> None:
    console = _CapturingConsole()
    with (
        patch.object(cli_main, "console", new=console),
        pytest.raises(typer.Exit) as excinfo,
    ):
        cli_main._finish_pack_mutation(None, (True, f"Installed 'p' from {_BRACKET}"))
    assert excinfo.value.exit_code == 0
    assert _BRACKET in console.lines[0]


# ── _render_runtime_error ─────────────────────────────────────────────


def test_runtime_error_panel_escapes_message() -> None:
    """Startup-error brackets survive: the message is escaped per line."""
    console = _CapturingConsole()
    exc = RuntimeError(f"bad value at {_BRACKET}: see logs")
    cli_main._render_runtime_error(console, exc, title="Startup Error")
    rendered = console.lines[0]
    assert _BRACKET in rendered


# ── shell-substitution / stdin failure paths ──────────────────────────


def test_shell_substitution_failure_escapes_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = f"echo {_BRACKET}"
    monkeypatch.setattr(
        cli_main,
        "console",
        _CapturingConsole(),
    )
    with (
        patch(
            "hosts.cli.expand.expand_shell_substitutions",
            lambda text: (text, [failing]),
        ),
        pytest.raises(typer.Exit) as excinfo,
    ):
        cli_main._expand_editor_message("anything")
    assert excinfo.value.exit_code == 2
    console = cli_main.console
    assert isinstance(console, _CapturingConsole)
    assert failing in console.lines[0]


def test_piped_stdin_error_escapes_exception_text() -> None:
    """The piped-stdin OSError message is escaped, not parsed as markup."""
    console = _CapturingConsole()
    message = f"read of dev/stdin failed at {_BRACKET}"
    exc = OSError(message)
    rendered = f"[red]Failed to read piped stdin: {cli_main.escape(str(exc))}[/red]"
    console.print(rendered)
    assert _BRACKET in console.lines[0]
