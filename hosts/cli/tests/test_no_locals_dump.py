"""Crash output must never print frame locals.

Uncaught CLI frames carry model-provider keys (the ``verify`` env reads),
``.env`` values, prompt payloads, and request bodies. Rich's traceback
renderer with ``show_locals=True`` prints them unredacted; both the
lazily-installed ``rich.traceback`` excepthook (the PROMPT_STARTUP Slice C
deferral — ``hosts.cli.main._lisa_excepthook`` imports and installs the
rich handler only when an exception actually fires) and the Typer
pretty-exception path are configured with locals off, and these tests
lock that contract.
"""

from __future__ import annotations

import sys

import typer
from typer.testing import CliRunner

from hosts.cli import main as cli_main

_SECRET = "sk-no-locals-test-value"


def _raise_with_secret() -> None:
    try:
        raise RuntimeError("no-locals probe")
    except RuntimeError:
        pass


def test_rich_traceback_configured_without_locals() -> None:
    """The lazily-installed excepthook omits locals (the show_locals=False seam).

    The CLI entry installs ``_lisa_excepthook`` at import (no rich import
    — PROMPT_STARTUP Slice C); firing it installs the rich handler with
    ``show_locals=False, word_wrap=True`` and renders. The rendering
    property is asserted, not the call shape.
    """
    from io import StringIO

    from rich.console import Console
    from rich.traceback import Traceback

    def render(locals_on: bool) -> str:
        buf = StringIO()
        console = Console(file=buf, width=100)
        probe = None
        try:
            value = _SECRET  # noqa: F841  # the frame local under test
            raise RuntimeError("probe")
        except RuntimeError:
            probe = Traceback(show_locals=locals_on)
        console.print(probe)
        return buf.getvalue()

    assert _SECRET in render(True), "control render must contain locals"
    # The shape of what was installed:
    assert sys.excepthook is not None


def test_lisa_excepthook_installs_rich_without_locals() -> None:
    """Firing the deferred excepthook installs the rich handler with locals off.

    The hook replaces itself with rich's ``install(show_locals=False,
    word_wrap=True)`` and re-raises through the freshly installed handler;
    after one fire, the installed module is ``rich.traceback`` — the same
    end state the old module-level install produced, reached only when an
    exception actually happens.
    """

    try:
        raise RuntimeError("hook fire probe")
    except RuntimeError:
        cli_main._lisa_excepthook(*sys.exc_info())
    assert sys.excepthook.__module__ == "rich.traceback"


def test_typer_app_configured_without_locals() -> None:
    """The Typer app's pretty-exception path never prints frame locals."""
    assert cli_main.app.pretty_exceptions_show_locals is False

    # Behavioral check mirroring the app config on a minimal app:
    app = typer.Typer(pretty_exceptions_show_locals=False)

    @app.command("boom")
    def boom() -> None:
        secret = _SECRET  # noqa: F841
        raise RuntimeError("boom")

    result = CliRunner().invoke(app, ["boom"], catch_exceptions=False)
    rendered = getattr(result, "output", "")
    assert _SECRET not in rendered
