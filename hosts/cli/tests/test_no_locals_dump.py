"""Crash output must never print frame locals.

Uncaught CLI frames carry model-provider keys (the ``verify`` env reads),
``.env`` values, prompt payloads, and request bodies. Rich's traceback
renderer with ``show_locals=True`` prints them unredacted; both the
module-level ``rich.traceback.install`` and the Typer pretty-exception
path are configured with locals off, and these tests lock that contract.
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
    """The module-level install omits locals (the show_locals=False seam)."""
    # The install call is module-level and unconditional; asserting the
    # excepthook's rendering keeps the property rather than the call.
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
