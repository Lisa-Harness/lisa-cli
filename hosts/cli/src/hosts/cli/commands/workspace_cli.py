"""``lisa diff`` — the working tree's diff as a typed one-shot command.

Reads through the workspace service (pure async git, no shell, no
mutation) and renders through the shared diff view. The comparison
scope is the first positional: ``staged`` (index vs HEAD), a git ref
(branch, tag, sha — compared against the working tree), or the default
working-tree-vs-HEAD.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from hosts.cli.application import _PANEL_INTROSPECTION, app, console
from hosts.cli.commands.diff_view import render_diff
from hosts.cli.workspace import read_diff


@app.command(
    "diff",
    rich_help_panel=_PANEL_INTROSPECTION,
    help=(
        "Show the working tree's git diff (read-only; no model call). "
        "Scope: 'staged' compares the index to HEAD, a ref (branch/tag/sha) "
        "compares that ref to the working tree; default is working tree vs HEAD."
    ),
)
def _diff_command(
    scope: str = typer.Argument(
        "",
        help=(
            "'staged' for the index vs HEAD, a git ref to compare against "
            "the working tree, or empty for working tree vs HEAD."
        ),
    ),
    max_bytes: int = typer.Option(
        200_000,
        "--max-bytes",
        help="Per-file diff byte cap; larger files are truncated with a marker.",
    ),
) -> None:
    """Render the requested git diff through the workspace service."""
    stripped = scope.strip()
    staged = stripped.lower() == "staged"
    ref = "" if staged or not stripped else stripped
    result = asyncio.run(
        read_diff(
            Path.cwd(),
            staged=staged,
            ref=ref,
            max_bytes=max_bytes,
        )
    )
    render_diff(result, console, title=ref or "Working tree")
