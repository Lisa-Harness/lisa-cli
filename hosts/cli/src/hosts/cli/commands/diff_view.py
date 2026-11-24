"""Rendering for workspace diffs (shared by `/diff` and `lisa diff`).

Tty consoles get per-file sections with colored hunks (adds/dels/context
using the approval panel's diff palette); pipes get the raw unified diff
verbatim — machine-readable, no decoration. Large files are truncated by
the service; the renderer only relays the honest markers.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

#: Per-file diff lines rendered before truncation takes over.
_MAX_RENDERED_DIFF_LINES = 400

_STYLE_ADD = "green3"
_STYLE_DEL = "indian_red"
_STYLE_HUNK = "steel_blue3"
_STYLE_META = "grey58"


def render_diff(result: Any, console: Console, *, title: str = "Working tree") -> None:
    """Render one DiffResult; tty gets sections, pipes get raw text."""
    from rich.rule import Rule  # lazy: presentation import (startup Slice C)

    if not result.files:
        console.print(f"[grey58]No changes ({result.scope}).[/grey58]")
        return
    if not console.is_terminal:
        for file_diff in result.files:
            if file_diff.diff:
                console.print(file_diff.diff, highlight=False, markup=False)
        if result.truncated:
            console.print("… diff truncated", highlight=False, markup=False)
        return
    console.print(Rule(title=f"Diff — {result.scope}", style="steel_blue3"))
    for file_diff in result.files:
        console.print(_file_section(file_diff))
    if result.truncated:
        console.print("[gold3]… some diffs truncated; open the file for the rest[/gold3]")


def _file_section(file_diff: Any) -> Any:
    """One file's section: header line plus colored diff body."""
    from rich.console import Group  # lazy: presentation import
    from rich.panel import Panel  # lazy: presentation import
    from rich.text import Text  # lazy: presentation import

    body: list[Any] = []
    if file_diff.reason:
        body.append(Text(f"{file_diff.reason}", style=_STYLE_META))
    if file_diff.diff:
        lines = file_diff.diff.splitlines()
        shown = lines[:_MAX_RENDERED_DIFF_LINES]
        for line in shown:
            if line.startswith("+"):
                body.append(Text(line, style=_STYLE_ADD))
            elif line.startswith("-"):
                body.append(Text(line, style=_STYLE_DEL))
            elif line.startswith("@@"):
                body.append(Text(line, style=_STYLE_HUNK))
            else:
                body.append(Text(line))
        if len(lines) > _MAX_RENDERED_DIFF_LINES:
            omitted = len(lines) - _MAX_RENDERED_DIFF_LINES
            body.append(Text(f"… {omitted} more lines (open the file)", style=_STYLE_META))
    return Panel(Group(*body), title=file_diff.path, border_style="grey35", padding=(0, 1))


__all__ = ["render_diff"]
