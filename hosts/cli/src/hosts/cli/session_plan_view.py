"""Read-only workflow-plan rendering for the interactive session.

The plan is derived from the durable thread (the single source of
truth) via :func:`kernel.workflow_view.derive_workflow_view` — the same
derivation the runtime uses to inject plan context — and rendered as a
checklist with the current step's orientation. Read-only by design:
the interactive session never mutates workflow state directly; turns do.
"""

from __future__ import annotations

from typing import Any

from kernel.workflow_view import derive_workflow_view, render_workflow_progress
from rich.panel import Panel
from rich.text import Text

_STYLE_STEP_DONE = "green3"
_STYLE_STEP_TODO = "grey58"
_STYLE_STEP_CURRENT = "gold3"


def render_plan_panel(runtime: Any) -> Panel:
    """Build the active plan panel from the runtime's thread tail.

    Reads the thread through the runtime's own read API (the durable
    store is the truth; the plan is never cached here). Synchronous by
    design: thread reads are file reads, and the Ctrl+T provider runs
    inline in the transport's toggle path.
    """
    from runtime.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=50)
    view = derive_workflow_view(entries)
    if view is None:
        return Panel(
            Text("No active workflow plan (a turn with a plan block starts one)."),
            title="Plan",
            border_style="grey58",
        )
    body = _plan_checklist(view)
    return Panel(body, title="Plan", border_style="steel_blue3")


#: The interactive surface's sync alias — the async surface command
#: renders the same panel through the identical derivation.
render_plan_panel_sync = render_plan_panel


def _plan_checklist(view: Any) -> Text:
    """A compact checklist: ✓/• per step, the current step highlighted."""
    completed = set(view.completed_steps)
    current = view.current_step_id
    text = Text()
    for index, step in enumerate(view.declared_steps):
        if index:
            text.append("\n")
        if step.step_id in completed:
            text.append("✓ ", style=_STYLE_STEP_DONE)
            text.append(step.mode_id, style=_STYLE_STEP_DONE)
        elif step.step_id == current:
            text.append("▸ ", style=_STYLE_STEP_CURRENT)
            text.append(step.mode_id, style=_STYLE_STEP_CURRENT)
        else:
            text.append("• ", style=_STYLE_STEP_TODO)
            text.append(step.mode_id, style=_STYLE_STEP_TODO)
        description = (step.step_query or "").strip()
        if description:
            text.append(f" — {description}", style=_STYLE_STEP_TODO)
    # The narrative orientation block (kernel-rendered) rides below the
    # checklist when the view provides one.
    progress = render_workflow_progress(view)
    if progress:
        text.append("\n\n")
        text.append(progress, style=_STYLE_STEP_TODO)
    return text


__all__ = ["render_plan_panel"]
