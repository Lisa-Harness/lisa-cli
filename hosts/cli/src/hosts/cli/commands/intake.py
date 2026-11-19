"""One-shot intake command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.markup import escape

from hosts.cli.application import _PANEL_LIFECYCLE, app, console
from hosts.cli.cli_runtime import _resolve_profiled_data_dir
from hosts.cli.options import DataDirOption, HostLogOption, LogLevelOption, ProfileOption


@app.command(
    "intake",
    short_help="Run one intake tick (collect, process, write state).",
    help="Run one intake tick: collect signals, process, write state.",
    rich_help_panel=_PANEL_LIFECYCLE,
)
def _intake_command(
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot intake process. Triggered by systemd timer or cron.

    Collects signals from installed pack signal connectors, runs processing
    loops, and writes the merged UnifiedState to ``<data-dir>/state/state.json``.
    The next turn reads that state so the model has fresh ambient data.
    """
    # Deferred config import (Slice C): ``runtime.config`` drags the
    # framework's eager ``runtime/__init__`` chain; the intake command
    # is the only consumer of the platform default data dir.
    from runtime.config import default_data_dir  # lazy: runtime.config (Slice C)

    resolved_data_dir = (
        Path(_resolve_profiled_data_dir(data_dir, profile)).expanduser()
        if _resolve_profiled_data_dir(data_dir, profile)
        else default_data_dir()
    )
    try:
        from runtime.intake import run_intake
    except ImportError:
        console.print("[red]intake module not available in this host.[/red]")
        raise typer.Exit(code=2) from None

    state = asyncio.run(run_intake(data_dir=resolved_data_dir))
    console.print(
        f"[green]Intake tick complete.[/green] "
        f"{len(state.context_items)} context item(s). "
        f"State written to {escape(str(resolved_data_dir / 'state' / 'state.json'))}"
    )
