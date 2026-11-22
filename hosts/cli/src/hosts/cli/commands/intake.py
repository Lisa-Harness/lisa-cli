"""One-shot intake command."""

from __future__ import annotations

import asyncio

import typer
from rich.markup import escape

from hosts.cli.application import _PANEL_LIFECYCLE, app, console
from hosts.cli.bootstrap import _resolve_data_dir
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
    resolved_data_dir = _resolve_data_dir(data_dir, profile)
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
