"""``lisa sessions`` — list resumable conversations from the thread stores.

Reads the durable thread stores through the conversation catalog (pure
module; no model calls, no pool) and renders a table of conversations:
id, turns (sealed + active), last activity, last mode, and the last
operator query as the summary. Construction follows the other one-shot
introspection commands: the runtime is built (config + session-plan
resolution decide the user slug and thread dir) but never started.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hosts.cli.application import _PANEL_INTROSPECTION, app, console
from hosts.cli.cli_runtime import _initialize_runtime_or_exit
from hosts.cli.options import (
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    NoLogOption,
    NoThreadOption,
    OverlaysDirOption,
    ProfileOption,
    SessionContextOption,
    StateSnapshotOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
)


def _sessions_table(records: list[Any], *, user_slug: str, thread_dir: Path) -> Any:
    """Build the Rich conversations table (tty) from catalog records."""
    from rich.table import Table

    table = Table(
        title=f"Conversations — {user_slug} ({thread_dir})",
        show_lines=False,
        expand=False,
    )
    table.add_column("Id", style="bold")
    table.add_column("Turns", justify="right")
    table.add_column("Last activity", style="grey58")
    table.add_column("Mode", style="grey58")
    table.add_column("Summary", overflow="fold")
    if not records:
        table.add_row("[grey58]— no conversations yet —", "", "", "", "")
        return table
    for record in records:
        table.add_row(
            record.conversation_id,
            str(record.turn_count),
            _short_timestamp(record.last_activity_utc),
            record.last_mode or "—",
            record.summary or "—",
        )
    return table


def _short_timestamp(timestamp: str) -> str:
    """ISO-8601 timestamp → ``YYYY-MM-DD HH:MM`` (UTC), or ``—``."""
    if not timestamp:
        return "—"
    # Keep the date+time prefix; drop seconds/fraction/timezone for the table.
    try:
        date_part, time_part = timestamp.split("T", 1)
    except ValueError:
        return timestamp[:16]
    return f"{date_part} {time_part[:5]}"


@app.command(
    "sessions",
    rich_help_panel=_PANEL_INTROSPECTION,
    help=(
        "List resumable conversations from the thread stores (no model call). "
        "Resume with `lisa --resume <id>` or the interactive `/resume <id>`."
    ),
)
def _sessions_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Render the conversation catalog for the resolved profile."""
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return

    from hosts.cli.conversations import scan_conversations

    # The scan is pure: runtime construction resolved the user slug and the
    # thread dir; no pool start, no model call, nothing durable is touched.
    records = scan_conversations(
        runtime.paths.thread_dir,
        user_slug=runtime.config.user_slug,
    )
    if not console.is_terminal:
        # Piped: plain lines stay machine-greppable — one per conversation.
        for record in records:
            console.print(
                f"{record.conversation_id}\t{record.turn_count}\t"
                f"{record.last_activity_utc}\t{record.last_mode}\t{record.summary}",
                highlight=False,
                markup=False,
            )
        if not records:
            console.print("no conversations", highlight=False, markup=False)
        return
    console.print(
        _sessions_table(
            records, user_slug=runtime.config.user_slug, thread_dir=runtime.paths.thread_dir
        )
    )
