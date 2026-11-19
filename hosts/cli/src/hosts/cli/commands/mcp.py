"""MCP server inspection command."""

from __future__ import annotations

import asyncio

import typer
from rich.markup import escape

from hosts.cli.application import _PANEL_INTROSPECTION, app, console
from hosts.cli.cli_runtime import _collect_mcp_tools, _initialize_runtime_or_exit
from hosts.cli.options import (
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    NoLogOption,
    NoThreadOption,
    ProfileOption,
    SessionContextOption,
    StateSnapshotOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
)


@app.command(
    "mcp",
    short_help="List MCP servers; drill into one server's tools.",
    help=(
        "List MCP servers with status, transport, URL/command, tool count, "
        "and last error. Pass a server id to drill into that server's tools "
        "(name, description, required args)."
    ),
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _mcp_command(
    target: str = typer.Argument(
        "",
        help="Optional server id to drill into that server's tools.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
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
    server_id = target.strip()
    # Warm servers only by default — cold servers stay disconnected so the
    # one-shot command never sequentially exits multiple cold stdio task
    # groups in one task (anyio cancel-scope poison, 2026-09-21). Drill-down
    # adds exactly ONE cold target to the same pass: a single cold context
    # open/close in the CLI's task is safe (verified), while a second
    # in-process runtime init is not (stale global tool catalog breaks the
    # operation-catalog re-validation on the second boot).
    mcp_health, mcp_tools_by_server = asyncio.run(
        _collect_mcp_tools(runtime, connect_server_ids={server_id} if server_id else None)
    )
    # Deferred render import (Slice C): the kernel.models-typed render
    # helpers load only on the mcp command's own path.
    from hosts.cli.render import (  # lazy: presentation import (Slice C)
        render_mcp_server_tools as _render_mcp_server_tools,
    )
    from hosts.cli.render import (
        render_mcp_servers as _render_mcp_servers,
    )

    if not mcp_health:
        console.print("[yellow]No MCP server tools discovered.[/yellow]")
        return
    if server_id:
        specs = mcp_tools_by_server.get(server_id, [])
        health = mcp_health.get(server_id)
        if health is None:
            console.print(
                f"[yellow]No MCP server named {escape(server_id)!r} is configured.[/yellow]"
            )
            return
        _render_mcp_server_tools(specs, health, console=console)
    else:
        _render_mcp_servers(mcp_health, console=console)
