"""Host-owned settings, token, and debug commands."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

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


@app.command(
    "interactive",
    rich_help_panel="Session",
    help=(
        "Run a persistent interactive session: one runtime, many turns. "
        "Approvals and model questions are decided in place. "
        "Ctrl-C stops a running turn; /quit or Ctrl-D exits."
    ),
)
def _interactive_command(
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
    """Dispatch the persistent interactive session (engine + surface in hosts.cli.interactive)."""
    from hosts.cli.interactive import interactive_command

    interactive_command(
        log_level=log_level,
        log=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot=state_snapshot,
        streaming=streaming,
        host_log=host_log,
    )


def _render_markdown_response(response_text: str, *, title: str) -> None:
    """Print a transport-neutral host-command reply as a Rich panel.

    Non-tty output (pipes, redirects, CI) prints the raw text verbatim —
    no Markdown re-render — so piped output stays machine-readable (the
    same contract as PromptTransport raw mode).
    """
    if not console.is_terminal:
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    from rich.markdown import Markdown  # lazy: defers startup cost
    from rich.panel import Panel  # lazy: presentation import (Slice C)

    console.print(
        Panel(
            Markdown(response_text, code_theme="github-dark", hyperlinks=True),
            title=title,
            border_style="cyan",
        )
    )


@app.command(
    "settings",
    rich_help_panel=_PANEL_INTROSPECTION,
    help="Show the effective runtime settings (one-shot /settings).",
)
def _settings_command(  # Typer exposes shared runtime options as CLI flags.
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
    """Render the effective runtime settings without a model turn."""
    _dispatch_constant_command(
        "/settings",
        title="Settings",
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


@app.command(
    "tokens",
    rich_help_panel=_PANEL_INTROSPECTION,
    help="Show session token usage and budget (one-shot /tokens).",
)
def _tokens_command(  # Typer exposes shared runtime options as CLI flags.
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
    """Render token accounting without a model turn."""
    _dispatch_constant_command(
        "/tokens",
        title="Tokens",
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


@app.command(
    "debug",
    short_help="Show the complete, unedited model input for a previous turn.",
    help=(
        "Show the complete, unedited model input for a previous turn "
        "(default: the last turn): `lisa debug [N]`."
    ),
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _debug_command(  # Typer exposes shared runtime options as CLI flags.
    turn: str = typer.Argument(
        "",
        help="Optional turn number whose model input to render.",
    ),
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
    """Render one turn's complete model input without a model turn."""
    _dispatch_constant_command(
        f"/debug {turn.strip()}".strip(),
        title="Debug",
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


def _dispatch_constant_command(command: str, *, title: str, **runtime_kwargs: Any) -> None:
    """Run one constant host command through a fresh runtime and render it.

    The command is one of the host's fixed admin commands (``/settings``,
    ``/tokens``, ``/debug N``) — no user-controlled grammar, so dispatching
    via ``handle_message`` costs nothing in parsing fidelity while keeping
    the runtime's own bookkeeping (thread entry, turn log) intact.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    from hosts.cli.prompt import _run_host_command_turn

    text = asyncio.run(_run_host_command_turn(runtime, command))
    _render_markdown_response(text, title=title)
