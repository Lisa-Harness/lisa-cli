"""Model, catalog, operation, tool, and response-history commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

import typer

from hosts.cli.application import _PANEL_INTROSPECTION, _STYLE_RESPONSE, app, console
from hosts.cli.cli_runtime import _aclose_runtime_bounded, _initialize_runtime_or_exit
from hosts.cli.options import (
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    ModelsAction,
    NoLogOption,
    NoThreadOption,
    ProfileOption,
    SessionContextOption,
    StateSnapshotOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
)

if TYPE_CHECKING:
    pass


@app.command("modes", help="List supported Lisa modes.")
def _modes_command(
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
    from hosts.cli.render import (  # lazy: presentation import (Slice C)
        render_modes as _render_modes,
    )

    _render_modes(runtime, console=console)


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------


@app.command(
    "models",
    short_help="Manage the model manifest.",
    help=(
        "Manage the model manifest. ACTION defaults to 'show'. "
        "Subcommands: show, discover, apply, help. "
        "(reload and edit only refresh transient state or print guidance; "
        "they have no durable effect as a one-shot CLI invocation.)"
    ),
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _models_command(
    action: Annotated[
        ModelsAction,
        typer.Argument(
            ...,
            help=(
                "Action: show (default), discover, apply, help. "
                "'discover' writes a proposal manifest; 'apply' promotes it to active."
            ),
        ),
    ] = "show",
    force: bool = typer.Option(
        False,
        "--force",
        help="With 'discover', overwrite an existing proposal manifest.",
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

    # Typer parsed/validated the action (Literal) — no normalization needed.
    #
    # 'show' renders the Rich table from the active manifest. A missing
    # manifest raises FileNotFoundError; that one absence is routed to the
    # core handler for the friendly "run discover first" guidance. Any other
    # failure (schema, permissions, render) propagates to the startup-error
    # panel instead of being retried through a second dispatch.
    if action == "show":
        from runtime.model_manifest import load_model_manifest

        try:
            load_model_manifest(runtime.paths.models_path)
        except FileNotFoundError:
            pass
        else:
            from hosts.cli.render import (  # lazy: presentation import (Slice C)
                render_models as _render_models,
            )

            _render_models(runtime, console=console)
            return

    # 'reload' and 'edit' only mutate transient in-process state or print
    # guidance — both pointless for a one-shot CLI process whose runtime
    # is discarded on exit. Surface that clearly instead of pretending
    # to do work; the durable CLI workflow is discover -> (edit file) ->
    # apply. Both remain available in the persistent REPL.
    if action in {"reload", "edit"}:
        console.print(
            "[yellow]'models " + action + "' has no durable effect as a one-shot "
            "CLI command: it only refreshes in-process state or prints guidance, and "
            "the CLI runtime is discarded on exit.[/yellow]\n"
            "Edit models.proposed.yaml by hand, then run 'lisa models apply'."
        )
        return

    # show (absent manifest), discover, apply, help delegate to the
    # transport-neutral core handler (the same one the REPL /models
    # command uses), so behavior stays identical across hosts. We
    # synthesize the equivalent '/models <sub> [--force]' slash string
    # and print the returned Markdown text in a Rich panel.
    from rich.markdown import Markdown  # lazy: presentation import (Slice C)
    from rich.panel import Panel  # lazy: presentation import (Slice C)
    from runtime.commands import handle_models_command

    if action == "discover" and force:
        stripped = "/models discover --force"
    else:
        stripped = f"/models {action}".strip()

    # The discover/apply/reload/show subcommands read or write model
    # manifests via the lisa-storage MCP server, which requires a live
    # MCP tool pool. _initialize_runtime only compiles configs; the pool
    # is started inside the host's event loop. Start it here, run the
    # handler, and tear the pool down in finally — all within one
    # asyncio.run loop so anyio cancel scopes are entered and exited in
    # the same task (per runtime.runtime invariants).
    async def _run_models_command() -> str:
        try:
            await runtime.start()
            return await handle_models_command(runtime, stripped, stripped)
        finally:
            await _aclose_runtime_bounded(runtime)

    response_text = asyncio.run(_run_models_command())
    console.print(
        Panel(
            Markdown(response_text, code_theme="github-dark", hyperlinks=True),
            title="Models",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# operations command
# ---------------------------------------------------------------------------


@app.command(
    "operations",
    short_help="List internal operations with IDs for allowlist authoring.",
    help="List internal operations with IDs for allowlist authoring.",
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _operations_command(
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
    from hosts.cli.render import (  # lazy: presentation import (Slice C)
        render_operations as _render_operations,
    )

    _render_operations(runtime, console=console)


# ---------------------------------------------------------------------------
# tools command
# ---------------------------------------------------------------------------


@app.command(
    "tools",
    short_help="List internal tools with IDs for allowlist authoring.",
    help=(
        "List internal tools with IDs for allowlist authoring. "
        "Use the `mcp` command to inspect MCP server tools."
    ),
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _tools_command(
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
    from hosts.cli.render import (  # lazy: presentation import (Slice C)
        render_tools as _render_tools,
    )

    _render_tools(runtime, console=console)


# ---------------------------------------------------------------------------
# last command
# ---------------------------------------------------------------------------


@app.command(
    "last",
    short_help="Reprint a previous response verbatim (no model call).",
    help=(
        "Reprint a previous response verbatim (default: the last model turn). "
        "Pass a turn number to reprint that turn's response: `lisa last 3`. "
        "Reads the durable thread — no model call."
    ),
    rich_help_panel=_PANEL_INTROSPECTION,
)
def _last_command(
    turn: str = typer.Argument(
        "",
        help="Optional turn number to reprint that turn's response.",
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
    from rich.markdown import Markdown  # lazy: defers startup cost
    from rich.rule import Rule  # lazy: defers startup cost
    from runtime.commands import (  # lazy: defers runtime.commands startup cost
        resolve_last_turn,
    )

    turn_index, response_text = resolve_last_turn(runtime, turn.strip())
    if turn_index is not None and not console.is_terminal:
        # Piped/redirected: a one-line header so the reprinted turn is
        # addressable in captured output, then the response verbatim —
        # no Markdown re-render (the raw contract; highlight/markup off
        # so Rich never interprets the body). Diagnostics (usage,
        # not-found, friendly messages) print bare as before.
        console.print(f"[ turn {turn_index} ]", style=_STYLE_RESPONSE, markup=False)
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    if not console.is_terminal:
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    # Interactive: the same treatment the original answer gets — a
    # ``Response · turn N`` rule separator plus Rich Markdown rendering
    # (colors, headings, code fences). The number is the global thread
    # turn index the prompt output already renders, so operators address
    # a reprinted turn the same way wherever it surfaces.
    console.print(Rule(title=f"Response · turn {turn_index}", style=_STYLE_RESPONSE))
    console.print(Markdown(response_text, code_theme="github-dark", hyperlinks=True))
