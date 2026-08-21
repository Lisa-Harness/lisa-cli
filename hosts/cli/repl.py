"""Interactive REPL for the CLI host.

This is the terminal transport's read-eval-print loop. It wraps the
transport-neutral :func:`host.handle_message` with a Rich-based
:class:`hosts.cli.transport.RichTransport` and a prompt_toolkit-backed
input session (with fallback to plain ``console.input``).

Lifecycle::

    runtime = HostRuntime(config)
    await runtime.start()           # starts MCP pool in this event loop
    await run_repl(runtime)         # runs the loop until /quit or EOF
    await runtime.aclose()          # closes MCP pool in same loop

The REPL is transport-specific: it owns the Rich console, the status line,
the prompt, and the session-exit summary. It delegates all routing and
turn execution to ``host.handle_message``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from host.callbacks import TransportCallbacks
from host.handle_message import handle_message
from host.operator_signal import OperatorSignal
from host.runtime import HostRuntime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.text import Text

from hosts.cli.keys import (
    TerminalSignalDispatcher,
    async_read_input,
    build_repl_key_bindings,
    install_loop_signal_handlers,
)
from hosts.cli.transport import RichTransport

logger = logging.getLogger(__name__)

# ── Visual style constants (CLI host palette) ─────────────────────────
STYLE_TEXT_PRIMARY = "bright_white"
STYLE_TEXT_MUTED = "grey58"
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_RULE = "grey42"
STYLE_PROMPT = "bold steel_blue3"

# ── REPL file history ────────────────────────────────────────────────
_REPL_HISTORY_FILE_NAME = "repl_history.txt"

# ── Host-only REPL shortcuts (not catalog modes) ────────────────────
_HOST_REPL_SHORTCUTS: tuple[str, ...] = (
    "/help",
    "/help capabilities",
    "/pack list",
    "/pack enable ",
    "/pack disable ",
    "/pack install ",
    "/pack uninstall ",
    "/models show",
    "/models discover",
    "/models apply",
    "/models reload",
    "/models edit",
    "/settings",
    "/tokens",
    "/mcp",
    "/mcp ",
    "/debug context",
    "/debug router",
    "/consolidate",
    "/intake",
    "/checkpoint",
    "/restore",
    "/checkpoints",
    "/quit",
)


# ── prompt_toolkit: optional dependency with graceful fallback ───────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory, History
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    AutoSuggestFromHistory = None
    Completer = None
    Completion = None
    WordCompleter = None
    Document = None
    FileHistory = None
    History = None
    KeyBindings = None
    CursorShape = None
    ModalCursorShapeConfig = None
    Style = None


class _SurrogateSafeFileHistory(History):
    """FileHistory wrapper that sanitises unpaired surrogates before persisting.

    On Windows, the console can emit unpaired surrogate characters for emoji.
    Python's utf-8 codec rejects those, which crashes prompt_toolkit's
    FileHistory.store_string. This wrapper strips them.
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._inner = FileHistory(filename)

    def load_history_strings(self) -> list[str]:
        return self._inner.load_history_strings()

    def store_string(self, string: str) -> None:
        # Strip lone surrogates (U+D800–U+DFFF) that Windows consoles can emit
        # for emoji. ``surrogateescape`` only handles byte-level surrogates,
        # not Unicode surrogates, so it raises UnicodeEncodeError here. Use
        # ``replace`` to substitute the U+FFFD replacement character instead.
        safe = string.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        self._inner.store_string(safe)

    # NOTE: do NOT override ``append_string``. The base ``History.append_string``
    # inserts into ``self._loaded_strings`` (the in-memory list that prompt_toolkit
    # reads via ``load()`` for up-arrow navigation) and then calls
    # ``self.store_string`` (which sanitises via the override above). Delegating to
    # ``self._inner.append_string`` instead — as a prior version did — updates the
    # *inner* FileHistory's in-memory list and writes the file, but leaves this
    # wrapper's ``_loaded_strings`` stale. The result: within a session, up-arrow
    # history omits the commands you just typed (they only reappear after a
    # restart that reloads from disk). The base implementation does the right
    # thing here.


def _repl_command_hints() -> tuple[str, ...]:
    """Build REPL completion hints from host shortcuts plus catalog mode commands."""
    from host.commands import mode_support_gap
    from kernel.ipsar_fsm import get_workflow_catalog

    catalog = get_workflow_catalog()
    mode_hints: list[str] = []
    for mode_id, mode in sorted(catalog.items()):
        if mode_support_gap(mode) is not None:
            continue
        if mode.command_pattern and mode_id != "default":
            mode_hints.append(f"/{mode_id} ")
    return (*_HOST_REPL_SHORTCUTS, *mode_hints)


async def run_repl(
    runtime: HostRuntime,
    *,
    console: Console | None = None,
) -> bool:
    """Run the interactive REPL loop.

    Args:
        runtime: A constructed ``HostRuntime`` whose ``start()`` has not
            yet been called (this function calls ``start()`` and ``aclose()``).
        console: Optional Rich Console. Defaults to a new ``Console()``.

    Returns:
        ``True`` if a restart was requested (caller may re-exec), ``False``
        on clean exit.
    """
    console = console or Console(legacy_windows=False)
    # Stash the runtime on the console so the status-line renderer can read
    # route preview + token counters without threading the runtime through
    # every status-segment call.
    console._lisa_runtime = runtime  # type: ignore[attr-defined]
    transport = RichTransport(console=console)

    # ── Signal dispatcher ──────────────────────────────────────────
    # Constructed once and threaded through the REPL: key bindings,
    # input reader, turn processor, and loop-level signal handlers all
    # share this one instance.
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    # ── Process-level signal handlers (POSIX backstop) ─────────────
    install_loop_signal_handlers(asyncio.get_running_loop(), dispatcher)

    # Set the initial route preview so the status line shows the
    # resolved model before the first turn is executed. When start_mode
    # is set in the session context, use it instead of "default".
    from host.commands import model_alias_for_mode, set_runtime_route_preview

    initial_mode = runtime.start_mode_id
    initial_model_alias = (
        model_alias_for_mode(initial_mode, manifest=runtime.manifest)
        if runtime.start_mode_id
        else "default"
    )
    set_runtime_route_preview(
        runtime,
        mode_id=initial_mode,
        preferred_model_id=initial_model_alias,
    )

    # Start the MCP pool in the background so the prompt appears immediately
    # while warm servers spawn. The first turn (or any slash command reading
    # the pool) awaits ensure_pool_ready() in handle_message before touching
    # the pool. Anyio cancel-scope binding is preserved: the task runs on
    # this event loop and is awaited on this same loop.
    try:
        runtime.start_pool_deferred()
    except Exception as exc:  # pylint: disable=broad-except
        _render_runtime_error(console, exc, title="MCP Startup Error")
        return False

    _print_startup_banner(console)
    console.print(
        Panel(
            _render_session_panel_text(runtime),
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )

    prompt_session = _build_repl_prompt_session(runtime, console, transport, dispatcher)
    if prompt_session is None:
        console.print(f"[{STYLE_TEXT_MUTED}]prompt_toolkit unavailable; using basic input mode.[/]")

    # Replay a pending resume message (set by `/pack ... --resume <msg>` in
    # the prior session via the LISA_RESTART_RESUME_MESSAGE env var) as the
    # first turn. Cleared after one replay.
    pending_resume = runtime.pending_resume_message
    runtime.pending_resume_message = None

    restart_requested = False
    try:
        if pending_resume:
            console.print(
                Panel(
                    f"Resuming after restart: {pending_resume}",
                    title="Resume",
                    border_style=STYLE_BORDER_SESSION,
                )
            )
            await _process_repl_turn(runtime, transport, console, pending_resume, dispatcher)
            if runtime.restart_required:
                _render_restart_required_notice_if_needed(runtime, console)
                if not _confirm_restart(runtime, console):
                    runtime.restart_required = False
                    runtime.restart_reason = ""
                    console.print(
                        f"[{STYLE_TEXT_MUTED}]Restart deferred. "
                        f"Run /quit then relaunch `lisa` manually when ready.[/]"
                    )
                else:
                    _render_session_exit_summary(runtime, console)
                    console.print(f"[{STYLE_TEXT_MUTED}]Applying restart now...[/]")
                    return True
        restart_requested = await _run_repl_loop(runtime, transport, console, prompt_session, dispatcher)
    finally:
        # Close the MCP pool. aclose() handles the anyio cross-task rule
        # internally: when the pool was started via start_pool_deferred(),
        # it delegates teardown to the long-lived start task (which owns the
        # MCP stdio_client anyio cancel scopes) and shields that await so a
        # Ctrl+C during shutdown doesn't orphan MCP subprocesses. Do NOT
        # wrap this in asyncio.shield here — that would run aclose in a
        # different task than the one that entered the pool's anyio cancel
        # scopes, tripping anyio's cross-task guard (see debugging.md).
        await runtime.aclose()

    return restart_requested


async def _run_repl_loop(
    runtime: HostRuntime,
    transport: TransportCallbacks,
    console: Console,
    prompt_session: Any,
    dispatcher: TerminalSignalDispatcher,
) -> bool:
    """Process REPL turns until the user quits or sends EOF."""
    while True:
        message = await async_read_input(console, prompt_session, dispatcher)
        if message is None:
            _render_session_exit_summary(runtime, console)
            return False
        if not message:
            continue
        if _is_repl_quit_command(message):
            _render_session_exit_summary(runtime, console)
            console.print(f"[{STYLE_TEXT_MUTED}]Session closed.[/]")
            return False
        await _process_repl_turn(runtime, transport, console, message)
        if runtime.restart_required:
            _render_restart_required_notice_if_needed(runtime, console)
            if not _confirm_restart(runtime, console):
                runtime.restart_required = False
                runtime.restart_reason = ""
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Restart deferred. "
                    f"Run /quit then relaunch `lisa` manually when ready.[/]"
                )
                continue
            _render_session_exit_summary(runtime, console)
            if os.name == "nt":
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Please restart `lisa` manually to "
                    f"reload packs and MCP manifests.[/]"
                )
            elif runtime.restart_reason:
                console.print(f"[{STYLE_TEXT_MUTED}]Applying restart now...[/]")
            else:
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Restarting is required before continuing this session.[/]"
                )
            return True


def _is_repl_quit_command(message: str) -> bool:
    """Return True if the message is a quit/exit command."""
    return message.lower() in {"exit", "quit", "/exit", "/quit"}


async def _process_repl_turn(
    runtime: HostRuntime,
    transport: TransportCallbacks,
    console: Console,
    message: str,
    dispatcher: TerminalSignalDispatcher,
) -> None:
    """Execute one REPL turn via handle_message and render the response."""
    turn_start = time.perf_counter()
    # Start the thinking spinner immediately so the terminal is not dead
    # between submit and the first pipeline stage callback. The spinner
    # is stopped by the transport on the first on_stage/on_response/
    # on_error/on_text_chunk. The stage is "dispatch" — the pre-pipeline
    # phase (command parse, start_mode bypass, current_mode_id selection).
    # There is no separate router; handoff is mid-turn via transition_stage.
    if isinstance(transport, RichTransport):
        # Pin the submitted prompt to the top of the Live region (ask 1).
        transport.start_thinking(
            "dispatch",
            session_tokens=runtime.tokens.session_total_tokens,
            message=message,
        )
    # ── Clear stale operator signals ───────────────────────────────────
    dispatcher.clear()

    try:
        result = await handle_message(runtime, message, transport)
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Ctrl+C mid-turn: cancel the in-flight turn and return the prompt
        # fast without tearing down the session.
        #
        # Two interrupt paths converge here:
        # 1. asyncio.run()'s SIGINT handler calls main_task.cancel(), which
        #    raises CancelledError inside this coroutine (Python 3.11+ on
        #    all platforms). CancelledError is BaseException, so without
        #    this handler it propagates past _run_repl_loop, hits
        #    run_repl's `finally: await runtime.aclose()`, and kills the
        #    whole session — losing the warm MCP pool.
        # 2. On some platforms/loops a KeyboardInterrupt may still surface
        #    inside the await; catch it here too so the behavior is
        #    consistent.
        #
        # We do NOT close the MCP pool here. The pool owns long-lived
        # subprocesses bound to this event loop (anyio cancel-scope rule
        # from debugging.md: a resource started in a loop must be closed
        # in that same loop). Cancelling the turn's await cancels the
        # in-flight tool/complete coroutines; the pool itself survives and
        # is reused on the next turn. MCPToolPool.call_tool and aclose
        # already handle CancelledError internally (kernel/tools/mcp_client.py).
        if isinstance(transport, RichTransport):
            transport.stop_thinking(flush_transcript=True)
        # Flush any half-streamed on_text_chunk output that was written
        # without a trailing newline before the interrupt.
        console.file.write("\n")
        console.print(f"[{STYLE_TEXT_MUTED}]⌧ turn cancelled[/]")
        logger.info("REPL turn cancelled by user (Ctrl+C)")
        dispatcher.note_interrupt()
        _reset_runtime_after_cancelled_turn(runtime)
        runtime.tokens.session_turn_count += 1
        runtime.tokens.session_cancelled_count += 1
        elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
        runtime.tokens.session_total_latency_ms += elapsed_ms
        console.print(Rule(style=STYLE_RULE))
        # Swallow the cancellation so the REPL loop re-renders the prompt.
        # Without this re-raise guard, asyncio's cancellation propagation
        # would unwind the whole task despite the except block.
        return
    except Exception as exc:  # pylint: disable=broad-except
        if isinstance(transport, RichTransport):
            transport.stop_thinking(flush_transcript=True)
        elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
        runtime.tokens.session_error_count += 1
        runtime.tokens.session_total_latency_ms += elapsed_ms
        logger.error("REPL turn failed: %s: %s", type(exc).__name__, exc)
        _render_runtime_error(console, exc, title="Turn Error")
        console.print(Rule(style=STYLE_RULE))
        return
    elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
    runtime.tokens.session_turn_count += 1
    runtime.tokens.session_success_count += 1
    runtime.tokens.session_total_latency_ms += elapsed_ms
    mode_id = runtime.current_mode_id
    runtime.tokens.session_mode_counts[mode_id] = (
        runtime.tokens.session_mode_counts.get(mode_id, 0) + 1
    )

    # The transport's on_response already rendered the response panel during
    # the turn. If on_response was a no-op (e.g. NullTransport), render here
    # as a fallback — but RichTransport already printed it.
    if result.text and not isinstance(transport, RichTransport):
        await transport.on_response(result.text)

    # ── Operator exit indicator ────────────────────────────────────────
    # Surface the pause/stop/terminate discriminator so the user knows the
    # turn ended via an operator signal (or a cap that degraded to one).
    # The kernel handles checkpoint stamping and WorkflowPosition marking;
    # this is a visual indicator only.
    _op_exit = getattr(result, "operator_exit", "")
    if _op_exit == "paused":
        console.print(f"[{STYLE_TEXT_MUTED}]⏸ turn paused — send input to resume[/]")
    elif _op_exit == "stopped":
        console.print(f"[{STYLE_TEXT_MUTED}]⏹ turn stopped — state preserved[/]")
    elif _op_exit == "terminated":
        console.print(f"[{STYLE_TEXT_MUTED}]⏹ workflow terminated[/]")

    # Compact turn-summary footer: mode · model · tokens · elapsed, plus a
    # handoff note when the model switched modes mid-turn. Sourced from
    # ``runtime.last_turn_context`` (populated by handle_message) and the
    # per-turn elapsed timer. Skipped when there's no completion context
    # (e.g. an early-exit host-command path).
    if not isinstance(transport, RichTransport):
        _render_turn_footer(console, runtime, elapsed_ms)

    console.print(Rule(style=STYLE_RULE))


# ── Rendering helpers ────────────────────────────────────────────────


def _render_turn_footer(console: Console, runtime: HostRuntime, elapsed_ms: float) -> None:
    """Render a one-line post-response summary: mode · model · tokens · time.

    Draws from ``runtime.last_turn_context`` (the ``completion`` dict
    populated by ``handle_message``) plus the per-turn elapsed timer.
    When a mid-turn mode handoff occurred, appends ``→  <new mode>`` so
    the user sees the transition that happened during the turn. Quiet
    (no-op) when no completion context is available.
    """
    ctx = runtime.last_turn_context
    if ctx is None or not getattr(ctx, "completion", None):
        return
    comp = ctx.completion or {}
    mode_id = str(comp.get("mode_id") or ctx.mode_id or "default")
    resolved = comp.get("resolved_model") or {}
    if not isinstance(resolved, dict):
        resolved = {}
    model_id = str(resolved.get("model_id") or "")
    total_tokens = comp.get("total_tokens")
    parts: list[str] = []
    parts.append(f"▾ {mode_id}")
    if model_id:
        parts.append(model_id)
    if total_tokens:
        parts.append(f"{total_tokens} tok")
    parts.append(f"{int(elapsed_ms)}ms")
    # Mid-turn handoff: the host already updated runtime.current_mode_id
    # from result.handoff_to. If it differs from the entry mode, surface
    # the transition on the footer.
    entry_mode = str(comp.get("mode_id") or "")
    current = runtime.current_mode_id
    if current and current != entry_mode and entry_mode:
        parts.append(f"→ {current}")
    footer = "  ·  ".join(parts)
    console.print(Text(footer, style=STYLE_TEXT_MUTED))


def _print_startup_banner(console: Console) -> None:
    """Print a compact startup banner."""
    title = Text()
    title.append("Ready", style="bold steel_blue3")
    title.append("  ")
    title.append("interactive terminal host", style=STYLE_TEXT_MUTED)
    console.print(title)
    console.print()


def _render_session_panel_text(runtime: HostRuntime) -> str:
    """Build the informative Session panel shown at REPL startup.

    Surfaces the resolved user context, start mode, default model alias,
    and a pointer to the pack-aware /help overview, instead of a static
    hint string. Keeps the CLI host free of host-layer command vocabulary
    — it only reads runtime accessors.
    """
    from host.commands import model_alias_for_mode

    start_mode = runtime.start_mode_id or "default"
    model_alias = model_alias_for_mode(start_mode, manifest=runtime.manifest) or "default"
    user = runtime.config.user_slug or "(unset)"
    return (
        f"User: {user}  |  Start mode: {start_mode}  |  Model: {model_alias}\n"
        "Type /help for the command overview, /help <command> for per-command "
        "guidance, /help capabilities for the live inventory, or /quit to exit."
    )


def _render_runtime_error(console: Console, exc: Exception, *, title: str = "Error") -> None:
    error_text = f"{type(exc).__name__}: {exc}"
    logger.error("%s: %s: %s", title, type(exc).__name__, exc)
    console.print(
        Panel(
            error_text,
            title=title,
            border_style="indian_red",
        )
    )


def _reset_runtime_after_cancelled_turn(runtime: HostRuntime) -> None:
    """Reset per-turn runtime state after a cancelled (Ctrl+C) turn.

    A cancelled ``handle_message`` may have partially advanced the route
    preview (mode/stage/model) and left per-turn token counters dirty.
    The next prompt must show a clean status line and start from a known
    state, so:

    - Reset per-turn token counters (``tokens.reset_turn()``).
    - Restore the route preview to the session's start mode with neutral
      stage/model placeholders (the same defaults ``HostRuntime.__init__``
      sets). The next turn's routing path will repopulate
      ``next_model_id``/``next_model_provider`` via
      ``set_runtime_route_preview``. We do NOT call that helper here
      because it depends on ``get_workflow_catalog()``, which requires
      content-MCP registration — a dependency that is not safe in a
      mid-interrupt teardown path (and breaks unit tests that exercise
      the cancel handler without a full host boot).
    - Clear ``last_turn_context`` so stale half-built state isn't surfaced
      by ``/debug context``.

    This does NOT touch the MCP pool, the thread, or session-level
    accumulators (turn/error/cancelled counts are updated by the caller).
    """
    runtime.tokens.reset_turn()
    runtime.last_turn_context = None
    runtime.current_mode_id = runtime.start_mode_id
    runtime.current_stage_id = "-"
    runtime.next_model_id = "-"
    runtime.next_model_provider = "-"


def _render_restart_required_notice_if_needed(runtime: HostRuntime, console: Console) -> None:
    """Render a one-shot restart-required notice after pack lifecycle changes."""
    if not runtime.restart_required:
        return
    reason = runtime.restart_reason.strip() or "runtime change"
    console.print(
        Panel(
            "Restart required to reload entry points, pack catalogs, and MCP manifests.\n"
            f"Reason: {reason}",
            title="Restart Required",
            border_style="gold3",
        )
    )


def _confirm_restart(runtime: HostRuntime, console: Console) -> bool:
    """Ask the user yes/no before exiting the session to apply a restart.

    Returns ``True`` to proceed with the restart (exit/relaunch), or
    ``False`` to defer and keep the current session alive. On EOF
    (Ctrl+D) or KeyboardInterrupt, the restart is deferred so the
    user is not forced out of the session unexpectedly.
    """
    reason = runtime.restart_reason.strip() or "runtime change"
    prompt_text = f"Exit now to reload ({reason})? [Y/n]"
    try:
        return bool(
            Confirm.ask(
                prompt_text,
                console=console,
                default=True,
                show_choices=False,
                show_default=False,
            )
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False


def _render_session_exit_summary(runtime: HostRuntime, console: Console) -> None:
    """Render the session-exit summary panel from the runtime's token counters."""
    from host.commands import format_token_stats

    summary = format_token_stats(runtime)
    console.print(
        Panel(
            Markdown(summary),
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )


class _SlashCommandCompleter(Completer):
    """Tab-complete slash commands, but only when the input starts with ``/``.

    When the buffer does not start with ``/``, Tab is a no-op (no dropdown,
    no suggestions). This keeps Tab useful for slash-command discovery
    without polluting free-form chat input.
    """

    def __init__(self, words: list[str]) -> None:
        self._inner = WordCompleter(words, ignore_case=True)

    def get_completions(
        self, document: Document, complete_event: Any
    ) -> Any:
        if not document.text_before_cursor.lstrip().startswith("/"):
            return
        yield from self._inner.get_completions(document, complete_event)


# Minimum visible height (in terminal rows) of the multi-line input area.
# The REPL is a chat box: reserving at least this many rows keeps the prompt
# from collapsing to a single line when empty, so multi-line paste and the
# newline continuation glyph (…) have room to breathe.
_REPL_MIN_INPUT_HEIGHT = 3


def _build_repl_prompt_session(
    runtime: HostRuntime,
    console: Console,
    transport: RichTransport,
    dispatcher: TerminalSignalDispatcher,
) -> Any:
    """Build a prompt_toolkit PromptSession with history + completion.

    Returns ``None`` when prompt_toolkit is unavailable. When available,
    the session is multi-line: Enter submits the turn, Ctrl+J inserts
    a newline (chat-box semantics). A bottom toolbar shows the session
    name, live char/line count, and control hints while the prompt is
    active.

    The input area reserves at least ``_REPL_MIN_INPUT_HEIGHT`` rows so the
    prompt never collapses to a single line (ask: chat-box minimum height).
    """
    if PromptSession is None or FileHistory is None or Completer is None or Document is None:
        return None
    history_path = runtime.data_dir / _REPL_HISTORY_FILE_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    bindings = build_repl_key_bindings(transport, dispatcher) if KeyBindings is not None else None
    return _ReplPromptSession(
        history=_SurrogateSafeFileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory() if AutoSuggestFromHistory is not None else None,
        completer=_SlashCommandCompleter(list(_repl_command_hints())),
        complete_while_typing=False,
        multiline=True,
        key_bindings=bindings,
        prompt_continuation=[("class:prompt-continuation", "… ")],
        bottom_toolbar=lambda: _repl_bottom_toolbar(
            transport,
            session_name=runtime.session_plan.manifest.name,
            mode_id=runtime.current_mode_id,
        ),
        style=Style.from_dict(_REPL_PROMPT_STYLE) if Style is not None else None,
    )


class _ReplPromptSession(PromptSession):  # type: ignore[misc]
    """``PromptSession`` with a minimum input-area height for the REPL chat box.

    prompt_toolkit's default ``_get_default_buffer_control_height`` returns
    ``Dimension()`` (min=1) when no completion menu is due, so a
    ``multiline=True`` prompt collapses to a single row when empty. This
    subclass raises the floor to ``_REPL_MIN_INPUT_HEIGHT`` rows while
    preserving the autocompletion-menu space reservation from the base
    implementation.
    """

    def _get_default_buffer_control_height(self) -> Any:  # noqa: D401
        """Return the buffer-control window height with a 3-row minimum."""
        base = super()._get_default_buffer_control_height()
        # Preserve completion-menu reservation, but never below the REPL floor.
        from prompt_toolkit.layout import Dimension

        return Dimension(
            min=max(base.min or 1, _REPL_MIN_INPUT_HEIGHT),
            max=base.max,
            preferred=base.preferred,
        )


# prompt_toolkit style: maps the class: names used in the prompt and
# toolbar to hex colors (prompt_toolkit does not accept Rich color names
# like 'steel_blue3'). Steel_blue3 is ANSI-256 #74 -> #5fafff; grey58
# is #5f5f5f; grey39 is #626262.
_REPL_PROMPT_STYLE = {
    "prompt": "bold #5fafff",
    "prompt-continuation": "#5f5f5f",
    "bottom-toolbar": "reverse",
    "bottom-toolbar.text": "#626262",
    "bottom-toolbar.metric": "bold #5fafff",
    "bottom-toolbar.hint": "#878787",
}


def _repl_bottom_toolbar(
    transport: RichTransport, *, session_name: str = "", mode_id: str = ""
) -> Any:
    """Render the bottom toolbar: session name + mode + live char/line count + hints.

    Called by prompt_toolkit on every buffer mutation (and on explicit
    invalidation from the Ctrl+Y toggle). Returns
    ``FormattedText`` (a list of ``(style, text)`` tuples) so the toolbar
    updates in real time as the user types — and reflects the current
    thinking toggle state rather than a static hint string.
    """
    # Import inside the function so the module imports cleanly when
    # prompt_toolkit is absent (the toolbar is only used when it is present).
    from prompt_toolkit.formatted_text import FormattedText

    # prompt_toolkit stores the active buffer on the current application.
    # We reach it via get_app() rather than capturing a session closure so
    # the toolbar works regardless of which PromptSession calls it.
    try:
        from prompt_toolkit.application import get_app

        buffer = get_app().current_buffer
        text = buffer.text
    except Exception:  # pylint: disable=broad-except
        text = ""

    char_count = len(text)
    line_count = text.count("\n") + 1 if text else 0

    # Reflect the live toggle state in the hint segments so the toolbar
    # doubles as a status indicator, not just a static cheat-sheet.
    thinking_label = "thinking:off" if transport.thinking_collapsed else "thinking:on"

    # The session-context name (from its front matter) anchors the toolbar so
    # you always know which session plan is driving the current turn.
    session_label = session_name.strip() if session_name and session_name.strip() else "session"

    # The newline chord reflects runtime detection: Shift+Enter on
    # kitty/CSI-u terminals, Ctrl+J as the portable fallback (ask 4).
    newline_chord = "Shift+Enter" if transport.shift_enter_supported else "Ctrl+J"
    mode_label = mode_id.strip() if mode_id and mode_id.strip() else "default"
    return FormattedText(
        [
            ("class:bottom-toolbar", f" {session_label} "),
            ("class:bottom-toolbar.text", "│ "),
            ("class:bottom-toolbar.hint", "mode:"),
            ("class:bottom-toolbar.metric", f"{mode_label} "),
            ("class:bottom-toolbar.text", "│ "),
            (
                "class:bottom-toolbar",
                f" Enter submit · {newline_chord} newline · ↑/↓ history ",
            ),
            ("class:bottom-toolbar.text", "│ "),
            ("class:bottom-toolbar.hint", "Ctrl+Y "),
            ("class:bottom-toolbar.metric", thinking_label),
            ("class:bottom-toolbar.text", " · "),
            (
                "class:bottom-toolbar.metric",
                f"{line_count} line{'s' if line_count != 1 else ''}, {char_count} char{'s' if char_count != 1 else ''}",
            ),
            ("class:bottom-toolbar.text", " · "),
            ("class:bottom-toolbar.hint", "^P pause · ^O stop · ^\\ end"),
        ]
    )


__all__ = ["run_repl"]
