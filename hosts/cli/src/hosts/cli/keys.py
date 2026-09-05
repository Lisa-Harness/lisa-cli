"""Terminal signal dispatch and key-binding module for the CLI REPL.

Consolidates all physical keystroke and OS signal handling into a single,
typed layer. The :class:`TerminalSignalDispatcher` is constructed once per
session; it owns references to the runtime, transport, and console, and is
the single authority for translating user intent into
:class:`host.operator_signal.OperatorSignal` values.

Before this module existed, signal handling was spread across three
disconnected paths in :mod:`hosts.cli.repl`:
- Mid-turn ``CancelledError`` / ``KeyboardInterrupt`` (prompt_toolkit not
  focused).
- Prompt-level ``KeyboardInterrupt`` / ``EOFError`` (prompt_toolkit
  focused).
- Operator key bindings writing bare ``str`` values into
  ``runtime.operator_signal_value``.

This module provides:
- :class:`TerminalSignalDispatcher` — the single dispatch hub.
- :func:`build_repl_key_bindings` — prompt_toolkit bindings (extracted
  from :func:`hosts.cli.repl._build_repl_key_bindings`).
- :func:`install_loop_signal_handlers` — POSIX signal backstops.
- :func:`async_read_input` — prompt-level input with interrupt-aware
  exit (extracted from :func:`hosts.cli.repl._read_repl_input`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import Any

from host.operator_signal import OperatorSignal
from host.runtime import HostRuntime
from rich.console import Console

from hosts.cli.transport import RichTransport

logger = logging.getLogger(__name__)

#: prompt_toolkit filter helper; imported lazily (optional dependency).
try:
    from prompt_toolkit.filters import Condition as _PTCondition
except ImportError:  # pragma: no cover - optional dependency
    _PTCondition = None  # type: ignore[assignment,misc]

#: Window within which a second SIGINT/Ctrl+C constitutes a double-tap and
#: escalates to TERMINATE (force-quit) instead of cooperative cancel.
_DOUBLE_TAP_WINDOW_S = 2.0


# ── prompt_toolkit optional imports ─────────────────────────────────
try:
    from prompt_toolkit.key_binding import KeyBindings as _PTKeyBindings
except ImportError:  # pragma: no cover - optional dependency
    _PTKeyBindings = None  # type: ignore[assignment,misc]


# ── Style constants (imported from repl.py at call time to avoid cycles) ──
_STYLE_RULE = "grey42"


# ── Dispatcher ──────────────────────────────────────────────────────


class TerminalSignalDispatcher:
    """Single authority for translating user keystrokes into operator signals.

    Owns references to the runtime (for writing ``operator_signal_value``),
    the transport (for toggle/invalidation), and the console (for rendering).
    Constructed once in ``run_repl`` and passed to key bindings and input
    reader.

    Double-tap escalation:
        The first ``request_interrupt()`` within ``_DOUBLE_TAP_WINDOW_S``
        writes ``INTERRUPT`` (cooperative cancel, pool survives). A second
        call within the same window escalates to ``TERMINATE`` (force-quit).
        Calls outside the window reset the counter.
    """

    __slots__ = (
        "_runtime",
        "_transport",
        "_console",
        "_last_interrupt_at",
        "_interrupt_count",
    )

    def __init__(
        self,
        runtime: HostRuntime,
        transport: RichTransport,
        console: Console,
    ) -> None:
        self._runtime = runtime
        self._transport = transport
        self._console = console
        self._last_interrupt_at: float = 0.0
        self._interrupt_count: int = 0

    # ── Public dispatch methods (called from key bindings) ──────────

    def request_pause(self) -> None:
        """Write PAUSE to ``runtime.operator_signal_value``."""
        self._runtime.operator_signal_value = OperatorSignal.PAUSE
        logger.debug("operator signal: PAUSE")

    def request_stop(self) -> None:
        """Write STOP to ``runtime.operator_signal_value``."""
        self._runtime.operator_signal_value = OperatorSignal.STOP
        logger.debug("operator signal: STOP")

    def request_terminate(self) -> None:
        """Write TERMINATE to ``runtime.operator_signal_value``."""
        self._runtime.operator_signal_value = OperatorSignal.TERMINATE
        logger.debug("operator signal: TERMINATE")

    def request_toggle_thinking(self) -> None:
        """Toggle thinking visibility in the transport."""
        self._transport.toggle_thinking()

    def render_approval_view(self) -> None:
        """Re-render the armed approval panel (the 'v' view binding).

        Reads the pending payload from the transport and re-renders it here —
        the dispatcher owns the console, so this keeps key bindings from
        reaching private plumbing.
        """
        payload = self._transport.pending_approval
        if payload is None:
            return
        from hosts.cli.render import render_approval_panel  # noqa: PLC0415

        render_approval_panel(payload, console=self._console)

    def request_interrupt(self) -> None:
        """Request a cooperative cancel (first Ctrl+C) with double-tap escalation.

        First call within ``_DOUBLE_TAP_WINDOW_S`` → ``INTERRUPT``.
        Second call within the same window → ``TERMINATE`` (force quit).
        Calls outside the window reset the counter.
        """
        now = time.monotonic()
        if now - self._last_interrupt_at <= _DOUBLE_TAP_WINDOW_S:
            self._interrupt_count += 1
        else:
            self._interrupt_count = 1
        self._last_interrupt_at = now

        if self._interrupt_count >= 2:
            self._runtime.operator_signal_value = OperatorSignal.TERMINATE
            logger.debug("operator signal: TERMINATE (double-tap Ctrl+C)")
        else:
            self._runtime.operator_signal_value = OperatorSignal.INTERRUPT
            logger.debug("operator signal: INTERRUPT (Ctrl+C)")

    def request_eof(self) -> None:
        """Write EOF to ``runtime.operator_signal_value``."""
        self._runtime.operator_signal_value = OperatorSignal.EOF
        logger.debug("operator signal: EOF")

    def request_suspend(self) -> None:
        """Request process suspension (Ctrl+Z).

        On POSIX: flush transport, then send SIGTSTP to self.
        On Windows: Ctrl+Z is EOF — exit the session.
        """
        if os.name == "nt":
            self.request_eof()
        else:
            self._runtime.operator_signal_value = OperatorSignal.SUSPEND
            logger.debug("operator signal: SUSPEND (Ctrl+Z)")
            # Flush the transport's Live region before backgrounding so the
            # terminal state is clean on resume.
            self._transport.stop_thinking(flush_transcript=True)
            # SIGTSTP is POSIX-only; a no-op when the attribute is missing
            # (e.g. on Windows, or when the test suite monkeypatches os.name).
            if hasattr(signal, "SIGTSTP"):
                os.kill(os.getpid(), signal.SIGTSTP)

    def note_interrupt(self) -> None:
        """Called from ``_process_repl_turn``'s CancelledError handler.

        Does NOT write a signal (the interrupt already happened via SIGINT).
        Instead updates the double-tap counter so a second SIGINT within the
        window escalates.
        """
        now = time.monotonic()
        if now - self._last_interrupt_at <= _DOUBLE_TAP_WINDOW_S:
            self._interrupt_count += 1
        else:
            self._interrupt_count = 1
        self._last_interrupt_at = now

    def clear(self) -> None:
        """Reset ``operator_signal_value`` to ``NONE`` at turn start."""
        self._runtime.operator_signal_value = OperatorSignal.NONE

    # ── Process-level signal handlers (POSIX only) ───────────────────

    def on_sigint(self) -> None:
        """Handle SIGINT at the process level (backstop when prompt_toolkit
        is not focused).

        First signal → cooperative ``INTERRUPT``.
        Second within ``_DOUBLE_TAP_WINDOW_S`` → ``TERMINATE`` + hard cancel
        the main task. The window and escalation mirror
        :meth:`request_interrupt`.
        """
        self.request_interrupt()
        if self._runtime.operator_signal_value is OperatorSignal.TERMINATE:
            # Hard-cancel the main task for immediate teardown.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

    def on_sigterm(self) -> None:
        """Handle SIGTERM: graceful TERMINATE, then cancel."""
        self._runtime.operator_signal_value = OperatorSignal.TERMINATE
        logger.debug("operator signal: TERMINATE (SIGTERM)")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task():
                task.cancel()

    def on_sigtstp(self) -> None:
        """Handle SIGTSTP: flush and suspend."""
        self.request_suspend()


# ── Key bindings builder ────────────────────────────────────────────


def build_repl_key_bindings(
    transport: RichTransport,
    dispatcher: TerminalSignalDispatcher,
) -> Any:
    """Build chat-box key bindings for the multi-line REPL prompt.

    - Enter: submit the current buffer (chat-box semantics).
    - Shift+Enter: insert a newline (kitty/CSI-u terminals only).
    - Ctrl+J: insert a newline — portable fallback.
    - Ctrl+Y: toggle thinking visibility.
    - Ctrl+C: cooperative cancel first press; force-quit on second press
      within 2 seconds. At the prompt, exits the session.
    - Ctrl+D: EOF — exit the session (mirrors Ctrl+C at prompt).
    - Ctrl+Z: suspend on POSIX; EOF on Windows.
    - Ctrl+P: pause the in-flight turn.
    - Ctrl+O: stop the turn, return prompt, keep workflow state.
    - Ctrl+\\: terminate the workflow.

    Without these bindings, prompt_toolkit's default ``multiline=True``
    behaviour is the reverse (Enter inserts a newline; Esc→Enter submits),
    which surprises chat users.
    """
    if _PTKeyBindings is None:
        return None
    bindings = _PTKeyBindings()

    @bindings.add("enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    # Shift+Enter — only fires on kitty/CSI-u terminals.
    try:

        @bindings.add("s-enter")
        def _newline_shift_enter(event: Any) -> None:
            transport.shift_enter_supported = True
            event.current_buffer.insert_text("\n")
    except ValueError:
        pass

    @bindings.add("c-j")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-y")
    def _toggle_thinking(event: Any) -> None:
        dispatcher.request_toggle_thinking()
        _invalidate_repl_toolbar(event)

    # ── Terminal signals ─────────────────────────────────────────

    @bindings.add("c-c")
    def _interrupt(event: Any) -> None:
        """Ctrl+C: cooperative cancel.

        When at the prompt (empty buffer), treat as "exit session."
        Mid-turn: the dispatcher's double-tap escalation handles the
        first/second press distinction.
        """
        dispatcher.request_interrupt()
        _invalidate_repl_toolbar(event)
        buffer = event.current_buffer
        if buffer.text.strip() == "":
            # At the prompt → exit the session (same as EOF).
            event.app.exit(result=None)

    @bindings.add("c-d")
    def _eof(event: Any) -> None:
        """Ctrl+D: exit the session."""
        dispatcher.request_eof()
        event.app.exit(result=None)

    @bindings.add("c-z")
    def _suspend(event: Any) -> None:
        """Ctrl+Z: suspend (POSIX) or EOF (Windows)."""
        # On Windows, Ctrl+Z is the console EOF marker and prompt_toolkit
        # may already handle it. We still route through the dispatcher so
        # the signal log is consistent and the behavior is documented.
        dispatcher.request_suspend()
        _invalidate_repl_toolbar(event)
        if os.name == "nt":
            event.app.exit(result=None)

    # ── Operator signals (pause / stop / terminate) ───────────────

    @bindings.add("c-p")
    def _pause_turn(event: Any) -> None:
        dispatcher.request_pause()
        _invalidate_repl_toolbar(event)

    # Ctrl+. would be the intuitive mnemonic, but prompt_toolkit's key
    # parser rejects ``c-.`` (the ``c-`` prefix only parses control-letter
    # sequences, not punctuation), raising ValueError at binding-
    # construction time and crashing REPL startup. Ctrl+O is the closest
    # free control-letter slot.
    @bindings.add("c-o")
    def _stop_turn(event: Any) -> None:
        dispatcher.request_stop()
        _invalidate_repl_toolbar(event)

    @bindings.add("c-\\")
    def _terminate_turn(event: Any) -> None:
        dispatcher.request_terminate()
        _invalidate_repl_toolbar(event)

    _add_approval_bindings(bindings, transport, dispatcher)

    return bindings


def _current_repl_input() -> str:
    """Return the current REPL input buffer text ("" when unreadable)."""
    try:
        from prompt_toolkit.application import get_app  # noqa: PLC0415

        return str(get_app().current_buffer.text or "")
    except Exception:  # pylint: disable=broad-except
        return ""


def _add_approval_bindings(
    bindings: Any,
    transport: RichTransport,
    dispatcher: TerminalSignalDispatcher,
) -> None:
    """Attach the y/n/a/v/Esc approval-decision bindings to ``bindings``.

    When a turn pauses on an approval, the transport stashes the payload in
    ``transport.pending_approval`` and the approval panel shows y/n/a/v. These
    bindings are gated by a Condition so they intercept ONLY while a decision is
    armed and the input box is empty — otherwise y/n/a flow through as typed
    text. Deciding submits a slash command as the prompt result, which the REPL
    loop routes through handle_message -> the host (/approve, /deny, or
    /approve --remember). Split out of ``build_repl_key_bindings`` to keep that
    function under the lint branch/statement ceilings.
    """
    if _PTCondition is None:
        return
    _armed = _PTCondition(
        lambda: transport.pending_approval is not None and not _current_repl_input()
    )

    def _arm(command: str) -> None:
        # Consume the arm and submit the decision as the prompt's result.
        request_id = (transport.pending_approval or {}).get("request_id", "")
        transport.pending_approval = None
        return command.format(id=request_id)

    def _make_decision(template: str) -> Any:
        def _handler(event: Any) -> None:
            text = _arm(template)
            event.app.exit(result=text)

        return _handler

    bindings.add("y", filter=_armed)(_make_decision("/approve {id}"))
    bindings.add("n", filter=_armed)(_make_decision("/deny {id}"))
    bindings.add("a", filter=_armed)(_make_decision("/approve {id} --remember"))

    @bindings.add("v", filter=_armed)
    def _approval_view(event: Any) -> None:
        # Re-render the panel (the payload already carries the full command).
        dispatcher.render_approval_view()

    @bindings.add("escape", filter=_armed)
    def _approval_dismiss(event: Any) -> None:
        # Esc dismisses the panel without deciding: disarm and return to the
        # (empty) prompt so the operator can read/scroll and use /approve.
        transport.pending_approval = None


def _invalidate_repl_toolbar(event: Any) -> None:
    """Force prompt_toolkit to re-render the bottom toolbar after a toggle.

    The toolbar callback fires automatically only on buffer mutations; the
    thinking toggle mutates transport state without touching the buffer, so
    without an explicit invalidation the stale hint string stays on screen
    until the user types again.
    """
    try:
        event.app.invalidate()
    except Exception:  # pylint: disable=broad-except
        pass


# ── Process-level signal handlers (POSIX backstop) ───────────────────


def install_loop_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    dispatcher: TerminalSignalDispatcher,
) -> None:
    """Install loop-level signal handlers as a backstop for POSIX systems.

    These fire when prompt_toolkit is NOT focused (e.g. mid-turn, or when
    the plain ``console.input()`` fallback is active). When prompt_toolkit
    IS focused, the ``c-c`` / ``c-z`` key bindings fire first and these
    handlers are not reached.

    On Windows, ``loop.add_signal_handler`` does not support SIGINT; the
    ``c-c`` binding and ``asyncio.run()``'s default cancellation are the
    only paths. SIGBREAK is not wired — Ctrl+Break already hard-kills.
    """
    if os.name == "nt":
        return

    try:
        loop.add_signal_handler(signal.SIGINT, dispatcher.on_sigint)
    except (ValueError, NotImplementedError):
        pass

    try:
        loop.add_signal_handler(signal.SIGTSTP, dispatcher.on_sigtstp)
    except (ValueError, NotImplementedError):
        pass

    try:
        loop.add_signal_handler(signal.SIGTERM, dispatcher.on_sigterm)
    except (ValueError, NotImplementedError):
        pass


# ── Prompt-level input reader ───────────────────────────────────────


async def async_read_input(
    console: Console,
    prompt_session: Any,
    dispatcher: TerminalSignalDispatcher,
) -> str | None:
    """Read one line of REPL input, returning None on EOF or Ctrl+C.

    A Ctrl+C at the input prompt (not mid-turn) is treated as an exit
    request, mirroring EOF (Ctrl+D). This prevents the infinite loop
    where repeated Ctrl+C on an empty prompt returns ``""`` and
    ``_run_repl_loop``'s ``if not message: continue`` loops forever.

    Mid-turn Ctrl+C is handled separately by ``_process_repl_turn``'s
    ``CancelledError``/``KeyboardInterrupt`` handler, which cancels the
    turn and returns to the prompt — that path never reaches here.
    """
    # Deferred import to avoid top-level dependency on prompt_toolkit when
    # it is unavailable (the plain-console fallback path works without it).
    from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig as _ModalCursorShapeConfig

    try:
        if prompt_session is not None:
            cursor = _ModalCursorShapeConfig() if _ModalCursorShapeConfig is not None else None
            return (
                await prompt_session.prompt_async(
                    [("class:prompt", "❯ ")],
                    cursor=cursor,
                )
            ).strip()
        return console.input("[bold steel_blue3]❯ [/]").strip()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C at the prompt → exit the session (same as EOF). Do NOT
        # return "" — that makes _run_repl_loop loop forever on an empty
        # prompt with no way out except /quit.
        dispatcher.request_eof()
        console.print("", markup=False)
        from rich.rule import Rule

        console.print(Rule(style="grey42"))
        return None
    except EOFError:
        dispatcher.request_eof()
        from rich.rule import Rule

        console.print(Rule(style="grey42"))
        return None


__all__ = [
    "TerminalSignalDispatcher",
    "build_repl_key_bindings",
    "install_loop_signal_handlers",
    "async_read_input",
]
