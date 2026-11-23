"""Persistent interactive Lisa session — one runtime, many turns.

``lisa interactive`` is the opt-in persistent surface (TUI roadmap
Milestone 1): boot one runtime, serve successive operator turns on the
warm MCP pool, decide approvals in place, answer model questions in the
same session, and close once on exit. The one-shot ``lisa prompt`` path is
untouched and remains the pipeline interface.

Structure (roadmap "architecture to establish"):

* :mod:`hosts.cli.session` — the toolkit-independent engine (runtime
  lifecycle, turn dispatch, approval decision, cancellation recovery).
* this module — the terminal surface: reading messages, rendering turn
  outcomes, collecting approval decisions, Ctrl-C interruption, quit and
  restart handling. Every turn goes through the engine; this module never
  drives the runtime directly.

Input reuses the one-shot editor machinery
(:func:`hosts.cli.entry.read_prompt_message_tui`): same key model (Enter
adds a line, Ctrl-J sends, chat-box via ``--swap-keys``), same history
file, same grapheme-safe editing, same ``$(...)`` expansion.

Interruption and exit
---------------------
While a turn runs, two stop paths exist, deliberately different:

* **Esc — cooperative stop.** The activity key reader maps a lone Esc to
  :meth:`InteractiveSession.request_stop` (an
  :class:`runtime.operator_signal.OperatorSignal` the kernel polls once
  per iteration). The turn finishes its current step, the kernel
  preserves the loop state, and the outcome classifies ``stopped``; the
  next message redirects the turn. Nothing is force-killed, so the warm
  pool stays valid — this is the supported stop for arbitrary in-flight
  tools (Milestone 2's safe-lifecycle contract).
* **Ctrl-C — cancel.** The in-flight turn's awaits are cancelled, the
  engine resets per-turn state, and the prompt returns. Use when the
  turn itself is wedged (a cooperative stop would never be polled).

While the editor is open, Ctrl-C aborts entry; a second consecutive Ctrl-C
aborts the session (the one-shot prompt's contract, so a hurried exit can
never leak a stray keystroke to the shell). EOF (Ctrl-D on an empty
message) ends the session cleanly. SIGINT during teardown is suppressed so
the bounded close is never abandoned mid-way.

Approvals
---------
A turn that pauses on approval renders the approval panel (the transport
draws it from ``data["approval"]``); this surface then collects a decision
inline and dispatches it through
:meth:`InteractiveSession.decide_approval`, which runs the post-approval
continuation turn on the same runtime. The decision vocabulary matches the
typed decision path: allow once, allow with a remembered command prefix,
deny, or defer (the request id is durable; ``lisa approve <id>`` in another
shell still works). Ctrl-C during the decision read defers — a keystroke
must never decide an approval for the operator.

Commands
--------
There is no slash-command grammar here beyond the two session-local
controls (``/help``, ``/quit``). Host operations remain typed subcommands
in a separate process (``lisa settings``, ``lisa models``, ...) — the
interactive session deliberately does not fork the command registry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.text import Text

from hosts.cli.application import console as app_console
from hosts.cli.entry import prompt_history_path, read_prompt_message_tui
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
from hosts.cli.session import InteractiveSession

if TYPE_CHECKING:
    from hosts.cli.session import TurnOutcome

logger = logging.getLogger(__name__)

_QUIT_WORDS = {"exit", "quit", "/exit", "/quit"}
_HELP_WORDS = {"/help"}

#: Session-local banner hints (the editor's own hint line carries the key
#: model; these name what is unique to the persistent session).
_BANNER_HINTS = (
    "/help · /quit · Esc stops a running turn · Ctrl-D exits",
    "approvals are decided in place; the request id survives if you defer",
)


def _is_quit(message: str) -> bool:
    """True for the session-local quit controls (never model content)."""
    return message.strip().lower() in _QUIT_WORDS


def _is_help(message: str) -> bool:
    """True for the session-local help listing."""
    return message.strip().lower() in _HELP_WORDS


def _print_help(console: Console) -> None:
    """Print the session-local control list (host commands stay real subcommands)."""
    console.print(
        Text(
            "\n".join(
                [
                    "Session controls:",
                    "  /help               this list",
                    "  /quit               end the session (also: exit, quit, Ctrl-D)",
                    "  Esc                 stop the running turn; next message redirects it",
                    "  Ctrl-C              cancel the running turn; session stays alive",
                    "  Ctrl-C at prompt    first aborts entry; second exits",
                    "  Approvals           a = allow · r <prefix> = allow + remember",
                    "                     d = deny · empty = defer",
                    "",
                    "Host operations are typed subcommands outside the session:",
                    "  lisa settings · lisa tokens · lisa models · lisa policy · lisa last",
                ]
            ),
            style="grey58",
        )
    )


def _print_banner(console: Console, model_hint: str) -> None:
    from rich.panel import Panel

    console.print(
        Panel(
            Text.assemble(
                ("Lisa interactive session\n", "bold bright_white"),
                ("\n".join(_BANNER_HINTS), "grey58"),
            ),
            title=model_hint or "Interactive",
            border_style="steel_blue3",
        )
    )


def _parse_approval_answer(answer: str) -> tuple[str, str]:
    """Parse an approval answer into ``(choice, remember_prefix)``.

    Vocabulary (single letters, words, or ``r <prefix>``):
    ``a``/``allow``/``yes`` → allow once; ``d``/``deny``/``no`` → deny;
    ``r``/``remember`` (optionally followed by a prefix) → allow and persist
    a matching rule; empty → defer (the request stays pending). Anything
    unrecognized defers rather than guessing an irreversible decision.
    """
    stripped = answer.strip()
    if not stripped:
        return "defer", ""
    lowered = stripped.lower()
    if lowered in {"a", "allow", "y", "yes", "approve"}:
        return "allow", ""
    if lowered in {"d", "deny", "n", "no"}:
        return "deny", ""
    if lowered in {"r", "remember"} or lowered.startswith(("r ", "remember ")):
        prefix = stripped.split(None, 1)[1].strip() if " " in stripped else ""
        return "allow", prefix
    return "defer", ""


def _read_approval_answer(console: Console) -> str | None:
    """Read one line for an approval decision (plain reader; single line).

    Returns ``None`` on Ctrl-C or EOF — the caller defers, because a stray
    keystroke must never decide an approval.
    """
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return None


def _default_message_reader(console: Console, data_dir: str, profile: str) -> Any:
    """Build the editor-based message reader (runs the PT editor off-loop)."""

    async def _read() -> str | None:
        return await asyncio.to_thread(
            read_prompt_message_tui,
            console,
            history_path=prompt_history_path(data_dir, profile),
            swapped_keys=None,
            data_dir=data_dir,
            profile=profile,
        )

    return _read


async def run_interactive_session(
    *,
    console: Console,
    data_dir: str,
    profile: str,
    runtime_kwargs: dict[str, Any],
    read_message: Any | None = None,
    read_approval: Any | None = None,
) -> int:
    """Run one persistent session to operator exit; returns the exit code.

    Boots the runtime once, serves turns until quit/EOF, then closes once.
    The exit code mirrors the one-shot contract: 0 clean; 2 startup error
    (raised as ``typer.Exit`` by the constructor path); 1 teardown failure
    (the bounded close did not complete — the operator must know); 130 the
    conventional SIGINT-abort code for exit-via-Ctrl-C.
    """
    from hosts.cli.cli_runtime import _initialize_runtime_or_exit
    from hosts.cli.transport import PromptTransport

    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return 2
    transport = PromptTransport(console=console)
    transport.streaming_mode = runtime.config.streaming.mode
    from hosts.cli.prompt import _collect_model_pricing

    transport.model_prices, transport.model_cache_prices = _collect_model_pricing(runtime)

    session = InteractiveSession(runtime=runtime, transport=transport)
    # Cooperative stop: a lone Esc while the activity screen is up maps to
    # the engine's operator-stop request (kernel-polled at the next
    # iteration boundary; loop state preserved). Duck-typed so test
    # transports need not implement the seam.
    if hasattr(transport, "set_stop_handler"):
        transport.set_stop_handler(session.request_stop)
    reader = read_message or _default_message_reader(console, data_dir, profile)
    decider = read_approval or _read_approval_answer
    exit_code = 0
    try:
        await session.start()
        _print_banner(console, _model_hint(runtime))
        exit_code = await _session_loop(session, console, reader, decider)
    except asyncio.CancelledError:
        # SIGINT during a turn cancels the loop task; the engine already
        # recovered the turn (pool warm, state reset). Nothing else to do.
        pass
    finally:
        try:
            await session.aclose()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("interactive session close failed: %s", exc)
            console.print(f"[red]Session teardown did not complete cleanly: {exc}[/red]")
            exit_code = 1
    return exit_code


async def _session_loop(
    session: InteractiveSession,
    console: Console,
    reader: Any,
    decider: Any,
) -> int:
    """Serve turns until quit; returns the session's exit code."""
    ctrl_c_at_prompt = False
    while True:
        try:
            message = await reader()
        except KeyboardInterrupt:
            message = None  # Ctrl-C at entry == aborted entry
        if message is None:
            # Entry aborted (Ctrl-C in the editor): abort-only, like the
            # one-shot double-Ctrl-C contract — first press warns, second exits.
            if _handle_prompt_interrupt(console, ctrl_c_at_prompt):
                return 130
            ctrl_c_at_prompt = True
            continue
        ctrl_c_at_prompt = False
        step = _classify_session_message(message)
        if step == "quit":
            return 0
        if step == "help":
            _print_help(console)
            continue
        if step == "skip":
            continue
        expanded, failure = _expand_message(message)
        if failure:
            console.print(f"[red]Shell substitution failed: {failure}[/red]")
            continue
        await _run_one_turn(session, console, expanded, decider)
        if session.runtime.session.restart_required:
            _render_restart_notice(console, session.runtime.session.restart_reason)
            return 0


def _handle_prompt_interrupt(console: Console, ctrl_c_at_prompt: bool) -> bool:
    """Handle one Ctrl-C/aborted-entry: True means exit now (130).

    The first press prints the exit hint and returns False (keep looping);
    a second consecutive press returns True. The caller owns the
    consecutive-state and resets it on any successful read.
    """
    if ctrl_c_at_prompt:
        return True
    console.print("[grey58]⏹ Ctrl-C again to exit.[/grey58]")
    return False


def _classify_session_message(message: str) -> str:
    """Classify one operator message: ``turn`` (expandable), ``quit``, ``help``, or ``skip``."""
    if not message.strip():
        return "skip"
    if _is_quit(message):
        return "quit"
    if _is_help(message):
        return "help"
    return "turn"


def _expand_message(message: str) -> tuple[str, str]:
    """Expand ``$(...)`` substitutions; returns (expanded, failure_text)."""
    from hosts.cli.expand import expand_shell_substitutions

    expanded, failures = expand_shell_substitutions(message)
    if failures:
        shown = "; ".join(f"$({cmd})" for cmd in failures)
        return message, shown
    return expanded, ""


async def _run_one_turn(
    session: InteractiveSession,
    console: Console,
    message: str,
    decider: Any,
) -> None:
    """Run one turn, deciding approvals inline until the turn settles."""
    from hosts.cli.transport import PromptTransport

    transport = session.transport
    if isinstance(transport, PromptTransport):
        transport.start_turn(message)
    outcome = await session.run_turn(message)
    while outcome.kind == "approval":
        approval = outcome.approval or {}
        request_id = str(approval.get("request_id") or "")
        try:
            answer = await decider(console)
        except KeyboardInterrupt:
            # An interrupt during the decision read defers — a keystroke
            # must never decide an approval for the operator.
            answer = None
        choice, remember = _parse_approval_answer(answer) if answer is not None else ("defer", "")
        if choice == "defer":
            console.print(
                "[grey58]⏸ Approval deferred; decide with `lisa approve <id>` "
                "or answer the next prompt.[/grey58]"
            )
            return
        outcome = await session.decide_approval(
            request_id,
            approve=choice == "allow",
            remember=remember or None,
        )
    if outcome.kind == "yield" and outcome.yielded_question:
        console.print(
            Text(
                "Answer the question in your next message; the turn resumes from it.\n",
                style="grey58",
            )
        )
    _render_outcome(console, outcome)


def _render_outcome(console: Console, outcome: TurnOutcome) -> None:
    """Render the engine outcome (the transport already streamed the body)."""
    if outcome.kind == "paused":
        console.print("[grey58]⏸ turn paused — send input to resume[/grey58]")
        return
    if outcome.kind == "stopped":
        if outcome.operator_exit == "terminated":
            console.print("[grey58]⏹ workflow terminated — state kept, no resume[/grey58]")
        else:
            console.print("[grey58]⏹ turn stopped — next message redirects it[/grey58]")
        return
    if outcome.kind == "cancelled":
        console.print("[grey58]⌧ turn cancelled[/grey58]")
        return
    if outcome.kind == "error":
        console.print("[red]⨯ turn failed; the session stays alive[/red]")
        return
    from rich.rule import Rule

    console.print(Rule(style="grey42"))


def _render_restart_notice(console: Console, reason: str) -> None:
    console.print(
        f"[gold3]Restart required ({reason}). Exiting; run `lisa interactive` again "
        "to pick up the change.[/gold3]"
    )


def _model_hint(runtime: Any) -> str:
    """A compact route line for the banner title (mode · model when known)."""
    try:
        mode = runtime.session.current_mode_id or "default"
        next_model = str(runtime.session.next_model_id or "")
        manifest = runtime.session.manifest
        for model in getattr(manifest, "models", ()) or ():
            if str(getattr(model, "model_id", "")) == next_model:
                return f"{mode} · {model.model_id}"
        return mode
    except Exception:  # pylint: disable=broad-except
        return "Interactive"


def interactive_command(
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
    """``lisa interactive`` — persistent session (registered by commands/host.py)."""
    code = asyncio.run(
        run_interactive_session(
            console=app_console,
            data_dir=data_dir,
            profile=profile,
            runtime_kwargs={
                "log_level": log_level,
                "log_path": log,
                "no_log": no_log,
                "data_dir": data_dir,
                "profile": profile,
                "overlays_dir": overlays_dir,
                "thread_dir": thread_dir,
                "no_thread": no_thread,
                "session_context": session_context,
                "timezone": timezone,
                "state_snapshot_path": state_snapshot,
                "streaming": streaming,
                "host_log_path": host_log,
                "in_process_mcp": True,
            },
        )
    )
    if code:
        raise typer.Exit(code=code)
