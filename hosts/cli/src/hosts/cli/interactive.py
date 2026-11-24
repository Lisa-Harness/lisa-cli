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
from collections.abc import Callable
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
    "/help · /quit · Esc stops a running turn · Enter queues a follow-up",
    "approvals are decided in place; the request id survives if you defer",
)


class _SessionSwitch(Exception):
    """A conversation switch request unwinding the session loop.

    Raised by ``/resume <id>`` and ``/new``; carries the target
    conversation id (``"default"`` for a fresh conversation). The
    conversation cycle catches it, closes the current session, and
    rebuilds a runtime around the new identity.
    """

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id)
        self.conversation_id = conversation_id


def _print_help(console: Console) -> None:
    """Print the session-local control list (host commands stay real subcommands)."""
    console.print(
        Text(
            "\n".join(
                [
                    "Session controls:",
                    "  /help               this list",
                    "  /quit               end the session (also: exit, quit, Ctrl-D)",
                    "  /queue              list queued follow-ups (/q)",
                    "  /drop               drop the newest queued message",
                    "  /diff [staged|ref]  working-tree diff (read-only)",
                    "  /plan               the active workflow checklist (Ctrl+T pins it)",
                    "  /resume [id]        switch to a conversation (latest when bare)",
                    "  /new                start a fresh default conversation",
                    "  Esc                 stop the running turn; next message redirects it",
                    "  Ctrl-C              cancel the running turn; session stays alive",
                    "  Ctrl-C at prompt    first aborts entry; second exits",
                    "  while a turn runs   type to draft; Enter queues it as the next turn",
                    "  Approvals           a = allow · r <prefix> = allow + remember",
                    "                     d = deny · empty = defer",
                    "",
                    "Conversations: `lisa sessions` lists them; `lisa --resume <id>` starts here.",
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
    """Build the editor-based message reader (runs the PT editor off-loop).

    The returned callable takes one ``prefill`` argument — text the editor
    starts with (draft safety: an operator typing while a turn ran keeps
    their text; the PT reader seeds it into the editable buffer).
    """

    async def _read(prefill: str = "") -> str | None:
        return await asyncio.to_thread(
            read_prompt_message_tui,
            console,
            history_path=prompt_history_path(data_dir, profile),
            swapped_keys=None,
            data_dir=data_dir,
            profile=profile,
            prefill=prefill,
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
    latest: bool = False,
) -> int:
    """Run the interactive workspace until operator exit; returns the exit code.

    Boots a runtime for the starting ``conversation_id`` (from
    ``runtime_kwargs``), serves turns until quit/EOF, and honors
    ``/resume <id>`` and ``/new`` by closing the current session and
    rebuilding a runtime around the selected conversation — durable
    threads make the switch a clean rebuild, not a live path mutation.

    ``latest`` (``--continue`` / ``--resume`` with no id) picks the most
    recently active conversation before the first boot: the user slug is
    resolved from the session plan, the catalog picks the newest thread,
    and the session boots straight into it.

    The exit code mirrors the one-shot contract: 0 clean; 2 startup error
    (raised as ``typer.Exit`` by the constructor path); 1 teardown failure
    (the bounded close did not complete — the operator must know); 130 the
    conventional SIGINT-abort code for exit-via-Ctrl-C.
    """
    kwargs = dict(runtime_kwargs)
    if latest and not str(kwargs.get("conversation_id") or "").strip():
        resolved = _resolve_latest_conversation(kwargs)
        if resolved is not None:
            kwargs["conversation_id"] = resolved
    reader = read_message or _default_message_reader(console, data_dir, profile)
    decider = read_approval or _read_approval_answer
    exit_code = 0
    while True:
        session, switch, exit_code = await _run_conversation(console, kwargs, reader, decider)
        if switch is None:
            return exit_code
        # A conversation switch: close the old session cleanly and rebuild
        # with the new identity. ``conversation_id`` rides the runtime
        # kwargs (the env var alone cannot change mid-process).
        await _close_session(session, console)


async def _run_conversation(
    console: Console,
    runtime_kwargs: dict[str, Any],
    reader: Any,
    decider: Any,
) -> tuple[Any, str | None, int]:
    """Boot one runtime, serve its loop; returns (session, switch, exit code).

    The switch directive is ``"default"`` for ``/new`` or a conversation
    id for ``/resume <id>``; ``None`` means the operator exited. The exit
    code is the loop's code (0/130) or 1 on an unexpected session error.
    """
    from hosts.cli.cli_runtime import _initialize_runtime_or_exit
    from hosts.cli.transport import PromptTransport

    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return None, None, 2
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
    # Queue-while-working: Enter mid-turn queues the typed draft as the
    # next follow-up message (the engine owns the FIFO; the drain runs
    # after each settled turn). Duck-typed for test transports.
    if hasattr(transport, "set_submit_handler"):
        transport.set_submit_handler(_make_queue_submitter(session, transport, console))
    # Read-only plan panel: Ctrl+T toggles the derived checklist (the
    # surface owns thread derivation; the transport only renders).
    if hasattr(transport, "set_plan_provider"):
        transport.set_plan_provider(_make_plan_provider(session))
    try:
        await session.start()
        _print_banner(console, _model_hint(runtime))
        exit_code = await _session_loop(session, console, reader, decider)
        # Normal exit: close here — the conversation cycle only closes on
        # a switch (where the session must stay alive until the rebuild).
        if not await _close_session(session, console):
            exit_code = 1
        return session, None, exit_code
    except asyncio.CancelledError:
        # SIGINT during a turn cancels the loop task; the engine already
        # recovered the turn (pool warm, state reset). Close what we can.
        await _close_session(session, console)
        return session, None, 0
    except _SessionSwitch as switch:
        return session, switch.conversation_id, 0
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("interactive conversation failed: %s", exc)
        console.print(f"[red]Session error: {exc}[/red]")
        await _close_session(session, console)
        return session, None, 1


async def _close_session(session: Any, console: Console) -> bool:
    """Close one conversation's session; True when the close was clean.

    A teardown failure is loud (logged + red notice) and returns False so
    the caller can exit nonzero — never silently treated as success.
    """
    if session is None:
        return True
    try:
        await session.aclose()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("interactive session close failed: %s", exc)
        console.print(f"[red]Session teardown did not complete cleanly: {exc}[/red]")
        return False


def _resolve_latest_conversation(runtime_kwargs: dict[str, Any]) -> str | None:
    """The most recently active conversation id, or ``None`` when none exist.

    Resolves the user slug the same way the session boot does (a probe
    runtime construction — no pool start, no model, discarded afterward),
    then asks the catalog for the newest thread. Never raises: a probe or
    scan failure means "no latest to resume", not a broken session.
    """
    from hosts.cli.conversations import latest_conversation

    probe_kwargs = {
        key: value
        for key, value in runtime_kwargs.items()
        if key not in {"conversation_id", "in_process_mcp"}
    }
    try:
        from hosts.cli.cli_runtime import _initialize_runtime_or_exit

        runtime = _initialize_runtime_or_exit(**probe_kwargs)
        if runtime is None:
            return None
        record = latest_conversation(
            runtime.paths.thread_dir,
            user_slug=runtime.config.user_slug,
        )
        if record is None or record.turn_count <= 0:
            return None
        if record.conversation_id == "default":
            return None  # the default store is where an un-flagged boot lands anyway
        return record.conversation_id
    except Exception:  # pylint: disable=broad-except
        logger.debug("interactive: latest-conversation probe failed", exc_info=True)
        return None


def _make_plan_provider(session: InteractiveSession) -> Any:
    """Build the Ctrl+T plan provider: a synchronous checklist snapshot.

    The transport calls this on toggle (never per frame); derivation reads
    the durable thread through the runtime's read API. Failures render as
    an honest placeholder, never a broken toggle.
    """

    def _provide() -> Any:
        from hosts.cli.session_plan_view import render_plan_panel_sync

        return render_plan_panel_sync(session.runtime)

    return _provide


def _make_queue_submitter(
    session: InteractiveSession,
    transport: Any,
    console: Console,
) -> Any:
    """Build the mid-turn submit callback: queue the draft, stamp the count.

    The transport's key reader invokes this on Enter while a turn runs.
    ``$(...)`` expansion is deliberately NOT applied here: the queued text
    runs through the ordinary turn path later (expanding at drain time,
    when its side effects are observable), and a queue-time expansion
    failure would need its own error surface mid-frame.
    """

    def _submit(draft: str) -> None:
        try:
            count = session.queue_message(draft)
        except Exception as exc:  # pylint: disable=broad-except  # queue-limit/state
            console.print(f"[red]Could not queue the message: {exc}[/red]")
            return
        if hasattr(transport, "set_queued_count"):
            transport.set_queued_count(count)

    return _submit


async def _session_loop(
    session: InteractiveSession,
    console: Console,
    reader: Any,
    decider: Any,
) -> int:
    """Serve turns until quit; returns the session's exit code."""
    ctrl_c_at_prompt = False
    pending_draft: list[str] = []
    while True:
        try:
            message = await reader(pending_draft.pop(0) if pending_draft else "")
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
        handled = await _handle_session_message(session, console, message, decider)
        if handled in ("quit", "restart"):
            return 0
        if handled == "continue":
            continue
        # handled == "turn": preserve the composer draft for the next read.
        draft = _pop_transport_draft(session)
        if draft:
            pending_draft.append(draft)
        if await _drain_queue(session, console, decider):
            return 0


async def _handle_session_message(
    session: InteractiveSession,
    console: Console,
    message: str,
    decider: Any,
) -> str:
    """Dispatch one submitted message; returns ``turn``, ``continue``, ``quit``, or ``restart``."""
    step, arg = _classify_session_message(message)
    if step != "turn":
        return _handle_local_step(session, console, step, arg)
    expanded, failure = _expand_message(message)
    if failure:
        console.print(f"[red]Shell substitution failed: {failure}[/red]")
        return "continue"
    await _run_one_turn(session, console, expanded, decider)
    if session.runtime.session.restart_required:
        _render_restart_notice(console, session.runtime.session.restart_reason)
        return "restart"
    return "turn"


def _handle_local_step(
    session: InteractiveSession,
    console: Console,
    step: str,
    arg: str = "",
) -> str:
    """Serve one session-local control step; returns ``continue``/``quit``.

    ``resume``/``new`` raise :class:`_SessionSwitch` — the conversation
    cycle (not the message loop) owns closing and rebuilding the session.
    """
    if step == "quit":
        return "quit"
    display_steps: dict[str, Callable[[], object]] = {
        "help": lambda: _print_help(console),
        "queue": lambda: _render_queue(session, console),
        "queue_drop": lambda: _drop_last_queued(session, console),
        "diff": lambda: asyncio.ensure_future(_render_diff_command(session, console, arg)),
        "plan": lambda: asyncio.ensure_future(_render_plan_command(session, console)),
    }
    handler = display_steps.get(step)
    if handler is not None:
        handler()
        return "continue"
    if step == "new":
        raise _SessionSwitch("default")
    if step == "resume":
        target = _resolve_resume_target(session, console, arg)
        if target is not None:
            raise _SessionSwitch(target)
    return "continue"  # skip, or a resume that did not resolve


async def _render_diff_command(
    session: InteractiveSession,
    console: Console,
    scope: str,
) -> None:
    """``/diff [scope]`` — render the working tree through the workspace service."""
    from pathlib import Path

    from hosts.cli.commands.diff_view import render_diff
    from hosts.cli.workspace import read_diff

    stripped = scope.strip()
    staged = stripped.lower() == "staged"
    ref = "" if staged or not stripped else stripped
    result = await read_diff(Path.cwd(), staged=staged, ref=ref)
    title = ref or "Working tree"
    render_diff(result, console, title=title)


async def _render_plan_command(session: InteractiveSession, console: Console) -> None:
    """``/plan`` — render the active workflow plan from the durable thread.

    The view derives from the thread (single source of truth; never
    persisted), so the plan reflects what the conversation actually
    completed — read-only by design.
    """
    from hosts.cli.session_plan_view import render_plan_panel

    console.print(render_plan_panel(session.runtime))


def _resolve_resume_target(
    session: InteractiveSession,
    console: Console,
    arg: str,
) -> str | None:
    """Resolve a ``/resume`` argument to a conversation id, or ``None``.

    An explicit id resolves through the catalog (existence check with a
    helpful error for unknown ids); bare ``/resume`` picks the most
    recently active conversation other than the current one, if any.
    """
    from hosts.cli.conversations import latest_conversation, resolve_conversation

    runtime = session.runtime
    thread_dir = runtime.paths.thread_dir
    slug = runtime.config.user_slug
    current = runtime.config.conversation_id.strip() or "default"
    if arg:
        record = resolve_conversation(thread_dir, user_slug=slug, conversation_id=arg)
        if record is None:
            console.print(
                f"[red]No conversation {arg!r} found. Run `lisa sessions` to list them.[/red]"
            )
            return None
        return record.conversation_id
    record = latest_conversation(thread_dir, user_slug=slug)
    if record is None:
        console.print("[grey58]No conversations to resume yet.[/grey58]")
        return None
    if record.conversation_id == current:
        console.print("[grey58]That is already the current conversation.[/grey58]")
        return None
    return record.conversation_id


async def _drain_queue(
    session: InteractiveSession,
    console: Console,
    decider: Any,
) -> bool:
    """Run queued follow-ups FIFO; True when the session must restart and exit.

    A pending operator decision (approval/yield) pauses the drain — the
    operator's next message settles it, and the remaining queue keeps its
    order for the next drain. The check runs BEFORE each queued turn (the
    turn that queued these may itself have ended pending) and after it.
    """
    while True:
        if _pending_decision(session):
            return False
        queued = session._pop_queued()
        if queued is None:
            return False
        _stamp_queued_count(session)
        expanded, failure = _expand_message(queued)
        if failure:
            console.print(f"[red]Shell substitution failed: {failure}[/red]")
            continue
        await _run_one_turn(session, console, expanded, decider)
        if session.runtime.session.restart_required:
            _render_restart_notice(console, session.runtime.session.restart_reason)
            return True


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


def _classify_session_message(message: str) -> tuple[str, str]:
    """Classify one operator message into ``(step, arg)``.

    Steps: ``turn`` (expandable), ``quit``, ``help``, ``skip``, ``queue``,
    ``queue_drop``, ``resume <id>``, ``new``, ``diff [scope]``, ``plan``.
    The argument carries the target (resume id; diff scope: staged or a
    git ref), empty where the step's default applies.
    """
    stripped = message.strip()
    if not stripped:
        return "skip", ""
    lowered = stripped.lower()
    step = _WORD_STEPS.get(lowered)
    if step is not None:
        return step, ""
    if lowered.startswith("/resume"):
        return "resume", stripped.split(None, 1)[1].strip() if " " in stripped else ""
    if lowered.startswith("/diff"):
        return "diff", stripped.split(None, 1)[1].strip() if " " in stripped else ""
    return "turn", ""


#: Exact-match session-local words → steps. ``/resume <id>`` (with its
#: argument) and ordinary prose fall through to the classifier.
_WORD_STEPS: dict[str, str] = {
    **dict.fromkeys(_QUIT_WORDS, "quit"),
    **dict.fromkeys(_HELP_WORDS, "help"),
    "/queue": "queue",
    "/q": "queue",
    "/drop": "queue_drop",
    "/queue drop": "queue_drop",
    "/new": "new",
    "new session": "new",
    "new conversation": "new",
    "/resume": "resume",
    "/plan": "plan",
    "/diff": "diff",
}


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


def _render_queue(session: InteractiveSession, console: Console) -> None:
    """List queued follow-ups with their positions (oldest first)."""
    queued = session.queued_messages()
    if not queued:
        console.print("[grey58]Queue is empty.[/grey58]")
        return
    console.print(f"[grey58]Queued ({len(queued)}), oldest first:[/grey58]")
    for index, message in enumerate(queued, start=1):
        console.print(f"[grey58]  {index}.[/grey58] {message}")


def _drop_last_queued(session: InteractiveSession, console: Console) -> None:
    """Drop the newest queued message (the operator's most recent mistake)."""
    try:
        dropped = session.drop_queued(-1)
    except IndexError:
        console.print("[grey58]Queue is empty.[/grey58]")
        return
    console.print(f"[grey58]Dropped queued message: {dropped}[/grey58]")
    _stamp_queued_count(session)


def _stamp_queued_count(session: InteractiveSession) -> None:
    """Mirror the engine's queue length onto the transport's status chip."""
    transport = session.transport
    if hasattr(transport, "set_queued_count"):
        transport.set_queued_count(len(session.queued_messages()))


def _pop_transport_draft(session: InteractiveSession) -> str:
    """Take the in-turn composer's draft off the transport (draft safety)."""
    transport = session.transport
    pop = getattr(transport, "pop_draft", None)
    if pop is None:
        return ""
    draft = pop()
    return draft if isinstance(draft, str) else ""


def _pending_decision(session: InteractiveSession) -> bool:
    """True when the last outcome left an approval/yield the operator must settle."""
    outcome = session.last_outcome
    return outcome is not None and outcome.kind in {"approval", "yield"}


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
    conversation_id: str = "",
    latest: bool = False,
) -> None:
    """``lisa interactive`` — persistent session (registered by commands/host.py)."""
    code = asyncio.run(
        run_interactive_session(
            console=app_console,
            data_dir=data_dir,
            profile=profile,
            latest=latest,
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
                "conversation_id": conversation_id,
            },
        )
    )
    if code:
        raise typer.Exit(code=code)
