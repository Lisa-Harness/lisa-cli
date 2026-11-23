"""Toolkit-independent engine for a persistent Lisa interactive session.

One :class:`InteractiveSession` owns one :class:`~runtime.runtime.HostRuntime`
for its whole life: the runtime (with its warm MCP pool) starts once, serves
any number of operator turns, and closes exactly once. This module holds the
lifecycle and dispatch logic only — no Rich, Typer, prompt_toolkit, or any
terminal library may be imported here, so the engine is drivable from tests
and from any interactive surface.

Turn outcomes
-------------
:func:`run_turn` never raises for turn-level failures; it classifies the
outcome instead so a caller can render it and continue the session:

* ``"ok"`` — the turn completed with a response.
* ``"error"`` — the turn failed (model, provider, or dispatch error). The
  runtime and its pool stay alive; the operator can try again.
* ``"cancelled"`` — the operator interrupted the turn. In-flight awaits are
  cancelled, the per-turn runtime state is reset, and the pool is reused.
* ``"paused"`` — the turn ended through an operator pause signal or a cap
  that degraded to one; conversation state is preserved and the next
  message resumes from it.
* ``"stopped"`` — the operator asked the turn to stop (Esc /
  :meth:`InteractiveSession.request_stop`) and the kernel honored it at
  an iteration boundary. Same preservation as ``paused``; the exit
  discriminator on the outcome says ``stopped`` (resume by redirecting)
  or ``terminated`` (workflow marked complete, no resume).
* ``"approval"`` — the turn paused pending an approval decision. The engine
  hands the structured approval payload to the caller; the caller collects a
  decision and calls :meth:`InteractiveSession.decide_approval`, which runs
  the post-approval continuation turn on the SAME runtime.
* ``"yield"`` — the model asked the operator a question instead of
  finishing. The question is exposed on the outcome; the operator's next
  message is the answer and resumes the task.

Cancellation semantics
----------------------
SIGINT-style cancellation of ``handle_message`` cancels the in-flight
model/tool coroutines but must NOT close the MCP pool (the pool owns
subprocesses bound to this loop; anyio forbids closing them from another
task/loop). The old REPL proved this shape works (its cancelled turn kept
the pool warm). The engine resets the per-turn runtime state the cancelled
coroutine may have partially advanced (route preview, stage pointer), clears
the one-shot operator signal, and keeps session counters honest.

Startup and restart
-------------------
``start()`` mirrors the one-shot prompt path: bootstrap + runtime
construction happen BEFORE the event loop owns anything async; the runtime
starts inside :meth:`InteractiveSession.start` (same-loop rule). A pack
install/update can set ``runtime.session.restart_required`` — the engine
exposes it on the outcome after each turn so the surface can ask the
operator and stop the session rather than silently continuing with a stale
pool.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.callbacks import TransportCallbacks
    from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)


class SessionStateError(RuntimeError):
    """A turn was requested while the session engine is not in an idle state."""


class QueueLimitError(RuntimeError):
    """The session's queued-message bound was reached; the message was refused."""


#: Bound on queued messages. A queue is a short "next up" list for a running
#: turn, not a durable backlog; an unbounded one invites runaway memory from
#: a pasted script or a stuck key repeat.
_QUEUE_MAX_MESSAGES = 16


@dataclass(slots=True)
class TurnOutcome:
    """The structural result of one engine turn.

    ``kind`` is one of ``ok``, ``error``, ``cancelled``, ``paused``,
    ``stopped``, ``approval``, or ``yield`` (see the module docstring).
    The remaining fields carry what the surface needs to render and
    continue: the response text, the raw result ``data``, the structured
    approval payload on an approval pause, the model's question on a
    yield, the operator-exit discriminator (``stopped``/``terminated``
    on a cooperative stop), and whether the runtime is asking for a
    restart.
    """

    kind: str = "ok"
    text: str = ""
    data: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    yielded_question: str = ""
    operator_exit: str = ""
    restart_required: bool = False


@dataclass
class InteractiveSession:
    """Own one runtime across many operator turns on one event loop."""

    runtime: HostRuntime
    transport: TransportCallbacks
    #: Bounded close grace (seconds) matching the one-shot CLI's teardown bound.
    close_timeout_s: float = 30.0

    _started: bool = field(default=False, init=False, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)
    _turn_active: bool = field(default=False, init=False, repr=False, compare=False)
    #: FIFO of operator messages submitted while a turn runs (or between
    #: decision rounds). Drained by the surface after each settled turn;
    #: approval/yield states preempt the drain — the operator must settle
    #: the pending state first, and queued work stays queued.
    _queue: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    #: The last completed turn's outcome (``None`` before the first turn).
    #: Surfaces read it to decide whether the queue drain must pause for a
    #: pending operator decision; the engine never mutates it retroactively.
    last_outcome: TurnOutcome | None = field(default=None, init=False, repr=False, compare=False)
    #: The structured approval payload of the turn that last paused on one
    #: (``data["approval"]``), or ``None``. Cleared at the start of the next
    #: turn and refreshed by classification.
    _pending_approval: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _pending_yield_question: str = field(default="", init=False, repr=False, compare=False)

    @property
    def turn_running(self) -> bool:
        """True while a turn (or approval decision) is executing."""
        return self._turn_active

    def request_stop(self) -> None:
        """Ask the running turn to stop cooperatively (kernel operator signal).

        The signal is polled once per turn-loop iteration by the kernel
        (``TurnHooks.operator_signal`` → :class:`runtime.operator_signal.OperatorSignal`):
        the turn finishes its current step, the kernel persists the loop
        state, and the outcome comes back classified ``paused`` with
        ``operator_exit="stopped"`` (or ``"terminated"`` for the terminal
        variant). No state is destroyed — the next message resumes from
        the preserved loop — and nothing here forces an exception, so the
        warm MCP pool stays valid.

        This is the *only* supported stop path for arbitrary in-flight
        tools: a hard cancel of an unknown tool's coroutine says nothing
        about the tool's own children or the turn's persisted state, so
        the engine never does it implicitly (Milestone 2's contract).
        """
        session = getattr(self.runtime, "session", None)
        if session is None or not hasattr(session, "operator_signal_value"):
            return
        from runtime.operator_signal import OperatorSignal

        session.operator_signal_value = OperatorSignal.STOP

    def request_terminate(self) -> None:
        """Ask the running turn to stop AND mark the workflow complete.

        The terminal variant of :meth:`request_stop`: same cooperative
        polling, but the kernel marks the WorkflowPosition complete — no
        resume. Use for "abandon this task"; use :meth:`request_stop` for
        "stop here, I'll redirect".
        """
        session = getattr(self.runtime, "session", None)
        if session is None or not hasattr(session, "operator_signal_value"):
            return
        from runtime.operator_signal import OperatorSignal

        session.operator_signal_value = OperatorSignal.TERMINATE

    # ── Queued follow-ups (Milestone 3) ─────────────────────────────────

    def queue_message(self, message: str) -> int:
        """Queue a follow-up message to run after the active turn settles.

        Returns the queue's new length. Whitespace-only messages are
        ignored (return the current length) — an accidental Enter submits
        nothing. Raises :class:`SessionStateError` on a closed session and
        :class:`QueueLimitError` past the bound, so a runaway paste cannot
        grow the queue without the operator hearing about it.

        Queueing is safe at any moment (idle, mid-turn, paused-for-approval):
        the queue is a plain FIFO drained by the surface only after a turn
        settles without a pending decision.
        """
        if self._closed:
            raise SessionStateError("The session is closed; build a new session.")
        if not message.strip():
            return len(self._queue)
        if len(self._queue) >= _QUEUE_MAX_MESSAGES:
            raise QueueLimitError(
                f"Queue is full ({_QUEUE_MAX_MESSAGES} messages); "
                "wait for the running turn to settle."
            )
        self._queue.append(message)
        return len(self._queue)

    def queued_messages(self) -> list[str]:
        """A copy of the queued follow-ups, oldest first."""
        return list(self._queue)

    def drop_queued(self, index: int = -1) -> str:
        """Remove one queued message (default: the newest) and return it.

        Index follows list semantics: ``-1`` drops the newest, ``0`` the
        oldest. An out-of-range index raises ``IndexError`` — the caller
        refreshed from :meth:`queued_messages` and raced; re-read and retry.
        """
        return self._queue.pop(index)

    def _pop_queued(self) -> str | None:
        """Pop the oldest queued message, or ``None`` when the queue is empty."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    async def start(self) -> None:
        """Start the runtime's MCP pool inside the session's event loop.

        Idempotent: a second call is a no-op. Teardown happens once in
        :meth:`aclose`; the runtime is unusable after that (same contract as
        the one-shot path).
        """
        if self._closed:
            raise SessionStateError("The session is closed; build a new session.")
        if self._started:
            return
        await self.runtime.start()
        self._started = True

    async def run_turn(self, message: str) -> TurnOutcome:
        """Run one operator message (host command or model turn) on the live runtime.

        The engine never raises for turn-level failures — see the module
        docstring for the outcome kinds. Raises :class:`SessionStateError`
        only for engine misuse (turn before start, turn after close, or a
        second turn while one is already running).
        """
        self._require_ready()
        if self._turn_active:
            raise SessionStateError("A turn is already running; queue the message instead.")
        self._turn_active = True
        started_perf = perf_counter()
        try:
            # One-shot operator signal: consumed by the turn that observes it.
            self._clear_operator_signal()
            try:
                result = await self._handle_message(message)
            except asyncio.CancelledError:
                await self._recover_after_cancel()
                return self._remember_outcome(TurnOutcome(kind="cancelled"))
            except KeyboardInterrupt:
                await self._recover_after_cancel()
                return self._remember_outcome(TurnOutcome(kind="cancelled"))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("interactive turn failed: %s: %s", type(exc).__name__, exc)
                await self.transport.on_error(f"Turn failed: {type(exc).__name__}: {exc}", exc)
                self._record_session_turn(succeeded=False, started_perf=started_perf)
                return self._remember_outcome(TurnOutcome(kind="error"))
        finally:
            self._turn_active = False
        outcome = self._outcome_from_result(result)
        self._record_session_turn(
            succeeded=outcome.kind in {"ok", "paused", "stopped", "approval", "yield"},
            started_perf=started_perf,
        )
        self._pending_approval = None
        self._pending_yield_question = ""
        return self._remember_outcome(outcome)

    def _remember_outcome(self, outcome: TurnOutcome) -> TurnOutcome:
        """Record the turn's outcome for surface reads (queue-drain preemption)."""
        self.last_outcome = outcome
        return outcome

    async def decide_approval(
        self,
        request_id: str,
        *,
        approve: bool,
        remember: str | None = None,
        remember_default: bool = False,
    ) -> TurnOutcome:
        """Decide one approval, running the post-approval continuation turn.

        Uses the typed decision seam
        (:func:`runtime.handle_message.decide_pending_approval`) so the
        continuation turn reuses the single-turn machinery on the same
        runtime and pool. An empty ``request_id`` resolves to the single
        pending request; the caller is expected to pass the id from the
        outcome's approval payload (or let the engine resolve it).
        """
        self._require_ready()
        if self._turn_active:
            raise SessionStateError("A turn is already running.")
        self._turn_active = True
        try:
            from runtime.handle_message import decide_pending_approval

            resolved = request_id or await self._resolve_pending_request_id()
            if not resolved:
                return TurnOutcome(
                    kind="error",
                    text="No pending approval to decide. Run a turn first.",
                )
            try:
                result = await decide_pending_approval(
                    self.runtime,
                    resolved,
                    approve=approve,
                    transport=self.transport,
                    remember=remember,
                    remember_default=remember_default,
                )
            except asyncio.CancelledError:
                await self._recover_after_cancel()
                return self._remember_outcome(TurnOutcome(kind="cancelled"))
            except KeyboardInterrupt:
                await self._recover_after_cancel()
                return self._remember_outcome(TurnOutcome(kind="cancelled"))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("approval decision failed: %s: %s", type(exc).__name__, exc)
                await self.transport.on_error(
                    f"Approval decision failed: {type(exc).__name__}: {exc}", exc
                )
                return self._remember_outcome(TurnOutcome(kind="error"))
        finally:
            self._turn_active = False
        self._pending_approval = None
        self._pending_yield_question = ""
        return self._remember_outcome(self._outcome_from_result(result))

    def _require_ready(self) -> None:
        """Raise the engine state errors for a closed or not-yet-started session."""
        if self._closed:
            raise SessionStateError("The session is closed; build a new session.")
        if not self._started:
            raise SessionStateError("Call start() before running a turn.")

    async def aclose(self) -> None:
        """Close the runtime once, bounded, keeping the pool's task ownership.

        The bounded close runs ``runtime.aclose()`` in the calling task (the
        anyio cross-task rule — see :func:`hosts.cli.cli_runtime._aclose_runtime_bounded`).
        Fails closed on teardown problems: a timeout or close exception
        raises :class:`RuntimeTeardownError <hosts.cli.cli_runtime.RuntimeTeardownError>`
        so the surface exits nonzero instead of reporting success over a pool
        that never closed.
        """
        if self._closed:
            return
        if not self._started:
            self._closed = True
            return
        from hosts.cli.cli_runtime import _aclose_runtime_bounded

        await _aclose_runtime_bounded(self.runtime, timeout_s=self.close_timeout_s)
        self._closed = True
        self._started = False

    # ── Internal: dispatch and outcome classification ────────────────────

    async def _handle_message(self, message: str) -> Any:
        from runtime.handle_message import handle_message

        return await handle_message(self.runtime, message, self.transport)

    def _outcome_from_result(self, result: Any) -> TurnOutcome:
        """Map a ``HostCommandResult``/turn result onto the engine outcome.

        Approval payloads ride on ``data["approval"]`` (panel transports
        read the same field); a yield rides on ``yielded_question``/``data["yielded"]``.
        """
        data = getattr(result, "data", None) if result is not None else None
        data = data if isinstance(data, dict) else {}
        approval = data.get("approval")
        approval = approval if isinstance(approval, dict) else None
        yielded_question = str(getattr(result, "yielded_question", "") or "")
        if not approval and not yielded_question:
            nested_approval = None
            if isinstance(data.get("schema"), str) and str(data["schema"]).startswith(
                "lisa.approval.v1"
            ):
                nested_approval = data
            approval = nested_approval
        operator_exit = str(getattr(result, "operator_exit", "") or "")
        restart_required = bool(getattr(self.runtime.session, "restart_required", False))
        kind = "ok"
        if approval is not None:
            kind = "approval"
            self._pending_approval = approval
        elif yielded_question:
            kind = "yield"
            self._pending_yield_question = yielded_question
        elif operator_exit in {"stopped", "terminated"}:
            kind = "stopped"
        elif operator_exit == "paused":
            kind = "paused"
        elif bool(getattr(result, "is_error", False)):
            kind = "error"
        return TurnOutcome(
            kind=kind,
            text=str(getattr(result, "text", "") or ""),
            data=data,
            approval=approval,
            yielded_question=yielded_question,
            operator_exit=operator_exit,
            restart_required=restart_required,
        )

    async def _resolve_pending_request_id(self) -> str:
        """Return the single pending approval's request id, or ``""`` when ambiguous/none."""
        service = getattr(self.runtime.services, "approval_service", None)
        if service is None or not hasattr(service, "list_pending"):
            return ""
        from runtime.approval import approval_identity

        try:
            pending = await service.list_pending(approval_identity(self.runtime))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("interactive: pending-approval resolution failed: %s", exc)
            return ""
        if len(pending) == 1:
            return str(pending[0].invocation.request_id)
        return ""

    async def _recover_after_cancel(self) -> None:
        """Reset per-turn state after a cancelled turn, keeping the pool warm.

        ``handle_message``'s own error paths clear the operator signal on
        operation errors, but a cancelled coroutine can return before the
        clear runs. The same runtime then keeps serving turns — a partially
        advanced route preview or stage pointer must not bleed into the
        next message.
        """
        self._clear_operator_signal()
        session = getattr(self.runtime, "session", None)
        if session is None:
            return
        session.tokens.session_cancelled_count += 1
        if getattr(session, "auto_approved_this_turn", 0):
            session.auto_approved_this_turn = 0
        if getattr(session, "last_turn_context", None) is not None:
            session.last_turn_context = None

    def _record_session_turn(self, *, succeeded: bool, started_perf: float) -> None:
        """Count one session turn on the runtime's session ledger.

        Mirrors the retired REPL's turn accounting (success/error counters
        plus elapsed latency) through the runtime's ``record_session_turn``
        helper, so ``/tokens`` and the session-exit summary see interactive
        turns. Cancelled turns are counted by the caller directly (the
        helper models only succeeded/failed).
        """
        session = getattr(self.runtime, "session", None)
        if session is None:
            return
        elapsed_ms = (perf_counter() - started_perf) * 1000.0
        mode_id = str(getattr(session, "current_mode_id", "") or "") or None
        try:
            from runtime.token_accounting import record_session_turn

            state: dict[str, Any] = {
                "session_turn_count": session.tokens.session_turn_count,
                "session_success_count": session.tokens.session_success_count,
                "session_error_count": session.tokens.session_error_count,
                "session_total_latency_ms": session.tokens.session_total_latency_ms,
                "session_mode_counts": dict(session.tokens.session_mode_counts),
            }
            record_session_turn(state, succeeded=succeeded, latency_ms=elapsed_ms, mode_id=mode_id)
            session.tokens.session_turn_count = int(state["session_turn_count"])
            session.tokens.session_success_count = int(state["session_success_count"])
            session.tokens.session_error_count = int(state["session_error_count"])
            session.tokens.session_total_latency_ms = float(state["session_total_latency_ms"])
            session.tokens.session_mode_counts = dict(state.get("session_mode_counts") or {})
        except Exception:  # pylint: disable=broad-except
            # Counters are diagnostics; a ledger failure must not fail a turn.
            logger.debug("interactive: session turn accounting failed", exc_info=True)

    def _clear_operator_signal(self) -> None:
        """Clear the one-shot operator signal (anyio-safe, idempotent)."""
        session = getattr(self.runtime, "session", None)
        if session is not None and hasattr(session, "operator_signal_value"):
            session.operator_signal_value = ""
