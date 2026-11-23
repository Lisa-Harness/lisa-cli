"""Tests for the interactive session engine (:mod:`hosts.cli.session`).

The engine is toolkit-independent, so these tests drive it directly with a
stub runtime and a recording transport — no terminal, no MCP subprocess, no
model. The scripted-model integration suite (integration/) proves the real
multi-turn lifecycle; here we pin the engine's contracts:

- one start / one close across many turns (idempotence, state errors);
- outcome classification (ok / error / approval / yield / paused);
- approval decision dispatch through the typed seam;
- cancellation recovery keeps the runtime usable and counters honest;
- a failed turn does not poison the next one.

Runtime construction goes through the CLI's own constructor seam via
``runtime.testing.build_test_runtime`` — the same stub the decision tests
use.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from hosts.cli.session import InteractiveSession, SessionStateError


class _RecordingTransport:
    """Minimal ``TransportCallbacks`` stub capturing events."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.turns_started: list[str] = []

    def start_turn(self, message: str) -> None:
        self.turns_started.append(message)

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self.errors.append(message)


def _result(
    *,
    text: str = "ok",
    data: dict[str, Any] | None = None,
    is_error: bool = False,
    operator_exit: str = "",
    yielded_question: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        data=data,
        is_error=is_error,
        operator_exit=operator_exit,
        yielded_question=yielded_question,
    )


async def _stub_handle_message(runtime: Any, message: str, transport: Any) -> Any:
    runtime.session.messages.append(message)
    return _result(text=f"echo:{message}")


@pytest.fixture
def stub_runtime(tmp_path: Any) -> Any:
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    runtime.session.messages = []
    runtime.session._handle_calls = 0
    return runtime


def _make_session(stub_runtime: Any) -> InteractiveSession:
    transport = _RecordingTransport()
    return InteractiveSession(runtime=stub_runtime, transport=transport)


def _patch_handle(monkeypatch: pytest.MonkeyPatch, impl: Any) -> list[str]:
    """Patch ``runtime.handle_message.handle_message``; record call count."""
    import runtime.handle_message as hm_module

    calls: list[str] = []

    async def _wrapper(runtime: Any, message: str, transport: Any) -> Any:
        calls.append(message)
        return await impl(runtime, message, transport)

    monkeypatch.setattr(hm_module, "handle_message", _wrapper)
    return calls


# ── Lifecycle ────────────────────────────────────────────────────────


def test_start_is_idempotent_and_close_is_once(stub_runtime: Any) -> None:
    """start() twice is one runtime.start(); aclose() twice is one close."""
    starts: list[int] = []
    closes: list[int] = []

    async def _fake_start() -> None:
        starts.append(1)

    async def _fake_aclose() -> None:
        closes.append(1)

    stub_runtime.start = _fake_start  # type: ignore[method-assign]
    stub_runtime.aclose = _fake_aclose  # type: ignore[method-assign]
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        await session.start()
        await session.aclose()
        await session.aclose()

    asyncio.run(scenario())
    assert len(starts) == 1
    assert len(closes) == 1


def test_turn_before_start_is_a_state_error(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        with pytest.raises(SessionStateError):
            await session.run_turn("hi")

    asyncio.run(scenario())


def test_turn_after_close_is_a_state_error(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        await session.aclose()
        with pytest.raises(SessionStateError):
            await session.run_turn("hi")

    asyncio.run(scenario())


def test_overlapping_turn_is_rejected(stub_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second turn while one runs is a state error, not a deadlock."""
    session = _make_session(stub_runtime)
    inner_entered = asyncio.Event()

    async def _slow_turn(runtime: Any, message: str, transport: Any) -> Any:
        inner_entered.set()
        await asyncio.sleep(0.2)
        return _result()

    _patch_handle(monkeypatch, _slow_turn)

    async def scenario() -> None:
        await session.start()
        first = asyncio.create_task(session.run_turn("first"))
        await asyncio.wait_for(inner_entered.wait(), timeout=5)
        with pytest.raises(SessionStateError):
            await session.run_turn("second")
        await first

    asyncio.run(scenario())


# ── Outcome classification ───────────────────────────────────────────


def test_ok_turn_classification(stub_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_handle(monkeypatch, _stub_handle_message)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("hello")
        assert outcome.kind == "ok"
        assert outcome.text == "echo:hello"
        assert outcome.restart_required is False
        assert calls == ["hello"]

    asyncio.run(scenario())


def test_stop_request_sets_operator_signal(stub_runtime: Any) -> None:
    """request_stop writes the kernel-polled STOP operator signal."""
    from runtime.operator_signal import OperatorSignal

    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        assert stub_runtime.session.operator_signal_value == OperatorSignal.NONE
        session.request_stop()
        assert stub_runtime.session.operator_signal_value == OperatorSignal.STOP

    asyncio.run(scenario())


def test_terminate_request_sets_terminal_signal(stub_runtime: Any) -> None:
    """request_terminate writes the TERMINATE signal (workflow marked complete)."""
    from runtime.operator_signal import OperatorSignal

    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        session.request_terminate()
        assert stub_runtime.session.operator_signal_value == OperatorSignal.TERMINATE

    asyncio.run(scenario())


def test_stopped_exit_classifies_as_stopped(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _stopped_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(operator_exit="stopped")

    _patch_handle(monkeypatch, _stopped_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("work")
        assert outcome.kind == "stopped"
        assert outcome.operator_exit == "stopped"

    asyncio.run(scenario())


# ── Queue mechanics (Milestone 3) ─────────────────────────────────────


def test_queue_message_fifo_and_inspection(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        assert session.queue_message("one") == 1
        assert session.queue_message("two") == 2
        assert session.queued_messages() == ["one", "two"]

    asyncio.run(scenario())


def test_queue_message_ignores_whitespace(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        assert session.queue_message("   \n  ") == 0
        assert session.queued_messages() == []

    asyncio.run(scenario())


def test_queue_message_bounded(stub_runtime: Any) -> None:
    """The queue refuses past its bound so a runaway paste cannot grow it."""
    from hosts.cli.session import _QUEUE_MAX_MESSAGES, QueueLimitError

    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        for i in range(_QUEUE_MAX_MESSAGES):
            session.queue_message(f"m{i}")
        with pytest.raises(QueueLimitError):
            session.queue_message("overflow")
        assert len(session.queued_messages()) == _QUEUE_MAX_MESSAGES

    asyncio.run(scenario())


def test_queue_message_closed_session_raises(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        await session.aclose()
        with pytest.raises(SessionStateError):
            session.queue_message("late")

    asyncio.run(scenario())


def test_drop_queued_newest_and_oldest(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        session.queue_message("first")
        session.queue_message("second")
        assert session.drop_queued() == "second"
        assert session.queued_messages() == ["first"]
        assert session.drop_queued(0) == "first"
        assert session.queued_messages() == []
        with pytest.raises(IndexError):
            session.drop_queued()

    asyncio.run(scenario())


def test_last_outcome_tracks_each_turn(stub_runtime: Any, monkeypatch: Any) -> None:
    """The engine's last_outcome surfaces the most recent turn's classification."""
    _patch_handle(monkeypatch, _stub_handle_message)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        assert session.last_outcome is None
        outcome = await session.run_turn("probe")
        assert session.last_outcome is outcome

    asyncio.run(scenario())


def test_stop_requests_tolerate_a_sessionless_runtime() -> None:
    """A runtime without a session seam: the stop requests are no-ops."""

    class _BareRuntime:
        async def start(self) -> None:
            return None

    session = InteractiveSession(
        runtime=_BareRuntime(),
        transport=_RecordingTransport(),  # type: ignore[arg-type]
    )
    session.request_stop()
    session.request_terminate()

    # And a session without the operator-signal attribute:
    class _PartialSession:
        pass

    class _PartialRuntime:
        session = _PartialSession()  # type: ignore[assignment]
        services = None  # type: ignore[assignment]

        async def start(self) -> None:
            return None

    partial = InteractiveSession(
        runtime=_PartialRuntime(),
        transport=_RecordingTransport(),  # type: ignore[arg-type]
    )
    partial.request_stop()
    partial.request_terminate()


def test_terminated_exit_classifies_as_stopped(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _terminated_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(operator_exit="terminated")

    _patch_handle(monkeypatch, _terminated_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("work")
        assert outcome.kind == "stopped"
        assert outcome.operator_exit == "terminated"

    asyncio.run(scenario())


def test_stop_signal_cleared_before_next_turn(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_turn clears a stale STOP signal so it cannot leak into the next turn."""

    async def _observing_turn(runtime: Any, message: str, transport: Any) -> Any:
        from runtime.operator_signal import OperatorSignal

        assert runtime.session.operator_signal_value == OperatorSignal.NONE
        return _result()

    _patch_handle(monkeypatch, _observing_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        from runtime.operator_signal import OperatorSignal

        await session.start()
        stub_runtime.session.operator_signal_value = OperatorSignal.STOP
        outcome = await session.run_turn("next")
        assert outcome.kind == "ok"

    asyncio.run(scenario())


def test_error_turn_is_classified_not_raised(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _failing(runtime: Any, message: str, transport: Any) -> Any:
        raise RuntimeError("provider exploded")

    _patch_handle(monkeypatch, _failing)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("hello")
        assert outcome.kind == "error"
        assert session.transport.errors, "the transport must see the failure"
        # The session stays usable.
        _patch_handle(monkeypatch, _stub_handle_message)
        outcome2 = await session.run_turn("again")
        assert outcome2.kind == "ok"

    asyncio.run(scenario())


def test_approval_payload_classification(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval = {"request_id": "req-1", "target_id": "bash_run", "command": "echo hi"}

    async def _paused_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(text="paused", data={"approval": approval})

    _patch_handle(monkeypatch, _paused_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("run it")
        assert outcome.kind == "approval"
        assert outcome.approval == approval
        assert outcome.approval is not None and outcome.approval["request_id"] == "req-1"

    asyncio.run(scenario())


def test_yield_question_classification(stub_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _yielding_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(yielded_question="What file?")

    _patch_handle(monkeypatch, _yielding_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("do it")
        assert outcome.kind == "yield"
        assert outcome.yielded_question == "What file?"

    asyncio.run(scenario())


def test_operator_exit_is_paused(stub_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(operator_exit="paused")

    _patch_handle(monkeypatch, _paused_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("work")
        assert outcome.kind == "paused"
        assert outcome.operator_exit == "paused"

    asyncio.run(scenario())


# ── Approval decision dispatch ────────────────────────────────────────


def test_decide_approval_uses_typed_seam(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions: list[tuple[str, bool, dict[str, Any]]] = []

    async def _fake_decide(
        runtime: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        decisions.append((request_id, approve, dict(kwargs)))
        return _result(text="approved and continued")

    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "decide_pending_approval", _fake_decide)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.decide_approval("req-9", approve=True, remember="git status")
        assert outcome.kind == "ok"
        assert outcome.text == "approved and continued"
        assert decisions == [("req-9", True, {"remember": "git status", "remember_default": False})]

    asyncio.run(scenario())


def test_decide_approval_resolves_single_pending(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty request id resolves to the single pending approval."""
    resolved_ids: list[str] = []

    async def _fake_decide(
        runtime: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        resolved_ids.append(request_id)
        return _result(text="denied")

    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "decide_pending_approval", _fake_decide)

    class _PendingRecord:
        invocation = SimpleNamespace(request_id="pending-42")

    class _Service:
        async def list_pending(self, identity: Any) -> list[Any]:
            return [_PendingRecord()]

    stub_runtime.services.approval_service = _Service()
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.decide_approval("", approve=False)
        assert outcome.kind == "ok"
        assert resolved_ids == ["pending-42"]

    asyncio.run(scenario())


def test_decide_approval_no_pending_is_an_error(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.handle_message as hm_module

    async def _fake_decide(
        runtime: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        raise AssertionError("must not be called")

    monkeypatch.setattr(hm_module, "decide_pending_approval", _fake_decide)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.decide_approval("", approve=True)
        assert outcome.kind == "error"
        assert "No pending approval" in outcome.text

    asyncio.run(scenario())


# ── Cancellation recovery ─────────────────────────────────────────────


def test_cancelled_turn_recovers_and_keeps_session_usable(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()

    async def _hanging_turn(runtime: Any, message: str, transport: Any) -> Any:
        entered.set()
        await asyncio.sleep(30)
        return _result()

    _patch_handle(monkeypatch, _hanging_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        task = asyncio.create_task(session.run_turn("hang"))
        await entered.wait()
        task.cancel()
        outcome = await task
        assert outcome.kind == "cancelled"
        # Session counters were updated and the runtime stayed alive.
        assert stub_runtime.session.tokens.session_cancelled_count == 1
        # The next turn runs on the same runtime.
        _patch_handle(monkeypatch, _stub_handle_message)
        outcome2 = await session.run_turn("next")
        assert outcome2.kind == "ok"
        assert outcome2.text == "echo:next"

    asyncio.run(scenario())


def test_operator_signal_cleared_after_cancel(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled turn leaves no stale operator signal for the next turn."""
    entered = asyncio.Event()

    async def _hanging_turn(runtime: Any, message: str, transport: Any) -> Any:
        entered.set()
        await asyncio.sleep(30)
        return _result()

    _patch_handle(monkeypatch, _hanging_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        from runtime.operator_signal import OperatorSignal

        await session.start()
        stub_runtime.session.operator_signal_value = OperatorSignal.INTERRUPT
        task = asyncio.create_task(session.run_turn("hang"))
        await entered.wait()
        task.cancel()
        await task
        assert stub_runtime.session.operator_signal_value == OperatorSignal.NONE

    asyncio.run(scenario())


def test_session_turn_ledger_counts_turns(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful turns are counted on the runtime's session ledger."""
    _patch_handle(monkeypatch, _stub_handle_message)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        await session.run_turn("one")
        await session.run_turn("two")
        assert stub_runtime.session.tokens.session_turn_count == 2
        assert stub_runtime.session.tokens.session_success_count == 2

    asyncio.run(scenario())


def test_restart_flag_surfaces_on_outcome(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_handle(monkeypatch, _stub_handle_message)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        stub_runtime.session.restart_required = True
        outcome = await session.run_turn("go")
        assert outcome.restart_required is True

    asyncio.run(scenario())


def test_turn_running_property_tracks_the_turn(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``turn_running`` is True inside the turn and False outside it."""
    observed: list[bool] = []
    session_ref: dict[str, InteractiveSession] = {}

    async def _observing(runtime: Any, message: str, transport: Any) -> Any:
        observed.append(session_ref["session"].turn_running)
        return _result()

    _patch_handle(monkeypatch, _observing)
    session = _make_session(stub_runtime)
    session_ref["session"] = session

    async def scenario() -> None:
        await session.start()
        assert session.turn_running is False
        await session.run_turn("probe")
        assert session.turn_running is False
        assert observed == [True]

    asyncio.run(scenario())


def test_nested_approval_schema_is_classified(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result carrying ``lisa.approval.v1`` at the top level is an approval."""

    async def _approval_turn(runtime: Any, message: str, transport: Any) -> Any:
        return _result(
            text="paused",
            data={"schema": "lisa.approval.v1", "request_id": "req-top"},
        )

    _patch_handle(monkeypatch, _approval_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("go")
        assert outcome.kind == "approval"
        assert outcome.approval is not None
        assert outcome.approval["request_id"] == "req-top"

    asyncio.run(scenario())


def test_keyboard_interrupt_is_classified_as_cancelled(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _interrupting_turn(runtime: Any, message: str, transport: Any) -> Any:
        raise KeyboardInterrupt()

    _patch_handle(monkeypatch, _interrupting_turn)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("go")
        assert outcome.kind == "cancelled"
        assert stub_runtime.session.tokens.session_cancelled_count == 1
        # The session stays usable.
        _patch_handle(monkeypatch, _stub_handle_message)
        outcome2 = await session.run_turn("next")
        assert outcome2.kind == "ok"

    asyncio.run(scenario())


def test_aclose_on_unstarted_session_is_a_no_op(stub_runtime: Any) -> None:
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.aclose()
        assert session._closed is True

    asyncio.run(scenario())


def test_resolve_pending_without_service_returns_empty(stub_runtime: Any) -> None:
    """No approval service on the runtime: resolution yields '' (ambiguous)."""
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        resolved = await session._resolve_pending_request_id()
        assert resolved == ""

    asyncio.run(scenario())


def test_resolve_pending_failure_returns_empty(stub_runtime: Any) -> None:
    """A failing approval service is tolerated: resolution yields ''."""

    class _BrokenService:
        async def list_pending(self, identity: Any) -> list[Any]:
            raise RuntimeError("store unavailable")

    stub_runtime.services.approval_service = _BrokenService()
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        resolved = await session._resolve_pending_request_id()
        assert resolved == ""

    asyncio.run(scenario())


def test_decide_approval_cancelled_recovers(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation during the decision dispatch is classified, not raised."""
    entered = asyncio.Event()

    async def _fake_decide(
        runtime: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        entered.set()
        await asyncio.sleep(30)

    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "decide_pending_approval", _fake_decide)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        task = asyncio.create_task(session.decide_approval("req-1", approve=True))
        await entered.wait()
        task.cancel()
        outcome = await task
        assert outcome.kind == "cancelled"
        assert stub_runtime.session.tokens.session_cancelled_count == 1

    asyncio.run(scenario())


def test_decide_approval_failure_is_classified(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _failing_decide(
        runtime: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        raise RuntimeError("decision store exploded")

    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "decide_pending_approval", _failing_decide)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        outcome = await session.decide_approval("req-1", approve=True)
        assert outcome.kind == "error"
        assert session.transport.errors, "the failure must reach the transport"

    asyncio.run(scenario())


def test_turn_after_failed_turn_records_error_count(
    stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed turn increments the session error ledger, not just a log line."""

    async def _failing(runtime: Any, message: str, transport: Any) -> Any:
        raise RuntimeError("boom")

    _patch_handle(monkeypatch, _failing)
    session = _make_session(stub_runtime)

    async def scenario() -> None:
        await session.start()
        await session.run_turn("explode")
        assert stub_runtime.session.tokens.session_error_count == 1
        assert stub_runtime.session.tokens.session_turn_count == 1
        _patch_handle(monkeypatch, _stub_handle_message)
        await session.run_turn("recover")
        assert stub_runtime.session.tokens.session_turn_count == 2
        assert stub_runtime.session.tokens.session_success_count == 1

    asyncio.run(scenario())


# ── Approval answer parsing (surface helper, pure) ────────────────────


def test_parse_approval_answer_vocabulary() -> None:
    from hosts.cli.interactive import _parse_approval_answer

    assert _parse_approval_answer("") == ("defer", "")
    assert _parse_approval_answer("a") == ("allow", "")
    assert _parse_approval_answer("allow") == ("allow", "")
    assert _parse_approval_answer("y") == ("allow", "")
    assert _parse_approval_answer("d") == ("deny", "")
    assert _parse_approval_answer("deny") == ("deny", "")
    assert _parse_approval_answer("n") == ("deny", "")
    assert _parse_approval_answer("r") == ("allow", "")
    assert _parse_approval_answer("r git status") == ("allow", "git status")
    assert _parse_approval_answer("remember git push") == ("allow", "git push")
    assert _parse_approval_answer("what?") == ("defer", "")


def test_classify_session_message() -> None:
    from hosts.cli.interactive import _classify_session_message

    assert _classify_session_message("  ") == "skip"
    assert _classify_session_message("quit") == "quit"
    assert _classify_session_message("/exit") == "quit"
    assert _classify_session_message("/help") == "help"
    assert _classify_session_message("fix the parser") == "turn"
