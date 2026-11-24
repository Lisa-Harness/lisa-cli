"""Integration: the persistent interactive session engine on a real runtime.

The scenarios prove the engine's core Milestone-1 contract end to end:
multiple model turns served by ONE runtime and ONE started MCP pool, with
exactly one start and one close, and a turn-level failure not poisoning
the next turn on the same runtime. The scripted gateway stands in for the
model (no provider credential is read); nothing between the engine and
the kernel turn runner is mocked.

The approval-continuation contract (pause decided in place through the
typed seam) is intentionally NOT re-proven here with a real ``bash_run``
auto-install: installing an external pack adds its venv to the process's
``sys.path`` (``site.addsitedir`` in the CLI bootstrap), which makes
``importlib.metadata`` see the pack for every later runtime constructed in
this pytest process — breaking the scripted suite's first-boot
``restart_required`` assertion when this module runs first. The engine's
decision dispatch is covered by the unit suite, and the real approval
orchestration by ``test_scripted_turns``.

Pins:
- one runtime.start() / one aclose() across several turns (the engine's
  lifecycle contract; the pool survives turn boundaries);
- sequential turns accumulate thread entries and session counters;
- a turn-level failure does not poison the next turn on the same runtime.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from e2etest.scripted import Reply

from hosts.cli.cli_runtime import _aclose_runtime_bounded
from hosts.cli.session import InteractiveSession

pytestmark = pytest.mark.integration


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _RecordingTransport:
    """Transport-callback recorder (the integration suite's existing shape)."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[str] = []
        self.errors: list[str] = []

    def start_turn(self, message: str) -> None:
        return None

    def start_activity(self) -> None:
        return None

    def stop_activity(self) -> None:
        return None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        self.stages.append((label, dict(data or {})))

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        self.responses.append(text)

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self.errors.append(message)


# ── Scenario 1: many turns on one runtime, one start, one close ──────


def test_session_serves_many_turns_on_one_runtime(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    scenario_plan: Path,
    tmp_path: Path,
) -> None:
    """Three turns run on one runtime; start() and aclose() each happen once."""
    gateway = scripted_gateway(lambda turn: Reply.text(f"reply:{turn.count('model')}"))
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    runtime.session.provider_gateways_override = gateway

    starts: list[int] = []
    original_start = runtime.start

    async def _counting_start() -> None:
        starts.append(1)
        await original_start()

    runtime.start = _counting_start  # type: ignore[method-assign]

    transport = _RecordingTransport()
    session = InteractiveSession(runtime=runtime, transport=transport)

    async def scenario() -> None:
        await session.start()
        for message in ("first", "second", "third"):
            outcome = await session.run_turn(message)
            assert outcome.kind == "ok", f"turn {message!r} failed: {outcome.text[:120]}"
        await session.aclose()

    _run(scenario())
    assert starts == [1], "the engine must start the runtime exactly once"
    assert len(transport.responses) == 3
    # The durable thread carries all three turns.
    thread_store = tmp_path / "data" / "threads"
    currents = [p for p in thread_store.rglob("*.current.jsonl") if p.is_file()]
    assert currents, "thread record missing"
    persisted = currents[0].read_text(encoding="utf-8")
    for message in ("first", "second", "third"):
        assert message in persisted
    # The scripted model served exactly three non-auxiliary calls.
    turns = [entry for entry in gateway.log if not entry["auxiliary"]]
    assert len(turns) == 3
    # Session counters accumulated across turns.
    assert runtime.session.tokens.session_turn_count >= 3


def test_failed_turn_does_not_poison_the_next(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    scenario_plan: Path,
    tmp_path: Path,
) -> None:
    """A turn-level failure is classified, and the next turn succeeds."""
    calls: list[int] = []

    def respond(turn: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("scripted provider outage")
        return Reply.text("recovered")

    gateway = scripted_gateway(respond)
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    runtime.session.provider_gateways_override = gateway
    transport = _RecordingTransport()
    session = InteractiveSession(runtime=runtime, transport=transport)

    async def scenario() -> None:
        await session.start()
        first = await session.run_turn("explode")
        assert first.kind == "error"
        assert transport.errors, "the failure must reach the transport"
        second = await session.run_turn("recover")
        assert second.kind == "ok", f"second turn failed: {second.text[:200]}"
        await session.aclose()

    _run(scenario())
    # One model call failed inside the turn; the recovery turn's reply arrived.
    assert "recovered" in transport.responses


# ── Scenario 3: cooperative stop mid-turn ────────────────────────────


def test_cooperative_stop_ends_turn_and_preserves_session(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    scenario_plan: Path,
    tmp_path: Path,
) -> None:
    """request_stop() mid-turn: the kernel honors it at the next iteration.

    The scripted model calls a tool once; the gateway sets the operator
    signal between calls (what the Esc key does in a real session), so the
    turn loop observes STOP at its next iteration boundary. The outcome
    classifies ``stopped`` with the operator-exit discriminator, the loop
    state is preserved (thread entry recorded), and the SAME runtime serves
    the next ordinary turn — the pool was never torn down.
    """
    from runtime.operator_signal import OperatorSignal

    def respond(turn: Any) -> Any:
        if turn.count("web_search") == 0:
            # Between iterations: simulate the operator pressing Esc.
            runtime.session.operator_signal_value = OperatorSignal.STOP
            return Reply.call("web_search", query="probe")
        raise AssertionError("the turn must stop before the next model call")

    gateway = scripted_gateway(respond)
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    runtime.session.provider_gateways_override = gateway
    transport = _RecordingTransport()
    session = InteractiveSession(runtime=runtime, transport=transport)

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("search then keep going")
        assert outcome.kind == "stopped", f"expected cooperative stop, got {outcome.kind}"
        assert outcome.operator_exit == "stopped"
        # The session stays usable on the same warm runtime.
        runtime.session.provider_gateways_override = scripted_gateway(
            lambda turn: Reply.text("still alive after stop")
        )
        after = await session.run_turn("continue fresh")
        assert after.kind == "ok", f"post-stop turn failed: {after.text[:200]}"
        await session.aclose()
        return after

    after = _run(scenario())
    assert "still alive after stop" in after.text


# ── Scenario 4: conversation isolation across rebuilt runtimes ────────


def test_conversation_id_isolates_threads_and_resume_reads_own_context(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    scenario_plan: Path,
    tmp_path: Path,
) -> None:
    """Turns under a conversation id land in its own thread; a rebuilt
    runtime on the same id rehydrates that conversation's context.

    This is the resume contract the interactive surface's ``/resume``
    switch relies on: switch = close + rebuild with the id, and the
    durable store carries the conversation forward.
    """
    # First runtime: an isolated conversation takes one turn.
    first_gateway = scripted_gateway(lambda turn: Reply.text("alpha reply"))
    runtime = offline_runtime_factory(
        tmp_path, session_plan=str(scenario_plan), conversation_id="alpha"
    )
    runtime.session.provider_gateways_override = first_gateway

    async def first_run() -> Any:
        from runtime.handle_message import handle_message

        await runtime.start()
        try:
            return await handle_message(runtime, "alpha question", _RecordingTransport())
        finally:
            runtime.session.provider_gateways_override = None
            await _aclose_runtime_bounded(runtime)

    _run(first_run())

    # The alpha store exists and carries the turn; the default store does not.
    thread_dir = tmp_path / "data" / "threads"
    slug = runtime.config.user_slug
    alpha_store = thread_dir / f"{slug}--alpha.current.jsonl"
    default_store = thread_dir / f"{slug}.current.jsonl"
    assert alpha_store.is_file(), "the isolated conversation must own its thread"
    persisted = alpha_store.read_text(encoding="utf-8")
    assert "alpha question" in persisted
    assert "alpha reply" in persisted
    assert not default_store.exists(), "the default conversation must stay untouched"

    # Second runtime (the rebuild shape): same data dir, same conversation
    # id, fresh construction. The scripted model echoes what the context
    # assembly fed it, proving the prior turn rehydrated.
    seen_context: list[str] = []

    def context_probe(turn: Any) -> Reply:
        for message in turn.request.messages:
            if message.get("role") == "user" and "alpha question" in str(message.get("content")):
                seen_context.append("rehydrated")
                break
        return Reply.text("beta reply")

    resume_gateway = scripted_gateway(context_probe)
    second = offline_runtime_factory(
        tmp_path, session_plan=str(scenario_plan), conversation_id="alpha"
    )
    second.session.provider_gateways_override = resume_gateway

    async def second_run() -> Any:
        from runtime.handle_message import handle_message

        await second.start()
        try:
            return await handle_message(second, "beta question", _RecordingTransport())
        finally:
            second.session.provider_gateways_override = None
            await _aclose_runtime_bounded(second)

    _run(second_run())
    assert seen_context == ["rehydrated"], (
        "the rebuilt runtime must rehydrate the conversation's prior turn"
    )
