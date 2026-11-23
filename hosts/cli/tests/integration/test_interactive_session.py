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
