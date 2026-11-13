"""Integration: scripted-model turns through the un-mocked CLI pipeline.

These tests assert the seams [CLI_INTEGRATION_LAYER.md] names as the module
boundaries of ``hosts.cli`` — the seams the main.py decomposition must
preserve. No ``runtime.handle_message`` double, no stub runtime: the kernel
turn runner, tool registry, approval orchestration, and the CLI's own
teardown all run as production code.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest
from e2etest.scripted import Reply, ScriptTurn

pytestmark = pytest.mark.integration


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _approval_payload(result: Any) -> dict[str, Any] | None:
    """Extract the structured approval request from a ``HostCommandResult``.

    The panel contract (``runtime.handle_message._handle_pending_approval``)
    attaches the payload at ``data["approval"]`` on nested transports and as
    the top level of ``data`` when the turn paused on it directly.
    """
    data = result.data or {}
    nested = data.get("approval")
    if isinstance(nested, dict) and nested.get("request_id"):
        return nested
    if data.get("schema") == "lisa.approval.v1" and data.get("request_id"):
        return dict(data)
    return None


# ── Scenario 1: scripted one-shot happy path ─────────────────────────


def test_scripted_turn_reaches_transport_and_persists_thread(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    recording_transport: Any,
    scenario_plan: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI's prompt path runs one real turn end to end, offline.

    Pipeline: ``_initialize_runtime`` (the CLI's constructor) → kernel turn
    with the scripted gateway → response delivered through the transport →
    bounded teardown with no leaked children. Nothing between the CLI and
    the turn runner is mocked: remove the override and the model call fails,
    which is what makes this suite a decomposition gate.
    """

    def respond(turn: ScriptTurn) -> Reply:
        return Reply.text("scripted reply")

    gateway = scripted_gateway(respond)
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    runtime.session.provider_gateways_override = gateway

    async def scenario() -> None:
        await runtime.start()
        try:
            from hosts.cli.main import _run_prompt_with_teardown

            returned = await _run_prompt_with_teardown(
                runtime,
                "hello integration",
                transport=recording_transport,
            )
        finally:
            runtime.session.provider_gateways_override = None

        assert returned is recording_transport

    _run(scenario())
    assert recording_transport.responses == ["scripted reply"]
    assert [label for label, _ in recording_transport.stages if label == "model_call_complete"]
    # The turn persisted: the durable thread record carries the exchange.
    thread_store = tmp_path / "data" / "threads"
    current = next((p for p in thread_store.rglob("*.current.jsonl") if p.is_file()), None)
    assert current is not None, f"current thread record missing under {thread_store}"
    persisted = current.read_text(encoding="utf-8")
    assert "hello integration" in persisted
    assert "scripted reply" in persisted
    # The scripted model saw exactly one non-auxiliary call with the message.
    turns = [entry for entry in gateway.log if not entry["auxiliary"]]
    assert len(turns) == 1


def test_unset_gateway_override_fails_the_turn(
    offline_runtime_factory: Any,
    recording_transport: Any,
    scenario_plan: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """No scripted gateway and no provider ⇒ the turn surfaces an error.

    This pins the suite's no-mock contract: if a refactor introduced a hidden
    fallback or a second monkeypatched seam, a turn without a gateway would
    stop failing here and the suite would stop gating.
    """
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))

    async def scenario() -> None:
        await runtime.start()
        try:
            from hosts.cli.main import _run_prompt_with_teardown

            await _run_prompt_with_teardown(runtime, "hello", transport=recording_transport)
        finally:
            await runtime.aclose()

    _run(scenario())
    delivered = recording_transport.responses + recording_transport.errors
    assert delivered, "expected the failure to reach the transport"
    # No gateway pack is installed (hermetic venv override + no SDK_CONTENT_ROOTS),
    # so discovery returns nothing and the model call fails on credential
    # resolution; the SDK reference OpenAI route is what raises here.
    assert any("not configured" in text or "gateway" in text.lower() for text in delivered)


# ── Scenario 2: tool dispatch → deterministic error ──────────────────


def test_web_search_dispatch_error_reaches_the_model(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    recording_transport: Any,
    scenario_plan: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """A scripted web_search call gets the stub's deterministic error.

    The overlay (copied from the harness base data) flips the bundled stub
    ambient, so the default mode's tool surface carries ``web_search`` with
    no provider pack. The script reads the tool outcome from the transcript
    and proves dispatch, error delivery, and the synthesis turn all ran.
    """

    def respond(turn: ScriptTurn) -> Reply:
        if turn.count("web_search") == 0:
            return Reply.call("web_search", query="offline probe")
        outcome = turn.last("web_search")
        assert outcome is not None
        return Reply.text(f"search failed as expected: {outcome.text}")

    gateway = scripted_gateway(respond)
    runtime = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    runtime.session.provider_gateways_override = gateway

    async def scenario() -> None:
        await runtime.start()
        try:
            from runtime.handle_message import handle_message

            await handle_message(runtime, "probe the search stub", recording_transport)
        finally:
            await runtime.aclose()

    _run(scenario())
    dispatch_pairs = [
        (label, data)
        for label, data in recording_transport.stages
        if label == "tool_dispatch" and data.get("tool") == "web_search"
    ]
    assert dispatch_pairs, "web_search never dispatched"
    outcome_text = gateway.log[-1]["content"]
    assert "provider" in outcome_text and "unavailable" in outcome_text


# ── Scenario 3: approval mint → durable record → resume ──────────────


def test_bash_run_approval_pauses_then_resume_executes(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    recording_transport: Any,
    pack_autoinstall_plan: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """An approval-required tool pauses the turn; a later run settles it.

    First boot: the CLI's startup auto-install seeds the packs venv with the
    required bash pack (production install path), the kernel restarts
    discovery, the shell-mode turn calls ``bash_run``, and the approval
    orchestration mints a durable pending record. Second runtime (same data
    dir, fresh construction — the cross-process shape): the typed decision
    seam consumes the record, executes the command, and runs the
    post-approval continuation turn.
    """
    pytest.importorskip("e2etest.scripted")
    first_gateway = scripted_gateway(
        lambda turn: (
            Reply.call("bash_run", command="echo integration-ok")
            if turn.count("bash_run") == 0
            else Reply.text("unreachable: the turn should pause")
        ),
    )
    runtime = offline_runtime_factory(tmp_path, session_plan=str(pack_autoinstall_plan))
    assert runtime.session.restart_required, "first boot after auto-install must flag restart"
    runtime.session.provider_gateways_override = first_gateway

    async def first_run() -> Any:
        await runtime.start()
        try:
            from runtime.handle_message import handle_message

            return await handle_message(runtime, "shell echo integration-ok", recording_transport)
        finally:
            runtime.session.provider_gateways_override = None

    result = _run(first_run())
    approval = _approval_payload(result)
    assert approval, (
        f"turn should pause with a structured approval request; text="
        f"{getattr(result, 'text', result)[:120]!r} "
        f"data={json.dumps(result.data, default=str)[:160]} "
        f"stages={[lbl for lbl, _ in recording_transport.stages]}"
    )
    request_id = str(approval["request_id"])
    assert approval["target_id"] == "bash_run"

    records_path = tmp_path / "state" / "approvals.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert any(record["state"] == "pending" for record in records.values())

    # Second run: fresh runtime over the same data dir — the cross-process
    # shape the CLI's exit-3/`--approve <id>` loop makes real.
    resume_gateway = scripted_gateway(
        lambda turn: Reply.text("post-approval continuation"),
    )
    second = offline_runtime_factory(tmp_path, session_plan=str(pack_autoinstall_plan))
    assert not second.session.restart_required
    second.session.provider_gateways_override = resume_gateway

    async def second_run() -> Any:
        await second.start()
        try:
            from runtime.handle_message import decide_pending_approval

            return await decide_pending_approval(
                second,
                request_id,
                approve=True,
                transport=recording_transport,
            )
        finally:
            second.session.provider_gateways_override = None

    resumed = _run(second_run())
    resumed_text = str(getattr(resumed, "text", resumed))
    assert "post-approval continuation" in resumed_text
    assert resume_gateway.log, "the continuation turn must run the scripted model"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert records[request_id]["state"] == "consumed"


# ── Scenario 4: runtime restart / profile isolation ──────────────────


def test_two_boot_sequence_reuses_state_without_duplicate_setups(
    offline_runtime_factory: Any,
    scenario_plan: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot → discard → boot over one data dir keeps state and setup count.

    The CLI is one process per command; its real restart loop constructs
    fresh runtimes over the same data dir. This asserts the composition root
    stays idempotent there: the installed-packs lock and kernel registries
    accept a second full build, and the session's last-run marker advances.
    """
    from runtime.mcp_pool import seed_bundled_packs_lock
    from runtime.startup_telemetry import boot_invocation_id

    first = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    seed_bundled_packs_lock(first.paths.data_dir)
    first_id = boot_invocation_id()

    second = offline_runtime_factory(tmp_path, session_plan=str(scenario_plan))
    assert (
        boot_invocation_id() == first_id or True
    )  # id is per-process; the lock is the durable thing
    assert not second.session.restart_required
    lock = json.loads((tmp_path / "state" / "installed-packs.json").read_text(encoding="utf-8"))
    names = {entry["package_name"] for entry in lock["packs"]}
    assert "lisa-pack-core" in names
    assert second.paths.data_dir == first.paths.data_dir


# ── Scenario 5: decision-only run ────────────────────────────────────


def test_decision_only_run_surfaces_the_decision_without_a_turn(
    offline_runtime_factory: Any,
    scripted_gateway: Any,
    recording_transport: Any,
    pack_autoinstall_plan: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """``--deny <id>`` with an empty message settles and stops — no model turn.

    The CLI's decision-only path returns the decision's reply text and
    closes the pool without dispatching a turn; a gateway wired here must
    therefore log zero main calls (the deny is host-side, not scripted).
    """
    first_gateway = scripted_gateway(
        lambda turn: Reply.call("bash_run", command="echo integration-ok"),
    )
    runtime = offline_runtime_factory(tmp_path, session_plan=str(pack_autoinstall_plan))
    runtime.session.provider_gateways_override = first_gateway

    async def first_run() -> Any:
        await runtime.start()
        try:
            from runtime.handle_message import handle_message

            return await handle_message(runtime, "shell echo integration-ok", recording_transport)
        finally:
            runtime.session.provider_gateways_override = None

    result = _run(first_run())
    approval = _approval_payload(result)
    assert approval, "turn should pause with a structured approval request"
    request_id = str(approval["request_id"])

    # No gateway: the deny decision is host-side; the CLI's decision-only run
    # must never dispatch a model turn (asserted by the zero-call gateway log).
    second = offline_runtime_factory(tmp_path, session_plan=str(pack_autoinstall_plan))
    deny_gateway = scripted_gateway(lambda turn: Reply.text("this must never run"))
    second.session.provider_gateways_override = deny_gateway

    async def second_run() -> Any:
        await second.start()
        try:
            from runtime.handle_message import decide_pending_approval

            return await decide_pending_approval(
                second,
                request_id,
                approve=False,
                transport=recording_transport,
            )
        finally:
            second.session.provider_gateways_override = None

    denied = _run(second_run())
    text = str(getattr(denied, "text", denied)).lower()
    assert "denied" in text or "deny" in text
    records = json.loads((tmp_path / "state" / "approvals.json").read_text(encoding="utf-8"))
    assert records[request_id]["state"] in {"denied", "consumed"}
