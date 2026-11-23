"""Coverage-focused tests for interactive-session edge paths.

These pin the branches the engine/surface contract defines but the happy
paths never reach: closed-session starts, concurrent decision rejection,
KeyboardInterrupt during a decision, ledger failure tolerance, the
interactive approval panel's in-session hints, and the Typer registration
of ``lisa interactive``.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from hosts.cli import interactive as interactive_mod
from hosts.cli.session import InteractiveSession, SessionStateError


class _Transport:
    def start_turn(self, message: str) -> None:
        return None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        return None


def _record(*, is_error: bool = False) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(text="x", data={}, is_error=is_error, operator_exit="")


def _patch_handle(monkeypatch: pytest.MonkeyPatch, impl: Any) -> None:
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", impl)


def _console() -> Console:
    stream = StringIO()
    return Console(file=stream, width=80, force_terminal=False)


def test_start_after_close_is_a_state_error(tmp_path: Any, monkeypatch: Any) -> None:
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    session = InteractiveSession(runtime=runtime, transport=_Transport())

    async def scenario() -> None:
        await session.start()
        await session.aclose()
        with pytest.raises(SessionStateError):
            await session.start()

    asyncio.run(scenario())


def test_decide_while_turn_running_is_rejected(tmp_path: Any, monkeypatch: Any) -> None:
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    entered = asyncio.Event()

    async def _hanging(runtime_arg: Any, message: str, transport: Any) -> Any:
        entered.set()
        await asyncio.sleep(30)
        return _record()

    _patch_handle(monkeypatch, _hanging)
    session = InteractiveSession(runtime=runtime, transport=_Transport())

    async def scenario() -> None:
        await session.start()
        first = asyncio.create_task(session.run_turn("run"))
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(SessionStateError):
            await session.decide_approval("req-1", approve=True)
        first.cancel()
        await first

    asyncio.run(scenario())


def test_keyboard_interrupt_during_decision_is_cancelled(tmp_path: Any, monkeypatch: Any) -> None:
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")

    async def _interrupting_decide(
        runtime_arg: Any, request_id: str, *, approve: bool, transport: Any, **kwargs: Any
    ) -> Any:
        raise KeyboardInterrupt()

    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "decide_pending_approval", _interrupting_decide)
    session = InteractiveSession(runtime=runtime, transport=_Transport())

    async def scenario() -> None:
        await session.start()
        outcome = await session.decide_approval("req-1", approve=True)
        assert outcome.kind == "cancelled"
        assert runtime.session.tokens.session_cancelled_count == 1

    asyncio.run(scenario())


def test_recover_after_cancel_resets_transient_state(tmp_path: Any) -> None:
    """The cancel recovery clears auto-approval counts and turn context."""
    from types import SimpleNamespace

    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    runtime.session.auto_approved_this_turn = 3
    runtime.session.last_turn_context = SimpleNamespace(stale=True)
    session = InteractiveSession(runtime=runtime, transport=_Transport())

    async def scenario() -> None:
        await session.start()
        await session._recover_after_cancel()
        assert runtime.session.auto_approved_this_turn == 0
        assert runtime.session.last_turn_context is None
        assert runtime.session.tokens.session_cancelled_count == 1

    asyncio.run(scenario())


def test_ledger_failure_is_tolerated(tmp_path: Any, monkeypatch: Any) -> None:
    """A broken session ledger must not fail the turn."""

    async def _ok_turn(runtime_arg: Any, message: str, transport: Any) -> Any:
        return _record()

    _patch_handle(monkeypatch, _ok_turn)

    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    # Break the token counters so the ledger write raises mid-way.
    runtime.session.tokens.session_turn_count = None  # type: ignore[assignment]
    session = InteractiveSession(runtime=runtime, transport=_Transport())

    async def scenario() -> None:
        await session.start()
        outcome = await session.run_turn("go")
        assert outcome.kind == "ok"

    asyncio.run(scenario())


def test_drain_returns_true_on_restart_required() -> None:
    """A restart-required drained turn stops the session with the notice."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from runtime.testing import build_test_runtime

    from hosts.cli.session import InteractiveSession as _Engine

    console = _console()
    runtime = build_test_runtime(Path(tempfile.mkdtemp()), models_yaml="")
    session = _Engine(runtime=runtime, transport=_Transport())

    async def _fake_handle(rt: Any, message: str, transport: Any) -> Any:
        rt.session.restart_required = True
        rt.session.restart_reason = "pack install"
        return SimpleNamespace(text="ok", data={}, is_error=False, operator_exit="")

    import runtime.handle_message as hm

    original = hm.handle_message
    hm.handle_message = _fake_handle

    async def scenario() -> bool:
        await session.start()
        session.queue_message("trigger")
        return await interactive_mod._drain_queue(session, console, None)

    try:
        must_restart = asyncio.run(scenario())
    finally:
        hm.handle_message = original
    assert must_restart is True


def test_stop_requests_on_sessionless_runtime_are_noops() -> None:
    """request_stop/request_terminate tolerate a runtime without a session."""

    class _BareRuntime:
        session = None  # type: ignore[assignment]

        async def start(self) -> None:
            return None

    from hosts.cli.session import InteractiveSession

    session = InteractiveSession(runtime=_BareRuntime(), transport=_Transport())  # type: ignore[arg-type]
    session.request_stop()
    session.request_terminate()


def test_resolve_pending_without_approval_service(tmp_path: Any) -> None:
    """A runtime with no approval service resolves no pending id."""

    class _NoServiceRuntime:
        session = None  # type: ignore[assignment]
        services = None  # type: ignore[assignment]

        async def start(self) -> None:
            return None

    from hosts.cli.session import InteractiveSession

    session = InteractiveSession(runtime=_NoServiceRuntime(), transport=_Transport())  # type: ignore[arg-type]
    resolved = asyncio.run(session._resolve_pending_request_id())
    assert resolved == ""


def test_approval_panel_interactive_hints() -> None:
    """The in-session approval panel names the in-session decision keys."""
    from hosts.cli.render import render_approval_panel

    console = _console()
    render_approval_panel(
        {
            "request_id": "req-42",
            "pack_id": "lisa-pack-bash",
            "target_id": "bash_run",
            "command": "echo hi",
            "containment": "none",
        },
        console=console,
        interactive=True,
    )
    rendered = console.file.getvalue() if hasattr(console.file, "getvalue") else ""
    assert "allow once" in rendered
    assert "lisa approve req-42" not in rendered

    # One-shot hints stay unchanged for the default panel.
    console2 = _console()
    render_approval_panel({"request_id": "req-42", "containment": "none"}, console=console2)
    rendered2 = console2.file.getvalue() if hasattr(console2.file, "getvalue") else ""
    assert "lisa approve req-42" in rendered2


def test_interactive_command_registered() -> None:
    """The Typer app exposes ``lisa interactive`` in the help listing."""
    import hosts.cli.main  # noqa: F401  (imports register every command module)
    from hosts.cli.application import app as _app

    names = {getattr(cmd, "name", "") for cmd in _app.registered_commands}
    assert "interactive" in names, f"interactive not registered: {sorted(names)[:20]}"


def test_interactive_command_invocation_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Typer wrapper runs the session loop and forwards the kwargs."""
    import hosts.cli.main as cli_main

    seen: dict[str, Any] = {}

    async def _fake_session(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("hosts.cli.interactive.run_interactive_session", _fake_session)
    result: int | None = cli_main.app(args=["interactive"], standalone_mode=False)
    assert result in (None, 0)
    assert seen, "the wrapper must call the session runner"
    assert seen["runtime_kwargs"]["in_process_mcp"] is True


def test_interactive_command_invocation_maps_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero session exit code becomes the process exit code."""
    import hosts.cli.main as cli_main

    async def _failing_session(**kwargs: Any) -> int:
        return 3

    monkeypatch.setattr("hosts.cli.interactive.run_interactive_session", _failing_session)
    with pytest.raises(SystemExit) as excinfo:
        cli_main.app(args=["interactive"], standalone_mode=True)
    assert excinfo.value.code == 3


def test_help_and_banner_render() -> None:
    """The help listing and banner print without raising."""
    console = _console()
    interactive_mod._print_help(console)
    interactive_mod._print_banner(console, "default · m-1")
    interactive_mod._print_banner(console, "")


def test_read_approval_answer_defers_on_eof(monkeypatch: Any) -> None:
    """EOF/Ctrl-C during the decision read returns None (defer)."""
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(EOFError()))
    assert interactive_mod._read_approval_answer(_console()) is None
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert interactive_mod._read_approval_answer(_console()) is None


def test_read_approval_answer_returns_line(monkeypatch: Any) -> None:
    monkeypatch.setattr("builtins.input", lambda: "a")
    assert interactive_mod._read_approval_answer(_console()) == "a"
