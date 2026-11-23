"""Tests for the interactive session surface (:mod:`hosts.cli.interactive`).

The surface is driven end to end with scripted readers — no terminal, no
runtime construction (the runtime builder is stubbed, like the decision
tests): the scenarios pin the loop's observable behavior:

- quit words and EOF end the session cleanly (exit 0);
- ``/help`` prints the control list and keeps the session open;
- a turn outcome renders and the loop continues;
- an approval pause is decided inline (allow/remember/deny/defer) and the
  continuation runs through the engine;
- the double-Ctrl-C exit contract (first warns, second exits 130);
- a restart-required flag ends the session with the restart notice;
- ``$(...)`` expansion failures are reported without dropping the session;
- a teardown failure exits 1 instead of reporting success.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from hosts.cli import interactive as interactive_mod
from hosts.cli.session import InteractiveSession, TurnOutcome


class _StubRuntime:
    """Runtime stub with the session fields the surface reads."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.session = SimpleNamespace(
            current_mode_id="default",
            next_model_id="-",
            manifest=None,
            restart_required=False,
            restart_reason="",
            tokens=SimpleNamespace(session_cancelled_count=0),
        )
        self.config = SimpleNamespace(streaming=SimpleNamespace(mode="none"))

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _NullTransport:
    """Transport stub for the surface (the engine's dispatch is stubbed)."""

    def __init__(self, console: Any = None, **kwargs: Any) -> None:
        self.console = console

    def start_turn(self, message: str) -> None:
        return None


def _outcome(kind: str = "ok", **fields: Any) -> TurnOutcome:
    return TurnOutcome(kind=kind, **fields)


def _scripted_reader(messages: list[Any]) -> Any:
    """Build a reader that yields the given messages in order.

    ``KeyboardInterrupt`` instances in the list are raised instead of
    returned; everything else is returned verbatim.
    """
    queue = list(messages)
    raised = {"first_ctrl_c": False}

    async def _read() -> Any:
        if not queue:
            return None  # EOF ends the loop
        item = queue.pop(0)
        if isinstance(item, KeyboardInterrupt):
            raised["first_ctrl_c"] = True
            raise item
        return item

    return _read


@pytest.fixture
def console() -> Console:
    return Console(
        file=type("_F", (), {"write": lambda s, t: None, "flush": lambda s: None})(),
        width=80,
        force_terminal=False,
        _environ={},
    )


@pytest.fixture
def surface(monkeypatch: pytest.MonkeyPatch, console: Console) -> Any:
    """Patch the surface's constructor/transport seams for loop-only tests."""
    runtime = _StubRuntime()

    def _fake_init(**_kwargs: Any) -> Any:
        return runtime

    from hosts.cli import cli_runtime as cli_runtime_mod
    from hosts.cli import prompt as prompt_mod
    from hosts.cli import transport as transport_mod

    monkeypatch.setattr(cli_runtime_mod, "_initialize_runtime_or_exit", _fake_init)
    monkeypatch.setattr(transport_mod, "PromptTransport", _NullTransport)
    monkeypatch.setattr(prompt_mod, "_collect_model_pricing", lambda runtime: ({}, {}))

    class _Session(InteractiveSession):
        def __init__(self, *, runtime: Any = None, transport: Any = None) -> None:
            # Bypass the dataclass wiring: the surface's construction call
            # lands here; the stub keeps the seam simple.
            self.runtime = runtime or _StubRuntime()
            self.transport = transport or _NullTransport()
            self.outcomes: list[TurnOutcome] = []

        async def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        async def run_turn(self, message: str) -> TurnOutcome:
            self.outcomes.append(_outcome("ok", text=f"echo:{message}"))
            return self.outcomes[-1]

        async def decide_approval(self, request_id: str, **kwargs: Any) -> TurnOutcome:
            self.outcomes.append(_outcome("ok", text="continuation"))
            return self.outcomes[-1]

    monkeypatch.setattr(interactive_mod, "InteractiveSession", _Session)
    return SimpleNamespace(runtime=runtime, session_cls=_Session, console=console)


def _run_surface(surface: Any, reader: Any, decider: Any | None = None) -> int:
    return asyncio.run(
        interactive_mod.run_interactive_session(
            console=surface.console,
            data_dir="",
            profile="",
            runtime_kwargs={},
            read_message=reader,
            read_approval=decider,
        )
    )


# ── Loop control ──────────────────────────────────────────────────────


def test_quit_word_ends_session_cleanly(surface: Any) -> None:
    reader = _scripted_reader(["hello", "quit"])
    assert _run_surface(surface, reader) == 0


def test_eof_warns_then_second_exits(surface: Any) -> None:
    """EOF/Ctrl-D/empty-submit all read as ``None``; the second consecutive ends 130."""
    reader = _scripted_reader(["hello", None, None])
    assert _run_surface(surface, reader) == 130


def test_empty_submit_does_not_exit(surface: Any) -> None:
    """A single ``None`` (empty submit or stray Ctrl-D) only warns; a turn continues."""
    reader = _scripted_reader([None, "still here", "quit"])
    assert _run_surface(surface, reader) == 0


def test_help_keeps_session_open(surface: Any) -> None:
    reader = _scripted_reader(["/help", "quit"])
    assert _run_surface(surface, reader) == 0


def test_turns_run_and_continue(surface: Any) -> None:
    reader = _scripted_reader(["first", "second", "quit"])
    code = _run_surface(surface, reader)
    assert code == 0


def test_double_ctrl_c_exits_130(surface: Any) -> None:
    reader = _scripted_reader([KeyboardInterrupt(), KeyboardInterrupt()])
    assert _run_surface(surface, reader) == 130


def test_first_ctrl_c_only_warns(surface: Any) -> None:
    reader = _scripted_reader([KeyboardInterrupt(), "quit"])
    assert _run_surface(surface, reader) == 0


def test_aborted_entry_none_after_ctrl_c_exits_130(surface: Any) -> None:
    reader = _scripted_reader([KeyboardInterrupt(), None])
    assert _run_surface(surface, reader) == 130


# ── Approvals ─────────────────────────────────────────────────────────


def test_approval_allow_decides_inline(surface: Any) -> None:
    """An approval outcome is decided inline; the loop continues."""

    class _ApprovalSession(surface.session_cls):  # type: ignore[misc]
        async def run_turn(self, message: str) -> TurnOutcome:
            self.outcomes.append(
                _outcome("approval", approval={"request_id": "req-1"}, text="paused")
            )
            return self.outcomes[-1]

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _ApprovalSession
    try:
        answers = iter(["a"])

        async def _decider(_console: Any) -> Any:
            return next(answers)

        reader = _scripted_reader(["do it", "quit"])
        code = _run_surface(surface, reader, decider=_decider)
        assert code == 0
    finally:
        mod.InteractiveSession = original


def test_approval_ctrl_c_defers(surface: Any) -> None:
    """A Ctrl-C during the decision read defers — never decides for the operator."""

    class _ApprovalSession(surface.session_cls):  # type: ignore[misc]
        async def run_turn(self, message: str) -> TurnOutcome:
            self.outcomes.append(
                _outcome("approval", approval={"request_id": "req-1"}, text="paused")
            )
            return self.outcomes[-1]

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _ApprovalSession
    try:

        async def _decider(_console: Any) -> Any:
            raise KeyboardInterrupt()

        reader = _scripted_reader(["do it", "quit"])
        code = _run_surface(surface, reader, decider=_decider)
        assert code == 0
    finally:
        mod.InteractiveSession = original


def test_yield_outcome_prints_resume_hint(surface: Any) -> None:
    """A yield outcome tells the operator the next message answers it."""

    class _YieldSession(surface.session_cls):  # type: ignore[misc]
        async def run_turn(self, message: str) -> TurnOutcome:
            self.outcomes.append(_outcome("yield", yielded_question="What environment?", text=""))
            return self.outcomes[-1]

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _YieldSession
    try:
        reader = _scripted_reader(["deploy", "quit"])
        code = _run_surface(surface, reader)
        assert code == 0
    finally:
        mod.InteractiveSession = original


def test_restart_required_ends_session(surface: Any) -> None:
    """A restart_required runtime ends the session with the restart notice."""

    class _RestartSession(surface.session_cls):  # type: ignore[misc]
        async def run_turn(self, message: str) -> TurnOutcome:
            self.runtime.session.restart_required = True
            self.runtime.session.restart_reason = "pack install"
            return _outcome("ok", text="installed")

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _RestartSession
    try:
        reader = _scripted_reader(["install pack", "quit"])
        code = _run_surface(surface, reader)
        assert code == 0
    finally:
        mod.InteractiveSession = original


def test_teardown_failure_exits_nonzero(surface: Any) -> None:
    """A bounded-close failure is loud: exit 1, not silent success."""

    class _FailingCloseSession(surface.session_cls):  # type: ignore[misc]
        async def aclose(self) -> None:
            raise RuntimeError("pool teardown stalled")

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _FailingCloseSession
    try:
        reader = _scripted_reader(["quit"])
        code = _run_surface(surface, reader)
        assert code == 1
    finally:
        mod.InteractiveSession = original


def test_substitution_failure_keeps_session(surface: Any) -> None:
    """A failed $(...) substitution reports and keeps the loop alive."""
    reader = _scripted_reader(["$(definitely-not-a-command-xyz)", "quit"])
    code = _run_surface(surface, reader)
    assert code == 0


def test_render_outcome_lines(surface: Any) -> None:
    """Outcome renderers map each engine kind to its operator notice."""
    from rich.console import Console as _Console

    console = _Console(
        file=type("_F", (), {"write": lambda s, t: None, "flush": lambda s: None})(),
        width=80,
        force_terminal=False,
    )
    interactive_mod._render_outcome(console, _outcome("paused"))
    interactive_mod._render_outcome(console, _outcome("cancelled"))
    interactive_mod._render_outcome(console, _outcome("error"))
    interactive_mod._render_outcome(console, _outcome("ok"))
    interactive_mod._render_restart_notice(console, "pack install")
    interactive_mod._model_hint(surface.runtime)


def test_model_hint_with_manifest(surface: Any) -> None:
    """The banner title resolves mode · model when the manifest names it."""
    from types import SimpleNamespace as _NS

    runtime = _NS(
        session=_NS(
            current_mode_id="default",
            next_model_id="m-1",
            manifest=_NS(models=[_NS(model_id="m-1")]),
        )
    )
    assert interactive_mod._model_hint(runtime) == "default · m-1"


def test_approval_defer_keeps_request_pending(surface: Any) -> None:
    class _ApprovalSession(surface.session_cls):  # type: ignore[misc]
        async def run_turn(self, message: str) -> TurnOutcome:
            self.outcomes.append(
                _outcome("approval", approval={"request_id": "req-1"}, text="paused")
            )
            return self.outcomes[-1]

    import hosts.cli.interactive as mod

    original = mod.InteractiveSession
    mod.InteractiveSession = _ApprovalSession
    try:

        async def _decider(_console: Any) -> Any:
            return ""  # empty = defer

        reader = _scripted_reader(["do it", "quit"])
        code = _run_surface(surface, reader, decider=_decider)
        assert code == 0
    finally:
        mod.InteractiveSession = original


# ── Pure helpers ──────────────────────────────────────────────────────


def test_parse_approval_answer_vocabulary() -> None:
    assert interactive_mod._parse_approval_answer("") == ("defer", "")
    assert interactive_mod._parse_approval_answer("a") == ("allow", "")
    assert interactive_mod._parse_approval_answer("y") == ("allow", "")
    assert interactive_mod._parse_approval_answer("d") == ("deny", "")
    assert interactive_mod._parse_approval_answer("n") == ("deny", "")
    assert interactive_mod._parse_approval_answer("r") == ("allow", "")
    assert interactive_mod._parse_approval_answer("r git status") == ("allow", "git status")
    assert interactive_mod._parse_approval_answer("what?") == ("defer", "")


def test_classify_session_message() -> None:
    assert interactive_mod._classify_session_message("") == "skip"
    assert interactive_mod._classify_session_message("   ") == "skip"
    assert interactive_mod._classify_session_message("quit") == "quit"
    assert interactive_mod._classify_session_message("/exit") == "quit"
    assert interactive_mod._classify_session_message("/help") == "help"
    assert interactive_mod._classify_session_message("fix the parser") == "turn"


def test_expand_message_passthrough() -> None:
    expanded, failure = interactive_mod._expand_message("plain message")
    assert expanded == "plain message"
    assert failure == ""


def test_expand_message_failure_is_loud() -> None:
    expanded, failure = interactive_mod._expand_message("$(definitely-not-a-command-xyz)")
    assert failure, "a failed substitution must be reported, not silently dropped"
    assert expanded == "$(definitely-not-a-command-xyz)"
