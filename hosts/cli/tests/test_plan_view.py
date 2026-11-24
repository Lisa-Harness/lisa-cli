"""Tests for the read-only plan panel and its Ctrl+T toggle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hosts.cli import interactive as interactive_mod
from hosts.cli.session_plan_view import render_plan_panel


class _NullTransport:
    """Transport stub for handler-path tests."""

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


@pytest.fixture
def surface() -> Any:
    import io

    from rich.console import Console

    class _S:
        console = Console(file=io.StringIO(), width=100, force_terminal=False)

    return _S()


def _entry_with_plan(plan_markdown: str) -> dict[str, Any]:
    return {
        "timestamp_utc": "2026-11-23T10:00:00",
        "user_slug": "cli-user",
        "mode_id": "default",
        "query": "do the task",
        "response_text": plan_markdown,
        "turn": 0,
    }


def _runtime_with_thread(tmp_path: Any, entries: list[dict[str, Any]]) -> Any:
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    thread_path = runtime.paths.thread_path
    if thread_path is not None:
        # The thread path is the STORE BASE; entries live in
        # <base>.current.jsonl (the store's own convention).
        import json

        current = Path(f"{thread_path}.current.jsonl")
        current.parent.mkdir(parents=True, exist_ok=True)
        with current.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return runtime


def test_plan_panel_renders_steps_and_current(tmp_path: Any) -> None:
    plan = (
        "## Plan\n"
        "1. **search** — Find the right module\n"
        "   _reads_\n"
        "2. **edit** — Apply the fix\n"
        "   _writes: apply_patch_\n"
    )
    runtime = _runtime_with_thread(tmp_path, [_entry_with_plan(plan)])
    panel = render_plan_panel(runtime)
    import io

    from rich.console import Console

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    console.print(panel)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "search" in out
    assert "edit" in out
    assert "Plan" in out


def test_plan_panel_without_active_plan(tmp_path: Any) -> None:
    runtime = _runtime_with_thread(tmp_path, [_entry_with_plan("just prose, no plan")])
    panel = render_plan_panel(runtime)
    import io

    from rich.console import Console

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    console.print(panel)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "No active workflow plan" in out


def test_plan_panel_empty_thread(tmp_path: Any) -> None:
    runtime = _runtime_with_thread(tmp_path, [])
    panel = render_plan_panel(runtime)
    import io

    from rich.console import Console

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    console.print(panel)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "No active workflow plan" in out


# ── Ctrl+T toggle (transport) ─────────────────────────────────────────


def _make_tty_transport() -> Any:
    import io

    from rich.console import Console

    from hosts.cli.transport import PromptTransport

    console = Console(file=io.StringIO(), record=True, force_terminal=True, width=100)
    return PromptTransport(console=console), console


def test_ctrl_t_parses_as_toggle_plan() -> None:
    from hosts.cli.transport import _parse_scroll_keys

    actions, pending = _parse_scroll_keys(b"\x14")
    assert actions == [("toggle_plan", "")]
    assert pending == b""


def test_plan_toggle_flips_visibility_and_refreshes() -> None:
    transport, _console = _make_tty_transport()
    transport.set_plan_provider(lambda: "step one\nstep two")
    assert transport._plan_visible is False

    transport._scroll_activity(("toggle_plan", ""))
    assert transport._plan_visible is True
    assert transport._plan_text == "step one\nstep two"

    transport._scroll_activity(("toggle_plan", ""))
    assert transport._plan_visible is False


def test_plan_toggle_without_provider_is_inert() -> None:
    transport, _console = _make_tty_transport()
    transport._scroll_activity(("toggle_plan", ""))
    assert transport._plan_visible is False


def test_plan_provider_failure_renders_placeholder() -> None:
    transport, _console = _make_tty_transport()

    def _boom() -> str:
        raise RuntimeError("thread unreadable")

    transport.set_plan_provider(_boom)
    transport._scroll_activity(("toggle_plan", ""))
    assert transport._plan_visible is True
    assert transport._plan_text == "(plan unavailable)"


def test_frame_includes_plan_when_visible() -> None:
    transport, console = _make_tty_transport()
    transport._screen_mode = True
    transport.set_plan_provider(lambda: "the checklist")
    transport._scroll_activity(("toggle_plan", ""))

    frame = transport._activity_frame()

    rendered = console.render_lines(frame, console.options, pad=False)
    plain = "\n".join("".join(seg.text for seg in line) for line in rendered)
    assert "the checklist" in plain


def test_plan_checklist_marks_completed_and_current(tmp_path: Any) -> None:
    """The checklist renders both declared steps with their queries."""
    plan = (
        "## Plan\n"
        "1. **search** — Find it\n"
        "   _reads_\n"
        "2. **edit** — Apply it\n"
        "   _writes: apply_patch_\n"
    )
    runtime = _runtime_with_thread(tmp_path, [_entry_with_plan(plan)])
    panel = render_plan_panel(runtime)
    import io

    from rich.console import Console

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    console.print(panel)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "search" in out
    assert "edit" in out


def test_plan_and_diff_handlers_render(surface: Any) -> None:
    """``/plan`` and ``/diff`` handlers render through the real paths."""
    import asyncio
    import tempfile
    from pathlib import Path

    from runtime.testing import build_test_runtime

    from hosts.cli.session import InteractiveSession

    runtime = build_test_runtime(Path(tempfile.mkdtemp()), models_yaml="")
    session = InteractiveSession(runtime=runtime, transport=_NullTransport())
    result = asyncio.run(interactive_mod._render_plan_command(session, surface.console))
    assert result is None  # the handler prints; it returns nothing

    # /diff against a non-repo temp dir still renders the honest notice.
    asyncio.run(interactive_mod._render_diff_command(session, surface.console, ""))


def test_diff_handler_with_scope(surface: Any, monkeypatch: Any) -> None:
    """``/diff staged`` parses the scope and renders through the service."""
    import asyncio
    import tempfile
    from pathlib import Path

    from runtime.testing import build_test_runtime

    from hosts.cli import workspace as workspace_mod
    from hosts.cli.session import InteractiveSession
    from hosts.cli.workspace import DiffResult

    calls: list[dict[str, Any]] = []

    async def _fake_read_diff(root: Path, **kwargs: Any) -> Any:
        calls.append({"root": root, **kwargs})
        return DiffResult(scope="staged (index vs HEAD)", files=())

    monkeypatch.setattr(workspace_mod, "read_diff", _fake_read_diff)
    runtime = build_test_runtime(Path(tempfile.mkdtemp()), models_yaml="")
    session = InteractiveSession(runtime=runtime, transport=_NullTransport())
    asyncio.run(interactive_mod._render_diff_command(session, surface.console, "staged"))
    assert calls and calls[0]["staged"] is True


def test_plan_refresh_without_provider_returns_early() -> None:
    """A plan refresh with no provider wired is a no-op (one-shot path)."""
    transport, _console = _make_tty_transport()
    transport._refresh_plan_text()  # provider is None: returns before touching state
    assert transport._plan_text == ""


def test_plan_checklist_second_step_gets_newline(tmp_path: Any) -> None:
    """A multi-step plan renders each step on its own line."""
    plan = "## Plan\n1. **alpha** — First\n_reads_\n2. **beta** — Second\n_reads_\n"
    runtime = _runtime_with_thread(tmp_path, [_entry_with_plan(plan)])
    from kernel.workflow_view import derive_workflow_view
    from runtime.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=50)
    view = derive_workflow_view(entries)
    from hosts.cli.session_plan_view import _plan_checklist

    text = _plan_checklist(view)
    assert "alpha" in text.plain
    assert "beta" in text.plain
    assert "\n" in text.plain  # steps separated onto their own lines


def test_make_plan_provider_returns_the_panel_body(surface: Any) -> None:
    """The provider closure renders through the sync panel path."""
    import tempfile
    from pathlib import Path

    from runtime.testing import build_test_runtime

    from hosts.cli.session import InteractiveSession

    runtime = build_test_runtime(Path(tempfile.mkdtemp()), models_yaml="")
    session = InteractiveSession(runtime=runtime, transport=_NullTransport())
    provider = interactive_mod._make_plan_provider(session)
    panel = provider()
    assert panel is not None
    rendered = str(panel.renderable)
    assert "plan" in rendered.lower()


def test_latest_probe_early_returns(surface: Any, monkeypatch: Any) -> None:
    """The latest-conversation probe returns None on construction failure."""
    monkeypatch.setattr("hosts.cli.cli_runtime._initialize_runtime_or_exit", lambda **kw: None)
    assert interactive_mod._resolve_latest_conversation({}) is None


def _recording_transport(wired: list[str]) -> Any:
    """A transport stub class that records seam wiring."""

    class _RecordingTransport:
        def __init__(self, console: Any = None, **kwargs: Any) -> None:
            self.console = console
            self._stop_handler = None
            self._submit_handler = None
            self._plan_provider = None

        def set_stop_handler(self, handler: Any) -> None:
            self._stop_handler = handler

        def set_submit_handler(self, handler: Any) -> None:
            self._submit_handler = handler

        def set_plan_provider(self, provider: Any) -> None:
            self._plan_provider = provider
            wired.append("plan")

        async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
            return None

    return _RecordingTransport


def test_run_conversation_wires_plan_provider(surface: Any, monkeypatch: Any) -> None:
    """The real conversation runner wires the Ctrl+T plan provider on the transport."""
    import asyncio
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from hosts.cli import transport as transport_mod

    wired: list[str] = []

    class _StubSession:
        def __init__(self, runtime: Any, transport: Any) -> None:
            self.runtime = runtime
            self.transport = transport
            wired.append("session")

        async def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def request_stop(self) -> None:
            return None

    class _StubRuntime:
        session = SimpleNamespace(streaming=SimpleNamespace(mode="none"))
        config = SimpleNamespace(streaming=SimpleNamespace(mode="none"))

        async def start(self) -> None:
            return None

    async def _stub_reader(prefill: str = "") -> Any:
        return None

    async def _stub_loop(*args: Any, **kwargs: Any) -> int:
        return 0

    monkeypatch.setattr(transport_mod, "PromptTransport", _recording_transport(wired))
    monkeypatch.setattr(interactive_mod, "_session_loop", _stub_loop)
    monkeypatch.setattr(interactive_mod, "InteractiveSession", _StubSession)
    monkeypatch.setattr(
        "hosts.cli.cli_runtime._initialize_runtime_or_exit", lambda **kwargs: _StubRuntime()
    )
    monkeypatch.setattr("hosts.cli.prompt._collect_model_pricing", lambda runtime: ({}, {}))

    asyncio.run(
        interactive_mod._run_conversation(
            surface.console,
            {"data_dir": str(Path(tempfile.mkdtemp()))},
            _stub_reader,
            None,
        )
    )
    assert "plan" in wired, "the runner must wire the plan provider"
    assert "session" in wired
