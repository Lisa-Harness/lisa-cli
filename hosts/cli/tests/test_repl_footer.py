"""Tests for the ``_render_turn_footer`` post-response summary in the REPL.

The footer renders a one-line ``mode · model · tokens · elapsed`` summary
sourced from ``runtime.last_turn_context`` plus a handoff note when the
model switched modes mid-turn.
"""

from __future__ import annotations

# pylint: disable=protected-access
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from host.runtime import HostRuntime, TurnContext
from hosts.cli.repl import _render_turn_footer
from rich.console import Console


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _make_console() -> Console:
    # A StringIO-backed console records output via capture() (quiet=True
    # suppresses capture entirely).
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_footer_quiet_when_no_completion_context(tmp_path: Path) -> None:
    """No last_turn_context (or no completion dict) means no footer."""
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = None
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=340.0)
    assert capture.get().strip() == ""


def test_footer_renders_mode_model_tokens_time(tmp_path: Path) -> None:
    """The footer shows mode, model, total tokens, and elapsed ms."""
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = TurnContext(
        mode_id="coding",
        completion={
            "mode_id": "coding",
            "total_tokens": 1200,
            "resolved_model": {"model_id": "haiku", "provider": "anthropic"},
        },
    )
    runtime.current_mode_id = "coding"
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=340.0)
    out = capture.get().strip()
    assert "▾ coding" in out
    assert "haiku" in out
    assert "1200 tok" in out
    assert "340ms" in out
    # No handoff — current == entry mode.
    assert "→" not in out


def test_footer_shows_handoff_when_mode_changed_mid_turn(tmp_path: Path) -> None:
    """When runtime.current_mode_id differs from the entry mode, the footer
    appends ``→  <new mode>`` so the mid-turn transition is visible.
    """
    runtime = _make_runtime(tmp_path)
    console = _make_console()
    runtime.last_turn_context = TurnContext(
        mode_id="default",
        completion={
            "mode_id": "default",
            "total_tokens": 900,
            "resolved_model": {"model_id": "haiku", "provider": "anthropic"},
        },
    )
    # Simulate handle_message having applied a mid-turn handoff.
    runtime.current_mode_id = "coding"
    with console.capture() as capture:
        _render_turn_footer(console, runtime, elapsed_ms=500.0)
    out = capture.get().strip()
    assert "▾ default" in out
    assert "→ coding" in out


def test_repl_toolbar_shows_ctrl_j_when_shift_enter_unsupported() -> None:
    """The toolbar shows 'Ctrl+J newline' until an s-enter event fires (ask 4)."""
    from hosts.cli.repl import _repl_bottom_toolbar
    from hosts.cli.transport import RichTransport

    console = _make_console()
    transport = RichTransport(console=console)
    transport.shift_enter_supported = False
    toolbar = _repl_bottom_toolbar(transport, session_name="test")
    plain = "".join(text for _, text in toolbar)
    assert "Ctrl+J newline" in plain
    assert "Shift+Enter" not in plain


def test_repl_toolbar_shows_shift_enter_once_supported() -> None:
    """Once shift_enter_supported is set, the toolbar shows 'Shift+Enter newline' (ask 4)."""
    from hosts.cli.repl import _repl_bottom_toolbar
    from hosts.cli.transport import RichTransport

    console = _make_console()
    transport = RichTransport(console=console)
    transport.shift_enter_supported = True
    toolbar = _repl_bottom_toolbar(transport, session_name="test")
    plain = "".join(text for _, text in toolbar)
    assert "Shift+Enter newline" in plain
    assert "Ctrl+J" not in plain


def test_repl_toolbar_shows_mode_and_omits_tab_complete() -> None:
    """The toolbar surfaces the current mode and drops 'Tab complete' (ask)."""
    from hosts.cli.repl import _repl_bottom_toolbar
    from hosts.cli.transport import RichTransport

    console = _make_console()
    transport = RichTransport(console=console)
    toolbar = _repl_bottom_toolbar(transport, session_name="test", mode_id="coding")
    plain = "".join(text for _, text in toolbar)
    assert "mode:coding" in plain
    assert "Tab complete" not in plain


def test_repl_toolbar_mode_defaults_to_default() -> None:
    """An empty mode_id renders as 'default' in the toolbar."""
    from hosts.cli.repl import _repl_bottom_toolbar
    from hosts.cli.transport import RichTransport

    console = _make_console()
    transport = RichTransport(console=console)
    toolbar = _repl_bottom_toolbar(transport, session_name="test")
    plain = "".join(text for _, text in toolbar)
    assert "mode:default" in plain


def test_shift_enter_registration_failure_keeps_ctrl_j_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older prompt_toolkit must not block REPL startup over ``s-enter``."""
    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport

    registered_keys: list[str] = []

    class _BindingsRejectingShiftEnter:
        def add(self, key: str) -> Any:
            if key == "s-enter":
                raise ValueError("Invalid key: s-enter")
            registered_keys.append(key)

            def _decorate(callback: Any) -> Any:
                return callback

            return _decorate

    monkeypatch.setattr(cli_repl, "KeyBindings", _BindingsRejectingShiftEnter)
    transport = RichTransport(console=_make_console())

    bindings = cli_repl._build_repl_key_bindings(transport)

    assert bindings is not None
    assert "enter" in registered_keys
    assert "c-j" in registered_keys
    assert transport.shift_enter_supported is False


@pytest.mark.asyncio
async def test_rich_turn_uses_transport_workflow_summary_not_footer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rich transport owns the completed-turn summary and footer is not repeated."""
    from hosts.cli import repl as cli_repl
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    footer_calls: list[tuple[object, ...]] = []

    async def _handle_message(*args: Any) -> SimpleNamespace:
        callback_transport = args[2]
        await callback_transport.on_response("answer")
        return SimpleNamespace(text="answer")

    def _record_footer(*args: object) -> None:
        footer_calls.append(args)

    monkeypatch.setattr(cli_repl, "handle_message", _handle_message)
    monkeypatch.setattr(cli_repl, "_render_turn_footer", _record_footer)

    await cli_repl._process_repl_turn(runtime, transport, console, "hello")

    assert footer_calls == []
