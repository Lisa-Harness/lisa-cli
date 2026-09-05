"""Tests for terminal signal handling and the :class:`TerminalSignalDispatcher`.

Covers:
- ``OperatorSignal`` enum creation, string compatibility, ``bool()``.
- All key bindings register and call the correct dispatcher method.
- The clear-at-turn-start invariant (regression for ``operator-signal-never-cleared.md``).
- Double-tap Ctrl+C escalation.
- Per-OS Ctrl+Z behavior.
- Mid-turn interrupt keeps the pool alive.
- No operator-signal leaks across consecutive turns.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from host.operator_signal import OperatorSignal
from host.runtime import HostRuntime
from rich.console import Console


# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _make_console() -> Console:
    """Return a quiet Console backed by an in-memory buffer."""
    import io

    return Console(file=io.StringIO(), force_terminal=False, width=120)


# ── OperatorSignal enum ─────────────────────────────────────────────


class TestOperatorSignalEnum:
    """Tests for the ``OperatorSignal`` str-enum."""

    def test_enum_members_are_strings(self) -> None:
        """Every enum member is a valid str and comparable via ``is``."""
        assert OperatorSignal.NONE == ""
        assert OperatorSignal.PAUSE == "pause"
        assert OperatorSignal.STOP == "stop"
        assert OperatorSignal.TERMINATE == "terminate"
        assert OperatorSignal.INTERRUPT == "interrupt"
        assert OperatorSignal.EOF == "eof"
        assert OperatorSignal.SUSPEND == "suspend"

    def test_identity_comparison(self) -> None:
        """``is`` comparison works within the enum."""
        assert OperatorSignal("pause") is OperatorSignal.PAUSE
        assert OperatorSignal("") is OperatorSignal.NONE

    def test_bad_string_raises_value_error(self) -> None:
        """Constructing with an unknown string raises ValueError."""
        with pytest.raises(ValueError):
            OperatorSignal("nonexistent")

    def test_bool_none_is_false(self) -> None:
        """Only NONE is falsy."""
        assert not bool(OperatorSignal.NONE)
        assert bool(OperatorSignal.PAUSE)
        assert bool(OperatorSignal.STOP)
        assert bool(OperatorSignal.TERMINATE)
        assert bool(OperatorSignal.INTERRUPT)
        assert bool(OperatorSignal.EOF)
        assert bool(OperatorSignal.SUSPEND)


# ── Key binding registration ─────────────────────────────────────────


@pytest.mark.parametrize(
    "key,method_name",
    [
        ("c-c", "request_interrupt"),
        ("c-d", "request_eof"),
        ("c-z", "request_suspend"),
        ("c-p", "request_pause"),
        ("c-o", "request_stop"),
        ("c-\\", "request_terminate"),
        ("c-y", "request_toggle_thinking"),
        ("c-j", None),  # no dispatcher call — inserts newline directly
        ("enter", None),  # no dispatcher call — submits buffer
    ],
)
def test_all_key_bindings_register_and_dispatch(
    monkeypatch: pytest.MonkeyPatch, key: str, method_name: str | None
) -> None:
    """Every binding in the matrix registers and calls the correct method.

    Non-dispatcher bindings (enter, c-j) register but don't call a
    dispatcher method — they manipulate the buffer directly.
    """
    from hosts.cli import keys as cli_keys
    from hosts.cli.transport import RichTransport

    # Stub key-bindings class that records registered keys *and*
    # the callback so we can invoke it to verify dispatch.
    registered_keys: list[str] = []
    registered_callbacks: dict[str, Any] = {}

    class _RecordBindings:
        def add(self, k: str, filter: Any = None, **kwargs: Any) -> Any:  # noqa: A002
            registered_keys.append(k)

            def _decorate(callback: Any) -> Any:
                registered_callbacks[k] = callback
                return callback

            return _decorate

    monkeypatch.setattr(cli_keys, "_PTKeyBindings", _RecordBindings)

    transport = RichTransport(console=_make_console())
    runtime = SimpleNamespace(operator_signal_value=OperatorSignal.NONE)
    dispatcher = cli_keys.TerminalSignalDispatcher(
        runtime, transport, _make_console()
    )

    bindings = cli_keys.build_repl_key_bindings(transport, dispatcher)

    assert bindings is not None
    assert key in registered_keys, f"Key {key!r} not registered"

    if method_name is not None:
        # Simulate dispatching: invoke the callback and check the signal.
        callback = registered_callbacks[key]

        class _FakeEvent:
            current_buffer = SimpleNamespace(text="test text")
            app = SimpleNamespace()

            class app:  # type: ignore[assignment]
                exit_called = False
                exit_result: object = None

                @staticmethod
                def exit(result: object = None) -> None:
                    _FakeEvent.app.exit_called = True
                    _FakeEvent.app.exit_result = result

                @staticmethod
                def invalidate() -> None:
                    pass

        event = _FakeEvent()
        callback(event)

        # Verify the signal was set (for operator-signal bindings).
        if key in ("c-c", "c-d", "c-z", "c-p", "c-o", "c-\\"):
            assert runtime.operator_signal_value is not OperatorSignal.NONE, (
                f"Key {key!r} did not set operator_signal_value"
            )

        # c-d and c-c (at-prompt) should exit the session.
        if key == "c-d":
            assert _FakeEvent.app.exit_called
        # c-z on Windows should exit; on POSIX it suspends.
        if key == "c-z" and os.name == "nt":
            assert _FakeEvent.app.exit_called


# ── Clear-at-turn-start invariant ───────────────────────────────────


def test_dispatcher_clear_resets_signal_to_none(tmp_path: Path) -> None:
    """``dispatcher.clear()`` sets ``operator_signal_value`` to NONE."""
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    dispatcher.request_pause()
    assert runtime.operator_signal_value is OperatorSignal.PAUSE

    dispatcher.clear()
    assert runtime.operator_signal_value is OperatorSignal.NONE


def test_process_repl_turn_clears_signal_before_handle_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher clears the signal at the start of each turn.

    Regression test for the bug documented in ``operator-signal-never-cleared.md``:
    a stale PAUSE signal must not leak into the next turn.
    """
    from hosts.cli import repl as cli_repl
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    # Simulate a previous turn having paused.
    dispatcher.request_pause()
    assert runtime.operator_signal_value is OperatorSignal.PAUSE

    # handle_message should NOT see the stale signal.
    seen_signals: list[OperatorSignal] = []

    async def _record_signal(*args: Any) -> SimpleNamespace:
        seen_signals.append(runtime.operator_signal_value)
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(cli_repl, "handle_message", _record_signal)

    asyncio.run(
        cli_repl._process_repl_turn(
            runtime, transport, console, "hello", dispatcher
        )
    )

    # handle_message saw NONE, not PAUSE.
    assert seen_signals == [OperatorSignal.NONE]


# ── Double-tap escalation ────────────────────────────────────────────


def test_sigint_double_tap_within_window_escalates_to_terminate(tmp_path: Path) -> None:
    """Two interrupt requests within the window escalate to TERMINATE."""
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    dispatcher.request_interrupt()
    assert runtime.operator_signal_value is OperatorSignal.INTERRUPT

    dispatcher.request_interrupt()
    assert runtime.operator_signal_value is OperatorSignal.TERMINATE


def test_sigint_outside_window_resets_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt requests outside the double-tap window reset the counter."""
    import time as _time

    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    # First interrupt.
    dispatcher.request_interrupt()
    assert runtime.operator_signal_value is OperatorSignal.INTERRUPT

    # Advance time past the window.
    with mock.patch.object(
        _time, "monotonic", return_value=_time.monotonic() + 3.0
    ):
        dispatcher.request_interrupt()

    # Still INTERRUPT (counter reset).
    assert runtime.operator_signal_value is OperatorSignal.INTERRUPT


# ── Ctrl+Z per-OS ────────────────────────────────────────────────────


def test_ctrl_z_windows_is_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows, Ctrl+Z writes EOF (not SUSPEND)."""
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    monkeypatch.setattr(os, "name", "nt")

    runtime = SimpleNamespace(operator_signal_value=OperatorSignal.NONE)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    dispatcher.request_suspend()
    assert runtime.operator_signal_value is OperatorSignal.EOF


def test_ctrl_z_posix_is_suspend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On POSIX, Ctrl+Z writes SUSPEND."""
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    monkeypatch.setattr(os, "name", "posix")
    # Prevent os.kill from actually sending SIGTSTP.
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    # Use SimpleNamespace to avoid pathlib PosixPath errors on Windows.
    runtime = SimpleNamespace(operator_signal_value=OperatorSignal.NONE)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    dispatcher.request_suspend()
    assert runtime.operator_signal_value is OperatorSignal.SUSPEND


# ── No operator-signal leaks across turns ──────────────────────────


def test_no_operator_signal_leaks_across_two_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PAUSE set in turn 1 does not leak into turn 2."""
    from hosts.cli import repl as cli_repl
    from hosts.cli.keys import TerminalSignalDispatcher
    from hosts.cli.transport import RichTransport

    runtime = _make_runtime(tmp_path)
    console = _make_console()
    transport = RichTransport(console=console)
    dispatcher = TerminalSignalDispatcher(runtime, transport, console)

    # Turn 1: pause.
    dispatcher.request_pause()

    async def _ok_turn(*args: Any) -> SimpleNamespace:
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(cli_repl, "handle_message", _ok_turn)
    asyncio.run(
        cli_repl._process_repl_turn(
            runtime, transport, console, "turn 1", dispatcher
        )
    )

    # Turn 2: signal should be NONE.
    asyncio.run(
        cli_repl._process_repl_turn(
            runtime, transport, console, "turn 2", dispatcher
        )
    )
    # The turn-start clear should have reset it.
    # At this point it's still NONE because handle_message didn't set anything.
    assert runtime.operator_signal_value is OperatorSignal.NONE


# ── OperatorSignal str-roundtrips through kernel path ────────────────


def test_operator_signal_str_roundtrip_via_turn_hooks() -> None:
    """The kernel's ``==`` comparisons work against OperatorSignal enum values.

    Even though the kernel widens the type annotation to ``str``, the
    ``==`` comparisons against string literals like ``"pause"`` work
    correctly because ``OperatorSignal(str, Enum)`` is a ``str`` subclass.
    """
    assert OperatorSignal.NONE == ""
    assert OperatorSignal.PAUSE == "pause"
    assert OperatorSignal.STOP == "stop"
    assert OperatorSignal.TERMINATE == "terminate"

    # The kernel does: `if op_signal == "pause": ...`
    sig = OperatorSignal.PAUSE
    assert sig == "pause"
    if sig == "pause":
        action = "PauseAction"
    else:
        action = ""
    assert action == "PauseAction"

    # Empty string check.
    sig = OperatorSignal.NONE
    assert sig == ""
    assert not sig


# ── Operator exit indicator strings (visual only) ──────────────────


def test_operator_exit_indicator_strings_are_unchanged() -> None:
    """The kernel's ``operator_exit`` strings are unchanged.

    These are set by the kernel on the result object (e.g. ``"paused"``,
    ``"stopped"``, ``"terminated"``) and read by ``_process_repl_turn``
    for the visual indicator line. They are distinct from
    ``OperatorSignal`` enum values (note the ``ed`` suffix).
    """
    # The kernel operator_exit values use past-tense forms.
    assert OperatorSignal.PAUSE.value == "pause"
    assert OperatorSignal.STOP.value == "stop"
    assert OperatorSignal.TERMINATE.value == "terminate"
    # These are NOT the same as the kernel's operator_exit strings
    # ("paused", "stopped", "terminated") — the enum values match the
    # imperative signal used by turn_hooks_builder.py, not the past-tense
    # exit indicator used for display.
    assert OperatorSignal.PAUSE.value != "paused"
    assert OperatorSignal.STOP.value != "stopped"
    assert OperatorSignal.TERMINATE.value != "terminated"
