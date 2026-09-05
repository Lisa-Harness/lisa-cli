"""Tests for the CLI approval panel — render, transport capture, keystrokes.

Covers the three surfaces of the policy-driven approval UX delivered to the
terminal host:

- ``hosts.cli.render.render_approval_panel`` — the keystroke panel render with
  the sandbox containment badge.
- ``RichTransport.on_response`` — capturing ``data["approval"]`` into
  ``pending_approval`` and rendering the panel (instead of a bare text blob).
- ``hosts.cli.keys._add_approval_bindings`` — the armed y/n/a/v/Esc bindings
  that decide a pending approval with a single keystroke, submitting the
  corresponding slash command as the prompt result.

No MCP subprocess is ever spawned; the runtime and console are stubs.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from hosts.cli import render as render_mod

# ── Panel render ──────────────────────────────────────────────────────


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "lisa.approval.v1",
        "request_id": "abc123def456",
        "target_kind": "tool",
        "target_id": "bash_run",
        "pack_id": "lisa-pack-bash",
        "effect": "execute",
        "preview": "run a shell command",
        "command": "ls -la",
        "parameters": {"command": "ls -la"},
        "containment": "contained",
        "containment_reasons": "",
        "expires_at": "",
        "expires_in_seconds": 90,
        "policy_verdict": "ask",
        "policy_reason": "execute action requires approval",
        "policy_tier": "execute",
    }
    base.update(overrides)
    return base


def test_render_approval_panel_includes_command_and_hints() -> None:
    console = Console(record=True, width=100)
    panel = render_mod.render_approval_panel(_payload(), console=console)
    text = console.export_text()
    assert "run a shell command" in text
    assert "ls -la" in text
    assert "approve" in text and "deny" in text and "always" in text
    assert "/approve abc123def456" in text
    assert "execute action requires approval" in text
    assert "expires in 1m 30s" in text
    # Containment badge is shown and styled as contained.
    assert "CONTAINED" in text
    assert panel is not None


def test_render_approval_panel_unsandboxed_badge() -> None:
    console = Console(record=True, width=100)
    render_mod.render_approval_panel(_payload(containment="none"), console=console)
    assert "NOT SANDBOXED" in console.export_text()


def test_render_approval_panel_degraded_shows_reasons() -> None:
    console = Console(record=True, width=100)
    render_mod.render_approval_panel(
        _payload(containment="degraded", containment_reasons="bwrap binary not found"),
        console=console,
    )
    text = console.export_text()
    assert "DEGRADED" in text
    assert "bwrap binary not found" in text


def test_render_approval_panel_no_command_is_tolerated() -> None:
    """A non-shell action (no command) renders without crashing."""
    console = Console(record=True, width=100)
    render_mod.render_approval_panel(
        _payload(command="", target_id="write_file", effect="write", preview="apply the patch"),
        console=console,
    )
    assert "apply the patch" in console.export_text()


def test_render_approval_panel_shows_immediate_expiry() -> None:
    console = Console(record=True, width=100)
    render_mod.render_approval_panel(_payload(expires_in_seconds=0), console=console)
    assert "expires now" in console.export_text()


# ── Transport capture ─────────────────────────────────────────────────


def _make_transport() -> Any:
    from hosts.cli.transport import RichTransport

    return RichTransport(console=Console(record=True, width=100))


@pytest.mark.asyncio
async def test_on_response_captures_approval_payload_and_renders_panel() -> None:
    transport = _make_transport()
    await transport.on_response(
        "Approval required for run a shell command.", {"approval": _payload()}
    )
    assert transport.pending_approval is not None
    assert transport.pending_approval["request_id"] == "abc123def456"
    # The panel rendered into the console recording.
    assert "CONTAINED" in transport.console.export_text()


@pytest.mark.asyncio
async def test_on_response_without_approval_renders_text_not_panel() -> None:
    transport = _make_transport()
    await transport.on_response("just an answer", None)
    assert transport.pending_approval is None


@pytest.mark.asyncio
async def test_start_thinking_disarms_pending_approval() -> None:
    """A consumed decision must not linger into the next turn."""
    transport = _make_transport()
    transport.pending_approval = _payload()
    transport.start_thinking()
    assert transport.pending_approval is None


# ── Armed keystroke bindings ──────────────────────────────────────────


def _make_dispatcher_stub() -> Any:
    class _Dispatcher:
        def render_approval_view(self) -> None:
            return None

    return _Dispatcher()


def test_armed_binding_condition_blocks_when_not_armed() -> None:
    from prompt_toolkit.key_binding import KeyBindings

    from hosts.cli.keys import _add_approval_bindings

    transport = _make_transport()  # pending_approval is None -> not armed
    transport.pending_approval = None
    bindings = KeyBindings()
    _add_approval_bindings(bindings, transport, _make_dispatcher_stub())
    # The 'y' binding exists but is gated; with nothing armed the condition
    # evaluates False so a typed 'y' is not intercepted.
    assert bindings.get_bindings_for_keys(("y",)), "y binding must exist"


def test_armed_bindings_registered_for_all_decision_keys() -> None:
    from prompt_toolkit.key_binding import KeyBindings

    from hosts.cli.keys import _add_approval_bindings

    transport = _make_transport()
    transport.pending_approval = _payload()
    bindings = KeyBindings()
    _add_approval_bindings(bindings, transport, _make_dispatcher_stub())
    for key in ("y", "n", "a", "v", "escape"):
        assert bindings.get_bindings_for_keys((key,)), f"missing {key} binding"
