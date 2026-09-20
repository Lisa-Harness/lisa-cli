"""Tests for the CLI approval panel — render and transport capture.

Covers the two surfaces of the policy-driven approval UX delivered to
the once-mode host:

- ``hosts.cli.render.render_approval_panel`` — the panel render with the
  sandbox containment badge (once mode renders it with ``keystrokes=False``;
  the decision hints point at `lisa once "approve <id>"`).
- ``OnceTransport.on_response`` — capturing ``data["approval"]`` into
  ``pending_approval`` + ``approval_unresolved`` and rendering the panel
  (instead of a bare text blob) on a tty console, or the fail-fast stderr
  notice when piped/redirected.

No MCP subprocess is ever spawned; the console is a recording stub.
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
    from hosts.cli.transport import OnceTransport

    # force_terminal=True so the transport takes its interactive branch and
    # renders the panel (a non-tty console gets the fail-fast stderr notice).
    return OnceTransport(console=Console(record=True, width=100, force_terminal=True))


@pytest.mark.asyncio
async def test_on_response_captures_approval_payload_and_renders_panel() -> None:
    transport = _make_transport()
    await transport.on_response(
        "Approval required for run a shell command.", {"approval": _payload()}
    )
    assert transport.pending_approval is not None
    assert transport.pending_approval["request_id"] == "abc123def456"
    assert transport.approval_unresolved is True
    # The panel rendered into the console recording.
    assert "CONTAINED" in transport.console.export_text()


@pytest.mark.asyncio
async def test_on_response_without_approval_renders_text_not_panel() -> None:
    transport = _make_transport()
    await transport.on_response("just an answer", None)
    assert transport.pending_approval is None
    assert transport.approval_unresolved is False


@pytest.mark.asyncio
async def test_start_thinking_disarms_pending_approval() -> None:
    """A consumed decision must not linger into the next turn."""
    transport = _make_transport()
    transport.pending_approval = _payload()
    transport.start_thinking()
    assert transport.pending_approval is None

