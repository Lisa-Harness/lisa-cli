"""Tests for the prompt-mode transport.

``prompt`` is one process, one turn — not a session with a screen — so
``PromptTransport`` renders a transient one-line activity indicator and
prints each tool/model event to scrollback as it *completes*. Tokens are
per-turn (the turn/session split collapses in a one-process invocation)
and the ⚡ footer replaces the Workflow panel.

Consoles here are non-tty (``Console(file=io.StringIO())``), so the Live
activity line never starts and assertions see only the printed lines.
"""

from __future__ import annotations

import io

import pytest

from rich.console import Console

from hosts.cli.transport import PromptTransport


def _make_transport() -> tuple[PromptTransport, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    return PromptTransport(console=console), buffer


@pytest.mark.asyncio
async def test_tool_completion_prints_one_line_with_display_label() -> None:
    """A completed tool call prints one line: ✓, tool, kernel display label, ms.

    The display label comes pre-rendered from the tool's ``display_template``
    (bash_run's ``{command}``) — the transport shows it, never re-derives it.
    """
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "ls"}, "display": "ls -la /etc"},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 120}
    )
    out = buffer.getvalue()
    # The completion line carries glyph, tool, display label, ✓ + duration, once.
    assert out.count("→ tool bash_run ls -la /etc ✓ 120ms") == 1
    # Dispatch printed nothing (only the activity spinner, absent on non-tty);
    # the single row is the completion update.
    assert out.count("bash_run") == 1


@pytest.mark.asyncio
async def test_arg_fallback_when_no_display_label() -> None:
    """Without a display label the transport previews the primary argument."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "read_file", "tool_kind": "pack_tool", "arguments": {"path": "/tmp/x"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "read_file", "ok": True, "duration_ms": 5}
    )
    assert "→ tool read_file /tmp/x ✓ 5ms" in buffer.getvalue()


@pytest.mark.asyncio
async def test_no_mode_prefix_on_event_lines() -> None:
    """Event rows render without the repeated ``[mode]`` column (compact)."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "pwd"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 3}
    )
    out = buffer.getvalue()
    assert "[ops]" not in out
    assert "bash_run pwd" in out


@pytest.mark.asyncio
async def test_handoff_prints_compact_line() -> None:
    """A mid-turn mode change prints one ↦ line, not the REPL's bracket form."""
    transport, buffer = _make_transport()
    await transport.on_stage("model_call", {"model_id": "m", "mode_id": "ops"})
    await transport.on_stage("model_call", {"model_id": "m", "mode_id": "fix"})
    out = buffer.getvalue()
    assert "↦ ops → fix" in out
    assert "handoff:" not in out


@pytest.mark.asyncio
async def test_tool_failure_marks_line_and_footer() -> None:
    """A failed tool call prints ✗ and the footer counts it."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "false"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": False, "duration_ms": 7}
    )
    await transport.on_response("answer text")
    assert "→ tool bash_run false ✗ 7ms" in buffer.getvalue()
    assert "1 tool (0 ✓ 1 ✗)" in buffer.getvalue()


@pytest.mark.asyncio
async def test_footer_is_single_compact_line() -> None:
    """The post-response summary is one ⚡ line with per-turn totals only."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call",
        {
            "model_id": "llama3.2",
            "provider": "ollama",
            "mode_id": "ops",
            "iteration": 1,
        },
    )
    await transport.on_stage(
        "model_call_complete",
        {"duration_ms": 340, "total_tokens": 4100, "iteration": 1},
    )
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "ls"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 120}
    )
    await transport.on_response("the answer")
    out = buffer.getvalue()
    footer_lines = [ln for ln in out.splitlines() if ln.strip().startswith("⚡")]
    assert len(footer_lines) == 1
    footer = footer_lines[0]
    assert "2 models" not in footer
    assert "1 model" in footer
    assert "1 tool" in footer
    assert "4.1k tok" in footer
    # A prompt run collapses the REPL-era turn/session split: exactly one token figure.
    assert "session" not in footer
    assert "turn" not in footer


@pytest.mark.asyncio
async def test_quiet_prints_response_only() -> None:
    """Quiet mode suppresses the activity line, per-tool lines, and the footer."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console, quiet=True)
    transport.start_turn("do a thing")
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "ls"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 3}
    )
    await transport.on_response("clean answer")
    out = buffer.getvalue()
    assert "clean answer" in out
    assert "bash_run" not in out
    assert "⚡" not in out
    assert "λ" not in out
