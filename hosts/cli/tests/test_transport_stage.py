"""Tests for the PromptTransport ``on_stage`` event-line rendering.

``prompt`` is one process, one turn. The transport prints each completed
tool/model/fork event as a real scrollback line (no Live transcript), so
these tests drive the event pairs and assert on what printed. Consoles are
non-tty (``Console(file=io.StringIO())``), so the activity Live never
starts and assertions see only the printed lines.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from hosts.cli.transport import PromptTransport


def _make_transport(**kwargs: Any) -> tuple[PromptTransport, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    return PromptTransport(console=console, **kwargs), buffer


# ── Fork rendering ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fork_events_render_header_and_children() -> None:
    """fork_start/child_start/child_complete/complete render one line each."""
    transport, buffer = _make_transport()
    await transport.on_stage("fork_start", {"children": 2})
    await transport.on_stage(
        "fork_child_start", {"child_index": 0, "mode_id": "ops", "stage_id": "exec"}
    )
    await transport.on_stage(
        "fork_child_start", {"child_index": 1, "mode_id": "fix", "stage_id": "write"}
    )
    out = buffer.getvalue()
    assert "stage: fork (2 children)" in out
    # Child-start lines do not print — only fork_start and completions do.
    assert out.count("stage: fork (2 children)") == 1
    await transport.on_stage(
        "fork_child_complete", {"child_index": 0, "status": "ok", "duration_ms": 12}
    )
    await transport.on_stage(
        "fork_child_complete", {"child_index": 1, "status": "ok", "duration_ms": 20}
    )
    out = buffer.getvalue()
    # All children done: one combined per-child table printed.
    assert "#0 ops · exec · ok · 12ms" in out
    assert "#1 fix · write · ok · 20ms" in out
    await transport.on_stage("fork_complete", {"ok": 2, "failed": 0})
    # fork_complete with no live children prints nothing extra.
    assert buffer.getvalue() == out


@pytest.mark.asyncio
async def test_start_thinking_resets_fork_and_counters() -> None:
    """start_thinking zeroes per-turn state so counters never leak across turns."""
    transport, buffer = _make_transport()
    await transport.on_stage("fork_start", {"children": 1})
    transport.start_thinking("dispatch")
    await transport.on_stage("fork_start", {"children": 3})
    assert "stage: fork (3 children)" in buffer.getvalue()
    assert transport._fork_total == 3  # pylint: disable=protected-access
    assert transport._model_call_count == 0  # pylint: disable=protected-access


# ── Model-call rendering ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_call_failure_prints_error_line() -> None:
    """model_call_failed prints one ✗-marked line immediately."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage("model_call_failed", {"model_id": "llama3.2", "duration_ms": 99})
    out = buffer.getvalue()
    assert "stage: model_call llama3.2 ✗ 99ms" in out


@pytest.mark.asyncio
async def test_model_call_complete_accumulates_tokens_and_duration() -> None:
    """model_call_complete accumulates tokens; the footer reports them."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage(
        "model_call_complete", {"duration_ms": 340, "total_tokens": 4100, "iteration": 1}
    )
    await transport.on_response("the answer")
    assert "4.1k tok" in buffer.getvalue()


@pytest.mark.asyncio
async def test_footer_billed_tokens_surfaces_cached_turns() -> None:
    """Cache-heavy turns surface the billed (uncached+completion) basis.

    Regression for the 2026-10-12 performance review: the walk figure
    (prompt+completion, cache hits at full) read 7.6M on a turn whose
    billed basis was 216k — alarming and useless. With cached tokens
    reported, the footer adds the billed figure next to the walk.
    """
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "glm-5.3-flash:cloud", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage(
        "model_call_complete",
        {
            "duration_ms": 340,
            "prompt_tokens": 100000,
            "cached_tokens": 99750,
            "completion_tokens": 3000,
            "total_tokens": 103000,
            "iteration": 1,
        },
    )
    await transport.on_response("done")
    out = buffer.getvalue()
    footer_lines = [ln for ln in out.splitlines() if ln.strip().startswith("⚡")]
    assert footer_lines, "footer must print"
    footer = footer_lines[-1]
    assert "103k tok" in footer
    assert "billed 3.25k" in footer


@pytest.mark.asyncio
async def test_footer_billed_segment_absent_without_cache_hits() -> None:
    """No cached tokens = no billed segment (the walk IS the billed basis)."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage(
        "model_call_complete", {"duration_ms": 340, "total_tokens": 4100, "iteration": 1}
    )
    await transport.on_response("done")
    assert "billed" not in buffer.getvalue()


# ── Mode tracking + handoff ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_prefix_tracked_and_handoff_prints_compact_line() -> None:
    """A mid-turn mode change prints one ↦ line (compact, no brackets)."""
    transport, buffer = _make_transport()
    await transport.on_stage("model_call", {"model_id": "m", "mode_id": "ops"})
    await transport.on_stage("model_call", {"model_id": "m", "mode_id": "fix"})
    out = buffer.getvalue()
    assert "↦ ops → fix" in out
    assert "handoff:" not in out


@pytest.mark.asyncio
async def test_event_lines_render_without_mode_column() -> None:
    """Event lines never repeat the ``[mode]`` column (activity line owns it)."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "pwd"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 3}
    )
    out = buffer.getvalue()
    assert "[ops]" not in out
    assert "→ tool bash_run pwd ✓ 3ms" in out


# ── Tool output streaming ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_output_lines_print_to_scrollback() -> None:
    """Streamed tool-output lines print with the dim rail as they arrive."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "sleep 1"},
        },
    )
    await transport.on_stage("tool_output", {"text": "hello\nworld", "stream": "stdout"})
    out = buffer.getvalue()
    assert "│ hello" in out
    assert "│ world" in out


@pytest.mark.asyncio
async def test_tool_output_stderr_lines_prefixed() -> None:
    """stderr stream lines gain a [stderr] marker so origins stay readable."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "sh -c 'echo err >&2'"},
        },
    )
    await transport.on_stage("tool_output", {"text": "err", "stream": "stderr"})
    assert "│ [stderr] err" in buffer.getvalue()


# ── Streamed deltas ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_chunks_stage_without_rendering() -> None:
    """Answer deltas stage; on_response renders once from the full text."""
    transport, buffer = _make_transport()
    await transport.on_text_chunk("partial", reasoning="")
    assert "partial" not in buffer.getvalue()
    await transport.on_response("the full answer")
    assert "the full answer" in buffer.getvalue()


@pytest.mark.asyncio
async def test_reasoning_chars_counted_not_printed() -> None:
    """Reasoning deltas count chars but never print inline."""
    transport, buffer = _make_transport()
    await transport.on_text_chunk("", reasoning="thinking")
    assert "thinking" not in buffer.getvalue()
    assert transport._status_reasoning_chars == 8  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_on_error_flushes_staged_answer_then_panel() -> None:
    """A mid-turn error flushes partial staged answer before the panel."""
    transport, buffer = _make_transport()
    await transport.on_text_chunk("half an answer", reasoning="")
    await transport.on_error("the tool exploded", None)
    out = buffer.getvalue()
    assert "half an answer" in out
    assert "the tool exploded" in out
    # The staging buffer is cleared: a second error does not re-print it.
    buffer.seek(0)
    buffer.truncate(0)
    await transport.on_error("again", None)
    assert "half an answer" not in buffer.getvalue()
