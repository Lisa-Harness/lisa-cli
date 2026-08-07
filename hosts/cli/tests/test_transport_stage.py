"""Tests for the RichTransport ``on_stage`` rendering.

Covers the flowing-transcript stage rows and the fork fan-out per-child
rendering that the kernel's ``on_stage`` telemetry hook drives.
"""

from __future__ import annotations

import io

import pytest
from hosts.cli.transport import RichTransport, _Row
from rich.console import Console
from rich.text import Text


@pytest.mark.asyncio
async def test_on_stage_renders_detailed_suffix() -> None:
    """Stage events append transcript rows; metadata suffixes render only on
    the power/debug tiers.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {
            "mode_id": "default",
            "model_id": "haiku",
            "provider": "anthropic",
            "iteration": 1,
            "tool_calls_so_far": 0,
        },
    )
    transport.stop_thinking()
    await transport.on_stage(
        "model_call_complete",
        {
            "mode_id": "default",
            "model_id": "haiku",
            "iteration": 1,
            "total_tokens": 120,
            "duration_ms": 340,
        },
    )
    # Rows accumulated in the transcript.
    assert any(r.kind == "model" for r in transport._rows)


@pytest.mark.asyncio
async def test_on_stage_fork_renders_one_line_per_child() -> None:
    """Fork fan-out events track one entry per child index; fork_complete
    keeps the per-child table visible.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")

    await transport.on_stage("fork_start", {"children": 2})
    assert transport._fork_total == 2
    assert transport._fork_children == {}

    await transport.on_stage(
        "fork_child_start",
        {
            "child_index": 0,
            "mode_id": "coding",
            "stage_id": "implement",
            "query_preview": "write the thing",
        },
    )
    await transport.on_stage(
        "fork_child_start",
        {"child_index": 1, "mode_id": "markdown", "stage_id": "", "query_preview": "doc it"},
    )
    assert set(transport._fork_children.keys()) == {0, 1}
    assert transport._fork_children[0]["status"] == "running"

    await transport.on_stage(
        "fork_child_complete", {"child_index": 0, "status": "ok", "duration_ms": 42}
    )
    assert transport._fork_children[0]["status"] == "ok"
    assert transport._fork_children[0]["duration_ms"] == 42

    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})
    assert transport._fork_children != {}
    assert transport._fork_total == 0
    transport.start_thinking("dispatch")
    assert transport._fork_children == {}


@pytest.mark.asyncio
async def test_start_thinking_resets_fork_state() -> None:
    """start_thinking clears any stale fork tracking from a prior turn."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport._fork_total = 5
    transport._fork_children = {0: {"status": "running"}}

    transport.start_thinking("model_call")

    assert transport._fork_total == 0
    assert transport._fork_children == {}


@pytest.mark.asyncio
async def test_render_fork_stage_format() -> None:
    """The fork stage text is one header line plus one line per child."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport._fork_total = 2
    transport._fork_children = {
        0: {
            "mode_id": "coding",
            "stage_id": "implement",
            "status": "running",
            "query_preview": "write the thing",
            "duration_ms": None,
        },
        1: {
            "mode_id": "markdown",
            "stage_id": "",
            "status": "ok",
            "query_preview": "",
            "duration_ms": 120,
        },
    }

    text = transport._render_fork_stage()

    lines = text.splitlines()
    assert lines[0] == "stage: fork (2 children)"
    assert "#0 coding · implement · running · write the thing" in lines[1]
    assert "#1 markdown · ok · 120ms" in lines[2]


@pytest.mark.asyncio
async def test_on_stage_no_spinner_prints_inline() -> None:
    """When no Live region is active, on_stage prints the row inline without raising."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    assert transport._status is None
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})


@pytest.mark.asyncio
async def test_mode_prefix_tracked_from_model_call() -> None:
    """model_call seeds _current_mode and the [mode] prefix appears on rows."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    assert transport._current_mode == ""

    await transport.on_stage(
        "model_call",
        {"mode_id": "coding", "model_id": "haiku", "provider": "anthropic", "iteration": 1},
    )
    assert transport._current_mode == "coding"

    line = transport._render_status_line("model_call", {"mode_id": "coding"})
    assert line.startswith("[coding] ")
    assert "mode_id=" not in line


@pytest.mark.asyncio
async def test_mode_prefix_updates_on_mid_turn_handoff() -> None:
    """A mid-turn handoff updates _current_mode so the [mode] prefix reflects the new owner."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")

    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert transport._current_mode == "default"

    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "handoff",
            "handoff_to": "coding",
            "ok": True,
            "iteration": 1,
        },
    )
    assert transport._current_mode == "coding"
    line = transport._render_status_line("model_call", {"mode_id": "coding"})
    assert line.startswith("[coding] ")


@pytest.mark.asyncio
async def test_tool_dispatch_renders_legible_kind_glyphs() -> None:
    """tool_dispatch renders a glyph + tool name per tool_kind."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport._current_mode = ""

    line = transport._render_tool_dispatch({"tool": "write_file", "tool_kind": "operation"})
    assert line == "→ op write_file"

    line = transport._render_tool_dispatch({"tool": "brave_search", "tool_kind": "mcp_tool"})
    assert line == "→ mcp brave_search"

    line = transport._render_tool_dispatch(
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "handoff",
            "handoff_to": "markdown",
        }
    )
    assert "↦ transition_stage → markdown" in line

    line = transport._render_tool_dispatch(
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "fork",
            "fork_children": 3,
        }
    )
    assert "↦ transition_stage fork (3 children)" in line


@pytest.mark.asyncio
async def test_tool_row_updates_in_place_on_completion() -> None:
    """tool_dispatch_complete updates the open tool row in place with its
    status glyph and duration instead of appending a second row.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "tool_dispatch",
        {"tool": "read_file", "tool_kind": "mcp_tool", "path": "hosts/cli/transport.py"},
    )
    tool_rows_before = [r for r in transport._rows if r.kind == "tool"]
    assert len(tool_rows_before) == 1
    assert "…" not in tool_rows_before[0].text  # no status yet

    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "read_file", "ok": True, "duration_ms": 400}
    )
    tool_rows_after = [r for r in transport._rows if r.kind == "tool"]
    # Still exactly one tool row — updated in place, not appended.
    assert len(tool_rows_after) == 1
    assert tool_rows_after[0] is tool_rows_before[0]
    assert "✓" in tool_rows_after[0].text
    assert "400ms" in tool_rows_after[0].text


@pytest.mark.asyncio
async def test_model_call_updates_in_place_on_completion() -> None:
    """model_call_complete updates the open model row in place with its
    status glyph and duration instead of appending a second row — the
    same in-place pattern as tool_dispatch_complete.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "haiku", "provider": "anthropic", "iteration": 1},
    )
    model_rows_before = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows_before) == 1
    assert "✓" not in model_rows_before[0].text  # no status yet

    await transport.on_stage(
        "model_call_complete",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1, "duration_ms": 340},
    )
    model_rows_after = [r for r in transport._rows if r.kind == "model"]
    # Still exactly one model row — updated in place, not appended.
    assert len(model_rows_after) == 1
    assert model_rows_after[0] is model_rows_before[0]
    assert "✓" in model_rows_after[0].text
    assert "340ms" in model_rows_after[0].text


@pytest.mark.asyncio
async def test_model_call_failed_marks_row_with_x() -> None:
    """model_call_failed updates the open model row in place with ✗."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1},
    )
    await transport.on_stage(
        "model_call_failed",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1},
    )
    model_rows = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows) == 1  # updated in place, not appended
    assert "✗" in model_rows[0].text


@pytest.mark.asyncio
async def test_fork_complete_keeps_per_child_table() -> None:
    """fork_complete keeps the per-child table with ok/failed totals in the header."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")

    await transport.on_stage("fork_start", {"children": 2})
    await transport.on_stage(
        "fork_child_start",
        {"child_index": 0, "mode_id": "coding", "stage_id": "implement", "query_preview": "write"},
    )
    await transport.on_stage(
        "fork_child_start",
        {"child_index": 1, "mode_id": "markdown", "stage_id": "", "query_preview": "doc"},
    )
    await transport.on_stage(
        "fork_child_complete", {"child_index": 0, "status": "ok", "duration_ms": 42}
    )
    await transport.on_stage(
        "fork_child_complete", {"child_index": 1, "status": "no_tools", "duration_ms": 30}
    )
    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})

    text = transport._render_fork_stage(summary=(1, 1))
    lines = text.splitlines()
    assert lines[0] == "stage: fork complete (1 ok, 1 failed)"
    assert "#0 coding · implement · ok · write · 42ms" in lines[1]
    assert "#1 markdown · no_tools · doc · 30ms" in lines[2]


@pytest.mark.asyncio
async def test_transcript_accumulates_play_by_play() -> None:
    """on_stage accumulates every event into _rows so the user sees the full
    turn trajectory, with the last row active.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")

    await transport.on_stage(
        "model_call", {"mode_id": "default", "model_id": "glm", "iteration": 1}
    )
    await transport.on_stage(
        "tool_dispatch", {"tool": "search_web", "tool_kind": "mcp_tool", "iteration": 1}
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "search_web", "ok": True, "duration_ms": 1200}
    )
    await transport.on_stage(
        "model_call", {"mode_id": "default", "model_id": "glm", "iteration": 2}
    )

    model_rows = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows) >= 2
    tool_text = "\n".join(r.text for r in transport._rows if r.kind == "tool")
    assert "search_web" in tool_text
    # The last row carries the active spinner.
    assert transport._rows[-1].active is True


@pytest.mark.asyncio
async def test_start_thinking_clears_transcript() -> None:
    """start_thinking resets the transcript so a new turn starts clean, seeded
    with the initial dispatch row.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert len(transport._rows) > 1

    transport.start_thinking("dispatch")
    # Fresh transcript: exactly the dispatch seed row.
    assert len(transport._rows) == 1
    assert transport._rows[0].text == "stage: dispatch"


@pytest.mark.asyncio
async def test_stop_thinking_clears_status_sentinel() -> None:
    """stop_thinking leaves _status and _live as None."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    assert transport._status is not None

    transport.stop_thinking()
    assert transport._status is None
    assert transport._live is None


# ── thinking segments + answer staging (on_text_chunk) ──────────────


def _make_stream_transport() -> RichTransport:
    """Build a RichTransport writing to a quiet StringIO console."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    return RichTransport(console=console)


@pytest.mark.asyncio
async def test_reasoning_becomes_interspersed_thinking_row() -> None:
    """Reasoning deltas append to an interspersed thinking row in the
    transcript; the Live region stays up.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    live = transport._live

    await transport.on_text_chunk("", reasoning="Let me think")
    await transport.on_text_chunk("", reasoning=" more")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 1
    assert thinking_rows[0].text == "Let me think more"
    assert transport._live is live
    assert transport._status is not None


@pytest.mark.asyncio
async def test_tool_call_opens_new_thinking_segment() -> None:
    """A tool call between two reasoning deltas closes the open segment so the
    next reasoning opens a fresh interspersed segment.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="first think")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    await transport.on_stage("tool_dispatch", {"tool": "search", "tool_kind": "mcp_tool"})
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "search", "ok": True, "duration_ms": 10}
    )
    await transport.on_text_chunk("", reasoning="second think")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 2
    assert thinking_rows[0].text == "first think"
    assert thinking_rows[1].text == "second think"


@pytest.mark.asyncio
async def test_visible_answer_staged_not_rendered_as_final() -> None:
    """The visible-answer delta is staged (for mid-turn error flush) and is
    never rendered into the transcript — on_response renders the clean Panel.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("draft answer")

    assert transport._answer_staging.plain == "draft answer"
    # No stream/thinking row carries the visible answer.
    assert not any(r.kind == "thinking" and "draft answer" in r.text for r in transport._rows)


@pytest.mark.asyncio
async def test_start_thinking_resets_staging_and_thinking() -> None:
    """start_thinking clears staging and the transcript so a new turn is clean."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="some thinking")
    await transport.on_text_chunk("some answer")
    assert transport._answer_staging != Text()
    assert any(r.kind == "thinking" for r in transport._rows)

    transport.start_thinking("model_call")
    assert transport._answer_staging == Text()
    assert not any(r.kind == "thinking" for r in transport._rows)


@pytest.mark.asyncio
async def test_on_response_always_renders_clean_panel() -> None:
    """on_response always renders the clean Markdown Panel from the full
    response text — even when the turn streamed — so the final answer is
    readable, not interspersed with the calls.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="thinking")
    await transport.on_text_chunk("draft answer")

    await transport.on_response("# Final Answer")

    assert transport._live is None
    assert transport._status is None
    assert transport._answer_staging == Text()
    assert "Final Answer" in transport.console.file.getvalue()
    # The Response panel (border + title) is present.
    assert "Response" in transport.console.file.getvalue()


# ── toggle no-leak between turns (Issue 3) ────────────────────────────


def test_toggle_thinking_between_turns_does_not_print_into_input_area() -> None:
    """Toggling thinking with the Live region down must NOT reprint the
    transcript; only the toolbar reflects the new state."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call")
    transport._rows.append(_Row(kind="thinking", text="a reasoning segment"))
    transport.stop_thinking()

    out_before = sink.getvalue()
    transport.toggle_thinking()  # collapsed → expanded
    out_after = sink.getvalue()

    assert out_after == out_before
    assert transport.thinking_collapsed is False


@pytest.mark.asyncio
async def test_toggle_thinking_during_turn_expands_inflight_rows() -> None:
    """With the Live region up (mid-turn), toggling thinking re-renders the
    in-flight thinking rows expanded so the user sees the streamed reasoning."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="reasoning body")

    thinking_row = [r for r in transport._rows if r.kind == "thinking"][0]
    assert thinking_row.thinking_expanded is False  # collapsed by default

    transport.toggle_thinking()
    assert thinking_row.thinking_expanded is True
    # Live region still active and refreshed.
    assert transport._live is not None


@pytest.mark.asyncio
async def test_on_response_renders_workflow_summary() -> None:
    """Successful completion renders response and workflow summaries, not a raw transcript."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="internal activity")
    await transport.on_stage(
        "model_call", {"mode_id": "default", "iteration": 1, "duration_ms": 500}
    )
    await transport.on_stage(
        "model_call_complete", {"mode_id": "default", "iteration": 1, "duration_ms": 500}
    )
    await transport.on_stage("tool_dispatch", {"tool": "search", "tool_kind": "mcp_tool"})
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "search", "ok": True, "duration_ms": 200}
    )

    out_before = transport.console.file.getvalue()
    await transport.on_response("answer")

    out_after = transport.console.file.getvalue()[len(out_before) :]
    assert "Response" in out_after
    assert "Workflow" in out_after
    assert "Completed" in out_after
    assert "turn 0 tok" in out_after
    assert "session 0 tok" in out_after
    assert "1 model call" in out_after
    assert "1 tool call" in out_after
    assert "1 succeeded, 0 failed" in out_after
    assert "ok search 200ms" in out_after
    assert "internal activity" not in out_after


@pytest.mark.asyncio
async def test_stop_thinking_does_not_flush_answer() -> None:
    """stop_thinking flushes the transcript rows but NOT the staged answer —
    the clean Panel is rendered by on_response, not the stream.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("the staged answer")

    transport.stop_thinking()
    assert "the staged answer" not in transport.console.file.getvalue()


@pytest.mark.asyncio
async def test_on_error_flushes_partial_answer() -> None:
    """On a mid-turn error, the partial staged answer flushes to scrollback so
    partial work isn't silently lost.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("partial answer")

    await transport.on_error("boom")

    assert "partial answer" in transport.console.file.getvalue()
    assert "Error" in transport.console.file.getvalue()


# ── thinking collapsed/expanded (Ctrl+Y toggle) ──────────────────────


@pytest.mark.asyncio
async def test_thinking_rows_collapsed_by_default() -> None:
    """Thinking rows default to a compact summary (collapsed)."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="a long wall of reasoning text")

    assert transport.thinking_collapsed is True
    row = next(r for r in transport._rows if r.kind == "thinking")
    line = transport._row_line(row)
    assert "Thinking…" in line.plain
    assert "a long wall of reasoning text" not in line.plain


@pytest.mark.asyncio
async def test_toggle_thinking_expands_and_collapses_all_segments() -> None:
    """toggle_thinking flips every thinking row's expand state at once
    (segments are interspersed; the toggle is global).
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="first think")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    await transport.on_text_chunk("", reasoning="second think")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 2
    assert all(not r.thinking_expanded for r in thinking_rows)

    transport.toggle_thinking()
    assert transport.thinking_collapsed is False
    assert all(r.thinking_expanded for r in thinking_rows)
    # Expanded rows show the full body.
    assert "first think" in transport._row_line(thinking_rows[0]).plain

    transport.toggle_thinking()
    assert transport.thinking_collapsed is True
    assert all(not r.thinking_expanded for r in thinking_rows)


# ── live height ───────────────────────────────────────────────────────


def test_live_height_floor_and_reserve() -> None:
    """Live height is terminal height minus the reserve, floored at the minimum."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.console.size = (80, 24)
    assert transport._live_height() == 24 - 4
    transport.console.size = (80, 5)
    assert transport._live_height() == 6


# ── live status header ───────────────────────────────────────────────


def test_status_header_renders_mode_model_caps() -> None:
    """The status header shows mode, model, and the always-on cap strip.

    The cap strip renders on every tier (not just power/debug): tools,
    iterations, and turn tokens against their caps. With no caps set, the
    compact ``N<count>`` form is used for each dimension.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", session_tokens=500)
    transport._current_mode = "coding"
    transport._status_current_model = "haiku"
    transport._status_model_provider = "anthropic"
    transport._status_turn_tokens = 120
    transport._status_session_tokens = 620

    text = transport._render_status_header()
    plain = text.plain
    assert "mode coding" in plain
    assert "anthropic:haiku" in plain
    # Both token totals are explicit while work is in flight.
    assert "models 0" in plain
    assert "turn 120 tok" in plain
    assert "session 620 tok" in plain
    assert "N0 tools" in plain
    assert "N0 iter" in plain


def test_status_header_renders_caps_with_limits() -> None:
    """The cap strip shows count/cap ratios when caps are set (ask 3)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", session_tokens=19500)
    transport._current_mode = "coding"
    transport._status_current_model = "haiku"
    transport._max_tool_calls = 8
    transport._tool_call_count = 4
    transport._max_iterations = 5
    transport._current_iter = 2
    transport._status_turn_tokens = 12000
    plain = transport._render_status_header().plain
    assert "4/8 tools" in plain
    assert "2/5 iter" in plain
    assert "turn 12000 tok" in plain
    assert "session 19500 tok" in plain


def test_status_header_session_seeded_from_baseline() -> None:
    """start_thinking(session_tokens=...) seeds the cumulative session total."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", session_tokens=1234)
    assert transport._status_session_tokens == 1234
    assert transport._status_turn_tokens == 0
    assert transport._status_current_model == ""


@pytest.mark.asyncio
async def test_model_call_updates_live_status_header() -> None:
    """model_call sets the active model; model_call_complete adds token totals."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("model_call", session_tokens=100)

    await transport.on_stage(
        "model_call",
        {"mode_id": "coding", "model_id": "haiku", "provider": "anthropic", "iteration": 1},
    )
    assert transport._status_current_model == "haiku"
    assert transport._status_model_provider == "anthropic"
    assert transport._status_turn_tokens == 0
    assert transport._status_session_tokens == 100

    await transport.on_stage(
        "model_call_complete",
        {"model_id": "haiku", "iteration": 1, "_state.total_tokens": 120, "duration_ms": 340},
    )
    assert transport._status_turn_tokens == 120
    assert transport._status_session_tokens == 220
    header = transport._render_status_header().plain
    assert "models 1" in header
    assert "turn 120 tok" in header
    assert "session 220 tok" in header
    model_row = next(row for row in transport._rows if row.kind == "model")
    assert "120 tok" in model_row.text


def test_aggregate_progress_blends_caps() -> None:
    """_aggregate_progress returns the max of consumed-cap ratios (ask 2)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    # No caps set -> None (bar suppressed).
    assert transport._aggregate_progress() is None
    # Tool cap only.
    transport._max_tool_calls = 8
    transport._tool_call_count = 4
    assert transport._aggregate_progress() == 0.5
    # Iteration cap is the binding constraint (higher ratio).
    transport._max_iterations = 5
    transport._current_iter = 4
    assert transport._aggregate_progress() == 0.8
    # Handoff cap dominates.
    transport._max_handoffs = 3
    transport._handoff_count = 3
    assert transport._aggregate_progress() == 1.0


def test_progress_bar_gradient_fill() -> None:
    """_progress_bar fills left-to-right using the block ramp (ask 2)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    assert transport._progress_bar(0.0) == "        "  # 8 empty cells
    assert transport._progress_bar(1.0) == "████████"  # 8 full cells
    half = transport._progress_bar(0.5)
    assert half.startswith("████")
    assert half.endswith("    ")


def test_prompt_banner_pinned_when_message_supplied() -> None:
    """The prompt banner renders the submitted message (ask 1)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", message="restart the webapp")
    banner = transport._render_prompt_banner()
    assert "restart the webapp" in banner.plain
    assert banner.plain.startswith("❯")


def test_prompt_banner_empty_when_no_message() -> None:
    """No banner when start_thinking is called without a message (ask 1)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch")
    banner = transport._render_prompt_banner()
    assert banner.plain == ""


def test_live_display_includes_prompt_banner_and_header() -> None:
    """The Live display leads with the prompt banner then the status header (ask 1)."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", session_tokens=10, message="hello")
    group = transport._render_live_display()
    # Group.renderables[0] is the banner, [1] is the status header.
    renderables = list(group.renderables)
    assert "hello" in renderables[0].plain
    assert "mode" not in renderables[0].plain  # banner is just the prompt


@pytest.mark.asyncio
async def test_live_display_includes_status_header() -> None:
    """The Live display leads with the status header renderable."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console)
    transport.start_thinking("dispatch", session_tokens=10)
    transport._current_mode = "default"

    display = transport._render_live_display()
    renderables = list(display.renderables)
    assert len(renderables) >= 2
    header = renderables[0]
    assert "mode default" in header.plain
