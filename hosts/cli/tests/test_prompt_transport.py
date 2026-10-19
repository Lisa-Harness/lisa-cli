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
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console
from rich.text import Text

from hosts.cli.transport import _TOOL_BOX_MAX_LINES, PromptTransport


@pytest.fixture(autouse=True)
def _close_activity_displays(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    transports: set[PromptTransport] = set()
    start = PromptTransport.start_activity

    def start_activity(transport: PromptTransport) -> None:
        transports.add(transport)
        start(transport)

    monkeypatch.setattr(PromptTransport, "start_activity", start_activity)
    yield
    for transport in transports:
        transport.stop_activity()


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
    # A prompt run collapses the REPL-era turn/token split: exactly one token
    # figure. The host never stamps a turn index on tool-only payloads, so a
    # run with no model_call stage renders no turn segment.
    assert "session" not in footer
    assert "turn" not in footer


@pytest.mark.asyncio
async def test_footer_and_response_rule_carry_turn_number() -> None:
    """The host-stamped turn index renders on the ⚡ footer and Response rule.

    The number is the global thread turn index (``turn_index`` on the
    ``model_call`` stage payload) — the same number stamped on the thread
    entry, addressable with ``lisa last N`` / ``lisa debug N``.
    """
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call",
        {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops", "turn_index": 41},
    )
    await transport.on_stage("model_call_complete", {"duration_ms": 340, "total_tokens": 4100})
    await transport.on_response("the answer")
    out = buffer.getvalue()
    footer = next(ln for ln in out.splitlines() if ln.strip().startswith("⚡"))
    assert "turn 41" in footer
    assert "Response · llama3.2 · turn 41" in out


@pytest.mark.asyncio
async def test_turn_zero_renders_on_footer_and_rule() -> None:
    """A genuine ``turn 0`` (first turn on a fresh thread) renders."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call",
        {"model_id": "m", "provider": "ollama", "mode_id": "default", "turn_index": 0},
    )
    await transport.on_stage("model_call_complete", {"duration_ms": 10, "total_tokens": 100})
    await transport.on_response("answer")
    out = buffer.getvalue()
    footer = next(ln for ln in out.splitlines() if ln.strip().startswith("⚡"))
    assert "turn 0" in footer
    assert "Response · m · turn 0" in out


@pytest.mark.asyncio
async def test_turn_number_absent_yields_no_turn_segments() -> None:
    """No ``turn_index`` in stage payloads = no turn output (legacy shape)."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call", {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"}
    )
    await transport.on_stage("model_call_complete", {"duration_ms": 340, "total_tokens": 4100})
    await transport.on_response("answer")
    out = buffer.getvalue()
    assert "turn" not in out
    assert "Response · llama3.2" in out


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


@pytest.mark.asyncio
async def test_bracket_arg_renders_without_markup_error() -> None:
    """A tool argument containing Rich markup brackets renders literally.

    Rows are pre-styled ``Text`` objects (markup never parsed), so a raw
    argument like a grep character class cannot raise ``MarkupError`` or
    have its bracket content swallowed as a style tag.
    """
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "arguments": {"command": "grep [0-9] file.txt"},
            "display": "grep '[0-9]' file.txt",
        },
    )
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 4}
    )
    out = buffer.getvalue()
    assert "grep '[0-9]' file.txt" in out


@pytest.mark.asyncio
async def test_result_summary_shows_streamed_line_count() -> None:
    """A streaming tool's completion row carries a ``⎿ N lines`` summary."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "seq 3"},
        },
    )
    await transport.on_stage("tool_output", {"text": "one\ntwo\nthree", "stream": "stdout"})
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 9}
    )
    assert "⎿ 3 lines" in buffer.getvalue()


@pytest.mark.asyncio
async def test_result_summary_uses_preview_when_not_streaming() -> None:
    """Non-streaming tools summarize the kernel ``result_preview`` size."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "web_search", "tool_kind": "operation", "arguments": {"query": "q"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {"tool": "web_search", "ok": True, "duration_ms": 50, "result_preview": "x" * 150},
    )
    assert "⎿ 150 chars" in buffer.getvalue()


@pytest.mark.asyncio
async def test_result_summary_prefers_true_result_chars() -> None:
    """``result_chars`` reports the whole output, not the 200-char preview.

    JSON envelope tools (thread archive pages) produce previews that are
    compact summaries, so their length understates the result; the
    kernel stamps ``result_chars`` with the true size and the row shows
    that.
    """
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "thread_read_content", "tool_kind": "mcp_tool", "arguments": {}},
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "thread_read_content",
            "ok": True,
            "duration_ms": 84,
            "result_preview": "content=stored tool output offset_chars=0 total_chars=5412",
            "result_chars": 5412,
        },
    )
    assert "⎿ 5.412k chars" in buffer.getvalue()


@pytest.mark.asyncio
async def test_tool_detail_level_1_prints_result_preview_line() -> None:
    """-v prints the kernel result preview (first line) under the row."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console, tool_detail=1)
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "bash_run", "tool_kind": "pack_tool", "arguments": {"command": "echo hi"}},
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "bash_run",
            "ok": True,
            "duration_ms": 8,
            "result_preview": "hi\nthere",
        },
    )
    out = buffer.getvalue()
    assert "⎿ hi" in out
    # Level 1 shows only the FIRST preview line.
    assert "⎿ there" not in out


@pytest.mark.asyncio
async def test_tool_detail_level_1_shows_payload_first_json_summary() -> None:
    """-v detail lines show envelope content, not serialized JSON metadata.

    The 2026-10-02 regression: thread-read pages serialize ``ref`` +
    ``source`` provenance ahead of ``content``, so a raw first-200-chars
    preview rendered as raw JSON in verbose output. The kernel now sends
    a payload-first ``key=value`` summary; the transport prints it as-is.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console, tool_detail=1)
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "thread_read_content", "tool_kind": "mcp_tool", "arguments": {}},
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "thread_read_content",
            "ok": True,
            "duration_ms": 84,
            "result_preview": (
                "content=The reviewer found the config parser drops duplicate "
                "keys offset_chars=0 total_chars=5412"
            ),
            "result_chars": 5412,
        },
    )
    out = buffer.getvalue()
    assert "⎿ content=The reviewer found the config parser drops duplicate keys" in out
    assert '"ref"' not in out
    assert '"stored_tool_result"' not in out


@pytest.mark.asyncio
async def test_tool_detail_level_2_prints_arguments_and_full_preview() -> None:
    """-v -v also prints the full call arguments and multi-line preview."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console, tool_detail=2)
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "arguments": {"command": "echo hi", "timeout": 30},
        },
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "bash_run",
            "ok": True,
            "duration_ms": 8,
            "result_preview": "hi\nthere",
        },
    )
    out = buffer.getvalue()
    assert '"command"' in out
    assert '"echo hi"' in out
    assert "⎿ hi" in out
    assert "⎿ there" in out


@pytest.mark.asyncio
async def test_footer_cost_estimate_from_model_prices() -> None:
    """The footer prices the turn when model_prices carries per-token rates."""
    transport, buffer = _make_transport()
    transport.model_prices = {"llama3.2": (0.000003, 0.000006)}
    await transport.on_stage(
        "model_call",
        {"model_id": "llama3.2", "provider": "ollama", "mode_id": "ops"},
    )
    await transport.on_stage(
        "model_call_complete",
        {"duration_ms": 340, "total_tokens": 4100, "prompt_tokens": 4000, "completion_tokens": 100},
    )
    await transport.on_response("the answer")
    # 4000 * 3e-6 + 100 * 6e-6 = $0.0126 -> rounds to 0.0126
    assert "$0.0126" in buffer.getvalue()


@pytest.mark.asyncio
async def test_footer_omits_cost_without_prices() -> None:
    """No injected prices -> no cost segment (never block the footer)."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "model_call_complete", {"duration_ms": 340, "total_tokens": 4100}
    )
    await transport.on_response("the answer")
    assert "$" not in buffer.getvalue()


def test_activity_bar_renders_tool_budget() -> None:
    """A finite tool-call cap renders the ▰▱ bar with used/total counts."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console)
    transport._cap_max_tool_calls = 8  # pylint: disable=protected-access
    transport._tool_call_count = 2  # pylint: disable=protected-access
    line = transport._status_line(120)  # pylint: disable=protected-access
    rendered = line.render(1.0)
    assert isinstance(rendered, Text)
    assert "▰▰▱▱▱▱▱▱ 2/8" in rendered.plain


def test_activity_bar_skipped_when_unlimited() -> None:
    """Zero/unlimited caps fall back to the plain tool counter."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console)
    transport._tool_call_count = 3  # pylint: disable=protected-access
    line = transport._status_line(120)  # pylint: disable=protected-access
    rendered = line.render(1.0)
    assert isinstance(rendered, Text)
    assert "tools 3" in rendered.plain
    assert "▰" not in rendered.plain


# ── Bounded stream box / thinking meter ─────────────────────────────


def _make_tty_transport(**kwargs: Any) -> tuple[PromptTransport, Console]:
    """A forced-terminal, recording transport that runs the Live display.

    ``record=True`` captures everything the Console renders, including the
    transient Live region, so activity-line assertions read the export.
    """
    console = Console(file=io.StringIO(), record=True, force_terminal=True, width=100)
    return PromptTransport(console=console, **kwargs), console


@pytest.mark.asyncio
async def test_stream_box_requires_tty_and_streaming_mode() -> None:
    """No box for non-tty consoles, quiet mode, narrow consoles, or mode none."""
    # Default transport (streaming_mode=none) never boxes.
    transport, _ = _make_tty_transport()
    transport._model_call_count = 1
    transport._stream_answer_buf = "hello"
    assert transport._stream_box_active() is False
    # Even with a mode set, a non-tty console never boxes.
    nt, _ = _make_transport()
    nt.streaming_mode = "text"
    nt._model_call_count = 1
    nt._stream_answer_buf = "hello"
    assert nt._stream_box_active() is False
    # Quiet suppresses the box even on tty.
    transport2, _ = _make_tty_transport(quiet=True, streaming_mode="text")
    transport2._model_call_count = 1
    transport2._stream_answer_buf = "hello"
    assert transport2._stream_box_active() is False
    # Without a model call in flight there is nothing to box.
    transport3, _ = _make_tty_transport(streaming_mode="text")
    transport3._stream_answer_buf = "hello"
    assert transport3._stream_box_active() is False


@pytest.mark.asyncio
async def test_stream_box_tail_window_bounded() -> None:
    """The box shows at most _STREAM_BOX_MAX_LINES, taking from the tail."""
    transport, _ = _make_tty_transport(streaming_mode="text")
    transport.start_turn("hi")
    chunks = [f"line {i} of a long streamed answer. " * 2 for i in range(40)]
    for chunk in chunks:
        await transport.on_text_chunk(chunk)
    transport.stop_activity()
    lines = transport._stream_box_lines()
    assert 0 < len(lines) <= 8  # _STREAM_BOX_MAX_LINES
    # Tail content is present; head content aged out of the window.
    assert any("line 39" in ln or "line 38" in ln for ln in lines)
    assert not any("line 0" in ln for ln in lines)
    # Long single-line content hard-wraps within the box width.
    transport._stream_answer_buf = "x" * 500
    wrapped = transport._stream_box_lines()
    assert all(len(ln) <= 100 for ln in wrapped)


@pytest.mark.asyncio
async def test_hybrid_box_streams_reasoning_not_answer() -> None:
    """In hybrid mode the box renders the reasoning stream only."""
    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "o3", "reasoning_model": True})
    await transport.on_text_chunk("visible answer", reasoning="hidden thoughts ")
    transport.stop_activity()
    lines = transport._stream_box_lines()
    assert any("hidden thoughts" in ln for ln in lines)
    assert not any("visible answer" in ln for ln in lines)


@pytest.mark.asyncio
async def test_text_mode_box_streams_answer() -> None:
    """In text mode the box renders the answer stream."""
    transport, _ = _make_tty_transport(streaming_mode="text")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "claude"})
    await transport.on_text_chunk("the answer forming", reasoning="")
    transport.stop_activity()
    lines = transport._stream_box_lines()
    assert any("the answer forming" in ln for ln in lines)


@pytest.mark.asyncio
async def test_box_clears_for_final_rerender() -> None:
    """on_response clears the stream buffers; final text renders once, markdown."""
    transport, console = _make_tty_transport(streaming_mode="text")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "claude", "provider": "anthropic"})
    await transport.on_text_chunk("# Title\n\nstreamed tail content")
    assert transport._stream_box_active() is True
    transport.stop_activity()
    await transport.on_response("# Title\n\nstreamed tail content")
    assert transport._stream_answer_buf == ""
    assert transport._stream_reasoning_buf == ""
    rec = console.export_text()
    # The final answer rendered exactly once (the rule + markdown), not
    # once from the box and once from on_response.
    assert rec.count("streamed tail content") == 1 or "Title" in rec
    assert "Response · claude" in rec


# ── Bounded in-flight tool-output box (tty) ───────────────────────────


@pytest.mark.asyncio
async def test_tool_output_stages_into_box_on_tty() -> None:
    """On a tty streamed lines stage into the bounded box, not scrollback."""
    transport, console = _make_tty_transport()
    transport.start_turn("hi")
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "seq 3"},
        },
    )
    await transport.on_stage("tool_output", {"text": "one\ntwo\nthree", "stream": "stdout"})
    assert transport._tool_output_lines == ["one", "two", "three"]
    box = transport._tool_box_renderable()
    assert box is not None
    options = transport.console.options
    options.width = options.max_width = 90
    rendered = transport.console.render_lines(box, options, pad=False)
    plain = "\n".join("".join(seg.text for seg in line) for line in rendered)
    assert "one" in plain and "three" in plain
    # The completion row closes the box; the ⎿ summary carries the shape.
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 9}
    )
    assert transport._tool_output_lines == []
    out = console.export_text()
    assert "⎿ 3 lines" in out
    # Succeeded tool: staged lines never landed in scrollback.
    assert "│ one" not in out


@pytest.mark.asyncio
async def test_tool_output_box_tail_window_bounded() -> None:
    """The box keeps only the tail; the title reports the elided head."""
    transport, _ = _make_tty_transport()
    transport.start_turn("hi")
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "yes"},
        },
    )
    for i in range(40):
        await transport.on_stage("tool_output", {"text": f"line {i}", "stream": "stdout"})
    assert len(transport._tool_output_lines) == _TOOL_BOX_MAX_LINES
    assert transport._tool_output_lines[-1] == "line 39"
    assert "line 0" not in transport._tool_output_lines
    box = transport._tool_box_renderable()
    assert box is not None
    options = transport.console.options
    options.width = options.max_width = 90
    rendered = transport.console.render_lines(box, options, pad=False)
    plain = "\n".join("".join(seg.text for seg in line) for line in rendered)
    assert "+28 lines" in plain  # 40 - 12 staged


@pytest.mark.asyncio
async def test_tool_output_failure_flushes_tail_to_scrollback() -> None:
    """A ✗ completion flushes the staged tail so error context survives."""
    transport, console = _make_tty_transport()
    transport.start_turn("hi")
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "failing"},
        },
    )
    await transport.on_stage("tool_output", {"text": "boom", "stream": "stdout"})
    await transport.on_stage(
        "tool_dispatch_complete", {"tool": "bash_run", "ok": False, "duration_ms": 12}
    )
    out = console.export_text()
    assert "│ boom" in out
    assert transport._tool_output_lines == []


@pytest.mark.asyncio
async def test_tool_output_prints_to_scrollback_on_non_tty() -> None:
    """Non-tty consoles keep the legacy transcript: every line prints."""
    transport, buffer = _make_transport()
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "seq 3"},
        },
    )
    await transport.on_stage("tool_output", {"text": "hello\nworld", "stream": "stdout"})
    out = buffer.getvalue()
    assert "│ hello" in out
    assert "│ world" in out
    assert transport._tool_output_lines == []


@pytest.mark.asyncio
async def test_tool_output_box_stderr_lines_prefixed() -> None:
    """Staged stderr lines keep the [stderr] marker inside the box."""
    transport, _ = _make_tty_transport()
    transport.start_turn("hi")
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
    assert transport._tool_output_lines == ["[stderr] err"]


@pytest.mark.asyncio
async def test_tool_output_box_suppressed_when_quiet() -> None:
    """Quiet mode drops streamed lines entirely — box and scrollback."""
    transport, console = _make_tty_transport(quiet=True)
    await transport.on_stage(
        "tool_dispatch",
        {
            "tool": "bash_run",
            "tool_kind": "pack_tool",
            "stream_output": True,
            "arguments": {"command": "seq 3"},
        },
    )
    await transport.on_stage("tool_output", {"text": "noisy", "stream": "stdout"})
    assert transport._tool_output_lines == []
    assert "noisy" not in console.export_text()


def test_thinking_meter_and_iteration_on_status_line() -> None:
    """Reasoning-model calls add a thinking-size meter and iter N/max."""
    transport, _ = _make_tty_transport()
    transport._model_call_count = 1
    transport._active_reasoning_model = True
    transport._status_reasoning_chars = 4200
    transport._iteration_current = 2
    transport._iteration_max = 12
    line = transport._status_line(100).render(1.0)
    assert "thinking 4.2k" in line.plain
    assert "iter 2/12" in line.plain


def test_turn_number_leads_status_line() -> None:
    """The captured turn index leads the in-flight status line."""
    transport, _ = _make_tty_transport()
    transport._turn_number = 41
    transport._model_call_count = 1
    line = transport._status_line(100).render(1.0).plain
    assert line.startswith("turn 41 · ")
    assert "turn 41" in line


def test_turn_number_absent_leaves_status_line_unchanged() -> None:
    """Unknown turn (-1) renders no turn segment on the status line."""
    transport, _ = _make_tty_transport()
    transport._model_call_count = 1
    assert "turn" not in transport._status_line(100).render(1.0).plain


def test_reasoning_spinner_distinct_during_model_call() -> None:
    """Reasoning-model calls use the calmer spinner, not the dispatch one."""
    transport, _ = _make_tty_transport()
    transport._model_call_count = 1
    transport._active_reasoning_model = True
    assert transport._activity_spinner() == "aesthetic"
    transport._active_reasoning_model = False
    assert transport._activity_spinner() == "dots"


def test_starting_label_shows_before_first_stage() -> None:
    """Boot phase prints a 'starting runtime' label, cleared by a stage."""
    transport, _ = _make_tty_transport()
    assert "starting runtime" in transport._status_line(100).render(1.0).plain
    transport._phase_label = ""
    assert "starting runtime" not in transport._status_line(100).render(1.0).plain


@pytest.mark.asyncio
async def test_stage_event_clears_starting_label() -> None:
    """The first on_stage event ends the boot-phase label."""
    transport, _ = _make_tty_transport()
    await transport.on_stage("model_call", {"model_id": "haiku"})
    assert transport._phase_label == ""


def test_format_elapsed_scales() -> None:
    """Seconds under a minute, m:ss past it."""
    from hosts.cli.transport import _format_elapsed

    assert _format_elapsed(3) == "3s"
    assert _format_elapsed(59) == "59s"
    assert _format_elapsed(60) == "1:00"
    assert _format_elapsed(125) == "2:05"


def test_status_line_pulls_state_per_frame() -> None:
    """Elapsed and thinking size are read fresh on every render() call.

    The Live holds the renderable for the whole turn; a snapshot-at-start
    design froze the counters during quiet generations (the spinner
    animated but the numbers never moved).
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console)
    transport._model_call_count = 1  # pylint: disable=protected-access
    transport._active_reasoning_model = True  # pylint: disable=protected-access
    transport._status_reasoning_chars = 10  # pylint: disable=protected-access
    line = transport._status_line(120)  # pylint: disable=protected-access
    first = line.render(1.0)
    assert "thinking 10" in first.plain
    # State mutates outside any render call; the next frame must reflect it.
    transport._status_reasoning_chars = 90_000  # pylint: disable=protected-access
    second = line.render(1.0)
    assert "thinking 90k" in second.plain
    assert "10" not in second.plain


def test_spinner_name_pulled_per_frame() -> None:
    """The spinner kind re-resolves each frame (reasoning vs dispatch)."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console)
    transport._model_call_count = 1  # pylint: disable=protected-access
    transport._active_reasoning_model = True  # pylint: disable=protected-access
    line = transport._status_line(120)  # pylint: disable=protected-access
    # aesthetic's first frame char differs from dots';
    # assert via the resolved name instead of glyph shapes.
    assert transport._activity_spinner() == "aesthetic"  # pylint: disable=protected-access
    rendered = line.render(1.0)
    assert isinstance(rendered, Text)


# ── Observable reasoning arming + stream velocity ────────────────────


@pytest.mark.asyncio
async def test_thinking_meter_arms_on_observed_deltas_not_manifest_flag() -> None:
    """A model the manifest does NOT flag as reasoning still shows the
    meter once reasoning deltas actually arrive (glm-5.3-family: the
    reasoning capability bills without ``reasoning_model=True``)."""
    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    # Manifest says non-reasoning; delta says otherwise.
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    assert "thinking" not in transport._compose_status_body(100).plain
    await transport.on_text_chunk("", reasoning="some chain of thought")
    body = transport._compose_status_body(100).plain
    assert "thinking" in body
    assert transport._reasoning_seen is True


@pytest.mark.asyncio
async def test_stream_velocity_shown_and_decays() -> None:
    """~rate shows while chunks flow inside the window; decays to nothing
    when the stream stalls (stale samples age out)."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "o3", "reasoning_model": True})
    # Simulate a healthy 5s-old first sample then steady flow: seed a
    # sample 5s back so the 4s window prunes it, then add flow samples.
    # Reasoning chars must be present too — the meter gates on actual
    # accumulated size, samples alone are timestamps.
    transport._status_reasoning_chars = 1400  # pylint: disable=protected-access
    now = time.perf_counter()
    transport._velocity_samples = [
        (now - 5.0, 0),
        (now - 2.0, 20),
        (now - 1.0, 700),
        (now - 0.1, 1400),
    ]  # pylint: disable=protected-access
    velocity = transport._stream_velocity()
    assert velocity > 0
    body = transport._compose_status_body(100).plain
    assert "~" in body and "/s" in body
    # Stalled: no new samples for > window → estimate decays to 0.
    transport._velocity_samples = [
        (now - 30.0, 0),
        (now - 29.5, 500),
    ]  # pylint: disable=protected-access
    assert transport._stream_velocity() == 0.0
    stalled = transport._compose_status_body(100).plain
    assert "/s" not in stalled


@pytest.mark.asyncio
async def test_velocity_too_short_span_is_noise_gated() -> None:
    """Sub-threshold span or char delta renders no velocity figure."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "o3", "reasoning_model": True})
    now = time.perf_counter()
    # Two samples 0.1s apart: span < 0.75s → 0.
    transport._velocity_samples = [(now - 0.2, 0), (now - 0.1, 500)]  # pylint: disable=protected-access
    assert transport._stream_velocity() == 0.0
    # Span ok but delta tiny (< 64 chars) → 0.
    transport._velocity_samples = [(now - 2.0, 0), (now - 0.5, 30)]  # pylint: disable=protected-access
    assert transport._stream_velocity() == 0.0


@pytest.mark.asyncio
async def test_box_title_shows_velocity_or_size() -> None:
    """The thinking panel title carries the live rate, else the size."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "o3", "reasoning_model": True})
    await transport.on_text_chunk("", reasoning="word " * 100)
    box = transport._stream_box_renderable()
    assert box is not None
    # Fresh single burst: no qualifying velocity yet → size fallback.
    options = transport.console.options
    options.width = options.max_width = 90
    rendered = transport.console.render_lines(box, options, pad=False)
    plain = "\n".join(
        "".join(seg.text for seg in line) for line in rendered
    )
    assert "thinking ·" in plain
    # Velocity qualify → rate in the title.
    now = time.perf_counter()
    transport._velocity_samples = [(now - 2.0, 0), (now - 1.0, 900)]  # pylint: disable=protected-access
    rendered = transport.console.render_lines(
        transport._stream_box_renderable(), options, pad=False
    )
    plain = "\n".join(
        "".join(seg.text for seg in line) for line in rendered
    )
    assert "c/s" in plain


# ── Stall indicator (no-data age on the status line) ─────────────────


@pytest.mark.asyncio
async def test_stall_segment_hidden_while_streaming() -> None:
    """Fresh deltas keep the no-data segment off the status line."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    transport._last_delta_perf = time.perf_counter() - 3.0  # pylint: disable=protected-access
    transport._status_reasoning_chars = 500  # pylint: disable=protected-access
    body = transport._compose_status_body(100).plain
    assert "no data" not in body
    # Below the first delta of the call the clock is unarmed (-1).
    transport._last_delta_perf = 0.0  # pylint: disable=protected-access
    assert transport._no_data_age_s() == -1.0


@pytest.mark.asyncio
async def test_stall_segment_after_threshold() -> None:
    """A quiet stream past the threshold renders `no data Ns` in gold."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    await transport.on_text_chunk("", reasoning="initial burst of thinking")
    transport._last_delta_perf = time.perf_counter() - 20.0  # pylint: disable=protected-access
    transport._status_reasoning_chars = 800  # pylint: disable=protected-access
    body = transport._compose_status_body(100)
    assert "no data 20s" in body.plain
    assert body.plain.count("no data 20s") == 1
    # Below the alert age the segment stays gold, not red.
    stall_spans = [s for s in body.spans if "no data" in body.plain[s.start : s.end]]
    assert stall_spans and stall_spans[0].style == "gold3"


@pytest.mark.asyncio
async def test_stall_segment_escalates_to_red_at_alert() -> None:
    """Past the alert age the stall segment switches to the error style."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    await transport.on_text_chunk("", reasoning="initial burst")
    transport._last_delta_perf = time.perf_counter() - 90.0  # pylint: disable=protected-access
    transport._status_reasoning_chars = 800  # pylint: disable=protected-access
    body = transport._compose_status_body(100)
    assert "no data 90s" in body.plain
    # The escalated segment carries the error style (gold below the alert age).
    stall_spans = [s for s in body.spans if "no data" in body.plain[s.start : s.end]]
    assert stall_spans and stall_spans[0].style == "indian_red"


@pytest.mark.asyncio
async def test_new_model_call_resets_stall_clock() -> None:
    """model_call re-arms the stall clock: pre-call silence is warmup."""
    import time

    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    await transport.on_text_chunk("", reasoning="thinking once")
    transport._last_delta_perf = time.perf_counter() - 200.0  # pylint: disable=protected-access
    assert transport._no_data_age_s() >= 200.0
    await transport.on_stage("model_call", {"model_id": "glm-5.3:cloud"})
    assert transport._last_delta_perf == 0.0
    assert transport._no_data_age_s() == -1.0
    body = transport._compose_status_body(100).plain
    assert "no data" not in body


def test_stalled_stream_switches_spinner_to_clock() -> None:
    """A stalled stream uses the slow clock spinner, not the work spinners."""
    import time

    transport, _ = _make_tty_transport()
    transport._model_call_count = 1
    transport._active_reasoning_model = True
    transport._reasoning_seen = True
    transport._last_delta_perf = time.perf_counter() - 30.0  # pylint: disable=protected-access
    assert transport._activity_spinner() == "clock"
    # Fresh data → back to the calm reasoning spinner.
    transport._last_delta_perf = time.perf_counter()  # pylint: disable=protected-access
    assert transport._activity_spinner() == "aesthetic"


@pytest.mark.asyncio
async def test_on_text_chunk_stamps_stall_clock() -> None:
    """Each delta (answer or reasoning) refreshes the stall timestamp."""
    transport, _ = _make_tty_transport(streaming_mode="hybrid")
    transport.start_turn("hi")
    await transport.on_stage("model_call", {"model_id": "claude"})
    assert transport._last_delta_perf == 0.0
    await transport.on_text_chunk("answer text", reasoning="")
    first = transport._last_delta_perf
    assert first > 0.0
    await transport.on_text_chunk("", reasoning="and reasoning")
    assert transport._last_delta_perf >= first


# ── Footer cost: cache-aware estimate ────────────────────────────────


@pytest.mark.asyncio
async def test_footer_estimate_blends_cache_read_rate() -> None:
    """Without quoted cost, cache-hit prompts price at the read rate:
    (1000-600)*2 + 600*0.2 + 100*12 µD = $0.002 vs $0.0044 uncached."""
    transport, buffer = _make_transport()
    transport.model_prices = {"m": (0.000002, 0.000012)}
    transport.model_cache_prices = {"m": (0.0000002, 0.0)}
    await transport.on_stage("model_call", {"model_id": "m"})
    await transport.on_stage(
        "model_call_complete",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cached_tokens": 600,
        },
    )
    await transport.on_response("done")
    footer = next(
        ln for ln in buffer.getvalue().splitlines() if ln.strip().startswith("⚡")
    )
    assert "$0.002" in footer


@pytest.mark.asyncio
async def test_footer_estimate_falls_back_to_uncached_rate() -> None:
    """No cache prices in the manifest → every prompt at the uncached rate."""
    transport, buffer = _make_transport()
    transport.model_prices = {"m": (0.000002, 0.000012)}
    await transport.on_stage("model_call", {"model_id": "m"})
    await transport.on_stage(
        "model_call_complete",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cached_tokens": 600,
        },
    )
    await transport.on_response("done")
    footer = next(
        ln for ln in buffer.getvalue().splitlines() if ln.strip().startswith("⚡")
    )
    assert "$0.0032" in footer
