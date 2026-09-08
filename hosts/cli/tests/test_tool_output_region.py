"""Tests for the streaming tool-output live region (``tool_output`` events).

The transport renders a scrolling output region beneath the in-flight tool
row for tools whose ``.tool.yml`` declares ``stream_output``; lines attach to
the open tool row, the region keeps only the most recent tail, and a
completed streaming call collapses to a one-line summary.
"""

from __future__ import annotations

import asyncio
import io

from hosts.cli.transport import _OUTPUT_REGION_MAX_LINES, RichTransport, _Row
from rich.console import Console


def _make_transport() -> RichTransport:
    """Build a RichTransport writing to a quiet StringIO console."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    return RichTransport(console=console)


def _drive(coro_fn) -> None:
    asyncio.run(coro_fn())


class TestToolOutputRouting:
    """``tool_output`` stage events land on the open tool row."""

    def test_lines_attach_to_open_tool_row(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage(
                "tool_dispatch",
                {"tool": "bash_run", "tool_kind": "pack_tool", "stream_output": True},
            )
            await transport.on_stage("tool_output", {"text": "compiling a.c", "stream": "stdout"})
            await transport.on_stage("tool_output", {"text": "compiling b.c", "stream": "stdout"})

        _drive(drive)
        assert len(transport._rows) == 2
        row = transport._rows[1]
        assert row.kind == "tool"
        assert list(row.output_lines) == ["compiling a.c", "compiling b.c"]
        assert row.output_total == 2
        assert transport._tool_output_streaming is True

    def test_multiline_text_splits_into_lines(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "one\ntwo\nthree\n"})

        _drive(drive)
        row = transport._rows[1]
        assert list(row.output_lines) == ["one", "two", "three"]

    def test_tail_scrolls_beyond_cap(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            for i in range(_OUTPUT_REGION_MAX_LINES + 6):
                await transport.on_stage("tool_output", {"text": f"line-{i}"})

        _drive(drive)
        row = transport._rows[1]
        assert row.output_total == _OUTPUT_REGION_MAX_LINES + 6
        assert len(row.output_lines) == _OUTPUT_REGION_MAX_LINES
        assert row.output_lines[-1] == f"line-{_OUTPUT_REGION_MAX_LINES + 5}"
        assert row.output_lines[0] == f"line-{6}"  # oldest retained line

    def test_no_tool_row_drops_line(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            # Output arrives with no tool row open at all.
            await transport.on_stage("tool_output", {"text": "orphan line"})

        _drive(drive)
        assert transport._rows  # dispatch seed row only
        assert all(r.kind != "tool" for r in transport._rows)

    def test_empty_text_ignored(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": ""})

        _drive(drive)
        row = transport._rows[1]
        assert row.output_total == 0

    def test_flag_armed_only_when_declared(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            # A tool WITHOUT stream_output declared: no region, ever.
            await transport.on_stage("tool_dispatch", {"tool": "web_search"})
            await transport.on_stage("tool_output", {"text": "late line"})

        _drive(drive)
        # The line still attaches to the tool row (late-arrival fallback)
        # but the live region flag was never armed by dispatch.
        row = transport._rows[1]
        assert list(row.output_lines) == ["late line"]

    def test_completion_resets_streaming_flag(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "work"})
            await transport.on_stage(
                "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 10}
            )

        _drive(drive)
        assert transport._open_tool_row is None
        assert transport._tool_output_streaming is False

    def test_completion_summary_stored(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "x"})
            await transport.on_stage(
                "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 5}
            )

        _drive(drive)
        row = transport._rows[1]
        assert row.data["_output_summary"] == "↳ 1 line of output"

    def test_failed_call_summary_notes_failure(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "boom"})
            await transport.on_stage(
                "tool_dispatch_complete", {"tool": "bash_run", "ok": False, "duration_ms": 5}
            )

        _drive(drive)
        summary = transport._rows[1].data["_output_summary"]
        assert "1 line" in summary
        assert "failed" in summary

    def test_turn_start_resets_streaming_state(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "stale"})

        _drive(drive)
        transport.start_thinking("model_call")
        assert transport._tool_output_streaming is False
        assert transport._open_tool_row is None


class TestOutputRegionRendering:
    """The live display renders the region beneath the in-flight tool row."""

    def test_region_group_contains_buffered_lines(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "visible line"})

        _drive(drive)
        row = transport._rows[1]
        region = transport._render_output_region(row)
        assert region is not None
        # Group renderables carry the rail prefix.
        rendered = [r.plain for r in region.renderables]
        assert any("│ visible line" in line for line in rendered)

    def test_region_none_when_nothing_buffered(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")
        row = _Row(kind="tool", text="→ tool bash_run")
        assert transport._render_output_region(row) is None

    def test_row_line_carries_output_summary(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "bash_run", "stream_output": True})
            await transport.on_stage("tool_output", {"text": "x"})
            await transport.on_stage(
                "tool_dispatch_complete", {"tool": "bash_run", "ok": True, "duration_ms": 5}
            )

        _drive(drive)
        line = transport._row_line(transport._rows[1])
        assert "↳ 1 line of output" in line.plain

    def test_non_streaming_tool_rows_unchanged(self) -> None:
        transport = _make_transport()
        transport.start_thinking("model_call")

        async def drive() -> None:
            await transport.on_stage("tool_dispatch", {"tool": "web_search"})
            await transport.on_stage(
                "tool_dispatch_complete", {"tool": "web_search", "ok": True, "duration_ms": 5}
            )

        _drive(drive)
        line = transport._row_line(transport._rows[1])
        assert "↳" not in line.plain


class TestOnceTransportStreamLines:
    """Once mode prints streamed lines to real scrollback."""

    def test_stream_lines_printed(self) -> None:
        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, width=100)
        from hosts.cli.transport import OnceTransport

        transport = OnceTransport(console=console)
        transport.start_thinking("dispatch")

        async def drive() -> None:
            await transport.on_stage("tool_output", {"text": "streamed stdout", "stream": "stdout"})
            await transport.on_stage("tool_output", {"text": "streamed err", "stream": "stderr"})

        _drive(drive)
        out = sink.getvalue()
        assert "│ streamed stdout" in out
        assert "[stderr] streamed err" in out

    def test_quiet_drops_stream_lines(self) -> None:
        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, width=100)
        from hosts.cli.transport import OnceTransport

        transport = OnceTransport(console=console, quiet=True)
        transport.start_thinking("dispatch")

        async def drive() -> None:
            await transport.on_stage("tool_output", {"text": "hidden", "stream": "stdout"})

        _drive(drive)
        assert "hidden" not in sink.getvalue()