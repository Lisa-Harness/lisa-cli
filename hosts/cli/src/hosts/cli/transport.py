"""Rich-based TransportCallbacks implementation for the CLI host.

This is the terminal-transport implementation of
:class:`host.callbacks.TransportCallbacks`. The CLI host constructs a
``RichTransport`` and passes it to ``host.handle_message``; the runtime
calls these methods during turn execution to deliver stage updates, streamed
text chunks, final responses, debug context, and errors.

Rich is allowed here — this is the terminal host. The core runtime never
imports Rich; it only calls these methods on the injected transport.

Layout: one flowing transcript
------------------------------
The turn renders as a single flowing column of rows — no panes, headers, or
boxes. Each stage event becomes one left-aligned line; tool calls are
one-liners that update in place as they complete; the streamed answer flows
into the transcript and *persists* (the fully formatted Markdown panel appears
only when the turn did not stream).

Stage-line legend
-----------------
* ``[mode]`` prefix — the mode that currently owns the turn; updates live on a
  mid-turn handoff.
* ``→ op / → tool / → mcp / ↦`` — tool-kind glyphs (in-process operation, pack
  tool, MCP call, transition).
* ``✓ / ✗ / …`` — tool-completion status glyphs.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, ClassVar

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

logger = logging.getLogger(__name__)

# ── Visual style constants (kept in sync with the Lisa palette) ──
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_ERROR = "indian_red"
STYLE_STAGE = "grey58"
STYLE_TEXT_MUTED = "grey58"
STYLE_TEXT_PRIMARY = "bright_white"
STYLE_GRAPHIC = "grey58"
STYLE_ROW_TRANSITION = "gold3"
STYLE_ROW_MODEL = "dark_sea_green4"
STYLE_ROW_TOOL = "cornflower_blue"
STYLE_ROW_ERROR = "indian_red"
STYLE_STREAM_ANSWER = "bright_white"
STYLE_STREAM_REASONING = "grey42"
STYLE_OUTPUT_REGION = "grey42"
STYLE_OUTPUT_REGION_BORDER = "grey35"

#: Lines reserved below the Live region for post-turn content that must stay
#: visible: the turn footer (1), the Rule separator (1), the bottom toolbar
#: (1), and the input prompt line (1).
_RESERVED_BELOW_LINES = 4

#: Floor for the number of transcript rows kept visible in the Live region.
#: Guarantees a usable view on very short terminals / non-tty consoles.
_MIN_VISIBLE_ROWS = 6

#: Minimum model-call iterations sampled before the ETA estimate appears in
#: the status header. Below this the rolling average is too noisy to be useful.
_ETA_MIN_ITERATIONS = 2

#: Block ramp used for the progress bar. Index 0 is the empty cell; index 7 is
#: the full block.
_BLOCK_RAMP = " ▁▂▃▄▅▆▇█"


#: Glyphs for tool-dispatch completion status.
_TOOL_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…", "": "·"}

#: Maximum length of the primary tool argument shown inline on a tool row.
_ARG_PREVIEW_MAX = 40

#: Lines of a streaming tool's output kept visible in the live region. The
#: buffer scrolls: older lines are dropped as new ones arrive, mirroring a
#: terminal tail. The full output still reaches the model via ToolResult —
#: this is an operator-view cap only.
_OUTPUT_REGION_MAX_LINES = 12

#: Per-line width cap for streaming output lines so long lines wrap instead
#: of blowing out the region (final width clamps against the live console).
_OUTPUT_LINE_MAX = 200


def _truncate(value: str, limit: int = _ARG_PREVIEW_MAX) -> str:
    """Truncate ``value`` to ``limit`` chars with an ellipsis."""
    value = value.replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _format_tokens(tokens: int) -> str:
    """Format a token count compactly for the status/footer (``256k``)."""
    tokens = int(tokens or 0)
    if tokens <= 0:
        return "—"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:g}k"
    return str(tokens)


class _TrailingSpinner:
    """A Live-compatible renderable that puts the spinner *after* the line.

    The fixed line content renders first (so it stays aligned with every other
    row in the transcript) and a single spinner frame glyph is appended after a
    one-space gap as a trailing activity indicator.
    """

    __slots__ = ("_line", "_spinner", "_style")

    def __init__(self, line: Text, spinner_name: str, style: str) -> None:
        self._line = line
        self._spinner = Spinner(spinner_name, style=style)
        self._style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.render(console.get_time())

    def render(self, time: float) -> RenderableType:
        frame = self._spinner.render(time)
        glyph = frame.plain if isinstance(frame, Text) else ""
        out = Text()
        out.append(self._line)
        if glyph:
            out.append(" ")
            out.append(glyph, style=self._style)
        return out


@dataclass
class _Row:
    """One row in the transcript.

    ``kind`` selects the style; ``text`` is the status words; ``active`` marks
    the spinner-bearing row; ``data`` retains the raw payload.

    For ``kind == "thinking"`` the row carries one reasoning segment: ``text``
    is the segment body (kept collapsed/expanded per the transport flag) and
    ``thinking_expanded`` is the per-row expand state. Multiple thinking rows
    intersperse with tool-call rows in arrival order.

    For ``kind == "tool"`` a streaming tool (one whose ``.tool.yml`` declares
    ``stream_output``) accumulates its live operator-visible output lines in
    ``output_lines`` while the call is in flight; the region renders beneath
    the tool row and collapses to a one-line summary on completion.
    """

    kind: str
    text: str
    active: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    thinking_expanded: bool = False
    output_lines: deque[str] = field(default_factory=deque)
    output_total: int = 0


class RichTransport:
    """TransportCallbacks implementation backed by a Rich Console.

    The Live region is a single flowing column of transcript rows — the turn
    trajectory rendered as it happens. Each stage event appends (or updates)
    one row; the streamed answer flows in and persists across ``stop_thinking``.
    The active (most recent) row carries a live spinner.

    On turn completion the Live region stops and the transcript — including the
    streamed answer — is flushed to scrollback as the permanent record. The
    fully formatted Markdown panel is printed by ``on_response`` only when the
    turn did not already stream.

    Attributes:
        console: The Rich Console to render to. Defaults to a shared
            ``Console()``; callers may pass a custom one (testing, redirection).
    """

    def __init__(self, *, console: Console | None = None) -> None:
        self.console: Console = console or Console()
        # When True (default), each thinking row renders a compact
        # ``▸ Thinking… N char`` summary; when False, the full segment body is
        # shown. ``Ctrl+Y`` toggles ALL thinking rows together (they are
        # interspersed with tool calls, so a single global flag is the right
        # model, not a per-segment one).
        self.thinking_collapsed: bool = True
        # ``_status`` is a non-None sentinel while the Live region is active and
        # ``None`` when stopped — the contract used by REPL cancel handling and
        # tests. The actual display is owned by ``_live``.
        self._status: Live | None = None
        self._live: Live | None = None
        self._thinking_stage: str = ""
        self._thinking_spinner: str = "dots"
        # Whether any on_text_chunk visible-answer delta arrived this turn —
        # used by on_response to decide between re-rendering the staged stream
        # and printing the clean Panel. Reset in start_thinking.
        self._streamed_during_turn: bool = False
        # The visible-answer deltas stream into a staging buffer that is NEVER
        # shown as the final answer — on completion the Live region tears down
        # and a clean Markdown Panel renders the full answer. The buffer exists
        # only so a mid-turn error can flush partial work to scrollback.
        self._answer_staging: Text = Text()
        # Index of the in-flight thinking row, so consecutive reasoning deltas
        # append to one segment; a tool call or model_call opens a new segment.
        self._open_thinking_row: int | None = None
        # Fork fan-out live state, populated by on_stage("fork_*") so the
        # transcript can render one row per child. Reset on start_thinking.
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0
        # The mode that currently owns the turn, tracked across stage events.
        self._current_mode: str = ""
        # Pending approval payload captured from ``data["approval"]`` on the
        # final response. When non-None the approval panel just rendered and the
        # y/n/a/v keystroke bindings (keys.py) are armed to decide it without
        # the operator retyping the request id. Cleared on the next turn start.
        # When False, per-row ``[mode]`` prefixes are suppressed — the mode
        # already appears once on the REPL status header and the once-mode
        # activity line, so repeating it down every transcript row is noise.
        self.with_prefix: bool = True
        # Mode-only change while no Live region is up (REPL) is silent; once
        # mode emits a single handoff line itself.
        self.pending_approval: dict[str, Any] | None = None
        # ── The transcript: one ordered list of rows ────────────────────
        self._rows: list[_Row] = []
        # Per-iteration model-call durations for the current turn (sparkline).
        self._iteration_durations: list[float] = []
        # Number of model calls started in the current turn. This is distinct
        # from workflow iteration because a turn can change modes or fork.
        self._model_call_count: int = 0
        # Running counters for the graphic column.
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        self._tool_ok_count: int = 0
        self._tool_error_count: int = 0
        # ── Live status-header state (persistent, updated per stage) ─────
        # A fixed status bar rendered at the top of the Live region so
        # mode / model / token count stay visible and update instantly while
        # the turn runs — the guard against runaway sessions. The header is
        # owned by the transport, so it re-renders on every on_stage refresh
        # without waiting for the next prompt-toolkit status-line pass.
        self._status_current_model: str = ""
        self._status_model_provider: str = ""
        self._status_turn_tokens: int = 0
        self._status_session_tokens: int = 0
        # ── Pinned-prompt + glanceable-progress state (ask 1 & 2) ────────
        # The submitted message, pinned to the top of the Live region for the
        # whole turn so the user's question stays visible during long jobs.
        # Reset to "" in stop_thinking. Empty for slash-command turns that
        # supply no message.
        self._turn_prompt: str = ""
        # perf_counter() captured at start_thinking so the status header can
        # show elapsed seconds without a wall-clock dependency.
        self._turn_started_perf: float = 0.0
        # Compact once mode: suppress the Workflow summary Panel and heavy
        # transcript chrome — the activity line carries mode/model/progress
        # and a single ⚡ footer line replaces the grid (ask: once output is
        # vertically efficient, per-turn, pipe-friendly).
        self.compact: bool = False
        # ── Per-turn caps (ask 3) ────────────────────────────────────────
        # The max_* caps surfaced by stage events, stored so the always-on
        # status-header cap strip and _aggregate_progress can read them
        # without waiting for a specific stage event. 0 means "unset /
        # unlimited" and renders in the compact N<count> form.
        self._max_tool_calls: int = 0
        self._max_iterations: int = 0
        self._max_handoffs: int = 0
        self._current_iter: int = 0
        # Index of the in-flight tool row, so its completion updates in place.
        self._open_tool_row: int | None = None
        # Index of the in-flight model row, so its completion updates in place.
        self._open_model_row: int | None = None
        # Whether the in-flight tool row carries a live output region (the
        # tool declared stream_output and lines are arriving). Kept as a
        # flag because the row index alone doesn't say whether to render
        # the region.
        self._tool_output_streaming: bool = False
        # The streamed model output for the current turn as a single flowing
        # Text (visible answer bright, reasoning dimmed). Persists across
        # stop_thinking — the transcript flushes it to scrollback.
        self._stream_text: Text = Text()
        # ── Shift+Enter runtime detection (ask 4) ────────────────────────
        # True once an ``s-enter`` key event has fired in the REPL — i.e. the
        # terminal speaks the kitty/CSI-u keyboard protocol and can
        # distinguish Shift+Enter from Enter. The bottom toolbar reads this
        # to show the newline chord that will actually work on this terminal
        # rather than a static (and possibly lying) hint.
        self.shift_enter_supported: bool = False

    # ── Live region lifecycle ───────────────────────────────────────

    def start_thinking(
        self,
        stage: str = "dispatch",
        *,
        session_tokens: int = 0,
        message: str = "",
    ) -> None:
        """Start the flowing transcript so the user sees the turn is active.

        The default stage is ``dispatch`` — the pre-pipeline phase where
        ``handle_message`` parses commands and selects ``current_mode_id``.
        It uses a distinct spinner (``dots2``) so "still deciding" reads
        differently from "model is generating" once the first ``model_call``
        arrives and switches to ``dots``.

        ``session_tokens`` seeds the live status header's cumulative session
        token count (the runtime's pre-turn total), so the header shows the
        real running total rather than starting at zero each turn.

        ``message`` is the user's submitted prompt; it is pinned to the top of
        the Live region for the whole turn (ask 1) so the question stays
        visible during long jobs. Empty for slash-command turns.
        """
        self.stop_thinking()
        self._thinking_stage = stage
        self._streamed_during_turn = False
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        # A decision was made (or is about to be made) — disarm the approval
        # keystrokes from the prior turn so a stale pending state can't fire.
        self.pending_approval = None
        self._rows = []
        self._iteration_durations = []
        self._model_call_count = 0
        self._handoff_count = 0
        self._tool_call_count = 0
        self._tool_ok_count = 0
        self._tool_error_count = 0
        self._status_current_model = ""
        self._status_model_provider = ""
        self._status_turn_tokens = 0
        self._status_session_tokens = session_tokens
        self._turn_prompt = message
        self._turn_started_perf = perf_counter()
        self._max_tool_calls = 0
        self._max_iterations = 0
        self._max_handoffs = 0
        self._current_iter = 0
        self._open_tool_row = None
        self._open_model_row = None
        self._open_thinking_row = None
        self._tool_output_streaming = False
        self._answer_staging = Text()
        is_dispatch = stage == "dispatch"
        self._thinking_spinner = "dots2" if is_dispatch else "dots"
        # Seed the transcript with the initial dispatch row so the Live region
        # shows something immediately. Mark it active so the spinner lands here.
        seed = _Row(kind="dispatch", text=self._render_status_line(stage), active=True)
        self._rows.append(seed)
        live: Live | None = Live(
            self._render_live_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        if live is not None:
            try:
                live.start()
            except Exception:  # pylint: disable=broad-except
                # Non-tty / redirected consoles can reject Live; fall back to
                # plain inline printing via _refresh_live.
                live = None
        self._live = live
        self._status = live

    def stop_thinking(
        self,
        *,
        flush_transcript: bool = False,
        flush_stream: bool = False,
    ) -> None:
        """Stop the thinking display if it is running.

        By default this removes the transient working display without writing
        it to scrollback, keeping successful completion focused on the final
        response and workflow summary. Set ``flush_transcript`` for diagnostic
        outcomes (errors and cancellations) that need to preserve the ordered
        tool/model/thinking activity. The streamed answer is never flushed on
        a successful response; when ``flush_stream`` is True, partial answer
        text is retained with the diagnostic transcript.
        """
        live = self._live
        status = self._status
        self._live = None
        self._status = None
        if live is not None:
            try:
                live.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        if flush_transcript and live is not None:
            for row in self._rows:
                try:
                    self.console.print(self._row_line(row))
                    # A streaming tool's captured tail flushes with the row
                    # so a diagnostic transcript retains what the command
                    # produced before the failure/cancellation.
                    if row.kind == "tool" and row.output_lines:
                        for out_line in row.output_lines:
                            self.console.print(
                                Text(f"│ {out_line}", style=STYLE_OUTPUT_REGION)
                            )
                except Exception:  # pylint: disable=broad-except
                    pass
            if flush_stream and self._answer_staging:
                try:
                    self.console.print(self._answer_staging)
                except Exception:  # pylint: disable=broad-except
                    pass
        if status is not None and status is not live:
            try:
                status.stop()
            except Exception:  # pylint: disable=broad-except
                pass

    # ── Rendering ───────────────────────────────────────────────────

    def _live_height(self) -> int:
        """Number of transcript rows visible in the Live region.

        Terminal height minus ``_RESERVED_BELOW_LINES`` so post-turn content
        doesn't push the transcript off-screen. Floored at ``_MIN_VISIBLE_ROWS``.
        """
        height = self.console.size.height or 0
        if height <= 0:
            return _MIN_VISIBLE_ROWS
        return max(_MIN_VISIBLE_ROWS, height - _RESERVED_BELOW_LINES)

    def _render_status_header(self) -> Text:
        """Render the persistent live status bar (one line).

        This is the always-on guard against runaway sessions: it sits at the
        top of the Live region and updates on every stage event, so the mode,
        active model, per-turn caps, aggregate progress, elapsed/ETA, and
        session token total are visible and current for the whole turn — not
        just once above the input prompt.

        ``mode <m> · <provider>:<model> · [<bar> <pct>% ·] models <N> ·
        <tools>/<cap> tools · <iter>/<cap> iter · turn <turn_tok> tok ·
        session <session_tok> tok · <elapsed>s[ · ~<eta>s left]``.

        The progress bar only appears when at least one per-turn cap (tools,
        iterations, handoffs) is set. ETA appears only once ≥2 model
        iterations have been sampled.
        """
        parts: list[Text] = []
        if self._current_mode:
            parts.append(Text(f"mode {self._current_mode}", style="bold " + STYLE_STAGE))
        if self._status_current_model:
            model = self._status_current_model
            if self._status_model_provider:
                model = f"{self._status_model_provider}:{model}"
            parts.append(Text(model, style="bold " + STYLE_STAGE))
        # Aggregate progress bar: only when per-turn caps are set.
        progress = self._aggregate_progress()
        if progress is not None:
            pct = round(progress * 100)
            parts.append(
                Text(f"{self._progress_bar(progress)} {pct}%", style="bold steel_blue3")
            )
        # Cap strip: always-on counts/caps.
        cap_parts: list[str] = [f"models {self._model_call_count}"]
        cap_parts.append(self._format_cap(self._tool_call_count, self._max_tool_calls, "tools"))
        cap_parts.append(self._format_cap(self._current_iter, self._max_iterations, "iter"))
        cap_parts.append(f"turn {self._status_turn_tokens} tok")
        parts.append(Text(" · ".join(cap_parts), style=STYLE_TEXT_MUTED))
        # Session tokens.
        parts.append(Text(f"session {self._status_session_tokens} tok", style=STYLE_TEXT_MUTED))
        # Elapsed / ETA.
        elapsed = self._elapsed_seconds()
        eta = self._eta_seconds()
        timer = f"{elapsed}s"
        if eta is not None:
            timer += f" · ~{eta}s left"
        parts.append(Text(timer, style=STYLE_TEXT_MUTED))
        return Text(" · ").join(parts)


    def _aggregate_progress(self) -> float | None:
        """Blend consumed-cap ratios into a single 0–1 progress value.

        Uses the per-turn caps tracked internally: tool calls, model
        iterations, and handoffs. Each dimension contributes its consumed
        ratio (count/cap); dimensions with cap 0 (unset/unlimited) drop out.
        Returns the **max** of the available ratios — the binding constraint
        reads as the progress, which matches the "guard against runaway"
        intent. Returns ``None`` when no caps are set at all (the bar is
        suppressed in that case).
        """
        ratios: list[float] = []
        if self._max_tool_calls > 0:
            ratios.append(self._tool_call_count / self._max_tool_calls)
        if self._max_iterations > 0:
            ratios.append(self._current_iter / self._max_iterations)
        if self._max_handoffs > 0:
            ratios.append(self._handoff_count / self._max_handoffs)
        if not ratios:
            return None
        return max(0.0, min(1.0, max(ratios)))

    def _progress_bar(self, ratio: float, *, width: int = 8) -> str:
        """Render ``ratio`` (0–1) as a gradient block bar of ``width`` cells.

        Reuses the ``_BLOCK_RAMP`` ramp (`` ▁▂▃▄▅▆▇█``) so the progress bar
        shares its visual language with the duration sparklines and pip
        counters. The filled portion graduates left-to-right; the unfilled
        tail is the empty cell.
        """
        ratio = max(0.0, min(1.0, ratio))
        filled = round(ratio * width)
        if filled >= width:
            return _BLOCK_RAMP[-1] * width
        if filled <= 0:
            return _BLOCK_RAMP[0] * width
        return _BLOCK_RAMP[-1] * filled + _BLOCK_RAMP[0] * (width - filled)

    def _format_cap(self, count: int, cap: int, label: str) -> str:
        """Format a count/cap pair for the status header cap strip.

        ``cap > 0`` → ``<count>/<cap> <label>`` (the faithful ratio). ``cap <= 0``
        (unset/unlimited) → ``N<count> <label>`` (the compact counter).
        """
        if cap > 0:
            return f"{count}/{cap} {label}"
        return f"N{count} {label}"


    def _elapsed_seconds(self) -> int:
        """Whole seconds elapsed since ``start_thinking`` for this turn."""
        if self._turn_started_perf <= 0:
            return 0
        return int(perf_counter() - self._turn_started_perf)

    def _eta_seconds(self) -> int | None:
        """Estimate seconds remaining for the turn, or ``None`` if unknown.

        Uses the rolling average of ``_iteration_durations`` and the remaining
        iteration estimate (``max_iterations - current_iter``). Returns
        ``None`` until ≥2 iterations have been sampled (no wild guesses on the
        first iteration) or when no iteration cap is set.
        """
        if self._max_iterations <= 0 or len(self._iteration_durations) < _ETA_MIN_ITERATIONS:
            return None
        remaining = self._max_iterations - self._current_iter
        if remaining <= 0:
            return 0
        avg_ms = sum(self._iteration_durations) / len(self._iteration_durations)
        return max(0, int(avg_ms * remaining / 1000.0))

    def _render_prompt_banner(self) -> Text:
        """Render the pinned submitted-prompt banner (ask 1).

        The user's submitted message pinned to the top of the Live region
        for the duration of the turn, with newlines preserved so multi-line
        prompts stay multi-line. Each line is truncated to terminal width
        so it never wraps. Empty (returns a blank Text) when no prompt was
        supplied — slash-command turns that call ``start_thinking`` without
        a message show no banner.
        """
        if not self._turn_prompt:
            return Text("")
        width = self.console.size.width or 80
        prompt_text = self._turn_prompt.strip()
        line = Text()
        line.append("❯ ", style="bold steel_blue3")
        first = True
        for raw_line in prompt_text.split("\n"):
            body = _truncate(raw_line.strip(), max(0, width - 4))
            if not first:
                line.append("\n… ", style="bold steel_blue3")
            line.append(body, style="bold " + STYLE_TEXT_PRIMARY)
            first = False
        return line

    def _render_live_display(self) -> Group:
        """Build the Live renderable: prompt banner + status header + transcript.

        Layout (top → bottom):
        1. The pinned prompt banner (ask 1) — the user's submitted question,
           always visible during the turn.
        2. The live status header — mode · model · progress bar · cap strip ·
           session tokens · elapsed/ETA on one line.
        3. The transcript rows, most recent at the bottom; the active (last)
           row gets a trailing spinner. Thinking rows render collapsed (a
           compact ``▸ Thinking… N char`` summary) or expanded (the full dimmed
           segment body) per the global ``thinking_collapsed`` flag.
        4. Directly beneath the in-flight tool row, a streaming tool's live
           output region — the most recent ``_OUTPUT_REGION_MAX_LINES`` lines
           of the running command, indented so the tool row reads as the
           region's owner. Scrolls with the transcript; the buffer keeps only
           the tail (like ``tail -f``).
        """
        renderables: list[Any] = []
        banner = self._render_prompt_banner()
        if banner.plain:
            renderables.append(banner)
        renderables.append(self._render_status_header())
        rows = self._rows[-self._live_height() :]
        for i, row in enumerate(rows):
            is_active = row.active and self._live is not None and i == len(rows) - 1
            line = self._row_line(row)
            if is_active:
                renderables.append(_TrailingSpinner(line, self._thinking_spinner, STYLE_GRAPHIC))
            else:
                renderables.append(line)
            streaming_row = row.kind == "tool" and self._tool_output_streaming
            if streaming_row and self._open_tool_row is not None:
                # The region belongs to the IN-FLIGHT tool row; map the global
                # row index to the visible window index before rendering.
                open_idx = self._open_tool_row - (len(self._rows) - len(rows))
                if i == open_idx:
                    region = self._render_output_region(row)
                    if region is not None:
                        renderables.append(region)
        return Group(*renderables)

    def _render_output_region(self, row: _Row) -> RenderableType | None:
        """Render the live scrolling output region for a streaming tool call.

        Each buffered line renders as ``│ <line>`` — a thin left rail marks
        the region's extent without boxing it (the rail matches the dimmed
        row style, so the output reads as subordinate to the tool line).
        Returns ``None`` when there is nothing buffered yet, so an
        about-to-stream call shows no empty region.
        """
        if not row.output_lines:
            return None
        lines: list[Text] = []
        width = self.console.size.width or 80
        for out_line in row.output_lines:
            body = _truncate(out_line, max(0, min(_OUTPUT_LINE_MAX, width - 6)))
            lines.append(Text(f"│ {body}", style=STYLE_OUTPUT_REGION))
        return Group(*lines)

    def _row_line(self, row: _Row) -> Text:
        """Render one transcript row.

        Tool/model/transition/fork rows render as ``<text>``. Thinking rows
        render collapsed (a compact summary) or expanded (the full dimmed
        segment body) per the row's expand flag. A completed streaming tool
        row carries its ``↳ N lines of output`` summary inline.
        """
        if row.kind == "thinking":
            return self._render_thinking_row(row)
        text = row.text
        summary = row.data.get("_output_summary")
        if summary:
            text = f"{text}  {summary}"
        return Text(text, style=self._row_style(row.kind))

    def _render_thinking_row(self, row: _Row) -> Text:
        """Render one thinking segment: collapsed summary or expanded body."""
        if row.thinking_expanded and row.text:
            return Text(row.text, style=STYLE_STREAM_REASONING)
        char_count = len(row.text)
        label = "▸ Thinking…" if char_count else "▸ (thinking…)"
        return Text(f"{label} {char_count} char", style=STYLE_GRAPHIC)

    @staticmethod
    def _row_style(kind: str) -> str:
        return {
            "transition": STYLE_ROW_TRANSITION,
            "model": STYLE_ROW_MODEL,
            "tool": STYLE_ROW_TOOL,
            "fork": STYLE_ROW_TOOL,
            "error": STYLE_ROW_ERROR,
        }.get(kind, STYLE_STAGE)

    def _stream_line_count(self) -> int:
        """Visible thinking-char total across all thinking rows (for tests).

        Kept so existing tests that drove the old single-stream buffer can assert
        reasoning was captured. Returns the count of thinking segments whose
        bodies are non-empty.
        """
        return sum(1 for r in self._rows if r.kind == "thinking" and r.text)

    def toggle_thinking(self) -> None:
        """Collapse/expand ALL thinking rows together (Ctrl+Y).

        Thinking segments are interspersed with tool calls in the transcript;
        the toggle flips every visible thinking row's expand state at once so
        the user can inspect the model's reasoning across the whole turn
        without hunting for each segment.

        Between turns (Live region inactive) the toggle is a state-only change:
        it does NOT reprint the transcript, so no spurious row leaks into the
        input area. The bottom toolbar reflects the new state (the REPL
        invalidates it on the keystroke). During a turn (Live active) the
        in-flight thinking rows re-render expanded/collapsed immediately.
        """
        self.thinking_collapsed = not self.thinking_collapsed
        for row in self._rows:
            if row.kind == "thinking":
                row.thinking_expanded = not self.thinking_collapsed
        self._redraw_live()

    def _redraw_live(self) -> None:
        """Re-render the Live region if active; otherwise do nothing.

        Used by the thinking toggle: between turns the Live region is down and
        reprinting the last transcript row would pollute the input area with a
        stale call line. The toggle's feedback surface is the bottom toolbar,
        not the transcript.
        """
        if self._live is not None:
            try:
                self._live.update(self._render_live_display())
            except Exception:  # pylint: disable=broad-except
                pass

    def _refresh_live(self) -> None:
        """Re-render the Live region if active; otherwise print the latest row inline.

        Used by row-arrival paths (stage events, streamed deltas): when the
        Live region is up, refresh it; in the headless/test path (no Live),
        print the latest row so the transcript still flows to stdout.
        """
        if self._live is not None:
            try:
                self._live.update(self._render_live_display())
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            # No Live region (headless / test path): print the latest row inline.
            row = self._rows[-1] if self._rows else None
            if row is not None:
                self.console.print(self._row_line(row))

    # ── Stage events ────────────────────────────────────────────────

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Route one stage event into the transcript and refresh.

        * ``model_call`` opens a model row; ``model_call_complete`` /
          ``model_call_failed`` update it in place with ✓/✗ + duration
          (mirrors the tool-dispatch in-place update pattern).
        * ``tool_dispatch`` opens a tool row; ``tool_dispatch_complete`` updates
          it in place with its status glyph and duration.
        * ``fork_*`` → fork rows, one per child.
        * Mode transitions (from ``model_call`` / handoff) → transition rows.

        Bookkeeping-only events render rows only on the ``debug`` tier; on
        ``clean``/``power`` they still update internal counters.
        """
        data = data or {}

        # A tool/model/fork event closes the open thinking segment so the next
        # reasoning delta opens a fresh interspersed segment.
        if label in (
            "model_call",
            "model_call_complete",
            "model_call_failed",
            "tool_dispatch",
            "tool_dispatch_complete",
            "fork_start",
            "fork_child_start",
            "fork_child_complete",
            "fork_complete",
        ):
            self._open_thinking_row = None

        # ── Live mode tracking ──────────────────────────────────────
        prev_mode = self._current_mode
        if label == "model_call" and data.get("mode_id"):
            self._current_mode = str(data["mode_id"])
        elif (
            label == "tool_dispatch_complete"
            and data.get("transition") == "handoff"
            and data.get("handoff_to")
        ):
            self._current_mode = str(data["handoff_to"])
        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            self._append_transition(prev_mode, self._current_mode, data)

        # ── Per-iteration duration sampling ─────────────────────────
        if label == "model_call_complete":
            d_raw = data.get("duration_ms")
            if d_raw not in (None, "", 0):
                try:
                    self._iteration_durations.append(float(d_raw))
                except (TypeError, ValueError):
                    pass

        # ── Fork fan-out ────────────────────────────────────────────
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
            return
        if label == "fork_child_start":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                self._fork_children[idx] = {
                    "mode_id": data.get("mode_id", ""),
                    "stage_id": data.get("stage_id", ""),
                    "status": "running",
                    "query_preview": data.get("query_preview", ""),
                }
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
            return
        if label == "fork_child_complete":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                existing = self._fork_children.get(idx, {})
                existing.update(
                    {"status": data.get("status", "ok"), "duration_ms": data.get("duration_ms")}
                )
                self._fork_children[idx] = existing
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
            return
        if label == "fork_complete":
            ok = int(data.get("ok", 0))
            failed = int(data.get("failed", 0))
            self._fork_total = 0
            self._append_row(_Row(kind="fork", text=self._render_fork_stage(summary=(ok, failed))))
            return

        # ── Model calls ─────────────────────────────────────────────
        if label in ("model_call", "model_call_complete", "model_call_failed"):
            self._route_model_call(label, data)
            return

        # ── Incremental tool output (stream_output tools) ───────────
        if label == "tool_output":
            self._handle_tool_output(data)
            return

        # ── Tool dispatch ───────────────────────────────────────────
        if label == "tool_dispatch":
            self._tool_call_count += 1
            self._open_tool_row = len(self._rows)
            self._tool_output_streaming = bool(data.get("stream_output"))
            max_calls = int(data.get("max_tool_calls") or 0)
            # Track the cap for the always-on status header cap strip (ask 3)
            # and the aggregate progress blend (ask 2). A later event may
            # raise it; keep the max seen so the cap never appears to shrink.
            if max_calls > self._max_tool_calls:
                self._max_tool_calls = max(self._max_tool_calls, max_calls)
            self._append_row(
                _Row(
                    kind="tool",
                    text=self._render_tool_dispatch(data),
                    data=data,
                )
            )
            return
        if label == "tool_dispatch_complete":
            self._complete_tool_row(data)
            return

        # ── Catch-all (bookkeeping) ─────────────────────────────────
        # Bookkeeping events update counters but render no row.
        self._mark_last_active()
        self._refresh_live()

    def _handle_tool_output(self, data: dict[str, Any]) -> None:
        """Route one streamed tool-output line into the live region.

        Lines attach to the in-flight tool row (opened by ``tool_dispatch``).
        If no tool row is open — e.g. the line raced ahead of the dispatch
        event or the dispatch was for a different tool kind — the line is
        attributed to the LAST tool row so late job output still lands
        somewhere visible; otherwise it is dropped (there is no sane owner).
        The region shows the most recent ``_OUTPUT_REGION_MAX_LINES`` lines
        (a scrolling tail, like ``tail -f``), with a counter for the total
        lines received this call.
        """
        text = str(data.get("text") or "")
        if not text:
            return
        idx = self._open_tool_row
        row: _Row | None = None
        if idx is not None and 0 <= idx < len(self._rows) and self._rows[idx].kind == "tool":
            row = self._rows[idx]
        else:
            for candidate in reversed(self._rows):
                if candidate.kind == "tool":
                    row = candidate
                    break
        if row is None:
            return
        stream = str(data.get("stream") or "stdout")
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                continue
            row.output_lines.append(line)
            row.output_total += 1
        while len(row.output_lines) > _OUTPUT_REGION_MAX_LINES:
            row.output_lines.popleft()
        row.data = {**row.data, "_last_output_stream": stream}
        self._tool_output_streaming = True
        self._refresh_live()

    def _handle_tool_output(self, data: dict[str, Any]) -> None:
        """Route one streamed tool-output line into the live region.

        Lines attach to the in-flight tool row (opened by ``tool_dispatch``).
        If no tool row is open — e.g. the line raced ahead of the dispatch
        event or belonged to an already-completed call — the line is
        attributed to the LAST tool row so late job output still lands
        somewhere visible; otherwise it is dropped (there is no sane owner).
        The region shows the most recent ``_OUTPUT_REGION_MAX_LINES`` lines
        (a scrolling tail, like ``tail -f``).
        """
        text = str(data.get("text") or "")
        if not text:
            return
        idx = self._open_tool_row
        row: _Row | None = None
        if idx is not None and 0 <= idx < len(self._rows) and self._rows[idx].kind == "tool":
            row = self._rows[idx]
        else:
            for candidate in reversed(self._rows):
                if candidate.kind == "tool":
                    row = candidate
                    break
        if row is None:
            return
        stream = str(data.get("stream") or "stdout")
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                continue
            row.output_lines.append(line)
            row.output_total += 1
        while len(row.output_lines) > _OUTPUT_REGION_MAX_LINES:
            row.output_lines.popleft()
        row.data = {**row.data, "_last_output_stream": stream}
        self._tool_output_streaming = True
        self._refresh_live()

    def _append_row(self, row: _Row) -> None:
        """Append ``row`` to the transcript, mark it active, and refresh."""
        for existing in self._rows:
            existing.active = False
        row.active = True
        self._rows.append(row)
        self._refresh_live()

    def _mark_last_active(self) -> None:
        for existing in self._rows:
            existing.active = False
        if self._rows:
            self._rows[-1].active = True

    def _append_transition(self, from_mode: str, to_mode: str, data: dict[str, Any]) -> None:
        """Append a mode-transition row.

        Compact mode renders the handoff as a bare ``↦ from → to`` line — the
        destination mode already surfaces on the activity line and (via
        ``handoff``) the workflow footer, so the bracket form is redundant.
        """
        arrow = "→" if from_mode else "⇥"
        max_handoffs = int(data.get("max_handoffs") or 0)
        # Track the cap for the always-on status header and aggregate progress.
        if max_handoffs > self._max_handoffs:
            self._max_handoffs = max(self._max_handoffs, max_handoffs)
        if self.compact:
            text = f"↦ {from_mode or 'start'} {arrow} {to_mode}"
        else:
            text = f"[{to_mode}] handoff: {from_mode or 'start'} {arrow} {to_mode}"
        self._append_row(
            _Row(
                kind="transition",
                text=text,
                data={**data, "from_mode": from_mode, "to_mode": to_mode},
            )
        )

    def _route_model_call(self, label: str, data: dict[str, Any]) -> None:
        """Route a model-call stage event to a single in-place-updated row.

        ``model_call`` opens a model row (storing its index in
        ``_open_model_row``); ``model_call_complete`` / ``model_call_failed``
        update that row in place with ✓/✗ + duration — the same pattern as
        ``tool_dispatch`` / ``tool_dispatch_complete``.
        """
        max_iter = int(data.get("max_iterations") or 0)
        iter_raw = data.get("iteration")
        if iter_raw in (None, "", 0):
            iter_raw = data.get("_state.iteration")
        iter_count = int(iter_raw) if iter_raw not in (None, "", 0) else 0

        # Track the iteration cap and current iteration for the always-on
        # status header cap strip (ask 3) and the aggregate progress blend
        # (ask 2). ``iter_count`` is the iteration about to run (on
        # model_call) or just completed (on model_call_complete).
        if max_iter > self._max_iterations:
            self._max_iterations = max(self._max_iterations, max_iter)
        if iter_count > self._current_iter:
            self._current_iter = max(self._current_iter, iter_count)

        # Update the persistent live status header so mode/model stay current
        # the instant the model call begins (model_call_complete below adds
        # token totals).
        if data.get("model_id") not in (None, "", 0):
            self._status_current_model = str(data["model_id"])
            provider = data.get("provider")
            self._status_model_provider = str(provider) if provider not in (None, "") else ""

        if label == "model_call":
            self._model_call_count += 1
            self._open_model_row = len(self._rows)
            self._append_row(
                _Row(
                    kind="model",
                    text=self._render_model_call(data),
                    data=data,
                )
            )
            return

        # model_call_complete or model_call_failed — update the open row in place.
        status_glyph = "✗" if label == "model_call_failed" else "✓"
        if label == "model_call_complete":
            # Accumulate the model call's token usage into the live header so
            # the running total updates instantly and guards runaway sessions.
            total_raw = data.get("_state.total_tokens") or data.get("total_tokens")
            if total_raw not in (None, "", 0):
                self._status_turn_tokens += int(total_raw)
                self._status_session_tokens += int(total_raw)

        idx = self._open_model_row
        if idx is not None and idx < len(self._rows):
            row = self._rows[idx]
            row.text = self._render_model_call(row.data, status_glyph=status_glyph, complete=data)
            row.data = {**row.data, **data}
        else:
            self._append_row(
                _Row(
                    kind="model",
                    text=self._render_model_call(data, status_glyph=status_glyph, complete=data),
                    data=data,
                )
            )
        self._open_model_row = None
        self._mark_last_active()
        self._refresh_live()

    def _render_model_call(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a model call as a one-liner: stage, model, iteration, status.

        ``[mode] stage: model_call haiku #1 ✓ 340ms``. On completion
        (``status_glyph`` set) the status and duration are appended.
        """
        prefix = f"[{self._current_mode}] " if self._current_mode else ""
        body = "stage: model_call"
        model_id = data.get("model_id")
        if model_id not in (None, "", 0):
            body += f" {model_id}"
        iter_raw = data.get("iteration") or data.get("_state.iteration")
        if iter_raw not in (None, "", 0):
            body += f" #{iter_raw}"
        if status_glyph:
            body += f" {status_glyph}"
        duration = (complete or data).get("duration_ms")
        if status_glyph and duration not in (None, "", 0):
            body += f" {round(float(duration))}ms"
        completion_data = complete or {}
        total_tokens = completion_data.get("_state.total_tokens") or completion_data.get(
            "total_tokens"
        )
        if total_tokens in (None, "", 0):
            total_tokens = data.get("_state.total_tokens") or data.get("total_tokens")
        if status_glyph and total_tokens not in (None, "", 0):
            body += f" {total_tokens} tok"
        return self._with_mode_prefix(prefix, body)

    def _complete_tool_row(self, data: dict[str, Any]) -> None:
        """Update the in-flight tool row in place with its status + duration."""
        ok = data.get("ok")
        status = "ok" if ok is True else ("error" if ok is False else "")
        if status == "ok":
            self._tool_ok_count += 1
        elif status == "error":
            self._tool_error_count += 1
        glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
        if self._open_tool_row is not None and self._open_tool_row < len(self._rows):
            row = self._rows[self._open_tool_row]
            if row.output_total:
                # A streaming call: collapse the live region to a one-line
                # completion summary (the full tail already scrolled past in
                # the Live region; the model still gets the full output).
                plural = "s" if row.output_total != 1 else ""
                summary = f"↳ {row.output_total} line{plural} of output"
                if status == "error":
                    summary = f"↳ {summary} (failed)"
                row.text = self._render_tool_dispatch(row.data, status_glyph=glyph, complete=data)
                row.data = {**row.data, **data, "_output_summary": summary}
            else:
                row.text = self._render_tool_dispatch(row.data, status_glyph=glyph, complete=data)
                row.data = {**row.data, **data}
        else:
            self._append_row(
                _Row(
                    kind="tool",
                    text=self._render_tool_dispatch(data, status_glyph=glyph, complete=data),
                    data=data,
                )
            )
            return
        self._open_tool_row = None
        self._tool_output_streaming = False
        self._mark_last_active()
        self._refresh_live()

    def _render_status_line(self, label: str, data: dict[str, Any] | None = None) -> str:
        """Render one status line, prefixed with the active mode.

        ``[mode] stage: <label>``. The mode prefix updates live on mid-turn
        handoff.
        """
        data = data or {}
        prefix = f"[{self._current_mode}] " if self._current_mode else ""
        return self._with_mode_prefix(prefix, f"stage: {label}")

    def _with_mode_prefix(self, prefix: str, body: str) -> str:
        """Apply the ``[mode]`` prefix unless row prefixes are suppressed.

        When ``with_prefix`` is False the mode lives on the REPL status
        header and the once-mode activity line, so repeating it down every
        transcript row is noise — the body renders un-prefixed.
        """
        return f"{prefix}{body}" if self.with_prefix else body

    # ── tool_dispatch legible rendering ──────────────────────────────
    _TOOL_KIND_GLYPH: ClassVar[dict[str, str]] = {
        "operation": "→ op",
        "pack_tool": "→ tool",
        "mcp_tool": "→ mcp",
        "transition": "↦",
    }

    def _render_tool_dispatch(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a tool call as a one-liner: glyph, tool, primary arg, status.

        ``→ mcp brave_search("query") ✓ 120ms``. On completion (``status_glyph``
        set) the status and duration are appended.
        """
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")
        prefix = f"[{self._current_mode}] " if self._current_mode else ""

        if kind == "transition":
            target = ""
            if data.get("handoff_to"):
                target = f"→ {data['handoff_to']}"
            elif data.get("transition") == "fork":
                n = data.get("fork_children") or 0
                target = f"fork ({n} children)"
            elif data.get("transitioned_to"):
                target = f"→ {data['transitioned_to']}"
            body = f"stage: tool_dispatch {glyph} {tool}"
            if target:
                body += f" {target}"
        else:
            arg = self._primary_arg_preview(data)
            body = f"{glyph} {tool}"
            if arg:
                body += f" {arg}"

        if status_glyph:
            body += f" {status_glyph}"
        duration = (complete or data).get("duration_ms")
        if status_glyph and duration not in (None, "", 0):
            body += f" {round(float(duration))}ms"
        return self._with_mode_prefix(prefix, body)

    @staticmethod
    def _primary_arg_preview(data: dict[str, Any]) -> str:
        """Extract the primary call argument as a short inline preview.

        Prefers the kernel-resolved ``display`` label (rendered from the
        tool's ``display_template`` — e.g. bash_run's command line), falling
        back to the first present among the well-known primary-argument keys
        for tools whose payload pre-dates the field or whose template failed.
        """
        display = data.get("display")
        if isinstance(display, str) and display.strip():
            return _truncate(display.strip())
        # Fallback: first non-empty primary-argument string. The kernel emits
        # the full argument dict under ``arguments``; a tool without a
        # ``display_template`` (or one whose template failed) previews its
        # primary argument here.
        arguments = data.get("arguments")
        sources: list[dict[str, Any]] = [data]
        if isinstance(arguments, dict):
            sources.append(arguments)
        for source in sources:
            for key in ("arg_preview", "query", "path", "command", "input"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return _truncate(value)
        return ""

    def _render_fork_stage(self, summary: tuple[int, int] | None = None) -> str:
        """Render the fork fan-out as one header line plus one line per child."""
        if not self._fork_children:
            if summary is not None:
                ok, failed = summary
                return f"stage: fork complete ({ok} ok, {failed} failed)"
            return f"stage: fork ({self._fork_total} children)"
        if summary is not None:
            ok, failed = summary
            header = f"stage: fork complete ({ok} ok, {failed} failed)"
        else:
            header = f"stage: fork ({self._fork_total} children)"
        lines = [header]
        for idx in sorted(self._fork_children):
            child = self._fork_children[idx]
            mode_id = child.get("mode_id", "?")
            stage_id = child.get("stage_id", "")
            status = child.get("status", "running")
            stage_part = f"{stage_id} · " if stage_id else ""
            line = f"  #{idx} {mode_id} · {stage_part}{status}"
            preview = child.get("query_preview", "")
            if preview:
                line += f" · {preview}"
            duration_ms = child.get("duration_ms")
            if duration_ms is not None:
                line += f" · {duration_ms}ms"
            lines.append(line)
        return "\n".join(lines)

    # ── Streaming / response / debug / error ────────────────────────

    def _print_once_footer(self) -> None:
        """Render the single-line once-mode footer.

        ``⚡ 22s · 5 tools · 2 models · 1 handoff · 8.4k tok`` — one line,
        per-turn only; the REPL-style token split (turn/session) collapses
        here because a ``once`` invocation is one process and one turn.
        """
        model_time_ms = round(sum(self._iteration_durations))
        tools_total = self._tool_call_count
        tool_failures = self._tool_error_count
        parts: list[str] = [f"⚡ {self._elapsed_seconds()}s"]
        parts.append(f"{tools_total} tool{'s' if tools_total != 1 else ''}")
        if tool_failures:
            parts[-1] += f" ({tools_total - tool_failures} ✓ {tool_failures} ✗)"
        parts.append(f"{self._model_call_count} model{'s' if self._model_call_count != 1 else ''}")
        if model_time_ms:
            parts[-1] += f" ({model_time_ms}ms)"
        if self._handoff_count:
            parts.append(f"{self._handoff_count} handoff{'s' if self._handoff_count != 1 else ''}")
            route = self._current_mode
            if route:
                parts[-1] += f" → {route}"
        if self._status_turn_tokens:
            parts.append(f"{_format_tokens(self._status_turn_tokens)} tok")
        self.console.print(Text(" · ".join(parts), style=STYLE_TEXT_MUTED))

    def _render_workflow_summary(self) -> None:
        """Render the post-response summary: compact ⚡ line once, grid in REPL."""
        if self.compact:
            self._print_once_footer()
            return
        from rich.table import Table

        tools = [r for r in self._rows if r.kind == "tool"]
        transitions = [r for r in self._rows if r.kind == "transition"]
        forks = [r for r in self._rows if r.kind == "fork"]

        ok = self._tool_ok_count
        failed = self._tool_error_count
        model_time_ms = round(sum(self._iteration_durations))
        route = self._current_mode or "default"
        model = self._status_current_model or "-"
        if self._status_model_provider:
            model = f"{self._status_model_provider}:{model}"

        summary = Table.grid(padding=(0, 1))
        summary.add_column(style=STYLE_TEXT_MUTED, no_wrap=True)
        summary.add_column()
        summary.add_row("Outcome", Text(f"Completed · {route} · {model}"))
        if transitions:
            route_parts = [str(transitions[0].data.get("from_mode") or route)]
            route_parts.extend(str(row.data.get("to_mode") or route) for row in transitions)
            summary.add_row("Route", Text(" → ".join(route_parts)))
        summary.add_row(
            "Tokens",
            Text(
                f"turn {self._status_turn_tokens} tok · session {self._status_session_tokens} tok"
            ),
        )
        summary.add_row(
            "Activity",
            Text(
                f"{self._model_call_count} model call{'s' if self._model_call_count != 1 else ''}"
                f" · {model_time_ms}ms · {len(tools)} tool call{'s' if len(tools) != 1 else ''} "
                f"({ok} succeeded, {failed} failed)"
                f" · {len(transitions)} handoff{'s' if len(transitions) != 1 else ''}"
            ),
        )
        if tools:
            events: list[str] = []
            for row in tools:
                outcome = "failed" if "✗" in row.text else "ok"
                tool_name = str(row.data.get("tool") or "tool")
                duration = row.data.get("duration_ms")
                suffix = f" {duration}ms" if duration not in (None, "", 0) else ""
                events.append(f"{outcome} {tool_name}{suffix}")
            summary.add_row("Tools", Text(" · ".join(events)))
        if forks:
            summary.add_row("Fork", Text(forks[-1].text.splitlines()[0]))
        self.console.print(Panel(summary, title="Workflow", border_style=STYLE_BORDER_RESPONSE))

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Route a streamed model delta.

        ``reasoning`` (thinking) becomes an interspersed thinking row in the
        transcript — consecutive reasoning deltas append to the open segment,
        and any non-reasoning event (tool call, model_call) closes it so the
        next reasoning opens a fresh segment. Multiple segments thus interleave
        with the tool calls in arrival order.

        ``text`` (the visible answer) is buffered into ``_answer_staging`` and
        is NEVER shown as the final answer. On completion ``on_response`` tears
        down the Live region and renders a clean Markdown Panel from the full
        response text — streaming the answer inline made the final unreadable
        with interspersed calls. The staging buffer only exists so a mid-turn
        error can flush partial work to scrollback.
        """
        if reasoning:
            # Append to the open thinking segment, or open a new one. A prior
            # tool/model event closed the open segment, so a new reasoning
            # delta after a tool call starts a fresh interspersed segment.
            if self._open_thinking_row is None or self._open_thinking_row >= len(self._rows):
                row = _Row(
                    kind="thinking", text=reasoning, thinking_expanded=not self.thinking_collapsed
                )
                self._append_row(row)
                self._open_thinking_row = len(self._rows) - 1
            else:
                self._rows[self._open_thinking_row].text += reasoning
            self._refresh_live()
            return
        if text:
            # Visible answer: stage only, never rendered as the final.
            self._answer_staging.append(text, style=STYLE_STREAM_ANSWER)
            # A visible-answer delta closes the open thinking segment so the
            # next reasoning opens a new one (the model emitted an answer
            # chunk, so the prior thinking block is complete).
            self._open_thinking_row = None
            self._refresh_live()

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Render the final response as clean Markdown (no vertical borders).

        Tears down the transient Live region without flushing raw reasoning or
        tool activity to scrollback, then renders the fully formatted answer
        and a clear workflow summary. The streamed answer is NOT reused:
        rendering the complete ``text`` keeps the final response readable.

        The answer is printed as bare ``Markdown`` beneath a horizontal ``Rule``
        header rather than inside a bordered ``Panel``. A Panel draws vertical
        box-drawing characters down both sides of every line, which makes
        copy/pasting the answer pull in those border glyphs. The Rule gives a
        clear visual separator (so the user still sees where the response
        begins) while leaving the content itself border-free and cleanly
        selectable.
        """
        self.stop_thinking(flush_transcript=False)
        self._answer_staging = Text()
        # Capture a pending approval and render the keystroke panel. The host
        # attaches ``data["approval"]`` when a turn paused on an action the
        # policy engine decided still needs the operator; the panel shows the
        # command + sandbox containment and arms the y/n/a/v bindings for a
        # single-keypress decision instead of retyping ``/approve <uuid>``.
        approval = (data or {}).get("approval")
        if isinstance(approval, dict) and approval.get("request_id"):
            from hosts.cli.render import render_approval_panel  # noqa: PLC0415

            self.pending_approval = approval
            render_approval_panel(approval, console=self.console)
            return
        if text:
            self.console.print(Rule(title="Response", style=STYLE_BORDER_RESPONSE))
            self.console.print(Markdown(text))
        self._render_workflow_summary()

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """No-op: the debug context panel was removed with the detail tiers.

        Kept to satisfy the ``TransportCallbacks`` protocol; the runtime only
        invokes it when debug mode is active, which the CLI no longer exposes.
        """
        return

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Render an error in a red Rich Panel.

        The transcript (including any partial staged answer) flushes to
        scrollback first so partial work isn't silently lost before the error
        panel shows.
        """
        self.stop_thinking(flush_transcript=True, flush_stream=True)
        self._answer_staging = Text()
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        self.console.print(Panel(body, title="Error", border_style=STYLE_BORDER_ERROR))


class OnceTransport(RichTransport):
    """Vertically efficient transport for the ``once`` command.

    Once mode is input/output in the caller's shell — another pipeline stage,
    not a session with a screen — so instead of a persistent Live transcript
    it renders:

    * a one-line transient activity indicator (spinner + mode + in-flight
      tool label + models/tools/elapsed) via an overwrite-in-place Live; and
    * each tool/model/fork row printed to real scrollback as it *completes*,
      so the trajectory survives piping/tee and never needs the transcript.

    Tokens are per-turn: "session vs turn" is a meaningless split when the
    whole invocation is one process, and the footer reports a single total.

    Non-tty consoles (``console.is_terminal`` False) skip the spinner entirely
    and emit only the completed-event lines — safe under ``| grep``.

    Raw mode (``raw=True``) writes the response verbatim to raw stdout — no
    Rule header, no Markdown re-render — so redirected/piped output stays
    machine-readable; all activity output lives on the transport console
    (stderr when the once command runs in raw mode). An approval that reaches
    the response un-decided sets ``approval_unresolved`` for the caller to
    exit non-zero instead of emitting an interactive prompt.
    """

    _ACTIVITY_WIDTH: int = 100

    def __init__(
        self, *, console: Console | None = None, quiet: bool = False, raw: bool = False
    ) -> None:
        super().__init__(console=console)
        self.compact = True
        self.with_prefix = False
        self.quiet = quiet
        self.raw = raw
        # Set when a response arrives carrying an unresolved approval request:
        # once mode has no prompt loop to decide approvals, so the caller exits
        # with _APPROVAL_REQUIRED_EXIT instead of printing approval text into
        # redirected output. Never cleared here; owned by the command path.
        self.approval_unresolved: bool = False
        # Set when the turn ended in a model yield (the model asked the
        # operator a question instead of finishing). The question goes to
        # stderr only; stdout gets nothing, so pipelines never receive the
        # question as if it were the requested artifact.
        self.yield_unresolved: bool = False
        self.yield_question: str = ""
        self._active_label: str = ""
        self._last_tool_data: dict[str, Any] | None = None
        self._last_tool_streaming: bool = False
        self._activity: Live | None = None
        self._status_reasoning_chars: int = 0

    # ── Lifecycle ───────────────────────────────────────────────────

    def start_turn(self, message: str) -> None:
        """Begin once-mode rendering: echo the command, arm the activity line.

        No-op when ``quiet`` (the response alone prints later).
        """
        if self.quiet:
            return
        width = self.console.size.width or self._ACTIVITY_WIDTH
        line = Text()
        line.append("λ ", style="bold steel_blue3")
        line.append(_truncate(message.strip(), max(0, width - 6)), style="bold " + STYLE_TEXT_PRIMARY)
        self.console.print(line)
        self.start_activity()

    def start_activity(self) -> None:
        """Show the one-line transient activity display (tty only)."""
        if self.quiet or self._activity is not None or not self.console.is_terminal:
            return
        activity = Live(
            self._activity_renderable(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        try:
            activity.start()
        except Exception:  # pylint: disable=broad-except
            return
        self._activity = activity

    def stop_activity(self) -> None:
        activity = self._activity
        self._activity = None
        if activity is not None:
            try:
                activity.stop()
            except Exception:  # pylint: disable=broad-except
                pass

    def start_thinking(self, stage: str = "dispatch", **kwargs: Any) -> None:
        """Once mode does not use the Live transcript; start the activity line."""
        self.start_activity()

    def stop_thinking(
        self,
        *,
        flush_transcript: bool = False,
        flush_stream: bool = False,
    ) -> None:
        # The activity indicator is always transient — flush flags do not
        # resurrect it once the answer/error is ready to print.
        self.stop_activity()

    # ── Stage events ────────────────────────────────────────────────

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Update counters + activity line; print tool rows only on completion."""
        if self.quiet:
            return
        data = data or {}
        prev_mode = self._current_mode
        if label == "model_call":
            model_id = data.get("model_id")
            if model_id not in (None, "", 0):
                provider = data.get("provider")
                self._status_current_model = str(model_id)
                self._status_model_provider = str(provider) if provider not in (None, "") else ""
            self._model_call_count += 1
            iter_raw = data.get("iteration") or data.get("_state.iteration")
            if iter_raw not in (None, "", 0) and int(iter_raw) > self._current_iter:
                self._current_iter = int(iter_raw)
            max_iter = int(data.get("max_iterations") or 0)
            self._max_iterations = max(self._max_iterations, max_iter)
            if data.get("mode_id"):
                self._current_mode = str(data["mode_id"])
        elif label == "model_call_complete":
            d_raw = data.get("duration_ms")
            if d_raw not in (None, "", 0):
                try:
                    self._iteration_durations.append(float(d_raw))
                except (TypeError, ValueError):
                    pass
            total_raw = data.get("_state.total_tokens") or data.get("total_tokens")
            if total_raw not in (None, "", 0):
                self._status_turn_tokens += int(total_raw)
        elif label == "model_call_failed":
            self._print_event_line(
                self._render_model_call(data, status_glyph="✗"), style=STYLE_ROW_ERROR
            )
        elif label == "tool_dispatch":
            self._tool_call_count += 1
            max_calls = int(data.get("max_tool_calls") or 0)
            self._max_tool_calls = max(self._max_tool_calls, max_calls)
            self._active_label = self._render_tool_dispatch(data)
            self._last_tool_data = data
            self._last_tool_streaming = bool(data.get("stream_output"))
        elif label == "tool_output":
            # Once mode has no Live transcript region — streamed lines print
            # to real scrollback as they arrive (piped runs keep them under
            # tee/grep; quiet mode drops them). A dim rail prefixes each so
            # they read as tool output, not model prose.
            text = str(data.get("text") or "")
            stream_label = str(data.get("stream") or "stdout")
            if text and not self.quiet:
                for raw_line in text.split("\n"):
                    line = raw_line.rstrip()
                    if line:
                        self._print_stream_line(line, stream=stream_label)
        elif label == "tool_dispatch_complete":
            ok = data.get("ok")
            status = "ok" if ok is True else ("error" if ok is False else "")
            if status == "ok":
                self._tool_ok_count += 1
            elif status == "error":
                self._tool_error_count += 1
            glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
            # The completion event does not re-send the call arguments, so
            # render the row from the dispatch payload and apply status/duration
            # from the completion (mirrors the base transport's row update).
            row_data = self._last_tool_data or data
            self._print_event_line(
                self._render_tool_dispatch(row_data, status_glyph=glyph, complete=data),
                style=STYLE_ROW_TOOL,
            )
            self._active_label = ""
            self._last_tool_data = None
            self._last_tool_streaming = False
        elif label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._print_event_line(self._render_fork_stage())
        elif label == "fork_child_start":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                self._fork_children[idx] = {
                    "mode_id": data.get("mode_id", ""),
                    "stage_id": data.get("stage_id", ""),
                    "status": "running",
                    "query_preview": data.get("query_preview", ""),
                }
                self._active_label = f"fork #{idx} {data.get('mode_id', '')}".strip()
        elif label == "fork_child_complete":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                existing = self._fork_children.get(idx, {})
                existing.update(
                    {
                        "status": data.get("status", "ok"),
                        "duration_ms": data.get("duration_ms"),
                    }
                )
                self._fork_children[idx] = existing
            self._active_label = ""
            if self._fork_children and all(
                c.get("status") != "running" for c in self._fork_children.values()
            ):
                self._print_event_line(self._render_fork_stage())
                self._fork_children = {}
        elif label == "fork_complete":
            if self._fork_children:
                self._print_event_line(self._render_fork_stage())
                self._fork_children = {}
            self._fork_total = 0
        # else: bookkeeping-only events update nothing visible in once mode.

        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            max_handoffs = int(data.get("max_handoffs") or 0)
            self._max_handoffs = max(self._max_handoffs, max_handoffs)
            self._print_event_line(
                f"↦ {prev_mode or 'start'} → {self._current_mode}",
                style=STYLE_ROW_TRANSITION,
            )
        self._redraw_activity()

    # ── Streaming / response ───────────────────────────────────────

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Stage answer deltas / count reasoning without rendering either.

        The answer renders once, fully formatted, from ``on_response``;
        thinking segments are a REPL affordance (Ctrl+Y) that once mode
        replaces with the compact footer.
        """
        if reasoning:
            self._status_reasoning_chars += len(reasoning)
        if text:
            self._answer_staging.append(text, style=STYLE_STREAM_ANSWER)
            self._streamed_during_turn = True

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Print the answer, then the compact ⚡ footer (or nothing if quiet).

        Raw mode writes ``text`` verbatim to raw stdout — no Rule header, no
        Markdown re-render — keeping redirected/piped output machine-readable;
        the ⚡ footer prints on the transport console (stderr in raw mode).

        An approval that reaches the response un-decided is a hard stop: once
        mode has no prompt loop to answer it, so the request id is stashed
        (``approval_unresolved``) and nothing is written to stdout — the
        caller exits non-zero rather than emitting an interactive panel (or
        the bare approval text) into redirected output.

        A model yield (the ``yield`` terminal action — the model asked the
        operator a question instead of finishing) is the same contract: the
        question is stashed (``yield_unresolved``), printed to stderr only,
        and stdout gets nothing — the caller exits non-zero instead of
        writing the question into redirected output as if it were the
        requested result.
        """
        self.stop_activity()
        self._answer_staging = Text()
        approval = (data or {}).get("approval")
        if isinstance(approval, dict) and approval.get("request_id"):
            from hosts.cli.render import render_approval_panel  # noqa: PLC0415

            self.pending_approval = approval
            self.approval_unresolved = True
            if self.console.is_terminal:
                # Once mode has no prompt loop: the panel shows the
                # `lisa once "approve <id>"` commands instead of key hints.
                render_approval_panel(approval, console=self.console, keystrokes=False)
            else:
                # Non-interactive surface (piped/redirected): fail fast on
                # stderr instead of a panel the operator can never answer.
                # Keep each decision hint short so a 120-col console never
                # wraps inside a copyable command.
                request_id = str(approval.get("request_id", ""))
                target = str(approval.get("target_id", "action"))
                err_console = Console(stderr=True, width=200)
                err_console.print(
                    f"[red]Approval required for {target} (request {request_id}).\n"
                    f'Decide without a session: lisa once "approve {request_id}" '
                    f'or lisa once "deny {request_id}"\n'
                    f'(add --remember <prefix> to approve-and-persist), '
                    f"then re-run the original command.[/red]"
                )
            return
        # A model yield (ask-the-operator question) is an unresolved turn, not
        # an answer. The host surfaces it structurally on
        # ``data["yielded"]`` (and ``HostCommandResult.yielded_question``).
        yield_question = str((data or {}).get("yielded") or "")
        if yield_question:
            self.yield_unresolved = True
            self.yield_question = yield_question
            self.console.print(
                Text(f"⏸ Model asked: {yield_question}", style=STYLE_ROW_TRANSITION)
            )
            if not self.console.is_terminal:
                Console(stderr=True).print(
                    "[red]The model asked a question instead of producing a "
                    "result. Once mode cannot answer it. Re-run with the "
                    "missing information in the prompt, or interactively.[/red]"
                )
            return
        if text:
            if self.raw:
                sys.stdout.write(text)
                if not text.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                self.console.print(Rule(title="Response", style=STYLE_BORDER_RESPONSE))
                self.console.print(Markdown(text))
        if text and not self.quiet and not self.raw:
            self.console.print()
        if not self.quiet and not self.raw:
            self._print_once_footer()
        elif self.raw and not self.quiet:
            # The footer is activity output: in raw mode it prints on the
            # transport console (stderr) so stdout stays machine-readable.
            self._print_once_footer()

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Stop the activity line and render the error panel.

        Completed event lines already printed to scrollback as they finished,
        so the trajectory up to the failure is preserved without a flush.
        """
        self.stop_activity()
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        self.console.print(Panel(body, title="Error", border_style=STYLE_BORDER_ERROR))

    # ── Internals ───────────────────────────────────────────────────

    def _print_event_line(self, text: str, *, style: str | None = None) -> None:
        """Print one completed scrollback line, replacing the activity line."""
        self.stop_activity()
        self.console.print(f"  {text}", style=style or STYLE_TEXT_MUTED)
        self.start_activity()

    def _print_stream_line(self, line: str, *, stream: str = "stdout") -> None:
        """Print one streamed tool-output line to scrollback (once mode).

        Unlike ``_print_event_line`` this does NOT stop/restart the activity
        line per line — a churning command would flicker the spinner. The
        activity Live redraws over its own region on the next refresh, so
        printed lines persist above it.
        """
        if stream == "stderr" and not line.startswith("[stderr]"):
            line = f"[stderr] {line}"
        self.console.print(Text(f"  │ {line}", style=STYLE_OUTPUT_REGION))

    def _redraw_activity(self) -> None:
        if self._activity is not None:
            try:
                self._activity.update(self._activity_renderable())
            except Exception:  # pylint: disable=broad-except
                pass

    def _activity_renderable(self) -> RenderableType:
        """The one-line transient activity display."""
        elapsed = self._elapsed_seconds()
        width = self.console.size.width or self._ACTIVITY_WIDTH
        parts: list[str] = []
        if self._current_mode:
            parts.append(f"[{self._current_mode}]")
        if self._status_current_model:
            model = self._status_current_model
            if self._status_model_provider:
                model = f"{self._status_model_provider}:{model}"
            parts.append(model)
        if self._active_label:
            parts.append(_truncate(self._active_label, 48))
        parts.append(f"models {self._model_call_count}")
        parts.append(f"tools {self._tool_call_count}")
        parts.append(f"{elapsed}s")
        body = _truncate(" · ".join(parts), max(0, width - 4))
        line = Text()
        line.append(body, style=STYLE_STAGE)
        return _TrailingSpinner(line, self._thinking_spinner, STYLE_GRAPHIC)


__all__ = ["OnceTransport", "RichTransport"]
