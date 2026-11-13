"""Prompt-mode Rich transport for the CLI host.

This is the terminal-transport implementation of
:class:`runtime.callbacks.TransportCallbacks` for ``lisa prompt`` — the CLI's
only turn path. A prompt run is input/output in the caller's shell: another
pipeline stage. On a tty it renders:

* a full-screen Live view with the prompt pinned at the top, tool activity
  scrolling through the middle, and thinking/status pinned at the bottom.

Tokens are per-turn: a ``prompt`` invocation is one process and one turn,
so the footer reports a single total (the turn/session split collapses).

Non-tty consoles (``console.is_terminal`` False) skip the Live view and
emit completed-event lines — safe under ``| grep`` and ``tee``.

Streamed tool output (``stream_output`` tools, e.g. bash_run) follows the
same tty split: non-tty consoles print every streamed line to scrollback as
it arrives; on a tty lines join the bounded middle transcript as they
arrive, so failures retain their output context without a separate output
box. Older transcript rows scroll out of view and are counted; when stdin
is a tty, ↑/↓, PgUp/PgDn, and Home/End scroll back through the full
transcript (End resumes following the tail).

Raw mode (``raw=True``) writes the response verbatim to raw stdout — no
Rule header, no Markdown re-render — so redirected/piped output stays
machine-readable; all activity output lives on the transport console
(stderr when the prompt command runs in raw mode).

Stage-line legend
-----------------
* ``→ op / → tool / → mcp / ↦`` — tool-kind glyphs (in-process operation,
  pack tool, MCP call, transition).
* ``✓ / ✗ / …`` — tool-completion status glyphs.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any, ClassVar

from rich.align import Align
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.layout import Layout
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
STYLE_ROW_SUCCESS = "green3"
STYLE_STREAM_ANSWER = "bright_white"
STYLE_STREAM_BOX = "grey50"
STYLE_STREAM_BOX_BORDER = "grey35"
STYLE_OUTPUT_REGION = "grey42"
STYLE_TOOL_NAME = "bold bright_white"
STYLE_ARG_PREVIEW = "grey58"
STYLE_DETAIL_KEY = "grey58"

#: Glyphs for tool-dispatch completion status.
_TOOL_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…", "": "·"}

#: Maximum length of the primary tool argument shown inline on a tool row.
_ARG_PREVIEW_MAX = 72

#: Minimum terminal width below which the right-aligned status column is
#: skipped (narrow consoles keep the compact left-aligned row).
_STATUS_ALIGN_MIN_WIDTH = 80

#: Column reserved for the right-aligned status+duration segment.
_STATUS_ALIGN_COLUMN = 24

#: Token-count formatting thresholds for the compact footer figures.
_TOKENS_MILLION = 1_000_000
_TOKENS_THOUSAND = 1_000

#: Budget-bar glyphs: filled pips, empty pips (rendered in stage color), and
#: the near-cap warning threshold ratio at which the bar switches to gold.
_BUDGET_BAR_FILLED = "▰"
_BUDGET_BAR_EMPTY = "▱"
_BUDGET_BAR_MAX_PIPS = 8
_BUDGET_BAR_WARN_RATIO = 0.75

#: Maximum pretty-printed argument lines shown at detail level 2.
_DETAIL_ARGUMENT_MAX_LINES = 12

#: Tool-detail verbosity levels (``--tool-detail`` / ``-v`` count).
_TOOL_DETAIL_SUMMARY = 1
_TOOL_DETAIL_FULL = 2

#: Elapsed-time format switch: below this many seconds the activity line
#: shows plain seconds; at/above it shows minutes:seconds.
_ELAPSED_MINUTE_SECONDS = 60
#: Bounded thinking/answer stream window: maximum lines kept visible inside
#: the transient activity panel (a tail window — older lines age out, so
#: scrollback never fills with streamed chatter).
_STREAM_BOX_MAX_LINES = 8
#: Minimum terminal width for rendering the stream box (narrow consoles
#: keep the plain one-line indicator).
_STREAM_BOX_MIN_WIDTH = 60
#: Characters below which the stream box renders bare (no rounded border)
#: to keep the activity display light during very short generations.
_STREAM_BOX_BORDER_CHARS = 240

#: Maximum prompt lines kept pinned in the interactive screen header.
_PROMPT_MAX_LINES = 6
#: Bounded number of transcript entries retained by the interactive screen.
_ACTIVITY_HISTORY_MAX_ROWS = 5000
#: Transcript lines kept free of streamed tool output so recent tool rows
#: stay visible while a chatty tool runs.
_ACTIVITY_MIN_TOOL_ROWS = 7
#: Streamed-output lines per tool kept when the transcript is copied into
#: scrollback after the turn (the newest lines; earlier ones are counted).
_SCROLLBACK_OUTPUT_MAX_LINES = 20


class _OutputLine(Text):
    """A streamed tool-output transcript row, collapsible at render time."""


_ESC = 0x1B
#: Byte range that terminates a CSI/SS3 escape sequence.
_CSI_FINAL_MIN = 0x40
_CSI_FINAL_MAX = 0x7E
#: Smallest transcript window that has room for both scroll markers.
_SCROLL_MARKER_MIN_ROWS = 3

_SCROLL_KEYS: dict[bytes, str] = {
    b"\x1b[A": "up",
    b"\x1bOA": "up",
    b"\x1b[B": "down",
    b"\x1bOB": "down",
    b"\x1b[5~": "pgup",
    b"\x1b[6~": "pgdn",
    b"\x1b[H": "home",
    b"\x1bOH": "home",
    b"\x1b[1~": "home",
    b"\x1b[F": "end",
    b"\x1bOF": "end",
    b"\x1b[4~": "end",
}


def _parse_scroll_keys(buf: bytes) -> tuple[list[str], bytes]:
    """Split ``buf`` into scroll actions; return them and any partial escape."""
    actions: list[str] = []
    i = 0
    while i < len(buf):
        if buf[i] != _ESC:
            i += 1
            continue
        end = i + 1
        if end >= len(buf):
            return actions, buf[i:]
        if buf[end] in b"[O":
            end += 1
            while end < len(buf) and not _CSI_FINAL_MIN <= buf[end] <= _CSI_FINAL_MAX:
                end += 1
            if end >= len(buf):
                return actions, buf[i:]
        action = _SCROLL_KEYS.get(buf[i : end + 1])
        if action:
            actions.append(action)
        i = end + 1
    return actions, b""


class _ScrollKeyReader:
    """Read scroll keys from a tty stdin while the live screen is up.

    Puts the terminal in cbreak mode (no echo, signals intact so Ctrl-C
    still aborts), reads on a daemon thread, and hands each action to the
    event loop that started it so transcript state is only touched there.
    """

    def __init__(self, fd: int, saved: list[Any], on_action: Callable[[str], None]) -> None:
        import asyncio
        import threading

        self._fd = fd
        self._saved = saved
        self._on_action = on_action
        try:
            self._loop: Any = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lisa-scroll-keys", daemon=True)
        self._thread.start()

    @classmethod
    def start(cls, on_action: Callable[[str], None]) -> _ScrollKeyReader | None:
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                return None
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return None
        return cls(fd, saved, on_action)

    def _dispatch(self, action: str) -> None:
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._on_action, action)
        else:
            self._on_action(action)

    def _run(self) -> None:
        import os
        import select

        pending = b""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    pending = b""
                    continue
                chunk = os.read(self._fd, 64)
            except OSError:
                return
            if not chunk:
                return
            actions, pending = _parse_scroll_keys(pending + chunk)
            for action in actions:
                self._dispatch(action)

    def stop(self) -> None:
        import termios

        self._stop.set()
        self._thread.join(timeout=0.5)
        with contextlib.suppress(Exception):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


#: Minimum distinct samples before a velocity estimate is trusted.
_VELOCITY_MIN_SAMPLES = 2
#: Stream velocity display window (seconds): the rolling chars/s estimate
#: is computed from samples inside this window.
_VELOCITY_WINDOW_S = 4.0
#: Minimum window span (seconds) before a velocity figure renders — below
#: this the estimate is noise.
_VELOCITY_MIN_SPAN_S = 0.75
#: Minimum sampled delta (chars) for a velocity figure to be worth showing.
_VELOCITY_MIN_CHARS = 64

#: Seconds without a delta before the generation reads as stalled. Below
#: this the stream is assumed to be pausing between chunks (reasoning
#: models regularly go quiet for several seconds while composing); at/above
#: it the status line shows a no-data age and switches the spinner, so a
#: stalled upstream reads as a stall, not a hung frame.
_STALL_AFTER_S = 15.0
#: No-data age (seconds) at which the stall segment escalates from gold
#: to red — roughly the point where an operator starts reaching for ^C.
_STALL_ALERT_S = 60.0


def _format_velocity(chars_per_s: float) -> str:
    """Format a stream rate compactly (``812c/s``, ``1.4k/s``)."""
    if chars_per_s >= _TOKENS_THOUSAND:
        return f"{chars_per_s / _TOKENS_THOUSAND:.1f}k/s"
    return f"{chars_per_s:.0f}c/s"


def _truncate(value: str, limit: int = _ARG_PREVIEW_MAX) -> str:
    """Truncate ``value`` to ``limit`` chars with an ellipsis."""
    value = value.replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _format_tokens(tokens: int) -> str:
    """Format a token count compactly for the footer (``256k``)."""
    tokens = int(tokens or 0)
    if tokens <= 0:
        return "—"
    if tokens >= _TOKENS_MILLION:
        return f"{tokens / _TOKENS_MILLION:g}M"
    if tokens >= _TOKENS_THOUSAND:
        return f"{tokens / _TOKENS_THOUSAND:g}k"
    return str(tokens)


def _budget_bar(used: int, total: int) -> str:
    """Render a compact consumed/total bar (``▰▰▱▱``).

    ``total`` <= 0 means unlimited — return "" so callers fall back to the
    plain counter (never fabricate a denominator). The bar warns gold at
    ``>= _BUDGET_BAR_WARN_RATIO`` consumption; the color is applied by the
    caller (the style choice differs per surface).
    """
    if total <= 0 or used < 0:
        return ""
    pips = max(1, min(_BUDGET_BAR_MAX_PIPS, total))
    filled = pips * used // total
    return _BUDGET_BAR_FILLED * filled + _BUDGET_BAR_EMPTY * (pips - filled)


def _format_elapsed(seconds: int) -> str:
    """Format whole-second elapsed time for the activity line.

    Under a minute it stays a plain second count (``3s``), past a minute
    it switches to minutes:seconds (``1:23``) so long generations stay
    legible at a glance.
    """
    seconds = max(0, int(seconds))
    if seconds < _ELAPSED_MINUTE_SECONDS:
        return f"{seconds}s"
    return f"{seconds // _ELAPSED_MINUTE_SECONDS}:{seconds % _ELAPSED_MINUTE_SECONDS:02d}"


class _TrailingSpinner:
    """A Live-compatible renderable that puts the spinner *after* the line.

    The line content is pulled fresh on every frame via ``_line_fn`` so
    live counters (elapsed, thinking size) tick between stage events
    instead of freezing at the value built when the Live started.
    """

    __slots__ = ("_line_fn", "_spinner_fn", "_style")

    def __init__(
        self,
        line_fn: Callable[[], Text],
        spinner_fn: Callable[[], str],
        style: str,
    ) -> None:
        self._line_fn = line_fn
        self._spinner_fn = spinner_fn
        self._style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.render(console.get_time())

    def render(self, time: float) -> RenderableType:
        name = self._spinner_fn() or "dots"
        spinner = Spinner(name, style=self._style)
        frame = spinner.render(time)
        glyph = frame.plain if isinstance(frame, Text) else ""
        out = Text()
        out.append(self._line_fn())
        if glyph:
            out.append(" ")
            out.append(glyph, style=self._style)
        return out


class PromptTransport:
    """Vertically efficient transport for the ``prompt`` command.

    See the module docstring for the rendering contract. The two
    post-turn flags the command path reads:

    * ``approval_unresolved`` — set when a response arrives carrying an
      un-decided approval request. A prompt run has no interactive loop to
      decide approvals, so the caller exits with the approval-required code
      instead of emitting approval text into redirected output. Never
      cleared here; owned by the command path.
    * ``yield_unresolved`` / ``yield_question`` — set when the turn ended
      in a model yield (the model asked the operator a question instead
      of finishing). The question goes to stderr only; stdout gets
      nothing, so pipelines never receive the question as if it were the
      requested artifact.
    """

    _ACTIVITY_WIDTH: int = 100

    # ── tool_dispatch legible rendering ──────────────────────────────
    _TOOL_KIND_GLYPH: ClassVar[dict[str, str]] = {
        "operation": "→ op",
        "pack_tool": "→ tool",
        "mcp_tool": "→ mcp",
        "transition": "↦",
    }

    def __init__(
        self,
        *,
        console: Console | None = None,
        quiet: bool = False,
        raw: bool = False,
        tool_detail: int = 0,
        streaming_mode: str = "none",
    ) -> None:
        self.console: Console = console or Console()
        self.quiet = quiet
        self.raw = raw
        #: Verbosity for tool-call detail. 0 = one-line rows (default);
        #: 1 = + result preview; 2 = + pretty-printed arguments and the full
        #: result preview.
        self.tool_detail = max(0, min(2, int(tool_detail)))
        #: Runtime ``--streaming`` mode (none|stage|text|hybrid). Controls
        #: the bounded stream box at the bottom of the activity display: ``text``
        #: streams the visible answer, ``hybrid`` streams reasoning (with a
        #: char-counter fallback when the provider sends none); ``none``
        #: and ``stage`` keep the plain one-line indicator.
        self.streaming_mode = (streaming_mode or "none").strip().lower()
        self._init_screen_state()
        self.approval_unresolved: bool = False
        self.yield_unresolved: bool = False
        self.yield_question: str = ""
        # Pending approval payload captured from ``data["approval"]`` on the
        # final response. When non-None the approval panel just rendered;
        # the command path reads it for exit-code decisions. Cleared on the
        # next turn start (a decision was made or is about to be).
        self.pending_approval: dict[str, Any] | None = None
        # ── Activity-line state ─────────────────────────────────────
        self._active_label: str = ""
        self._last_tool_data: dict[str, Any] | None = None
        self._last_tool_streaming: bool = False
        self._last_tool_depth: int = 0
        # Streamed-output line count for the in-flight tool (drives the
        # ``⎿ N lines`` result-shape summary on the completion row).
        self._last_tool_output_lines: int = 0
        self._activity: Live | None = None
        self._thinking_spinner: str = "dots"
        # ── Budget caps (stamped by the kernel on every stage payload; ──
        # 0 = unlimited). Captured for the activity-line progress bars.
        self._cap_max_iterations: int = 0
        self._cap_max_tool_calls: int = 0
        self._cap_max_handoffs: int = 0
        # ── Turn counters ───────────────────────────────────────────
        self._turn_started_perf: float = 0.0
        self._current_mode: str = ""
        self._reset_model_status()
        # Global thread turn number of the turn being rendered, stamped by
        # the host on the ``model_call`` stage payload (``turn_index``).
        # Rendered on the Response rule and the ⚡ footer; -1 = unknown. A
        # genuine ``turn 0`` (fresh thread) is valid and renders. Not reset
        # in ``start_thinking``: a pre-armed-decision run dispatches a second
        # message on the same transport and each capture replaces it.
        self._turn_number: int = -1
        self._init_status_counters()
        self._iteration_durations: list[float] = []
        self._model_call_count: int = 0
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        self._tool_ok_count: int = 0
        self._tool_error_count: int = 0
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0
        # Tool call hierarchy: stack of (tool_name, tool_kind, parent_index)
        # for rendering nested tool calls with indentation
        self._tool_call_stack: list[tuple[str, str, int]] = []
        # Per-model per-token prices (USD) injected by the command path:
        # ``{model_id: (prompt_price, completion_price)}``. Empty = the
        # footer omits the cost estimate (manifest unreadable/unknown model).
        self.model_prices: dict[str, tuple[float, float]] = {}
        # Per-model prompt-cache prices (USD/token) keyed the same way; the
        # footer's estimate blends the read rate for cache-hit prompts.
        # Empty/missing keys = the estimate treats every prompt as uncached.
        self.model_cache_prices: dict[str, tuple[float, float]] = {}
        # Whether any on_text_chunk visible-answer delta arrived this turn —
        # informational only: the answer renders once, fully (markdown), from
        # on_response regardless.
        self._streamed_during_turn: bool = False
        # The visible-answer deltas stage into this buffer purely so a
        # mid-turn error path can flush partial work; on_response clears it.
        self._answer_staging: Text = Text()
        # ── Bounded stream box (tail window) ────────────────────────
        # Plain-text accumulation of the streamed content shown in the
        # activity panel: the answer (``text`` mode) or the model's
        # thinking (``hybrid`` mode). Only a tail window is ever rendered,
        # so long generations keep a bounded footprint on screen.
        self._active_reasoning_model: bool = False
        self._stream_answer_buf: str = ""
        self._stream_answer_chars: int = 0
        self._stream_reasoning_buf: str = ""
        self._iteration_current: int = 0
        self._iteration_max: int = 0
        #: Pre-first-event label shown on the activity line while the
        #: runtime boots (profile/manifest/MCP warmup) — the window that
        #: otherwise reads as a bare spinner. Cleared on the first stage.
        self._phase_label: str = "starting runtime"
        # ── Stream velocity (observable, not manifest-trusted) ──────
        self._init_velocity_state()

    def _init_screen_state(self) -> None:
        self._screen_mode = False
        self._prompt_message = ""
        self._activity_rows: list[Text] = []
        self._activity_history_elided = 0
        self._screen_tool_contexts: list[tuple[int, dict[str, Any], int, int]] = []
        #: First visible transcript line while scrolled back; None follows the tail.
        self._activity_scroll_top: int | None = None
        self._activity_view_rows = 1
        self._key_reader: _ScrollKeyReader | None = None
        self._wrap_cache: dict[int, tuple[Text, int, list[Text]]] = {}
        #: Transcript rows already copied into normal scrollback.
        self._scrollback_flushed_rows = 0
        self._scrollback_prompt_printed = False
        self._scrollback_rule_printed = False

    def _init_status_counters(self) -> None:
        """Zero the per-turn token/cost counters shared by the footer.

        ``start_thinking`` re-runs this for each turn; constructing the
        transport runs it once for the pre-turn state.

        ``_status_turn_tokens`` is the *walk* (every call's ``total_tokens``),
        ``_status_billed_tokens`` the uncached basis, and the two cost fields
        carry the kernel's canonical estimate and the provider's own reported
        charge. The provider figure is authoritative when non-zero; when the
        kernel supplied neither cost field the footer falls back to pricing
        ``model_prices`` itself, so an older runtime still shows a number.
        """
        self._status_turn_tokens: int = 0
        self._status_prompt_tokens: int = 0
        self._status_completion_tokens: int = 0
        self._status_cached_tokens: int = 0
        self._status_billed_tokens: int = 0
        self._status_cost_usd: float = 0.0
        self._status_provider_cost_usd: float = 0.0
        # Model calls that reported a provider charge. A turn mixing
        # providers (some reporting cost, some silent) has a partial total,
        # which must not masquerade as the full bill.
        self._status_provider_cost_calls: int = 0
        # Per-turn sums of the counters the serving proxy declared on its
        # response headers (e.g. compression ``tokens_saved``), keyed by the
        # pack's own field names. Empty when no proxy is in the path — the
        # footer then renders no proxy segment at all.
        self._status_proxy_telemetry: dict[str, int] = {}
        self._status_reasoning_chars: int = 0

    def _reset_model_status(self) -> None:
        self._status_current_model: str = ""
        self._status_model_provider: str = ""
        self._status_proxy_id: str = ""
        self._status_proxy_via: str = ""

    def _init_velocity_state(self) -> None:
        """Initialize the velocity-sampling state (one statement block).

        The manifest ``reasoning_model`` flag lies for some providers
        (glm-5.3-family bills reasoning without flagging it), so
        "reasoning in flight" is derived from what actually arrives:
        any non-empty reasoning delta arms the meter for the call
        (``_reasoning_seen``). Rolling velocity samples are
        ``(perf_counter, total_stream_chars)`` pairs; chars/s =
        (last - first) / span across the window, pruned per read.
        """
        self._reasoning_seen: bool = False
        self._velocity_samples: list[tuple[float, int]] = []
        # Wall clock of the most recent delta (answer or reasoning) of the
        # in-flight model call; 0 before the first delta of a call. Ages
        # into the stall segment when a generation goes quiet (the
        # ``_STALL_AFTER_S`` / ``_STALL_ALERT_S`` thresholds).
        self._last_delta_perf: float = 0.0

    def start_turn(self, message: str) -> None:
        """Begin prompt-mode rendering: pin the prompt, arm the activity display.

        Interactive terminals retain the full prompt in the screen header.
        Non-terminal output keeps the compact legacy echo.
        """
        if self.quiet:
            return
        self._prompt_message = message.strip()
        self._activity_rows = []
        self._activity_history_elided = 0
        self._screen_tool_contexts = []
        self._activity_scroll_top = None
        self._scrollback_flushed_rows = 0
        self._scrollback_prompt_printed = False
        self._scrollback_rule_printed = False
        if not self.console.is_terminal:
            width = self.console.size.width or self._ACTIVITY_WIDTH
            line = Text()
            line.append("λ ", style="bold steel_blue3")
            line.append(
                _truncate(self._prompt_message, max(0, width - 6)),
                style="bold " + STYLE_TEXT_PRIMARY,
            )
            self.console.print(line)
        self.start_activity()

    def start_activity(self) -> None:
        """Show the interactive screen or the compact activity line."""
        if self.quiet or self._activity is not None or not self.console.is_terminal:
            return
        self._screen_mode = True
        activity = Live(
            self._activity_renderable(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
            screen=True,
        )
        activity.start(refresh=True)
        self._activity = activity
        self._key_reader = _ScrollKeyReader.start(self._scroll_activity)

    def stop_activity(self) -> None:
        reader = self._key_reader
        self._key_reader = None
        if reader is not None:
            reader.stop()
        activity = self._activity
        self._activity = None
        if activity is not None:
            with contextlib.suppress(Exception):
                activity.stop()
        self._flush_transcript_to_scrollback()

    def _flush_transcript_to_scrollback(self) -> None:
        """Copy the screen's prompt and tool rows into normal scrollback.

        The live screen runs on the alternate buffer, so without this the
        turn's tool history vanishes when it closes. Rows are copied once
        (later stops print only new rows); streamed-output blocks keep their
        last ``_SCROLLBACK_OUTPUT_MAX_LINES`` so the answer is not buried.
        """
        if not self._screen_mode or self.quiet:
            return
        if not self._scrollback_prompt_printed and self._prompt_message:
            header = self._prompt_header(max_lines=_PROMPT_MAX_LINES)
            self.console.print(Group(*header.renderables[:-1]))
            self._scrollback_prompt_printed = True
        start = self._scrollback_flushed_rows
        if start >= len(self._activity_rows):
            return
        if not self._scrollback_rule_printed:
            self.console.print(Rule(style=STYLE_ROW_TOOL))
            self._scrollback_rule_printed = True
            if self._activity_history_elided:
                self.console.print(
                    Text(f"… {self._activity_history_elided} oldest lines dropped"),
                    style=STYLE_TEXT_MUTED,
                )
        width = max(1, (self.console.size.width or self._ACTIVITY_WIDTH) - 2)
        lines = self._activity_lines(
            width=width,
            output_budget=_SCROLLBACK_OUTPUT_MAX_LINES,
            rows=self._activity_rows[start:],
        )
        self.console.print(Group(*lines))
        self._scrollback_flushed_rows = len(self._activity_rows)

    def start_thinking(self, stage: str = "dispatch", **kwargs: Any) -> None:
        """Prompt mode has no Live transcript; (re)arm the activity line.

        Accepts the TransportCallbacks kwargs (``session_tokens``,
        ``message``) for protocol compatibility and resets the per-turn
        counters. Any pending approval from a prior render is disarmed —
        a decision was made (or is about to be).
        """
        self.pending_approval = None
        self._iteration_durations = []
        self._model_call_count = 0
        self._handoff_count = 0
        self._tool_call_count = 0
        self._tool_ok_count = 0
        self._tool_error_count = 0
        self._status_turn_tokens = 0
        self._status_prompt_tokens = 0
        self._status_completion_tokens = 0
        self._status_cached_tokens = 0
        self._status_billed_tokens = 0
        self._status_cost_usd = 0.0
        self._status_provider_cost_usd = 0.0
        self._status_provider_cost_calls = 0
        self._status_proxy_telemetry = {}
        self._status_reasoning_chars = 0
        self._fork_children = {}
        self._fork_total = 0
        self._tool_call_stack = []
        self._screen_tool_contexts = []
        self._current_mode = ""
        self._reset_model_status()
        self._active_label = ""
        self._phase_label = "starting runtime"
        self._last_tool_data = None
        self._last_tool_streaming = False
        self._last_tool_output_lines = 0
        self._cap_max_iterations = 0
        self._cap_max_tool_calls = 0
        self._cap_max_handoffs = 0
        self._iteration_current = 0
        self._iteration_max = 0
        self._active_reasoning_model = False
        self._stream_answer_buf = ""
        self._stream_answer_chars = 0
        self._stream_reasoning_buf = ""
        self._turn_started_perf = perf_counter()
        is_dispatch = stage == "dispatch"
        self._thinking_spinner = "dots2" if is_dispatch else "dots"
        self._answer_staging = Text()
        self._streamed_during_turn = False
        self._reasoning_seen = False
        self._velocity_samples = []
        self._last_delta_perf = 0.0
        self.start_activity()

    def stop_thinking(
        self,
        *,
        flush_transcript: bool = False,
        flush_stream: bool = False,
    ) -> None:
        """Stop the activity indicator if it is running.

        The Live screen is transient; non-tty event lines remain in
        scrollback. The flush flags exist for TransportCallbacks
        compatibility and only flush a staged partial answer on diagnostics.
        """
        if flush_transcript and flush_stream and self._answer_staging:
            with contextlib.suppress(Exception):
                self.console.print(self._answer_staging)
        self._answer_staging = Text()
        self.stop_activity()

    # ── Stage events ────────────────────────────────────────────────

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Update counters + activity line; print tool rows only on completion."""
        if self.quiet:
            return
        data = data or {}
        prev_mode = self._current_mode
        # Budget caps ride every kernel stage payload (emit_stage setdefaults
        # them). Capture for the activity-line progress bars; 0 = unlimited.
        for cap_field, payload_key in (
            ("_cap_max_iterations", "max_iterations"),
            ("_cap_max_tool_calls", "max_tool_calls"),
            ("_cap_max_handoffs", "max_handoffs"),
        ):
            cap_raw = data.get(payload_key)
            if isinstance(cap_raw, int) and cap_raw > 0:
                setattr(self, cap_field, cap_raw)
        # The first real stage event ends the boot phase.
        self._phase_label = ""
        if label.startswith("model_call"):
            self._handle_model_stage(label, data)
        elif label.startswith("tool_"):
            self._handle_tool_stage(label, data)
        elif label.startswith("fork_"):
            self._handle_fork_stage(label, data)
        # else: bookkeeping-only events update nothing visible in prompt mode.

        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            handoff = Text("↦ ", style=STYLE_ROW_TRANSITION)
            handoff.append(
                f"{prev_mode or 'start'} → {self._current_mode}", style=STYLE_ROW_TRANSITION
            )
            self._print_event_line(handoff)
        self._redraw_activity()

    # ── Streaming / response ───────────────────────────────────────

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Stage answer deltas / count reasoning; feed the bounded stream box.

        Deltas accumulate into plain-text tail-window buffers that render
        inside the transient activity display (``text`` mode shows the
        answer stream; ``hybrid`` shows the thinking stream). Status lines
        stay put — only the tail of what the model is producing scrolls
        inside the box. The answer always re-renders once, fully
        markdown-formatted, from ``on_response``.

        Every non-empty delta records a velocity sample (wall clock vs
        total stream chars) so the status line can show the generation
        rate — the manifest ``reasoning_model`` flag is NOT trusted for
        arming the meter: a non-empty reasoning delta does that, since
        some providers (glm-5.3 family) bill reasoning without flagging
        the model.
        """
        now = perf_counter()
        if reasoning:
            self._status_reasoning_chars += len(reasoning)
            self._stream_reasoning_buf += reasoning
            self._reasoning_seen = True
        if text:
            self._stream_answer_buf += text
            self._stream_answer_chars += len(text)
            self._answer_staging.append(text, style=STYLE_STREAM_ANSWER)
            self._streamed_during_turn = True
        if reasoning or text:
            self._last_delta_perf = now
            self._velocity_samples.append((now, self._stream_total_chars()))
            self._redraw_activity()

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Print the answer, then the compact ⚡ footer (or nothing if quiet).

        Raw mode writes ``text`` verbatim to raw stdout — no Rule header, no
        Markdown re-render — keeping redirected/piped output machine-readable;
        the ⚡ footer prints on the transport console (stderr in raw mode).

        An approval that reaches the response un-decided is a hard stop: a
        prompt run has no interactive loop to answer it, so the request id
        is stashed (``approval_unresolved``) and nothing is written to
        stdout — the caller exits non-zero rather than emitting an
        interactive panel (or the bare approval text) into redirected
        output.

        A model yield (the ``yield`` terminal action — the model asked the
        operator a question instead of finishing) is the same contract: the
        question is stashed (``yield_unresolved``), printed to stderr only,
        and stdout gets nothing — the caller exits non-zero instead of
        writing the question into redirected output as if it were the
        requested result.
        """
        self.stop_activity()
        self._answer_staging = Text()
        self._stream_answer_buf = ""
        self._stream_reasoning_buf = ""
        approval = (data or {}).get("approval")
        if isinstance(approval, dict) and approval.get("request_id"):
            self._render_unresolved_approval(approval)
            return
        # A model yield (ask-the-operator question) is an unresolved turn, not
        # an answer. The host surfaces it structurally on
        # ``data["yielded"]`` (and ``HostCommandResult.yielded_question``),
        # with the yield context, resume mode, and remediation lines beside it.
        yield_question = str((data or {}).get("yielded") or "")
        if yield_question:
            self._render_yield_question(yield_question, dict(data or {}))
            return
        if text:
            if self.raw:
                sys.stdout.write(text)
                if not text.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                self._render_response_text(text)
        # The footer is activity output: in raw mode it prints on the
        # transport console (stderr) so stdout stays machine-readable.
        if not self.quiet:
            self._print_footer()

    def _render_yield_question(self, yield_question: str, data: dict[str, Any]) -> None:
        """Stash + render a model yield (stdout gets nothing).

        The yield is a question the operator must answer, not a completed
        turn — so it gets a dedicated panel: the question first (that is
        what the operator reads), then why the turn stopped
        (``data["yield_context"]``) and where a reply resumes
        (``data["yield_mode_id"]``), with the rendered remediation command
        lines (``data["remediation"]``) last so the operator can act on
        stalled work without re-deriving the command. Fields the runtime did
        not supply (an in-place model ask carries no context or target)
        render nothing rather than an empty label.
        """
        self.yield_unresolved = True
        self.yield_question = yield_question
        body: list[RenderableType] = [Markdown(yield_question)]
        context = str(data.get("yield_context") or "").strip()
        target_mode = str(data.get("yield_mode_id") or "").strip()
        remediation = [str(x).strip() for x in data.get("remediation") or () if str(x).strip()]
        meta: list[str] = []
        if context:
            meta.append(f"Why: {context}")
        if target_mode:
            meta.append(f"Reply resumes in mode: {target_mode}")
        if remediation:
            meta.append("Remediation: " + ", ".join(f"`{x}`" for x in remediation))
        if meta:
            body.append(Text())
            body.append(Text("\n".join(meta), style=STYLE_TEXT_MUTED))
        self.console.print(
            Panel(
                Group(*body),
                title="Asked you a question (turn not finished)",
                title_align="left",
                border_style=STYLE_ROW_TRANSITION,
                padding=(0, 1),
            )
        )
        if not self.console.is_terminal:
            Console(stderr=True).print(
                "[red]The model asked a question instead of producing a "
                "result. A one-shot prompt run cannot answer it. Re-run "
                "with the missing information in the prompt.[/red]"
            )

    def _render_response_text(self, text: str) -> None:
        """Render the final answer: themed Markdown under a turn/model rule.

        The rule title carries the global thread turn index when known —
        the same number stamped on the thread entry, so it can be replayed
        with ``lisa last N`` or inspected with ``lisa debug N`` without
        counting from the top.
        """
        rule_title = "Response"
        if self._status_current_model:
            rule_title = f"Response · {self._status_current_model}"
        if self._turn_number >= 0:
            rule_title += f" · turn {self._turn_number}"
        self.console.print(Rule(title=rule_title, style=STYLE_BORDER_RESPONSE))
        self.console.print(Markdown(text, code_theme="github-dark", hyperlinks=True))

    def _render_unresolved_approval(self, approval: dict[str, Any]) -> None:
        """Capture a pending approval and surface the decision path.

        A tty console renders the approval panel (the hints point at
        `lisa approve <id>`); a piped/redirected console
        fails fast on stderr instead of printing a panel the operator can
        never answer.
        """
        from hosts.cli.render import render_approval_panel

        self.pending_approval = approval
        self.approval_unresolved = True
        if self.console.is_terminal:
            render_approval_panel(approval, console=self.console)
            return
        # Keep each decision hint short so a 120-col console never wraps
        # inside a copyable command.
        request_id = str(approval.get("request_id", ""))
        target = str(approval.get("target_id", "action"))
        err_console = Console(stderr=True, width=200)
        err_console.print(
            f"[red]Approval required for {target} (request {request_id}).\n"
            f"Decide without a session: lisa approve {request_id} "
            f"or lisa deny {request_id}\n"
            f"(add --remember <prefix> to approve-and-persist), "
            f"then re-run the original command.[/red]"
        )

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """No-op: the debug context panel was removed with the detail tiers.

        Kept to satisfy the ``TransportCallbacks`` protocol; the runtime only
        invokes it when debug mode is active, which the CLI no longer exposes.
        """
        return

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Stop the activity view and render the error panel.

        Non-tty event lines are already in scrollback. Any staged partial
        answer flushes first so partial work isn't silently lost before the
        error panel shows.
        """
        self.stop_thinking(flush_transcript=True, flush_stream=True)
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        # Escape Rich markup in the error body to avoid MarkupError
        # when the message contains paths like /home/adam/Code/lisa
        from rich.markup import escape

        self.console.print(Panel(escape(body), title="Error", border_style=STYLE_BORDER_ERROR))

    # ── Internals ───────────────────────────────────────────────────

    def _capture_turn_number(self, data: dict[str, Any]) -> None:
        """Capture the host's global thread turn index from a stage payload.

        Rendered on the Response rule, the ⚡ footer, and the in-flight
        status line. Only a non-negative int replaces the sentinel; a
        genuine ``turn 0`` (fresh thread) is valid and renders.
        """
        turn_raw = data.get("turn_index")
        if isinstance(turn_raw, int) and turn_raw >= 0:
            self._turn_number = turn_raw

    def _capture_proxy_route(self, data: dict[str, Any]) -> None:
        """Capture the provider's proxy routing from a stage payload.

        ``proxy_route.via`` is ``"proxy"`` (requests flow through the named
        proxy pack, e.g. headroom) or ``"direct"``. A payload without the
        key clears the chip: transports must not keep showing a stale hop
        when the runtime stops stamping routing info.
        """
        route = data.get("proxy_route")
        if not isinstance(route, dict):
            self._status_proxy_id = ""
            self._status_proxy_via = ""
            return
        via = str(route.get("via") or "")
        self._status_proxy_via = via if via in ("proxy", "direct") else ""
        proxy_id = route.get("proxy_id") if via == "proxy" else route.get("bypassed_proxy_id")
        self._status_proxy_id = str(proxy_id or "")

    def _handle_model_stage(self, label: str, data: dict[str, Any]) -> None:
        """Route model_call* stage events.

        ``model_call`` captures the host's global thread turn index, tracks
        the active model/mode and counts the call;
        ``model_call_complete`` accumulates duration + tokens for the
        footer; ``model_call_failed`` prints the ✗ row immediately.
        """
        if label == "model_call":
            self._capture_turn_number(data)
            model_id = data.get("model_id")
            if model_id not in (None, "", 0):
                provider = data.get("provider")
                self._status_current_model = str(model_id)
                self._status_model_provider = str(provider) if provider not in (None, "") else ""
            self._capture_proxy_route(data)
            self._model_call_count += 1
            self._active_reasoning_model = bool(data.get("reasoning_model"))
            # A new model call resets the stall clock: pre-call silence is
            # transport warmup, not a stalled stream.
            self._last_delta_perf = 0.0
            with contextlib.suppress(TypeError, ValueError):
                iter_raw = data.get("_state.iteration") or data.get("iteration")
                if iter_raw not in (None, "", 0):
                    self._iteration_current = int(iter_raw)
            max_iter_raw = data.get("max_iterations")
            if isinstance(max_iter_raw, int) and max_iter_raw > 0:
                self._iteration_max = max_iter_raw
            if data.get("mode_id"):
                self._current_mode = str(data["mode_id"])
            return
        if label == "model_call_complete":
            d_raw = data.get("duration_ms")
            if d_raw not in (None, "", 0):
                with contextlib.suppress(TypeError, ValueError):
                    self._iteration_durations.append(float(d_raw))
            self._accumulate_call_usage(data)
            return
        if label == "model_call_failed":
            self._print_event_line(
                self._render_model_call(data, status_glyph="✗"), style=STYLE_ROW_ERROR
            )

    def _handle_tool_stage(self, label: str, data: dict[str, Any]) -> None:
        """Route tool_* stage events to the prompt-mode scrollback lines.

        ``tool_dispatch`` adds a running row to the interactive transcript;
        ``tool_output`` adds streamed lines as they arrive; and
        ``tool_dispatch_complete`` updates that row with status and duration.
        Non-tty output still prints completed rows to scrollback.
        """
        if label == "tool_dispatch":
            self._tool_call_count += 1
            # Track tool call hierarchy for nested visualization
            parent_idx = len(self._tool_call_stack) - 1 if self._tool_call_stack else -1
            tool_name = str(data.get("tool") or "")
            tool_kind = str(data.get("tool_kind") or "")
            self._tool_call_stack.append((tool_name, tool_kind, parent_idx))
            self._last_tool_depth = len(self._tool_call_stack) - 1
            # Update activity label with hierarchy depth indicator
            depth = len(self._tool_call_stack) - 1
            depth_indicator = "  " * depth + "└─ " if depth > 0 else ""
            self._active_label = f"{depth_indicator}{self._render_tool_dispatch(data).plain}"
            self._last_tool_data = data
            self._last_tool_streaming = bool(data.get("stream_output"))
            self._last_tool_output_lines = 0
            if self._screen_mode:
                self._print_event_line(self._render_tool_dispatch(data, status_glyph="…"))
                self._screen_tool_contexts.append(
                    (len(self._activity_rows) - 1, data, self._last_tool_depth, 0)
                )
            if self.tool_detail >= _TOOL_DETAIL_FULL:
                self._print_detail_arguments(data)
            return
        if label == "tool_output":
            self._handle_tool_output(data)
            return
        if label == "tool_dispatch_complete":
            self._print_completion_row(data)
            return

    def _handle_tool_output(self, data: dict[str, Any]) -> None:
        """Route streamed tool lines to the tty transcript or plain scrollback."""
        text = str(data.get("text") or "")
        stream_label = str(data.get("stream") or "stdout")
        if not text or self.quiet:
            return
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                continue
            self._last_tool_output_lines += 1
            if self._screen_mode and self._screen_tool_contexts:
                row_index, tool_data, depth, line_count = self._screen_tool_contexts[-1]
                self._screen_tool_contexts[-1] = (
                    row_index,
                    tool_data,
                    depth,
                    line_count + 1,
                )
            self._print_stream_line(line, stream=stream_label)

    def _print_completion_row(self, data: dict[str, Any]) -> None:
        """Print the one completed tool row (+ summary + detail lines)."""
        screen_row_index: int | None = None
        if self._screen_mode and self._screen_tool_contexts:
            context = self._screen_tool_contexts.pop()
            (
                screen_row_index,
                self._last_tool_data,
                self._last_tool_depth,
                self._last_tool_output_lines,
            ) = context
            if screen_row_index < 0:
                screen_row_index = None
        ok = data.get("ok")
        status = "ok" if ok is True else ("error" if ok is False else "")
        if status == "ok":
            self._tool_ok_count += 1
        elif status == "error":
            self._tool_error_count += 1
        # Pop from tool call stack for hierarchy tracking
        if self._tool_call_stack:
            self._tool_call_stack.pop()
        glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
        row_data = self._last_tool_data or data
        summary = self._result_shape_summary(row_data, data)
        row = self._render_tool_dispatch(
            row_data, status_glyph=glyph, complete=data, summary=summary
        )
        if screen_row_index is not None:
            self._replace_activity_row(screen_row_index, row)
        else:
            self._print_event_line(row)
        if self.tool_detail >= _TOOL_DETAIL_SUMMARY or status == "error":
            self._print_detail_lines(row_data, data)
        self._active_label = ""
        self._last_tool_data = None
        self._last_tool_streaming = False
        self._last_tool_output_lines = 0

    def _handle_fork_stage(self, label: str, data: dict[str, Any]) -> None:
        """Route fork_* stage events to the prompt-mode scrollback lines.

        ``fork_start`` seeds the child table and prints the header;
        ``fork_child_start`` records the child (activity label only, no
        print); ``fork_child_complete`` records the outcome and, when every
        child is done, prints the combined per-child table; ``fork_complete``
        flushes any remaining table.
        """
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._print_event_line(self._render_fork_stage())
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
                self._active_label = f"fork #{idx} {data.get('mode_id', '')}".strip()
            return
        if label == "fork_child_complete":
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
            return
        if label == "fork_complete":
            if self._fork_children:
                self._print_event_line(self._render_fork_stage())
                self._fork_children = {}
            self._fork_total = 0

    def _result_shape_summary(self, row_data: dict[str, Any], complete: dict[str, Any]) -> str:
        """One-line shape of what the tool returned (``⎿ 14 lines``).

        Prefers the streamed-line count observed this dispatch (tools that
        declare ``stream_output``), falls back to the kernel-provided
        ``result_chars`` (true output size) or a compact char count of the
        ``result_preview`` snippet, else "". The summary rides inline on the
        completion row — muted, so it reads as metadata rather than content.
        """
        if row_data.get("stream_output") and self._last_tool_output_lines > 0:
            n = self._last_tool_output_lines
            return f"⎿ {n} line{'s' if n != 1 else ''}"
        with contextlib.suppress(TypeError, ValueError):
            chars = int(complete.get("result_chars") or 0)
            if chars > 0:
                return f"⎿ {_format_tokens(chars)} chars"
        preview = str(complete.get("result_preview") or "").strip()
        if preview and preview != "Done":
            return f"⎿ {_format_tokens(len(preview))} chars"
        return ""

    def _print_detail_arguments(self, data: dict[str, Any]) -> None:
        """Detail level 2: print the full call arguments under the row."""
        arguments = data.get("arguments")
        if not isinstance(arguments, dict) or not arguments:
            return
        try:
            import json

            rendered = json.dumps(arguments, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return
        if self._screen_mode:
            self._print_event_line(Text("    Arguments", style=STYLE_TEXT_MUTED))
        for line in rendered.splitlines()[:_DETAIL_ARGUMENT_MAX_LINES]:
            prefix = "    │ " if self._screen_mode else "    "
            self._print_event_line(Text(f"{prefix}{line}", style=STYLE_ARG_PREVIEW))

    def _print_detail_lines(self, row_data: dict[str, Any], complete: dict[str, Any]) -> None:
        """Print the result preview for verbose tools or failed calls.

        The kernel sends ``result_preview`` on every completion: a compact
        payload-first summary for JSON-object outputs (content quoted
        first, then short scalars), else the first 200 chars of raw text.
        Failed calls show the first line even at the default detail level;
        ``-v`` also shows the first line and ``-vv`` the full snippet.
        """
        preview = str(complete.get("result_preview") or "").strip()
        if preview and preview != "Done":
            lines = preview.splitlines() or [preview]
            shown = lines if self.tool_detail >= _TOOL_DETAIL_FULL else lines[:_TOOL_DETAIL_SUMMARY]
            if self._screen_mode:
                self._print_event_line(Text("    Result", style=STYLE_TEXT_MUTED))
            for line in shown:
                prefix = "    │ " if self._screen_mode else "    ⎿ "
                self._print_event_line(Text(f"{prefix}{line}", style=STYLE_OUTPUT_REGION))

    def _print_event_line(self, text: Text | str, *, style: str | None = None) -> None:
        """Print one event line or append it to the interactive transcript.

        ``text`` may be a pre-styled ``Text`` (preferred — spans carry their
        own colors and markup is never parsed, so raw tool arguments can
        never raise ``MarkupError``) or a legacy plain string styled whole.
        """
        if self._screen_mode:
            row = Text()
            if isinstance(text, Text):
                row.append(text)
            else:
                row.append(text, style=style or STYLE_TEXT_MUTED)
            self._append_activity_row(row)
            self._redraw_activity(refresh=True)
            return
        self.stop_activity()
        if isinstance(text, Text):
            line = Text("  ")
            line.append(text)
            self.console.print(line)
        else:
            self.console.print(f"  {text}", style=style or STYLE_TEXT_MUTED)
        self.start_activity()

    def _print_stream_line(self, line: str, *, stream: str = "stdout") -> None:
        """Add one streamed tool-output line to the screen or scrollback."""
        if stream == "stderr" and not line.startswith("[stderr]"):
            line = f"[stderr] {line}"
        prefix = "    │ " if self._screen_mode else "  │ "
        row = _OutputLine(f"{prefix}{line}", style=STYLE_OUTPUT_REGION)
        if self._screen_mode:
            self._append_activity_row(row)
            self._redraw_activity()
        else:
            self._print_event_line(row)

    def _redraw_activity(self, *, refresh: bool = False) -> None:
        if self._activity is not None:
            with contextlib.suppress(Exception):
                self._activity.update(self._activity_renderable(), refresh=refresh)

    # ── Bounded stream box (thinking/answer tail window) ────────────

    def _stream_box_active(self) -> bool:
        """Whether the bounded stream box should render this turn.

        Gated on: tty, not quiet, an enabled streaming mode, a wide-enough
        console, and at least one model_call in a mode whose box content
        could still arrive. The box lives inside the transient Live, so it
        never touches scrollback.
        """
        if self.quiet or not self.console.is_terminal:
            return False
        if self.streaming_mode not in {"text", "hybrid"}:
            return False
        width = self.console.size.width or 0
        return width >= _STREAM_BOX_MIN_WIDTH and self._model_call_count > 0

    def _stream_box_lines(self, *, max_lines: int = _STREAM_BOX_MAX_LINES) -> list[str]:
        """Return the tail window of the active stream as wrapped lines.

        Picks the content source from the streaming mode and what has
        actually arrived: ``text`` mode → the answer stream; ``hybrid`` →
        the reasoning stream, falling back to nothing (the status-line
        thinking char counter carries that case). Lines wrap to the box
        width; the last ``_STREAM_BOX_MAX_LINES`` are returned, oldest
        first. Long single-line chunks (minified JSON, URLs) hard-wrap on
        their own — a box line never exceeds the console width.
        """
        if max_lines <= 0:
            return []
        if self.streaming_mode == "text":
            buf = self._stream_answer_buf
        else:
            buf = self._stream_reasoning_buf
        if not buf:
            return []
        width = max(20, (self.console.size.width or self._ACTIVITY_WIDTH) - 4)
        # Walk from the tail: keep only as much text as can fill the box.
        # Rough char budget before wrapping (wraps can only shrink visible
        # chars per line), with slack for hard-wrapped long words.
        budget = width * max_lines
        tail = buf[-budget:] if len(buf) > budget else buf
        lines: list[str] = []
        for para in tail.replace("\r\n", "\n").split("\n"):
            remaining = para
            if not remaining:
                lines.append("")
                continue
            while len(remaining) > width:
                lines.append(remaining[:width])
                remaining = remaining[width:]
            lines.append(remaining)
        return lines[-max_lines:]

    def _stream_box_renderable(
        self, *, max_lines: int = _STREAM_BOX_MAX_LINES
    ) -> RenderableType | None:
        """Render the active stream box as a bordered, dim tail window.

        The content is deliberately rendered as *plain dim text*, not the
        styled markdown of the final response — visually signalling "this
        is live preview, not the formatted answer." Under
        ``_STREAM_BOX_BORDER_CHARS`` of content the box renders bare (no
        border) to stay light for short generations.
        """
        lines = self._stream_box_lines(max_lines=max_lines)
        if not lines:
            return None
        body = Text("\n".join(lines), style=STYLE_STREAM_BOX)
        if self._stream_total_chars() < _STREAM_BOX_BORDER_CHARS:
            return body
        title = "streaming" if self.streaming_mode == "text" else "thinking"
        velocity = self._stream_velocity()
        if velocity > 0:
            title = f"{title} · ~{_format_velocity(velocity)}"
        else:
            title = f"{title} · {_format_tokens(self._stream_total_chars())}"
        return Panel(
            body,
            title=title,
            title_align="left",
            border_style=STYLE_STREAM_BOX_BORDER,
            padding=(0, 1),
        )

    def _stream_total_chars(self) -> int:
        """Total streamed chars this turn (answer or reasoning, by mode)."""
        if self.streaming_mode == "text":
            return self._stream_answer_chars
        return self._status_reasoning_chars

    def _stream_velocity(self) -> float:
        """Rolling stream rate (chars/s) over the recent-sample window.

        Samples older than ``_VELOCITY_WINDOW_S`` fall out of the
        estimate; below the min-span/min-chars thresholds it returns 0
        (the caller omits the figure rather than showing noise). A
        stalled stream naturally decays to 0 as the window slides —
        which is exactly the signal the operator wants.
        """
        now = perf_counter()
        window_start = now - _VELOCITY_WINDOW_S
        samples = [s for s in self._velocity_samples if s[0] >= window_start]
        self._velocity_samples = samples
        if len(samples) < _VELOCITY_MIN_SAMPLES:
            return 0.0
        first_t, first_c = samples[0]
        last_t, last_c = samples[-1]
        span = last_t - first_t
        if span < _VELOCITY_MIN_SPAN_S:
            return 0.0
        delta = last_c - first_c
        if delta < _VELOCITY_MIN_CHARS:
            return 0.0
        return delta / span

    def _activity_renderable(self) -> RenderableType:
        """Build the full-screen tty layout or compact non-tty activity row.

        The tty layout divides the screen into a prompt header, a tailing
        event transcript, and a thinking/status footer. Non-tty callers keep
        the compact transient activity row used by the legacy transport.
        """
        if self._screen_mode:
            return self._screen_layout()
        return self

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self._activity_frame()

    def _screen_layout(self) -> Layout:
        """Keep prompt and thinking pinned while the transcript scrolls."""
        height = self.console.size.height or 24
        width = self.console.size.width or self._ACTIVITY_WIDTH
        prompt_header = self._prompt_header(
            max_lines=min(_PROMPT_MAX_LINES, max(1, height // 3 - 1))
        )
        header_rows = len(prompt_header.renderables)
        footer_rows = min(11, max(2, height // 3))
        if header_rows + footer_rows >= height:
            footer_rows = max(1, height - header_rows - 1)
        history_rows = max(1, height - header_rows - footer_rows)
        self._activity_view_rows = history_rows

        footer_parts: list[RenderableType] = []
        stream_lines = footer_rows - 3
        if stream_lines > 0:
            box = self._stream_box_renderable(max_lines=stream_lines)
            if box is not None:
                footer_parts.append(box)
        footer_parts.append(self._status_line(width))

        layout = Layout()
        layout.split_column(
            Layout(
                prompt_header,
                name="prompt",
                size=header_rows,
            ),
            Layout(
                self._activity_history(max_rows=history_rows),
                name="activity",
                ratio=1,
                minimum_size=1,
            ),
            Layout(
                Align(Group(*footer_parts), vertical="bottom"),
                name="thinking",
                size=footer_rows,
            ),
        )
        return layout

    def _prompt_header(self, *, max_lines: int) -> Group:
        """Render the wrapped prompt (capped) above a separator rule.

        The header is sized to its content: one row per prompt line plus
        the rule, so a one-line prompt takes two rows.
        """
        width = max(1, (self.console.size.width or self._ACTIVITY_WIDTH) - 2)
        wrapped: list[Text] = list(
            Text(self._prompt_message).wrap(self.console, width, overflow="fold")
        ) or [Text("")]
        limit = max(1, max_lines)
        truncated = len(wrapped) > limit
        if truncated:
            wrapped = wrapped[:limit]
        rows: list[RenderableType] = []
        for index, line in enumerate(wrapped):
            row = Text("λ " if index == 0 else "  ", style="bold " + STYLE_ROW_TOOL)
            if truncated and index == limit - 1:
                row.append("… prompt continues", style=STYLE_TEXT_MUTED)
            else:
                row.append(line.plain, style="bold " + STYLE_TEXT_PRIMARY)
            rows.append(row)
        rows.append(Rule(style=STYLE_ROW_TOOL))
        return Group(*rows)

    def _wrapped_row(self, row: Text, width: int) -> list[Text]:
        """Wrap one transcript row, cached per row object and width."""
        cached = self._wrap_cache.get(id(row))
        if cached is not None and cached[0] is row and cached[1] == width:
            return cached[2]
        lines = list(row.wrap(self.console, width, overflow="fold"))
        if len(self._wrap_cache) > 2 * len(self._activity_rows) + 64:
            live = {id(r) for r in self._activity_rows}
            self._wrap_cache = {k: v for k, v in self._wrap_cache.items() if k in live}
        self._wrap_cache[id(row)] = (row, width, lines)
        return lines

    def _activity_lines(
        self, *, width: int, output_budget: int | None, rows: list[Text] | None = None
    ) -> list[Text]:
        """Wrap the transcript; collapse streamed-output blocks to ``output_budget``."""
        lines: list[Text] = []
        output_run: list[Text] = []

        def flush_output_run() -> None:
            # Collapse each block of streamed output to its tail so it never
            # crowds the tool rows out of the middle region.
            if output_budget is not None and len(output_run) > output_budget:
                hidden = len(output_run) - output_budget + 1
                if output_budget > 1:
                    lines.append(
                        Text(f"    │ … {hidden} earlier output lines", style=STYLE_TEXT_MUTED)
                    )
                    lines.extend(output_run[-(output_budget - 1) :])
                else:
                    lines.append(output_run[-1])
            else:
                lines.extend(output_run)
            output_run.clear()

        for row in self._activity_rows if rows is None else rows:
            wrapped = self._wrapped_row(row, width)
            if isinstance(row, _OutputLine):
                output_run.extend(wrapped)
                continue
            flush_output_run()
            lines.extend(wrapped)
        flush_output_run()
        return lines

    def _activity_history(self, *, max_rows: int) -> RenderableType:
        """Render the transcript tail, or the scrolled-back window."""
        max_rows = max(1, max_rows)
        width = max(1, (self.console.size.width or self._ACTIVITY_WIDTH) - 2)
        if self._activity_scroll_top is not None:
            return self._scrolled_activity_history(max_rows=max_rows, width=width)
        # One extra row is reserved for the "earlier lines" marker.
        output_budget = max(1, max_rows - _ACTIVITY_MIN_TOOL_ROWS - 1)
        lines = self._activity_lines(width=width, output_budget=output_budget)
        omitted = self._activity_history_elided
        if len(lines) > max_rows:
            omitted += len(lines) - max_rows
            lines = lines[-max_rows:]
        if omitted:
            hint = " · ↑ to scroll back" if self._key_reader is not None else ""
            marker = Text(f"… {omitted} earlier lines{hint}", style=STYLE_TEXT_MUTED)
            visible_tail = lines[-(max_rows - 1) :] if max_rows > 1 else []
            lines = [marker, *visible_tail]
        if not lines:
            return Text("Waiting for tool activity…", style=STYLE_TEXT_MUTED)
        return Group(*lines)

    def _scrolled_activity_history(self, *, max_rows: int, width: int) -> RenderableType:
        """Render the full (uncollapsed) transcript from the scroll position."""
        lines = self._activity_lines(width=width, output_budget=None)
        top = max(0, min(self._activity_scroll_top or 0, len(lines) - max_rows))
        self._activity_scroll_top = top
        window = lines[top : top + max_rows]
        if max_rows >= _SCROLL_MARKER_MIN_ROWS:
            above = top + (self._activity_history_elided if top == 0 else 0)
            if above:
                label = (
                    f"↑ {top} earlier lines"
                    if top
                    else f"… {self._activity_history_elided} oldest lines dropped"
                )
                window[0] = Text(label, style=STYLE_TEXT_MUTED)
            below = len(lines) - (top + max_rows)
            hint = "↑/↓ PgUp/PgDn scroll · End to follow"
            label = f"↓ {below} newer lines · {hint}" if below > 0 else f"— end · {hint}"
            if len(window) == max_rows:
                window[-1] = Text(label, style=STYLE_TEXT_MUTED)
            else:
                window.append(Text(label, style=STYLE_TEXT_MUTED))
        return Group(*window)

    def _scroll_activity(self, action: str) -> None:
        """Apply one scroll action to the middle transcript and redraw."""
        rows = max(1, self._activity_view_rows)
        width = max(1, (self.console.size.width or self._ACTIVITY_WIDTH) - 2)
        total = len(self._activity_lines(width=width, output_budget=None))
        bottom = max(0, total - rows)
        top = bottom if self._activity_scroll_top is None else self._activity_scroll_top
        page = max(1, rows - 2)
        step = {"up": -1, "down": 1, "pgup": -page, "pgdn": page}.get(action)
        if action == "home":
            new_top: int | None = 0
        elif action == "end" or step is None:
            new_top = None
        else:
            new_top = max(0, min(bottom, top + step))
            if new_top >= bottom and action in ("down", "pgdn"):
                new_top = None
        if new_top is not None and bottom == 0:
            new_top = None
        self._activity_scroll_top = new_top
        self._redraw_activity(refresh=True)

    def _append_activity_row(self, row: Text) -> None:
        self._activity_rows.append(row)
        excess = len(self._activity_rows) - _ACTIVITY_HISTORY_MAX_ROWS
        if excess > 0:
            del self._activity_rows[:excess]
            self._activity_history_elided += excess
            self._scrollback_flushed_rows = max(0, self._scrollback_flushed_rows - excess)
            self._screen_tool_contexts = [
                (index - excess if index >= excess else -1, data, depth, lines)
                for index, data, depth, lines in self._screen_tool_contexts
            ]

    def _replace_activity_row(self, index: int, row: Text) -> None:
        if 0 <= index < len(self._activity_rows):
            self._activity_rows[index] = row
            self._redraw_activity(refresh=True)

    def _activity_frame(self) -> RenderableType:
        """Build one frame of the activity display from current state."""
        transport = self
        width = transport.console.size.width or transport._ACTIVITY_WIDTH
        line = transport._status_line(width)
        boxes: list[RenderableType] = []
        box = transport._stream_box_renderable() if transport._stream_box_active() else None
        if box is not None:
            boxes.append(box)
        if len(boxes) == 1:
            return Group(line, boxes[0])
        if boxes:
            return Group(line, *boxes)
        return line

    def _status_line(self, width: int) -> _TrailingSpinner:
        """Compose the one-line status row with a trailing spinner.

        Returns a dynamic spinner renderable: its ``line_fn``/``spinner_fn``
        pull from live transport state each frame, so elapsed and the
        thinking meter tick during quiet generations.
        """
        return _TrailingSpinner(
            lambda: self._compose_status_body(width), self._activity_spinner, STYLE_GRAPHIC
        )

    def _compose_status_body(self, width: int) -> Text:
        """Build the status-line body from current state (one frame)."""
        parts: list[Text] = self._status_leading_parts()
        self._append_meter_parts(parts)
        self._append_stall_part(parts)
        if self._phase_label and not parts:
            parts.append(Text(self._phase_label, style=STYLE_STAGE))
        if self._iteration_max:
            parts.append(
                Text(f"iter {self._iteration_current}/{self._iteration_max}", style=STYLE_STAGE)
            )
        parts.append(Text(f"models {self._model_call_count}", style=STYLE_STAGE))
        parts.append(self._tools_counter_part())
        parts.append(Text(_format_elapsed(self._elapsed_seconds()), style=STYLE_STAGE))
        body = Text()
        for i, part in enumerate(parts):
            if i:
                body.append(" · ")
            body.append(part)
        body.truncate(max(0, width - 4))
        return body

    def _status_leading_parts(self) -> list[Text]:
        """Leading status segments: turn, mode, model, proxy hop, in-flight label."""
        parts: list[Text] = []
        if self._turn_number >= 0:
            parts.append(Text(f"turn {self._turn_number}", style=STYLE_ROW_MODEL))
        if self._current_mode:
            parts.append(Text(f"[{self._current_mode}]", style=STYLE_ROW_TRANSITION))
        if self._status_current_model:
            model = self._status_current_model
            if self._status_model_provider:
                model = f"{self._status_model_provider}:{model}"
            parts.append(Text(model, style=STYLE_STAGE))
        chip = self._proxy_chip_part()
        if chip is not None:
            parts.append(chip)
        if self._active_label:
            parts.append(Text(_truncate(self._active_label, 48), style=STYLE_STAGE))
        return parts

    def _proxy_chip_part(self) -> Text | None:
        """Proxy-pack routing chip, or ``None`` when the host stamped nothing.

        ``⤳ <proxy_id>`` when the provider routes through a ready proxy
        (e.g. headroom); nothing when unstamped or direct with no proxy
        declared. The chip keeps the model segment honest about where
        requests actually flow.
        """
        if not self._status_proxy_via:
            return None
        if self._status_proxy_via == "proxy" and self._status_proxy_id:
            proxy_id = self._status_proxy_id
            prefix, separator, suffix = proxy_id.partition(".")
            if separator and prefix == suffix:
                proxy_id = prefix
            return Text(f"⤳ {_truncate(proxy_id, 24)}", style=STYLE_STAGE)
        if self._status_proxy_via == "direct" and self._status_proxy_id:
            return Text(
                f"{_truncate(self._status_proxy_id, 24)} ↓ direct", style=STYLE_ROW_TRANSITION
            )
        return None

    def _append_meter_parts(self, parts: list[Text]) -> None:
        """Append the thinking meter segment when armed.

        The thinking meter arms on observed reasoning deltas — not the
        manifest flag (glm-5.3-family bills reasoning without flagging
        the model) — so the figure appears the moment reasoning actually
        arrives. The stream velocity lives only in the stream box title.
        """
        if not (self._active_reasoning_model or self._reasoning_seen):
            return
        total = self._status_reasoning_chars or self._stream_total_chars()
        if not total:
            return
        parts.append(Text(f"thinking {_format_tokens(total)}", style=STYLE_STAGE))

    def _no_data_age_s(self) -> float:
        """Seconds since the last delta of the in-flight generation, or -1.

        Returns -1 while no generation is in flight or before the first
        delta has arrived — the caller only reads a stall once streaming
        has *started then stopped*.
        """
        if self._last_delta_perf <= 0.0:
            return -1.0
        return max(0.0, perf_counter() - self._last_delta_perf)

    def _append_stall_part(self, parts: list[Text]) -> None:
        """Append the no-data age once a generation goes quiet.

        Streams regularly pause between chunks; only a pause that has
        crossed ``_STALL_AFTER_S`` reads as a stall. The segment escalates
        gold → red at ``_STALL_ALERT_S`` so a multi-minute upstream hang
        (the 2026-10-06 glm-5.3:cloud mid-stream freeze) is legible as
        *stalled*, not as a frozen frame.
        """
        age = self._no_data_age_s()
        if age < _STALL_AFTER_S:
            return
        style = STYLE_ROW_ERROR if age >= _STALL_ALERT_S else STYLE_ROW_TRANSITION
        # Visual stall indicator: pulsing block + age
        stall_glyph = "▮" if int(age) % 2 == 0 else "▯"
        parts.append(Text(f"{stall_glyph} stalled {age:.0f}s", style=style))

    def _activity_spinner(self) -> str:
        """Pick the spinner for the current phase.

        Active reasoning (observed deltas or the manifest flag) gets a
        visually distinct, calmer spinner so a long think doesn't read as
        a stalled fetch. A stalled stream switches to a slow clock —
        visibly "waiting", not "working".
        """
        if self._no_data_age_s() >= _STALL_AFTER_S:
            return "clock"
        if (self._active_reasoning_model or self._reasoning_seen) and self._model_call_count > 0:
            return "aesthetic"
        return self._thinking_spinner

    def _tools_counter_part(self) -> Text:
        """Tools counter with a budget bar when the kernel stamped a cap."""
        if self._cap_max_tool_calls > 0:
            bar = _budget_bar(self._tool_call_count, self._cap_max_tool_calls)
            bar_style = (
                STYLE_ROW_TRANSITION
                if self._tool_call_count >= _BUDGET_BAR_WARN_RATIO * self._cap_max_tool_calls
                else STYLE_STAGE
            )
            return Text(
                f"tools {bar} {self._tool_call_count}/{self._cap_max_tool_calls}",
                style=bar_style,
            )
        return Text(f"tools {self._tool_call_count}", style=STYLE_STAGE)

    def _elapsed_seconds(self) -> int:
        """Whole seconds elapsed since ``start_thinking`` for this turn."""
        if self._turn_started_perf <= 0:
            return 0
        return int(perf_counter() - self._turn_started_perf)

    def _print_footer(self) -> None:
        """Render the single-line prompt-mode footer.

        ``⚡ 22s · turn 41 · 5 tools · 2 models · 8.4k tok · $0.02`` — one
        line, per-turn only; a ``prompt`` invocation is one process and one
        turn, so the turn/session token split collapses to a single total.
        The ``turn N`` segment is the global thread turn index (the number
        stamped on the thread entry, and the number ``lisa last N`` /
        ``lisa debug N`` address). Each segment carries its own color:
        elapsed gold, turn green, tools blue (red annotation on failures),
        models green, tokens green, cost cyan. Plain text is identical to
        the legacy flat line.
        """
        model_time_ms = round(sum(self._iteration_durations))
        tools_total = self._tool_call_count
        tool_failures = self._tool_error_count
        footer = Text()
        footer.append(f"⚡ {self._elapsed_seconds()}s", style=STYLE_ROW_TRANSITION)
        if self._turn_number >= 0:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            footer.append(f"turn {self._turn_number}", style=STYLE_ROW_MODEL)
        footer.append(" · ", style=STYLE_TEXT_MUTED)
        footer.append(f"{tools_total} tool{'s' if tools_total != 1 else ''}", style=STYLE_ROW_TOOL)
        if tool_failures:
            footer.append(f" ({tools_total - tool_failures} ✓ ", style=STYLE_ROW_SUCCESS)
            footer.append(f"{tool_failures} ✗", style=STYLE_ROW_ERROR)
            footer.append(")", style=STYLE_ROW_SUCCESS)
        footer.append(" · ", style=STYLE_TEXT_MUTED)
        model_part = f"{self._model_call_count} model{'s' if self._model_call_count != 1 else ''}"
        if model_time_ms:
            model_part += f" ({model_time_ms}ms)"
        footer.append(model_part, style=STYLE_ROW_MODEL)
        if self._handoff_count:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            handoff_part = f"{self._handoff_count} handoff{'s' if self._handoff_count != 1 else ''}"
            route = self._current_mode
            if route:
                handoff_part += f" → {route}"
            footer.append(handoff_part, style="steel_blue3")
        if self._status_turn_tokens:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            footer.append(f"{_format_tokens(self._status_turn_tokens)} tok", style="green3")
            # Billed segment: the walk above counts every cache hit at full
            # size, which on a cache-heavy turn reads up to 30x the tokens the
            # provider actually bills. Always show the billed basis beside it
            # when the two differ — that is the number to sanity-check spend
            # against. When the turn had no cache hits the two are identical
            # and the segment would be noise.
            if 0 < self._status_billed_tokens < self._status_turn_tokens:
                footer.append(
                    f" · billed {_format_tokens(self._status_billed_tokens)}",
                    style="cyan",
                )
        cost, authoritative = self._resolve_turn_cost()
        if cost:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            # ``$`` marks a provider-reported charge (the real meter); the
            # ``~`` prefix marks the harness's own estimate from manifest
            # rates, so the two are never mistaken for each other.
            prefix = "$" if authoritative else "~$"
            footer.append(f"{prefix}{cost:g}", style="cyan")
        self._append_proxy_savings(footer)
        self.console.print(footer)

    def _append_proxy_savings(self, footer: Text) -> None:
        """Append the serving proxy's reported savings, when it declared any.

        A proxy in the path (compression, cache steering) reports what it
        saved per call on response headers; the kernel sums them and the
        footer shows the turn's total. The counter is whatever the proxy
        declared, so nothing is rendered unless a known name is present —
        an unrecognised counter still reaches ``lisa tokens`` and any pack
        reading the telemetry log, but the one-line footer only labels the
        fields it can describe without guessing at units.
        """

        saved = self._status_proxy_telemetry.get("tokens_saved")
        if not saved:
            return
        footer.append(" · ", style=STYLE_TEXT_MUTED)
        footer.append(f"saved {_format_tokens(saved)}", style="dark_sea_green4")

    def _accumulate_call_usage(self, data: dict[str, Any]) -> None:
        """Fold one completed model call's usage into the turn totals.

        Three figures accumulate independently, because they answer different
        questions and must not be conflated:

        * ``_status_turn_tokens`` — the *walk*: sum of every call's
          ``total_tokens``. Grows quadratically across an agentic turn
          because each iteration re-sends the whole conversation. Useful for
          "how much context did this turn chew through", misleading as a bill.
        * ``_status_billed_tokens`` — uncached prompt + completion, i.e. the
          tokens the provider charges at full rate.
        * ``_status_cost_usd`` / ``_status_provider_cost_usd`` — the kernel's
          canonical estimate and the provider's own reported charge.
        """

        total_raw = data.get("_state.total_tokens") or data.get("total_tokens")
        if total_raw not in (None, "", 0):
            with contextlib.suppress(TypeError, ValueError):
                self._status_turn_tokens += int(total_raw)
        prompt_now = 0
        with contextlib.suppress(TypeError, ValueError):
            prompt_now = int(data.get("prompt_tokens") or 0)
            self._status_prompt_tokens += prompt_now
        completion_now = 0
        with contextlib.suppress(TypeError, ValueError):
            completion_now = int(data.get("completion_tokens") or 0)
            self._status_completion_tokens += completion_now
        cached_now = 0
        with contextlib.suppress(TypeError, ValueError):
            cached_now = int(data.get("cached_tokens") or 0)
            self._status_cached_tokens += cached_now
        self._status_billed_tokens += max(0, prompt_now - cached_now) + completion_now
        with contextlib.suppress(TypeError, ValueError):
            self._status_cost_usd += float(data.get("cost_usd") or 0.0)
        provider_now = 0.0
        with contextlib.suppress(TypeError, ValueError):
            provider_now = float(data.get("provider_cost_usd") or 0.0)
        if provider_now > 0.0:
            self._status_provider_cost_usd += provider_now
            self._status_provider_cost_calls += 1
        # Proxy counters are summed without interpreting them: the key set is
        # whatever the serving proxy declared, and the footer renders only the
        # ones it knows how to label.
        telemetry = data.get("proxy_telemetry")
        if isinstance(telemetry, dict):
            for key, raw in telemetry.items():
                with contextlib.suppress(TypeError, ValueError):
                    self._status_proxy_telemetry[str(key)] = self._status_proxy_telemetry.get(
                        str(key), 0
                    ) + max(0, int(raw))

    def _resolve_turn_cost(self) -> tuple[float, bool]:
        """Return ``(usd, authoritative)`` for the turn's cost segment.

        Three sources, most trustworthy first:

        1. ``_status_provider_cost_usd`` — the provider's own reported
           charge (OpenRouter ``usage.cost``). Authoritative: it already
           reflects the provider's real cache read/write rates and any
           surcharges, so nothing here can be more accurate.
        2. ``_status_cost_usd`` — the kernel's canonical estimate from the
           manifest's per-token rates. Correct but list-price only.
        3. ``_estimate_turn_cost()`` — this transport pricing the counts it
           accumulated itself. Kept for runtimes whose stage payload predates
           the cost fields, so an older kernel still yields a figure.

        A provider charge wins only when *every* call in the turn reported
        one; a partial sum would understate the bill, so mixed turns fall
        back to the complete canonical estimate.
        """

        calls = self._model_call_count
        provider_total = self._status_provider_cost_usd
        if provider_total > 0.0 and calls > 0 and self._status_provider_cost_calls == calls:
            return round(provider_total, 4), True
        if self._status_cost_usd > 0.0:
            return round(self._status_cost_usd, 4), False
        return self._estimate_turn_cost(), False

    def _estimate_turn_cost(self) -> float:
        """Estimate the turn's USD cost from the manifest per-token prices.

        Uses the split prompt/completion token counts accumulated from
        ``model_call_complete`` events against the per-model prices injected
        by the command path (``model_prices``). Returns 0 when either side
        is missing — a missing manifest must never block the footer.

        When the manifest publishes cache prices (``model_cache_prices``),
        cache-hit prompt tokens are priced at the cache-read rate instead
        of the uncached prompt rate.
        """
        prices = self.model_prices.get(self._status_current_model)
        if not prices:
            return 0.0
        prompt_price, completion_price = prices
        cache_prices = self.model_cache_prices.get(self._status_current_model) or (0.0, 0.0)
        cache_read_price, _cache_write_price = cache_prices
        cached = min(self._status_cached_tokens, self._status_prompt_tokens)
        uncached_prompt = self._status_prompt_tokens - cached
        prompt_read = uncached_prompt * prompt_price + cached * (cache_read_price or prompt_price)
        cost = prompt_read + self._status_completion_tokens * completion_price
        # Sub-cent figures round to zero display; keep one significant digit.
        return round(cost, 4)

    # ── Model/tool row rendering (shared with event lines) ──────────

    def _render_model_call(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> Text:
        """Render a model call as a one-liner: stage, model, iteration, status.

        ``stage: model_call haiku #1 ✓ 340ms``. On completion
        (``status_glyph`` set) the status and duration are appended.

        Returns a styled ``Text``: the label segment is muted, the model id
        is bright, the status is success/error colored. Plain text is
        identical to the legacy string form.
        """
        out = Text("stage: model_call", style=STYLE_TEXT_MUTED)
        model_id = data.get("model_id")
        if model_id not in (None, "", 0):
            out.append(f" {model_id}", style=STYLE_TOOL_NAME)
        iter_raw = data.get("iteration") or data.get("_state.iteration")
        if iter_raw not in (None, "", 0):
            out.append(f" #{iter_raw}", style=STYLE_TEXT_MUTED)
        if status_glyph:
            out.append(f" {status_glyph}", style=STYLE_ROW_ERROR)
        duration = (complete or data).get("duration_ms")
        if status_glyph and duration not in (None, "", 0):
            out.append(f" {round(float(duration))}ms", style=STYLE_TEXT_MUTED)
        return out

    def _render_tool_dispatch(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
        summary: str = "",
    ) -> Text:
        """Render a tool call as a one-liner: glyph, tool, primary arg, status.

        Non-screen output stays in its legacy form. In the interactive
        screen, tool identity, argument, status, and duration are a
        left-to-right scan line rather than opposing edge columns.

        Returns a styled ``Text``: glyph in the tool-kind color, tool name
        bright, argument preview dim, status success/error colored. Plain
        text (non-tty) is identical to the legacy string form — the
        right-alignment pads with spaces, which reads the same either way.
        """
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        if status_glyph and self._screen_mode and kind != "transition":
            return self._render_screen_tool_dispatch(
                data,
                status_glyph=status_glyph,
                complete=complete,
                summary=summary,
            )
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")

        out = Text()
        if kind == "transition":
            self._render_transition_dispatch(out, data, glyph=glyph, tool=tool, complete=complete)
        else:
            self._render_plain_dispatch(out, data, glyph=glyph, tool=tool)

        if status_glyph:
            status_style = STYLE_ROW_ERROR if status_glyph == "✗" else STYLE_ROW_SUCCESS
            duration = (complete or data).get("duration_ms")
            tail = status_glyph
            if duration not in (None, "", 0):
                tail += f" {round(float(duration))}ms"
            self._append_aligned_status(out, tail, status_style, suffix=summary)
            if summary:
                out.append(" ", style=STYLE_TEXT_MUTED)
                out.append(summary, style=STYLE_TEXT_MUTED)
        return out

    def _render_screen_tool_dispatch(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str,
        complete: dict[str, Any] | None,
        summary: str,
    ) -> Text:
        """Render a tool as one left-to-right scan line in the tty transcript."""
        tool = str(data.get("tool") or "")
        status_style = {
            "✓": STYLE_ROW_SUCCESS,
            "✗": STYLE_ROW_ERROR,
            "…": STYLE_STAGE,
        }.get(status_glyph, STYLE_TEXT_MUTED)
        out = Text("  " * self._last_tool_depth)
        out.append(f"{status_glyph} ", style=status_style)
        out.append(tool, style=STYLE_TOOL_NAME)
        arg = self._primary_arg_preview(data)
        if arg:
            out.append(f" · {arg}", style=STYLE_ARG_PREVIEW)
        duration = (complete or data).get("duration_ms")
        if status_glyph == "…":
            out.append(" · running", style=STYLE_TEXT_MUTED)
        elif duration not in (None, "", 0):
            out.append(f" · {round(float(duration))}ms", style=STYLE_TEXT_MUTED)
        if summary:
            out.append(f" · {summary.removeprefix('⎿ ')}", style=STYLE_TEXT_MUTED)
        return out

    def _render_transition_dispatch(
        self,
        out: Text,
        data: dict[str, Any],
        *,
        glyph: str,
        tool: str,
        complete: dict[str, Any] | None = None,
    ) -> None:
        """Render the transition-stage dispatch line (handoff/fork marker).

        The transition target rides the *completion* event (the kernel
        knows where it routed only when the transition landed), so read
        it from ``complete`` falling back to the dispatch payload — the
        same pattern the duration read uses.
        """
        data = {**data, **(complete or {})}
        target = ""
        if data.get("handoff_to"):
            target = f"→ {data['handoff_to']}"
        elif data.get("transition") == "fork":
            n = data.get("fork_children") or 0
            target = f"fork ({n} children)"
        elif data.get("transitioned_to"):
            target = f"→ {data['transitioned_to']}"
        out.append("stage: tool_dispatch ", style=STYLE_TEXT_MUTED)
        out.append(glyph, style=STYLE_ROW_TRANSITION)
        out.append(f" {tool}", style=STYLE_TOOL_NAME)
        if target:
            out.append(f" {target}", style=STYLE_ROW_TRANSITION)

    def _render_plain_dispatch(
        self, out: Text, data: dict[str, Any], *, glyph: str, tool: str
    ) -> None:
        """Render the plain dispatch line with tool-hierarchy indentation."""
        out.append(glyph, style=STYLE_ROW_TOOL)
        # Add hierarchy indentation for nested tool calls
        depth = len(self._tool_call_stack) - 1
        if depth > 0:
            indent = "  " * depth + "└─ "
            out.append(indent, style=STYLE_TEXT_MUTED)
        out.append(f" {tool}", style=STYLE_TOOL_NAME)
        arg = self._primary_arg_preview(data)
        if arg:
            out.append(f" {arg}", style=STYLE_ARG_PREVIEW)

    def _append_aligned_status(self, row: Text, tail: str, style: str, *, suffix: str = "") -> None:
        """Append ``tail`` right-aligned in a fixed status column (tty only).

        Interactive terminals (>= ``_STATUS_ALIGN_MIN_WIDTH``) pad the row so
        the status glyphs line up vertically across consecutive tool rows —
        the Claude-Code "result column" scan. Non-tty consoles (piped, tee,
        tests) keep the legacy single-space compact join so scripted output
        and pinned assertions stay byte-identical. ``suffix`` (the result
        summary that follows the status) reserves its width in the padding
        so a long row never wraps onto a second line.
        """
        width = self.console.size.width or 0
        if not self.console.is_terminal or width < _STATUS_ALIGN_MIN_WIDTH:
            row.append(" ", style=style)
            row.append(tail, style=style)
            return
        reserved = len(tail) + len(suffix) + 4
        pad = max(1, _STATUS_ALIGN_COLUMN - len(tail) - 1)
        pad = max(pad, width - row.cell_len - reserved)
        if row.cell_len + pad + len(tail) + len(suffix) + 1 >= width:
            # Not enough room: keep the compact join rather than wrapping.
            row.append(" ", style=style)
            row.append(tail, style=style)
            return
        row.append(" " * pad)
        row.append(tail, style=style)

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

    def _render_fork_stage(self, summary: tuple[int, int] | None = None) -> Text:
        """Render the fork fan-out as one header line plus one line per child.

        Returns a multi-line styled ``Text``: header in transition gold,
        child rows muted with the mode id bright. Plain text is identical
        to the legacy string form (the tests pin it).
        """
        out = Text()
        if summary is not None:
            ok, failed = summary
            header = f"stage: fork complete ({ok} ok, {failed} failed)"
        else:
            header = f"stage: fork ({self._fork_total} children)"
        out.append(header, style=STYLE_ROW_TRANSITION)
        for idx in sorted(self._fork_children):
            child = self._fork_children[idx]
            mode_id = child.get("mode_id", "?")
            stage_id = child.get("stage_id", "")
            status = child.get("status", "running")
            out.append("\n")
            out.append(f"  #{idx} ", style=STYLE_TEXT_MUTED)
            out.append(mode_id, style=STYLE_TOOL_NAME)
            status_part = f" · {status}"
            if stage_id:
                status_part = f" · {stage_id} · {status}"
            out.append(status_part, style=STYLE_TEXT_MUTED)
            preview = child.get("query_preview", "")
            if preview:
                out.append(f" · {preview}", style=STYLE_ARG_PREVIEW)
            duration_ms = child.get("duration_ms")
            if duration_ms is not None:
                out.append(f" · {duration_ms}ms", style=STYLE_TEXT_MUTED)
        return out


__all__ = ["PromptTransport"]
