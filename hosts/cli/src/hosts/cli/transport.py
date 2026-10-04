"""Prompt-mode Rich transport for the CLI host.

This is the terminal-transport implementation of
:class:`runtime.callbacks.TransportCallbacks` for ``lisa prompt`` — the CLI's
only turn path. A prompt run is input/output in the caller's shell: another
pipeline stage, not a session with a screen. Instead of a persistent Live
transcript it renders:

* a one-line transient activity indicator (spinner + mode + in-flight
  tool label + models/tools/elapsed) via an overwrite-in-place Live; and
* each tool/model/fork row printed to real scrollback as it *completes*,
  so the trajectory survives piping/tee.

Tokens are per-turn: a ``prompt`` invocation is one process and one turn,
so the footer reports a single total (the turn/session split collapses).

Non-tty consoles (``console.is_terminal`` False) skip the spinner entirely
and emit only the completed-event lines — safe under ``| grep``.

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

    The fixed line content renders first and a single spinner frame glyph is
    appended after a one-space gap as a trailing activity indicator.
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
        #: 1 = + result-preview line under each completed row; 2 = + the
        #: full pretty-printed arguments under the row.
        self.tool_detail = max(0, min(2, int(tool_detail)))
        #: Runtime ``--streaming`` mode (none|stage|text|hybrid). Controls
        #: the bounded stream box inside the activity display: ``text``
        #: streams the visible answer, ``hybrid`` streams reasoning (with a
        #: char-counter fallback when the provider sends none); ``none``
        #: and ``stage`` keep the plain one-line indicator.
        self.streaming_mode = (streaming_mode or "none").strip().lower()
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
        self._status_current_model: str = ""
        self._status_model_provider: str = ""
        self._status_turn_tokens: int = 0
        self._status_prompt_tokens: int = 0
        self._status_completion_tokens: int = 0
        self._status_reasoning_chars: int = 0
        self._iteration_durations: list[float] = []
        self._model_call_count: int = 0
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        self._tool_ok_count: int = 0
        self._tool_error_count: int = 0
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0
        # Per-model per-token prices (USD) injected by the command path:
        # ``{model_id: (prompt_price, completion_price)}``. Empty = the
        # footer omits the cost estimate (manifest unreadable/unknown model).
        self.model_prices: dict[str, tuple[float, float]] = {}
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

    def start_turn(self, message: str) -> None:
        """Begin prompt-mode rendering: echo the command, arm the activity line.

        No-op when ``quiet`` (the response alone prints later).
        """
        if self.quiet:
            return
        width = self.console.size.width or self._ACTIVITY_WIDTH
        line = Text()
        line.append("λ ", style="bold steel_blue3")
        line.append(
            _truncate(message.strip(), max(0, width - 6)), style="bold " + STYLE_TEXT_PRIMARY
        )
        self.console.print(line)
        self.start_activity()

    def start_activity(self) -> None:
        """Show the one-line transient activity display (tty only)."""
        if self.quiet or self._activity is not None or not self.console.is_terminal:
            return
        activity = Live(
            self._activity_renderable(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        with contextlib.suppress(Exception):
            activity.start()
        self._activity = activity

    def stop_activity(self) -> None:
        activity = self._activity
        self._activity = None
        if activity is not None:
            with contextlib.suppress(Exception):
                activity.stop()

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
        self._status_reasoning_chars = 0
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        self._status_current_model = ""
        self._status_model_provider = ""
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
        self.start_activity()

    def stop_thinking(
        self,
        *,
        flush_transcript: bool = False,
        flush_stream: bool = False,
    ) -> None:
        """Stop the activity indicator if it is running.

        The activity line is always transient — the flush flags exist for
        TransportCallbacks compatibility and do not resurrect it once the
        answer/error is ready to print. Completed event lines already live
        in real scrollback, so only the staged partial answer may need
        flushing on a diagnostic (error) path.
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
        """
        if reasoning:
            self._status_reasoning_chars += len(reasoning)
            self._stream_reasoning_buf += reasoning
        if text:
            self._stream_answer_buf += text
            self._stream_answer_chars += len(text)
            self._answer_staging.append(text, style=STYLE_STREAM_ANSWER)
            self._streamed_during_turn = True
        if (reasoning or text) and self._stream_box_active():
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
        # ``data["yielded"]`` (and ``HostCommandResult.yielded_question``).
        yield_question = str((data or {}).get("yielded") or "")
        if yield_question:
            self._render_yield_question(yield_question)
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

    def _render_yield_question(self, yield_question: str) -> None:
        """Stash + render a model yield (stdout gets nothing)."""
        self.yield_unresolved = True
        self.yield_question = yield_question
        self.console.print(
            Text(f"⏸ Model asked: {yield_question}", style=STYLE_ROW_TRANSITION)
        )
        if not self.console.is_terminal:
            Console(stderr=True).print(
                "[red]The model asked a question instead of producing a "
                "result. A one-shot prompt run cannot answer it. Re-run "
                "with the missing information in the prompt.[/red]"
            )

    def _render_response_text(self, text: str) -> None:
        """Render the final answer: themed Markdown under a model-titled rule."""
        rule_title = "Response"
        if self._status_current_model:
            rule_title = f"Response · {self._status_current_model}"
        self.console.print(Rule(title=rule_title, style=STYLE_BORDER_RESPONSE))
        self.console.print(Markdown(text, code_theme="github-dark", hyperlinks=True))

    def _render_unresolved_approval(self, approval: dict[str, Any]) -> None:
        """Capture a pending approval and surface the decision path.

        A tty console renders the approval panel (the hints point at
        `lisa prompt "approve <id>"`); a piped/redirected console
        fails fast on stderr instead of printing a panel the operator can
        never answer.
        """
        from hosts.cli.render import render_approval_panel  # noqa: PLC0415

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
            f'Decide without a session: lisa prompt "approve {request_id}" '
            f'or lisa prompt "deny {request_id}"\n'
            f'(add --remember <prefix> to approve-and-persist), '
            f"then re-run the original command.[/red]"
        )

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """No-op: the debug context panel was removed with the detail tiers.

        Kept to satisfy the ``TransportCallbacks`` protocol; the runtime only
        invokes it when debug mode is active, which the CLI no longer exposes.
        """
        return

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Stop the activity line and render the error panel.

        Completed event lines already printed to scrollback as they finished,
        so the trajectory up to the failure is preserved without a flush.
        Any staged partial answer flushes first so partial work isn't
        silently lost before the error panel shows.
        """
        self.stop_thinking(flush_transcript=True, flush_stream=True)
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        self.console.print(Panel(body, title="Error", border_style=STYLE_BORDER_ERROR))

    # ── Internals ───────────────────────────────────────────────────

    def _handle_model_stage(self, label: str, data: dict[str, Any]) -> None:
        """Route model_call* stage events.

        ``model_call`` tracks the active model/mode and counts the call;
        ``model_call_complete`` accumulates duration + tokens for the
        footer; ``model_call_failed`` prints the ✗ row immediately.
        """
        if label == "model_call":
            model_id = data.get("model_id")
            if model_id not in (None, "", 0):
                provider = data.get("provider")
                self._status_current_model = str(model_id)
                self._status_model_provider = str(provider) if provider not in (None, "") else ""
            self._model_call_count += 1
            self._active_reasoning_model = bool(data.get("reasoning_model"))
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
            total_raw = data.get("_state.total_tokens") or data.get("total_tokens")
            if total_raw not in (None, "", 0):
                self._status_turn_tokens += int(total_raw)
            # Split counts feed the footer's cost estimate (manifest prices
            # are per prompt/completion token).
            with contextlib.suppress(TypeError, ValueError):
                self._status_prompt_tokens += int(data.get("prompt_tokens") or 0)
            with contextlib.suppress(TypeError, ValueError):
                self._status_completion_tokens += int(data.get("completion_tokens") or 0)
            return
        if label == "model_call_failed":
            self._print_event_line(
                self._render_model_call(data, status_glyph="✗"), style=STYLE_ROW_ERROR
            )

    def _handle_tool_stage(self, label: str, data: dict[str, Any]) -> None:
        """Route tool_* stage events to the prompt-mode scrollback lines.

        ``tool_dispatch`` arms the activity label only; ``tool_output``
        prints streamed lines as they arrive (a dim rail marks them as tool
        output, not model prose); ``tool_dispatch_complete`` prints the one
        completed row — rendered from the dispatch payload (the completion
        event does not re-send arguments) with status/duration applied.
        """
        if label == "tool_dispatch":
            self._tool_call_count += 1
            self._active_label = self._render_tool_dispatch(data).plain
            self._last_tool_data = data
            self._last_tool_streaming = bool(data.get("stream_output"))
            self._last_tool_output_lines = 0
            if self.tool_detail >= _TOOL_DETAIL_FULL:
                self._print_detail_arguments(data)
            return
        if label == "tool_output":
            # Prompt mode has no Live transcript region — streamed lines print
            # to real scrollback as they arrive (piped runs keep them under
            # tee/grep; quiet mode drops them).
            text = str(data.get("text") or "")
            stream_label = str(data.get("stream") or "stdout")
            if text and not self.quiet:
                for raw_line in text.split("\n"):
                    line = raw_line.rstrip()
                    if line:
                        self._last_tool_output_lines += 1
                        self._print_stream_line(line, stream=stream_label)
            return
        if label == "tool_dispatch_complete":
            self._print_completion_row(data)
            return

    def _print_completion_row(self, data: dict[str, Any]) -> None:
        """Print the one completed tool row (+ summary + detail lines)."""
        ok = data.get("ok")
        status = "ok" if ok is True else ("error" if ok is False else "")
        if status == "ok":
            self._tool_ok_count += 1
        elif status == "error":
            self._tool_error_count += 1
        glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
        row_data = self._last_tool_data or data
        summary = self._result_shape_summary(row_data, data)
        row = self._render_tool_dispatch(
            row_data, status_glyph=glyph, complete=data, summary=summary
        )
        self._print_event_line(row)
        if self.tool_detail >= _TOOL_DETAIL_SUMMARY:
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

    def _result_shape_summary(
        self, row_data: dict[str, Any], complete: dict[str, Any]
    ) -> str:
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
            import json  # noqa: PLC0415

            rendered = json.dumps(arguments, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return
        for line in rendered.splitlines()[:_DETAIL_ARGUMENT_MAX_LINES]:
            self.console.print(Text(f"    {line}", style=STYLE_ARG_PREVIEW))

    def _print_detail_lines(
        self, row_data: dict[str, Any], complete: dict[str, Any]
    ) -> None:
        """Detail level >= 1: print the result preview under the row.

        The kernel sends ``result_preview`` on every completion: a compact
        payload-first summary for JSON-object outputs (content quoted
        first, then short scalars), else the first 200 chars of raw text.
        At level 2 the full snippet shows as a plain indented block;
        level 1 shows only the first line.
        """
        preview = str(complete.get("result_preview") or "").strip()
        if preview and preview != "Done":
            lines = preview.splitlines() or [preview]
            shown = (
                lines if self.tool_detail >= _TOOL_DETAIL_FULL else lines[:_TOOL_DETAIL_SUMMARY]
            )
            for line in shown:
                self.console.print(
                    Text(f"    ⎿ {line}", style=STYLE_OUTPUT_REGION)
                )

    def _print_event_line(self, text: Text | str, *, style: str | None = None) -> None:
        """Print one completed scrollback line, replacing the activity line.

        ``text`` may be a pre-styled ``Text`` (preferred — spans carry their
        own colors and markup is never parsed, so raw tool arguments can
        never raise ``MarkupError``) or a legacy plain string styled whole.
        """
        self.stop_activity()
        if isinstance(text, Text):
            line = Text("  ")
            line.append(text)
            self.console.print(line)
        else:
            self.console.print(f"  {text}", style=style or STYLE_TEXT_MUTED)
        self.start_activity()

    def _print_stream_line(self, line: str, *, stream: str = "stdout") -> None:
        """Print one streamed tool-output line to scrollback (prompt mode).

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
            with contextlib.suppress(Exception):
                self._activity.update(self._activity_renderable())

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

    def _stream_box_lines(self) -> list[str]:
        """Return the tail window of the active stream as wrapped lines.

        Picks the content source from the streaming mode and what has
        actually arrived: ``text`` mode → the answer stream; ``hybrid`` →
        the reasoning stream, falling back to nothing (the status-line
        thinking char counter carries that case). Lines wrap to the box
        width; the last ``_STREAM_BOX_MAX_LINES`` are returned, oldest
        first. Long single-line chunks (minified JSON, URLs) hard-wrap on
        their own — a box line never exceeds the console width.
        """
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
        budget = width * _STREAM_BOX_MAX_LINES
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
        return lines[-_STREAM_BOX_MAX_LINES:]

    def _stream_box_renderable(self) -> RenderableType | None:
        """Render the active stream box as a bordered, dim tail window.

        The content is deliberately rendered as *plain dim text*, not the
        styled markdown of the final response — visually signalling "this
        is live preview, not the formatted answer." Under
        ``_STREAM_BOX_BORDER_CHARS`` of content the box renders bare (no
        border) to stay light for short generations.
        """
        lines = self._stream_box_lines()
        if not lines:
            return None
        body = Text("\n".join(lines), style=STYLE_STREAM_BOX)
        if self._stream_total_chars() < _STREAM_BOX_BORDER_CHARS:
            return body
        title = "streaming" if self.streaming_mode == "text" else "thinking"
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

    def _activity_renderable(self) -> RenderableType:
        """The transient activity display: status line + bounded stream box.

        The first row is the one-line status (mode · model · in-flight
        label · thinking-meter · iter counters · budget bars · elapsed)
        with a trailing spinner. Below it, when a streaming mode streams
        into the box, the tail window of the model's live output renders
        inside a bounded panel — so long generations show visible motion
        without pushing completed event rows (or the footer) too far back.

        Wide-tty surfaces additionally show a tool-call budget bar
        (``▰▰▱▱ 4/8``) when the kernel stamped a finite cap on the stage
        payloads — the denominator rides every event, so this costs one
        capture per turn. Unlimited caps fall back to the plain counter;
        the bar warns gold past ``_BUDGET_BAR_WARN_RATIO``.
        """
        width = self.console.size.width or self._ACTIVITY_WIDTH
        line = self._status_line(width)
        box = self._stream_box_renderable() if self._stream_box_active() else None
        if box is not None:
            return Group(line, box)
        return line

    def _status_line(self, width: int) -> _TrailingSpinner:
        """Compose the one-line status row with a trailing spinner."""
        parts: list[Text] = []
        if self._current_mode:
            parts.append(Text(f"[{self._current_mode}]", style=STYLE_ROW_TRANSITION))
        if self._status_current_model:
            model = self._status_current_model
            if self._status_model_provider:
                model = f"{self._status_model_provider}:{model}"
            parts.append(Text(model, style=STYLE_STAGE))
        if self._active_label:
            parts.append(Text(_truncate(self._active_label, 48), style=STYLE_STAGE))
        if self._active_reasoning_model and self._status_reasoning_chars:
            parts.append(
                Text(f"thinking {_format_tokens(self._status_reasoning_chars)}", style=STYLE_STAGE)
            )
        if self._phase_label and not parts and not self._active_label:
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
        return _TrailingSpinner(body, self._activity_spinner(), STYLE_GRAPHIC)

    def _activity_spinner(self) -> str:
        """Pick the spinner for the current phase.

        Reasoning-model calls get a visually distinct, calmer spinner so a
        long think doesn't read as a stalled fetch.
        """
        if self._active_reasoning_model and self._model_call_count > 0:
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

        ``⚡ 22s · 5 tools · 2 models · 1 handoff · 8.4k tok · $0.02`` — one
        line, per-turn only; a ``prompt`` invocation is one process and one
        turn, so the turn/session token split collapses to a single total.
        Each segment carries its own color: elapsed gold, tools blue (red
        annotation on failures), models green, tokens green, cost cyan.
        Plain text is identical to the legacy flat line.
        """
        model_time_ms = round(sum(self._iteration_durations))
        tools_total = self._tool_call_count
        tool_failures = self._tool_error_count
        footer = Text()
        footer.append(f"⚡ {self._elapsed_seconds()}s", style=STYLE_ROW_TRANSITION)
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
            handoff_part = (
                f"{self._handoff_count} handoff{'s' if self._handoff_count != 1 else ''}"
            )
            route = self._current_mode
            if route:
                handoff_part += f" → {route}"
            footer.append(handoff_part, style="steel_blue3")
        if self._status_turn_tokens:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            footer.append(f"{_format_tokens(self._status_turn_tokens)} tok", style="green3")
        cost = self._estimate_turn_cost()
        if cost:
            footer.append(" · ", style=STYLE_TEXT_MUTED)
            footer.append(f"${cost:g}", style="cyan")
        self.console.print(footer)

    def _estimate_turn_cost(self) -> float:
        """Estimate the turn's USD cost from the manifest per-token prices.

        Uses the split prompt/completion token counts accumulated from
        ``model_call_complete`` events against the per-model prices injected
        by the command path (``model_prices``). Returns 0 when either side
        is missing — a missing manifest must never block the footer.
        """
        prices = self.model_prices.get(self._status_current_model)
        if not prices:
            return 0.0
        prompt_price, completion_price = prices
        cost = (
            self._status_prompt_tokens * prompt_price
            + self._status_completion_tokens * completion_price
        )
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

        ``→ tool brave_search("query") ✓ 120ms``. On completion
        (``status_glyph`` set) the status and duration are appended,
        right-aligned on wide terminals so a column of completions scans
        vertically. ``summary`` (the ``⎿ N lines`` result shape) rides
        after the status; its length is reserved in the alignment math so
        the row never wraps.

        Returns a styled ``Text``: glyph in the tool-kind color, tool name
        bright, argument preview dim, status success/error colored. Plain
        text (non-tty) is identical to the legacy string form — the
        right-alignment pads with spaces, which reads the same either way.
        """
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")

        out = Text()
        if kind == "transition":
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
        else:
            out.append(glyph, style=STYLE_ROW_TOOL)
            out.append(f" {tool}", style=STYLE_TOOL_NAME)
            arg = self._primary_arg_preview(data)
            if arg:
                out.append(f" {arg}", style=STYLE_ARG_PREVIEW)

        if status_glyph:
            status_style = (
                STYLE_ROW_ERROR if status_glyph == "✗" else STYLE_ROW_SUCCESS
            )
            duration = (complete or data).get("duration_ms")
            tail = status_glyph
            if duration not in (None, "", 0):
                tail += f" {round(float(duration))}ms"
            self._append_aligned_status(out, tail, status_style, suffix=summary)
            if summary:
                out.append(" ", style=STYLE_TEXT_MUTED)
                out.append(summary, style=STYLE_TEXT_MUTED)
        return out

    def _append_aligned_status(
        self, row: Text, tail: str, style: str, *, suffix: str = ""
    ) -> None:
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