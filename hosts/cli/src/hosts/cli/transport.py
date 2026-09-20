"""Prompt-mode Rich transport for the CLI host.

This is the terminal-transport implementation of
:class:`host.callbacks.TransportCallbacks` for ``lisa prompt`` — the CLI's
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

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
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
STYLE_OUTPUT_REGION = "grey42"

#: Glyphs for tool-dispatch completion status.
_TOOL_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…", "": "·"}

#: Maximum length of the primary tool argument shown inline on a tool row.
_ARG_PREVIEW_MAX = 40

#: Token-count formatting thresholds for the compact footer figures.
_TOKENS_MILLION = 1_000_000
_TOKENS_THOUSAND = 1_000


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
        self, *, console: Console | None = None, quiet: bool = False, raw: bool = False
    ) -> None:
        self.console: Console = console or Console()
        self.quiet = quiet
        self.raw = raw
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
        self._activity: Live | None = None
        self._thinking_spinner: str = "dots"
        # ── Turn counters ───────────────────────────────────────────
        self._turn_started_perf: float = 0.0
        self._current_mode: str = ""
        self._status_current_model: str = ""
        self._status_model_provider: str = ""
        self._status_turn_tokens: int = 0
        self._status_reasoning_chars: int = 0
        self._iteration_durations: list[float] = []
        self._model_call_count: int = 0
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        self._tool_ok_count: int = 0
        self._tool_error_count: int = 0
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0
        # Whether any on_text_chunk visible-answer delta arrived this turn —
        # informational only: the answer renders once, fully, from
        # on_response regardless.
        self._streamed_during_turn: bool = False
        # The visible-answer deltas stage into this buffer purely so a
        # mid-turn error path can flush partial work; on_response clears it.
        self._answer_staging: Text = Text()

    # ── Lifecycle ───────────────────────────────────────────────────

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
            refresh_per_second=4,
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
        self._status_reasoning_chars = 0
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        self._status_current_model = ""
        self._status_model_provider = ""
        self._active_label = ""
        self._last_tool_data = None
        self._last_tool_streaming = False
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
        if label.startswith("model_call"):
            self._handle_model_stage(label, data)
        elif label.startswith("tool_"):
            self._handle_tool_stage(label, data)
        elif label.startswith("fork_"):
            self._handle_fork_stage(label, data)
        # else: bookkeeping-only events update nothing visible in prompt mode.

        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            self._print_event_line(
                f"↦ {prev_mode or 'start'} → {self._current_mode}",
                style=STYLE_ROW_TRANSITION,
            )
        self._redraw_activity()

    # ── Streaming / response ───────────────────────────────────────

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Stage answer deltas / count reasoning without rendering either.

        The answer renders once, fully formatted, from ``on_response``;
        thinking segments were a live-transcript affordance that prompt mode
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
        approval = (data or {}).get("approval")
        if isinstance(approval, dict) and approval.get("request_id"):
            self._render_unresolved_approval(approval)
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
                    "result. A one-shot prompt run cannot answer it. Re-run "
                    "with the missing information in the prompt.[/red]"
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
            self._print_footer()
        elif self.raw and not self.quiet:
            # The footer is activity output: in raw mode it prints on the
            # transport console (stderr) so stdout stays machine-readable.
            self._print_footer()

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
            self._active_label = self._render_tool_dispatch(data)
            self._last_tool_data = data
            self._last_tool_streaming = bool(data.get("stream_output"))
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
                        self._print_stream_line(line, stream=stream_label)
            return
        if label == "tool_dispatch_complete":
            ok = data.get("ok")
            status = "ok" if ok is True else ("error" if ok is False else "")
            if status == "ok":
                self._tool_ok_count += 1
            elif status == "error":
                self._tool_error_count += 1
            glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
            row_data = self._last_tool_data or data
            self._print_event_line(
                self._render_tool_dispatch(row_data, status_glyph=glyph, complete=data),
                style=STYLE_ROW_TOOL,
            )
            self._active_label = ""
            self._last_tool_data = None
            self._last_tool_streaming = False

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

    def _print_event_line(self, text: str, *, style: str | None = None) -> None:
        """Print one completed scrollback line, replacing the activity line."""
        self.stop_activity()
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

    def _elapsed_seconds(self) -> int:
        """Whole seconds elapsed since ``start_thinking`` for this turn."""
        if self._turn_started_perf <= 0:
            return 0
        return int(perf_counter() - self._turn_started_perf)

    def _print_footer(self) -> None:
        """Render the single-line prompt-mode footer.

        ``⚡ 22s · 5 tools · 2 models · 1 handoff · 8.4k tok`` — one line,
        per-turn only; a ``prompt`` invocation is one process and one turn,
        so the turn/session token split collapses to a single total.
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

    # ── Model/tool row rendering (shared with event lines) ──────────

    def _render_model_call(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a model call as a one-liner: stage, model, iteration, status.

        ``stage: model_call haiku #1 ✓ 340ms``. On completion
        (``status_glyph`` set) the status and duration are appended.
        """
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
        return body

    def _render_tool_dispatch(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a tool call as a one-liner: glyph, tool, primary arg, status.

        ``→ tool brave_search("query") ✓ 120ms``. On completion (``status_glyph``
        set) the status and duration are appended.
        """
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")

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
        return body

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


__all__ = ["PromptTransport"]