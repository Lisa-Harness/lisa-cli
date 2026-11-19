"""Shared Typer option types for CLI commands."""

from __future__ import annotations

from typing import Annotated, Literal

import typer

# ---------------------------------------------------------------------------
# These Annotated type aliases define each runtime option exactly once so every
# subcommand shares identical flag names, defaults, and help text.

#: Parse-time-validated pack subcommand (Typer rejects typos before any
#: runtime boot costs, and completion offers the values).
PackAction = Literal["list", "install", "uninstall", "update"]

#: Parse-time-validated models subcommand ('reload'/'edit' are accepted but
#: rejected in-body as no-ops for a one-shot CLI process).
ModelsAction = Literal["show", "discover", "apply", "help", "reload", "edit"]

LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        help=("Log threshold: debug, info, warning, or error. Overrides LISA_LOG_LEVEL."),
    ),
]
LogOption = Annotated[
    str,
    typer.Option(
        "--log",
        help="Append JSONL request/response turn logs to this file.",
    ),
]
NoLogOption = Annotated[
    bool,
    typer.Option(
        "--no-log",
        help="Disable turn logging entirely.",
    ),
]
AllPacksOption = Annotated[
    bool,
    typer.Option(
        "--all",
        help=(
            "Update every runtime-installed pack with a recorded install source (pack update only)."
        ),
    ),
]
HostLogOption = Annotated[
    str,
    typer.Option(
        "--host-log",
        help=(
            "Stdlib host log file (kernel/sdk startup and runtime logs). "
            "Default: <data-dir>/host.jsonl"
        ),
    ),
]
DataDirOption = Annotated[
    str,
    typer.Option(
        "--data-dir",
        help=(
            "Data root for artifacts, memory, and logs. "
            "Overrides LISA_DATA_DIR. "
            "Default: ~/.lisa (see default_data_dir())"
        ),
    ),
]
ProfileOption = Annotated[
    str,
    typer.Option(
        "--profile",
        help=(
            "Named data-dir profile: resolves to <profiles-root>/<name> "
            "(default ~/.lisa/profiles; relocatable via LISA_PROFILES_DIR). "
            "Overrides LISA_PROFILE; --data-dir still wins. Shares the "
            "base packs venv, isolates threads/memory/models/policy."
        ),
    ),
]
OverlaysDirOption = Annotated[
    str,
    typer.Option(
        "--overlays-dir",
        help=(
            "User-authored overlays directory for mode and content overrides. "
            "Default: <data-dir>/overlays"
        ),
    ),
]
ThreadDirOption = Annotated[
    str,
    typer.Option(
        "--thread-dir",
        help=(
            "Thread log directory. Overrides LISA_THREAD_DIR. "
            "Defaults to <data-dir>/threads when a session context is active."
        ),
    ),
]
NoThreadOption = Annotated[
    bool,
    typer.Option(
        "--no-thread",
        help="Disable local turn history thread logging.",
    ),
]
SessionContextOption = Annotated[
    str,
    typer.Option(
        "--session-context",
        help=(
            "Path to a Markdown session plan file. "
            "Overrides LISA_SESSION_CONTEXT_PATH. "
            "Default: <data-dir>/config/session_context.md, else the bundled "
            "core session."
        ),
    ),
]
TimezoneOption = Annotated[
    str,
    typer.Option(
        "--timezone",
        help=(
            "IANA timezone (e.g. UTC, America/New_York). "
            "Overrides LISA_TIMEZONE, then OS timezone env vars."
        ),
    ),
]
StateSnapshotOption = Annotated[
    str,
    typer.Option(
        "--state-snapshot",
        help="Local JSON snapshot file to inject compact context from.",
    ),
]
StreamingOption = Annotated[
    str,
    typer.Option(
        "--streaming",
        help=(
            "Streaming mode: none, stage, text, hybrid (default hybrid). "
            "text streams the answer, hybrid streams thinking, into a "
            "bounded box under the status line; the final answer always "
            "re-renders with full markdown. Append ':<ms>' for "
            "per-character delay (e.g. hybrid:50)."
        ),
    ),
]
QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress the activity line, per-tool lines, and the "
        "⚡ footer so the response is the only output (pipe-friendly).",
    ),
]
StdinOption = Annotated[
    bool,
    typer.Option(
        "--stdin/--no-stdin",
        help=(
            "Read piped stdin (when stdin is not a tty) and attach it to the "
            "turn as a <piped-input> block ahead of the message. Default: auto."
        ),
    ),
]
RawOption = Annotated[
    bool,
    typer.Option(
        "--raw",
        "-r",
        help=(
            "Write the response verbatim to stdout for redirection/piping — no "
            "Rule header, no Markdown re-render, no footer — move activity "
            "output to stderr, and prepend a raw-mode output contract to the "
            "turn so the model emits only the requested result."
        ),
    ),
]
ToolDetailOption = Annotated[
    int,
    typer.Option(
        "--tool-detail",
        "-v",
        count=True,
        help=(
            "Show more tool-call detail: -v adds the result preview under "
            "each completed tool row; -v -v also prints the full call "
            "arguments and the multi-line result snippet. Default: compact "
            "one-line rows."
        ),
    ),
]
PromptApproveOption = Annotated[
    str,
    typer.Option(
        "--approve",
        help=(
            "Pre-arm approval for the request id the upcoming turn raises: "
            "`lisa prompt --approve <request-id> 'continue'` dispatches "
            "/approve first, then runs the turn. Use with the durable id "
            "printed by a prior exit-3 run. `--approve` (bare) targets the "
            "single pending approval."
        ),
    ),
]
PromptDenyOption = Annotated[
    str,
    typer.Option(
        "--deny",
        help=(
            "Pre-arm a denial for the upcoming turn: "
            "`lisa prompt --deny <request-id> 'continue'` dispatches /deny first. "
            "`--deny` (bare) targets the single pending approval."
        ),
    ),
]
PromptRememberOption = Annotated[
    str,
    typer.Option(
        "--remember",
        help=(
            "With --approve: persist an allow rule for the decided action's "
            "command prefix (default: program + first subcommand token) so "
            "future matching runs auto-approve. "
            "Pass an explicit prefix to widen/narrow the scope."
        ),
    ),
]
PromptSwapKeysOption = Annotated[
    bool | None,
    typer.Option(
        "--swap-keys/--no-swap-keys",
        help=(
            "No-args editor key layout. Shipped default: Enter inserts a "
            "newline; Ctrl-J sends — Enter never fires the "
            "turn, so a hurried keypress cannot send a half-typed message "
            "nor leak to the shell after the run. --swap-keys reverses to "
            "chat-box (Enter sends); LISA_PROMPT_KEYS truthy does the same "
            "without the flag; --no-swap-keys pins the shipped default "
            "against a truthy env."
        ),
    ),
]
