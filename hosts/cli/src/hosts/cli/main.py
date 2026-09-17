"""hosts.cli.main — Typer app and CLI entry points for the terminal host.

This is the terminal-transport entry point for the Lisa framework. It
wraps the transport-neutral :mod:`host` runtime with Typer commands
and Rich/prompt_toolkit rendering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

# Load .env BEFORE injecting the packs venv site-packages so that
# LISA_DATA_DIR (and LISA_PACKS_VENV) are honored when resolving the
# packs venv (``<data_dir>/venv``) location. The loader only needs framework_root() and
# default_data_dir(), both of which are env-independent, so it can run
# safely before any host import. Search order:
#   (1) CWD — for repo-mode dev sessions
#   (2) ~/.lisa/.env — for tool-install mode (uv tool install lisa)
#   (3) project root — repo-mode back-compat
#   (4) hosts/little_lisa/.env — legacy repo-mode back-compat
from host.paths import (
    framework_root as _framework_root,
)
from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from dotenv import load_dotenv

    _FRAMEWORK_ROOT = _framework_root()
    from host.config import default_data_dir as _default_data_dir

    # Honor LISA_DATA_DIR here so tool-mode installs configured with a custom
    # data dir find their .env before any HostRuntime exists to do the full
    # config resolution. Falls back to the platform default (~/.lisa).
    _data_dir_override = os.getenv("LISA_DATA_DIR", "").strip()
    if _data_dir_override:
        _default_env_dir = Path(_data_dir_override).expanduser()
    else:
        _default_env_dir = _default_data_dir()

    # Module-level search order, shared with _verify_check_env (which reads
    # this constant so monkeypatched tests and the loader stay in sync).
    _ENV_SEARCH_CANDIDATES = (
        Path.cwd() / ".env",
        _default_env_dir / ".env",
        _FRAMEWORK_ROOT / ".env",
        _FRAMEWORK_ROOT / "hosts" / "little_lisa" / ".env",
    )

    for _env_candidate in _ENV_SEARCH_CANDIDATES:
        if _env_candidate.exists():
            load_dotenv(dotenv_path=_env_candidate, override=False)
            break
except ImportError:  # pragma: no cover — python-dotenv is optional
    pass

# Inject the packs venv site-packages into sys.path BEFORE importing
# host so that importlib.metadata.entry_points discovers packs
# installed via /pack install into the packs venv. This must run
# before any host / kernel import that triggers entry-point scanning.
from host.paths import (
    inject_packs_venv_site_packages as _inject_packs_venv_site_packages,
)

_inject_packs_venv_site_packages()

from host.config import default_data_dir
from hosts.cli.render import (
    render_mcp_server_tools,
    render_mcp_servers,
    render_models,
    render_modes,
    render_operations,
    render_pack_list_panel,
    render_tools,
)

# ``HostRuntime`` is used only as a string annotation (``from __future__
# import annotations`` stringifies signatures) across the Typer callbacks.
# The runtime construction happens inside ``_initialize_runtime``, which
# imports ``HostRuntime`` + the mcp_pool / state_provider symbols lazily so
# ``--version`` (which exits in the eager Typer callback before any command
# body runs) does not pay the ~2s ``host.runtime`` ->
# ``kernel.ipsar_fsm`` -> ``openai`` import chain.
if TYPE_CHECKING:
    from host.runtime import HostRuntime as HostRuntime

    from hosts.cli.transport import OnceTransport as OnceTransport

# ``handle_message`` is deferred to the turn-execution call sites
# (``_run_once_with_teardown``) so ``--version`` / ``modes`` / ``models`` /
# ``verify`` don't pay the ~3s ``openai`` + ``mcp`` + ``kernel.ipsar_fsm``
# import cost it drags in. See ``host/__init__.py``'s PEP 562
# ``__getattr__`` for the package-level lazy import.

from rich.traceback import install

install(show_locals=True)

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(
    no_args_is_help=False,
    help="Run the Lisa framework terminal host.",
)

# Exit code returned when a restart is required (e.g. after a pack lifecycle
# change). Scripts can detect this to distinguish "restart needed" from a
# clean exit. Matches the BSD sysexits.h value for EX_UNAVAILABLE.
_RESTART_REQUIRED_EXIT = 69

# Exit code returned when a `once` turn ends with an unresolved approval
# request. Once mode has no prompt loop to decide an approval, so the
# request stays durable and the process exits non-zero — scripts detect
# "needs approval" instead of mistaking it for a completed turn.
_APPROVAL_REQUIRED_EXIT = 3

# Exit code returned when a `once` turn ends in a model yield (the model
# asked the operator a question instead of finishing). The question is
# durable on the thread and printed to stderr; stdout gets nothing. Scripts
# detect "needs input" and re-run with the missing information in the prompt.
_YIELD_UNRESOLVED_EXIT = 4

# Env var carrying a message to replay as the first turn on the next launch
# after a restart. Set by `/pack install ... --resume <msg>` in
# host.commands; consumed once on startup by the CLI host.
_RESTART_RESUME_MESSAGE_ENV = "LISA_RESTART_RESUME_MESSAGE"

# Env var (or --once-unknown flag) enabling lazy routing: an unrecognized
# first positional is redirected to `lisa once <args...>` so users can run
# `lisa continue` or `lisa "do something else"` without typing `once`. The
# flag is redundant with the env var (people set it via env); it exists for
# discoverability and is handled in pre-parse before Typer sees argv.
_ONCE_UNKNOWN_ENV = "LISA_ONCE_UNKNOWN"
_ONCE_UNKNOWN_FLAG = "--once-unknown"

# Sentinel for --approve/--deny meaning "the (single) pending approval":
# resolved to the concrete request id before dispatch via the approval
# service's list_pending (recorded-owner identity makes cross-process
# pending records visible to a fresh process).
_PENDING_APPROVAL_TARGET = "pending"

# Grammar for `lisa once` decision subcommands (matched against the
# lowercased message). `policy` accepts a bare verb or up to two short
# arg tokens (subcommand word + value; the host validates the vocabulary).
# approve/deny require the first argument to look like a request id (the
# host mints 32-hex ids) or start a --remember flag whose value is
# free-form. Prose like "deny the claim" or "approve my leave request"
# fails the id shape and stays on the model path. The host re-validates
# the exact id anyway.
_REQUEST_ID_SHAPE = r"[0-9a-f]{8,64}"
_ONCE_HOST_COMMAND_RE = re.compile(
    r"^(?:"
    r"policy(?:\s+\S{1,32}){0,2}"
    r"|approve(?:\s+" + _REQUEST_ID_SHAPE + r")?(?:\s+--remember\b.*)?"
    r"|deny\s+" + _REQUEST_ID_SHAPE + r""
    r")$"
)


def _restart_current_process() -> None:
    """Restart the CLI process after a runtime change.

    - Unix (``os.name != "nt"``): true in-place re-exec via
      :func:`os.execv`. The process keeps its terminal, stdin/stdout, and
      environment; the new image re-enters :func:`main` with the same argv.
    - Windows (``os.name == "nt"``): do NOT spawn a child. A spawned
      process loses its parent terminal link and breaks the interactive
      REPL. Instead, print a clear restart-required notice and exit with
      :data:`_RESTART_REQUIRED_EXIT` so the user manually re-runs ``lisa``.

    Uses :data:`sys.argv` (not ``sys.orig_argv``) for args continuity â€”
    ``sys.orig_argv`` on Windows contains the Store launcher, not the
    venv interpreter, which corrupts re-execs.
    """
    if os.name == "nt":
        # Do not attempt to spawn a child process on Windows â€” it detaches
        # from the parent terminal and breaks the interactive REPL. Just
        # exit with a clear, script-detectable restart-required code.
        console.print(
            Panel(
                "A runtime change requires a restart to reload entry points,\n"
                "pack catalogs, and MCP manifests.\n\n"
                "Please re-run [bold]lisa[/bold] manually to continue.",
                title="Restart Required",
                border_style="gold3",
            )
        )
        raise typer.Exit(code=_RESTART_REQUIRED_EXIT)
    else:
        # True in-place exec replacement on Unix. The current process image
        # is replaced; the terminal, file descriptors, and environment are
        # preserved. os.execv does not return on success.
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("lisa-host-cli")
    except Exception:  # pylint: disable=broad-except
        return "0.1.0"


_VERSION_OPTION = typer.Option(
    False,
    "--version",
    help="Print the lisa version and exit.",
    is_eager=True,
)

# ---------------------------------------------------------------------------
# Shared Typer option aliases
# ---------------------------------------------------------------------------
# These Annotated type aliases define each runtime option exactly once so every
# subcommand shares identical flag names, defaults, and help text.

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
            "User context source: a slug name (content/user_contexts/<name>.md or downloaded from remote registry), "
            "a single remote URL, a full .md file path, or a directory (all files concatenated). "
            "Omit to skip session context."
        ),
    ),
]

SessionContextRefreshOption = Annotated[
    bool,
    typer.Option(
        "--session-context-refresh",
        help="Force re-fetch of remote session context, ignoring local cache.",
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
            "Optionally append ':<ms>' for per-character delay (e.g. hybrid:50)."
        ),
    ),
]
QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress the once-mode activity line, per-tool lines, and the "
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
OnceApproveOption = Annotated[
    str,
    typer.Option(
        "--approve",
        help=(
            "Pre-arm approval for the request id the upcoming turn raises: "
            "`lisa once --approve <request-id> 'continue'` dispatches "
            "/approve first, then runs the turn. Use with the durable id "
            "printed by a prior exit-3 run. `--approve` (bare) targets the "
            "single pending approval."
        ),
    ),
]
OnceDenyOption = Annotated[
    str,
    typer.Option(
        "--deny",
        help=(
            "Pre-arm a denial for the upcoming turn: "
            "`lisa once --deny <request-id> 'continue'` dispatches /deny first. "
            "`--deny` (bare) targets the single pending approval."
        ),
    ),
]
OnceRememberOption = Annotated[
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
# ---------------------------------------------------------------------------
# Runtime initialization (builds HostRuntime, not _RUNTIME_STATE)
# ---------------------------------------------------------------------------


def _initialize_runtime(
    *,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    overlays_dir: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str,
    timezone: str = "",
    session_context_refresh: bool = False,
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
    in_process_mcp: bool = False,
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Thin adapter over :func:`host.bootstrap.build_runtime` that
    stamps ``host_kind="cli"`` and forwards every CLI argument. The
    composition sequence (config -> runtime -> state snapshot -> framework
    wiring -> MCP config compilation) lives in the shared bootstrap so
    the CLI and web hosts cannot drift.

    The MCP pool is NOT started here â€” ``runtime.start()`` must be called
    inside the host's event loop (see :func:`hosts.cli.repl.run_repl`).
    """
    from host.bootstrap import build_runtime

    return build_runtime(
        host_kind="cli",
        log_level=log_level,
        log_path=log_path,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=user_slug,
        timezone=timezone,
        session_context=user_slug,
        session_context_refresh=session_context_refresh,
        state_snapshot_path=state_snapshot_path,
        streaming=streaming,
        host_log_path=host_log_path,
        in_process_mcp=in_process_mcp,
    )


def _initialize_runtime_or_exit(**kwargs: Any) -> HostRuntime | None:
    """Initialize runtime, rendering startup errors and exiting on failure.

    Returns the runtime on success, or ``None`` after calling
    ``typer.Exit`` on failure (the ``typer.Exit`` raises, so ``None`` is
    only reached if the exit is suppressed).
    """
    try:
        return _initialize_runtime(**kwargs)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        from hosts.cli.repl import _render_runtime_error

        _render_runtime_error(console, exc, title="Startup Error")
        raise typer.Exit(code=2) from exc


# ---------------------------------------------------------------------------
# Default no-args mode
# ---------------------------------------------------------------------------


def _run_default_no_args_mode(
    *,
    log_level: str,
    log: str,
    no_log: bool,
    data_dir: str,
    overlays_dir: str,
    thread_dir: str,
    no_thread: bool,
    session_context: str,
    session_context_refresh: bool,
    timezone: str,
    state_snapshot: str,
    streaming: str,
    host_log: str,
) -> None:
    """Launch the REPL when no subcommand is given.

    The default no-args mode is the interactive REPL.
    """
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    _consume_restart_resume_message(runtime)
    from hosts.cli.repl import run_repl

    #                                   --version/modes/verify fast
    restart_requested = asyncio.run(run_repl(runtime, console=console))
    if restart_requested:
        _restart_current_process()


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------


def _consume_restart_resume_message(runtime: HostRuntime) -> None:
    """Consume the resume-message env var on startup if present.

    If ``LISA_RESTART_RESUME_MESSAGE`` is set (by a `/pack ... --resume <msg>`
    command in the prior session), stash it on the runtime so the REPL can
    replay it as the first turn, then clear the env var to avoid replay
    loops on subsequent restarts.

    Only consumed in interactive REPL startup; one-shot ``once``/``pack``
    modes ignore it for deterministic one-shot UX.
    """
    resume_msg = os.environ.pop(_RESTART_RESUME_MESSAGE_ENV, "").strip()
    if resume_msg:
        runtime.session.pending_resume_message = resume_msg


def _exit_restart_required() -> None:
    """Print a restart-required notice and exit for one-shot commands.

    Used by ``once`` and ``pack install/uninstall`` after a successful
    pack lifecycle change. Unlike the REPL path, one-shot commands do
    NOT re-exec (re-running the same command would redo the mutation).
    Instead, print a clear notice and exit with
    :data:`_RESTART_REQUIRED_EXIT` on all platforms so the user manually
    restarts the interactive session.
    """
    console.print(
        Panel(
            "A runtime change requires a restart to reload entry points,\n"
            "pack catalogs, and MCP manifests.\n\n"
            "Please re-run [bold]lisa[/bold] to continue.",
            title="Restart Required",
            border_style="gold3",
        )
    )
    raise typer.Exit(code=_RESTART_REQUIRED_EXIT)


# ---------------------------------------------------------------------------
# Typer callback (no-args default)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _cli_callback(
    ctx: typer.Context,
    version: bool = _VERSION_OPTION,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "hybrid",
    host_log: HostLogOption = "",
) -> None:
    if version:
        console.print(f"lisa v{_get_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    _run_default_no_args_mode(
        log_level=log_level,
        log=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot=state_snapshot,
        streaming=streaming,
        host_log=host_log,
    )


# ---------------------------------------------------------------------------
# cli command (REPL)
# ---------------------------------------------------------------------------


@app.command("cli", help="Run the interactive Lisa terminal REPL.")
def _cli_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "hybrid",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    _consume_restart_resume_message(runtime)
    from hosts.cli.repl import run_repl

    #                                   --version/modes/verify fast
    restart_requested = asyncio.run(run_repl(runtime, console=console))
    if restart_requested:
        _restart_current_process()


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------

_VERIFY_PASS = "[green]PASS[/green]"
_VERIFY_WARN = "[yellow]WARN[/yellow]"
_VERIFY_FAIL = "[red]FAIL[/red]"
_KEY_REDACT_VISIBLE_CHARS = 4
_MIN_PYTHON_MAJOR = 3
_MIN_PYTHON_MINOR = 11
_HTTP_OK = 200
_OLLAMA_MODEL_PREVIEW_LIMIT = 5


def _redact_key(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= _KEY_REDACT_VISIBLE_CHARS:
        return "*" * len(value)
    return "*" * (len(value) - _KEY_REDACT_VISIBLE_CHARS) + value[-_KEY_REDACT_VISIBLE_CHARS:]


def _verify_check_python() -> tuple[str, str]:
    version = platform.python_version()
    major, minor, *_ = version.split(".")
    if int(major) >= _MIN_PYTHON_MAJOR and int(minor) >= _MIN_PYTHON_MINOR:
        return _VERIFY_PASS, f"Python {version}"
    return _VERIFY_FAIL, f"Python {version} (need >= 3.11)"


def _verify_check_env(runtime: HostRuntime) -> tuple[str, str]:
    # Check for .env in the same search order as the startup loader. The
    # loader's _ENV_SEARCH_CANDIDATES is module-level so it can be
    # monkeypatched (tests); CWD still resolves dynamically because pytest's
    # chdir affects it.
    for candidate in _ENV_SEARCH_CANDIDATES:
        if candidate.exists():
            return _VERIFY_PASS, f".env found ({candidate})"
    return _VERIFY_WARN, ".env not found. Create one at ~/.lisa/.env with your model provider keys."


def _verify_check_manifest(runtime: HostRuntime) -> tuple[str, str]:
    from host.model_manifest import load_model_manifest

    try:
        path = runtime.paths.models_path
        manifest = load_model_manifest(path)
        model_count = len(manifest.models)
        if model_count == 0:
            return _VERIFY_FAIL, "Model manifest loaded but contains zero models"
        model_ids = ", ".join(m.model_id for m in manifest.models)
        return _VERIFY_PASS, f"{model_count} model(s): {model_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Model manifest failed to load: {exc}"


def _verify_check_pack_catalog() -> tuple[str, str]:
    try:
        from kernel.ipsar_fsm import get_workflow_catalog

        catalog = get_workflow_catalog()
        mode_count = len(catalog)
        if mode_count == 0:
            return _VERIFY_WARN, "Pack catalog loaded but contains zero modes"
        mode_ids = ", ".join(sorted(catalog.keys()))
        return _VERIFY_PASS, f"{mode_count} mode(s): {mode_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Pack catalog failed to load: {exc}"


def _verify_check_ollama() -> tuple[str, str]:
    import os

    import httpx

    configured_host = (
        os.getenv("LISA_OLLAMA_HOST", "").strip()
        or os.getenv("OLLAMA_HOST", "").strip()
        or "127.0.0.1:11434"
    )

    async def _ping() -> tuple[str, str]:
        try:
            base_url = f"http://{configured_host}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == _HTTP_OK:
                    data = resp.json()
                    models = [m.get("name", "?") for m in data.get("models", [])]
                    if models:
                        preview = ", ".join(models[:_OLLAMA_MODEL_PREVIEW_LIMIT])
                        suffix = "..." if len(models) > _OLLAMA_MODEL_PREVIEW_LIMIT else ""
                        return _VERIFY_PASS, (
                            f"Ollama reachable, {len(models)} model(s) pulled: {preview}{suffix}"
                        )
                    return _VERIFY_WARN, "Ollama reachable but no models pulled"
                return _VERIFY_WARN, f"Ollama returned HTTP {resp.status_code}"
        except Exception as exc:
            return _VERIFY_WARN, f"Ollama not reachable at {configured_host}: {exc}"

    try:
        return asyncio.run(_ping())
    except Exception as exc:
        return _VERIFY_WARN, f"Ollama check failed: {exc}"


def _verify_check_provider_keys() -> tuple[str, str]:
    import os

    parts: list[str] = []
    # Check LISA_* vars first (the new host's env vars), then fall back to
    # the bare provider variables accepted by the host gateway provider.
    azure_key = (
        os.getenv("LISA_AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    azure_endpoint = (
        os.getenv("LISA_AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    if azure_key and azure_endpoint:
        parts.append(f"Azure OpenAI: {_redact_key(azure_key)} @ {azure_endpoint}")
    elif azure_key or azure_endpoint:
        parts.append("Azure OpenAI: partially configured")

    openai_key = (
        os.getenv("LISA_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if openai_key:
        parts.append(f"OpenAI: {_redact_key(openai_key)}")

    anthropic_key = (
        os.getenv("LISA_ANTHROPIC_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if anthropic_key:
        parts.append(f"Anthropic: {_redact_key(anthropic_key)}")

    google_key = (
        os.getenv("LISA_GOOGLE_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if google_key:
        parts.append(f"Google Gemini: {_redact_key(google_key)}")

    ollama_host = os.getenv("LISA_OLLAMA_HOST", "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        parts.append(f"Ollama: {ollama_host}")

    openrouter_key = (
        os.getenv("LISA_OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    if openrouter_key:
        parts.append(f"OpenRouter: {_redact_key(openrouter_key)}")

    aws_access_key = (
        os.getenv("LISA_AWS_ACCESS_KEY_ID", "").strip()
        or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    )
    aws_secret_key = (
        os.getenv("LISA_AWS_SECRET_ACCESS_KEY", "").strip()
        or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    aws_region = (
        os.getenv("LISA_AWS_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )
    if aws_access_key and aws_secret_key and aws_region:
        parts.append(f"AWS Bedrock: {_redact_key(aws_access_key)} @ {aws_region}")
    elif aws_access_key or aws_secret_key or aws_region:
        parts.append("AWS Bedrock: partially configured")

    if not parts:
        return _VERIFY_WARN, "No provider keys configured"
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_paths(runtime: HostRuntime) -> tuple[str, str]:
    parts: list[str] = []
    parts.append(f"Data dir: {runtime.paths.data_dir}")
    parts.append(f"Thread logs: {runtime.paths.thread_dir}")
    tp = runtime.paths.thread_path
    if tp:
        parts.append(f"Thread path: {tp}")
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_session_context(runtime: HostRuntime) -> tuple[str, str]:
    if not runtime.config.user_slug:
        return "[yellow]NONE[/yellow]", "No session context active"
    parts = [f"Context: {runtime.config.user_slug}"]
    ref = runtime.config.session_context.strip()
    if ref and ref != runtime.config.user_slug:
        parts.append(f"Path: {ref}")
    return _VERIFY_PASS, "; ".join(parts)


@app.command("verify", help="Run verification checks of the Lisa environment.")
def _verify_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    """Run verification checks of the Lisa environment."""
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    checks: list[tuple[str, str, str]] = [
        ("Python", *_verify_check_python()),
        (".env", *_verify_check_env(runtime)),
        ("Session context", *_verify_check_session_context(runtime)),
        ("Model manifest", *_verify_check_manifest(runtime)),
        ("Pack catalog", *_verify_check_pack_catalog()),
        ("Provider keys", *_verify_check_provider_keys()),
        ("Ollama", *_verify_check_ollama()),
        ("Paths", *_verify_check_paths(runtime)),
    ]

    table = Table(title="lisa verify")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold", no_wrap=False)

    failures = 0
    for name, status, detail in checks:
        table.add_row(name, status, detail)
        if "FAIL" in status:
            failures += 1

    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# modes command
# ---------------------------------------------------------------------------


@app.command("modes", help="List supported Lisa modes.")
def _modes_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    render_modes(runtime, console=console)


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------


@app.command(
    "models",
    help=(
        "Manage the model manifest. ACTION defaults to 'show'. "
        "Subcommands: show, discover, apply, help. "
        "(reload and edit are REPL-only; they have no durable effect as a "
        "one-shot CLI invocation.)"
    ),
)
def _models_command(
    action: str = typer.Argument(
        "show",
        help=(
            "Action: show (default), discover, apply, help. "
            "'discover' writes a proposal manifest; 'apply' promotes it to active."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="With 'discover', overwrite an existing proposal manifest.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return

    normalized = action.strip().lower()

    # 'show' uses the Rich table renderer; on failure (e.g. no active
    # manifest yet) it falls back to the core text renderer so the user
    # gets the same friendly "run discover first" guidance as the REPL.
    if normalized in {"", "show"}:
        try:
            render_models(runtime, console=console)
            return
        except Exception:  # pylint: disable=broad-except
            normalized = "show"

    # 'reload' and 'edit' only mutate transient in-process state or print
    # guidance — both pointless for a one-shot CLI process whose runtime
    # is discarded on exit. Surface that clearly instead of pretending
    # to do work; the durable CLI workflow is discover -> (edit file) ->
    # apply. Both remain available in the persistent REPL.
    if normalized in {"reload", "edit"}:
        console.print(
            "[yellow]'models " + normalized + "' has no durable effect as a one-shot "
            "CLI command: it only refreshes in-process state or prints guidance, and "
            "the CLI runtime is discarded on exit.[/yellow]\n"
            "Edit models.proposed.yaml by hand, then run 'lisa models apply'.\n"
            "Use 'lisa' (REPL) for /models " + normalized + "."
        )
        return

    # show (fallback), discover, apply, help delegate to the
    # transport-neutral core handler (the same one the REPL /models
    # command uses), so behavior stays identical across hosts. We
    # synthesize the equivalent '/models <sub> [--force]' slash string
    # and print the returned Markdown text in a Rich panel.
    from host.commands import handle_models_command
    from rich.markdown import Markdown

    if normalized == "discover" and force:
        stripped = "/models discover --force"
    else:
        stripped = f"/models {normalized}".strip()

    # The discover/apply/reload/show subcommands read or write model
    # manifests via the lisa-storage MCP server, which requires a live
    # MCP tool pool. _initialize_runtime only compiles configs; the pool
    # is started inside the host's event loop. Start it here, run the
    # handler, and tear the pool down in finally — all within one
    # asyncio.run loop so anyio cancel scopes are entered and exited in
    # the same task (per host.runtime invariants).
    async def _run_models_command() -> str:
        try:
            await runtime.start()
            return await handle_models_command(runtime, stripped, stripped)
        finally:
            await runtime.aclose()

    response_text = asyncio.run(_run_models_command())
    console.print(Panel(Markdown(response_text), title="Models", border_style="cyan"))


# ---------------------------------------------------------------------------
# operations command
# ---------------------------------------------------------------------------


@app.command("operations", help="List internal operations with IDs for allowlist authoring.")
def _operations_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    render_operations(runtime, console=console)


# ---------------------------------------------------------------------------
# tools command
# ---------------------------------------------------------------------------


@app.command(
    "tools",
    help=(
        "List internal tools with IDs for allowlist authoring. "
        "Use the `mcp` command to inspect MCP server tools."
    ),
)
def _tools_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    render_tools(runtime, console=console)


McpIntrospection = tuple[dict[str, Any], dict[str, list[MCPToolSpec]]]
"""One MCP introspection result: ``(health, tools_by_server)``.

``health`` is a ``dict[str, McpServerHealth]`` keyed by ``server_id``;
``tools_by_server`` is a ``dict[str, list[MCPToolSpec]]`` grouping the
discovered tool specs by their owning server. Both are collected by
warming the MCP pool (see :func:`_collect_mcp_tools`). The CLI render
layer reads health for the summary table and the specs for the per-server
tool detail.
"""


def _build_mcp_health(
    pool: Any,
    is_alias: dict[str, bool],
) -> dict[str, McpServerHealth]:
    """Build a per-server health dict from the live pool's introspection accessors.

    Mirrors :func:`host.introspect.gather_mcp_pool_health` but runs
    in the CLI's short-lived ``asyncio.run`` event loop (the pool is started
    and torn down within ``_collect_mcp_tools``). Reads the same accessors
    (``warm_server_ids``, ``warm_failed_ids``, ``cold_failed_ids``,
    ``permanently_failed_ids``, ``restart_counts``, ``last_errors``,
    ``tools_by_server``) so the CLI summary and the REPL ``/mcp`` command
    show identical state.
    """
    from host.introspect import gather_mcp_pool_health

    health = gather_mcp_pool_health(pool)
    # Annotate alias tools on the tool_names so the render layer can mark
    # them without a separate lookup. The specs themselves carry the real
    # server_id; this is purely a display hint.
    _ = is_alias  # reserved for a future render-layer annotation pass
    return health


async def _collect_mcp_tools(runtime: HostRuntime) -> McpIntrospection:
    """Warm the MCP pool, return per-server health + discovered tool specs.

    MCP tool names are only discoverable by calling ``tools/list`` on each
    server, which requires a live session. This starts the pool (warm + cold
    configs), reads the discovered tool specs and per-server health, and
    tears the pool down — all inside one ``asyncio.run`` event loop to honor
    anyio's same-loop contract (the pool's async context managers must be
    entered and exited in the same loop). Cold servers that fail to start
    are skipped by the pool; their tools are simply absent from the
    returned list.

    Returns an empty :data:`McpIntrospection` when the runtime has no MCP
    servers configured or when pool startup fails, so the ``tools``/``mcp``
    commands degrade to showing only the internal tool catalog instead of
    crashing.
    """
    if not runtime.resources.mcp_warm_configs and not runtime.resources.mcp_cold_configs:
        return ({}, {})
    try:
        await runtime.start()
        pool = runtime.resources.mcp_tool_pool
        if pool is None:
            return ({}, {})
        tools = await pool.list_tools()
        # Capture is_alias flags BEFORE aclose() clears _synthetic_handlers.
        is_alias = {spec.name: pool.is_synthetic_tool(spec.name) for spec in tools}
        tools_by_server: dict[str, list[MCPToolSpec]] = {}
        for spec in tools:
            tools_by_server.setdefault(spec.server_id, []).append(spec)
        health = _build_mcp_health(pool, is_alias)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("MCP tool collection failed: %s", exc)
        return ({}, {})
    finally:
        await runtime.aclose()
    return (health, tools_by_server)


# ---------------------------------------------------------------------------
# mcp command
# ---------------------------------------------------------------------------


@app.command(
    "mcp",
    help=(
        "List MCP servers with status, transport, URL/command, tool count, "
        "and last error. Pass a server id to drill into that server's tools "
        "(name, description, required args)."
    ),
)
def _mcp_command(
    target: str = typer.Argument(
        "",
        help="Optional server id to drill into that server's tools.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    mcp_health, mcp_tools_by_server = asyncio.run(_collect_mcp_tools(runtime))
    if not mcp_health:
        console.print("[yellow]No MCP server tools discovered.[/yellow]")
        return
    server_id = target.strip()
    if server_id:
        specs = mcp_tools_by_server.get(server_id, [])
        health = mcp_health.get(server_id)
        if health is None:
            console.print(f"[yellow]No MCP server named {server_id!r} is configured.[/yellow]")
            return
        render_mcp_server_tools(specs, health, console=console)
    else:
        render_mcp_servers(mcp_health, console=console)


# ---------------------------------------------------------------------------
# pack command
# ---------------------------------------------------------------------------


@app.command("pack", help="Manage runtime pack installs (list/install/uninstall/update).")
def _pack_command(
    action: str = typer.Argument(..., help="Action: list, install, uninstall, update."),
    target: str = typer.Argument(
        "",
        help="Install spec or package name (required for install/uninstall/update <pack>).",
    ),
    all_packs: AllPacksOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return

    normalized = action.strip().lower()
    if normalized == "list":
        render_pack_list_panel(runtime, console=console)
        return
    if normalized == "install":
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_install(runtime, target)))
        return
    if normalized == "uninstall":
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_uninstall(runtime, target)))
        return
    if normalized == "update":
        if all_packs:
            _finish_pack_mutation(runtime, asyncio.run(_run_pack_update_all(runtime)))
            return
        if not target.strip():
            console.print("[red]Usage: lisa pack update <pack> | lisa pack update --all.[/red]")
            raise typer.Exit(code=2)
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_update(runtime, target.strip())))
        return

    console.print("[red]Unknown action. Use: list, install, uninstall, update.[/red]")
    raise typer.Exit(code=2)


def _finish_pack_mutation(runtime: HostRuntime, result: tuple[bool, str]) -> None:
    """Print a pack lifecycle result and apply the exit-code contract.

    Failure prints the details in red and exits 1; success prints the
    details and exits via :func:`_exit_restart_required` when the runtime
    flagged a restart, else 0.
    """
    ok, details = result
    if not ok:
        console.print(f"[red]{details}[/red]")
        raise typer.Exit(code=1)
    console.print(details)
    if runtime.session.restart_required:
        _exit_restart_required()
    raise typer.Exit(code=0)


async def _run_pack_install(runtime: HostRuntime, spec: str) -> tuple[bool, str]:
    """Install a pack by spec (delegates to host pack management)."""
    from host.commands import run_pack_install

    return await run_pack_install(runtime, spec)


async def _run_pack_uninstall(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Uninstall a pack by package name."""
    from host.commands import run_pack_uninstall

    return await run_pack_uninstall(runtime, package_name)


async def _run_pack_update(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Upgrade one installed pack in place from its recorded install source."""
    from host.commands import run_pack_update

    return await run_pack_update(runtime, package_name)


async def _run_pack_update_all(runtime: HostRuntime) -> tuple[bool, str]:
    """Upgrade every runtime-installed pack with a recorded install source."""
    from host.commands import run_pack_update_all

    return await run_pack_update_all(runtime)


# ---------------------------------------------------------------------------
# once command
# ---------------------------------------------------------------------------

# Prepended to the turn message in --raw mode: the response is consumed
# verbatim as a machine artifact, so the model must emit only the result.
# The yield prohibition is explicit: a prior turn ended in the `yield`
# tool (asking the operator for missing input), which wrote the question
# prose into redirected output as if it were the artifact.
_RAW_CONTRACT = (
    "OUTPUT CONTRACT (raw mode): Your entire response text is consumed verbatim "
    "as a machine artifact (redirected to a file or piped to another tool). "
    "Emit ONLY the requested result — no preamble, no closing commentary, no "
    "restating the task, no follow-up questions, and do not wrap the whole "
    "output in a markdown code fence. "
    "Do NOT call the yield/ask tool and do not end with a question: if required "
    "input is missing, make the best-supported inference from the provided "
    "material and state the assumption in a trailing comment (e.g. '# NOTE: ...'), "
    "then still emit the full result. A question or an empty response is a "
    "failed turn."
)


def _once_prearmed_decision(
    approve: str, deny: str, remember: str
) -> str | None:
    """Build the pre-armed decision command for a once turn, or None.

    ``--approve``/``--deny`` accept either a request id or, bare, the
    sentinel "pending" (resolving to the single pending approval via the
    host's recorded-owner identity). ``--remember <prefix>`` only applies
    to approve; deny + remember is a usage error.
    """
    approve = approve.strip()
    deny = deny.strip()
    remember = remember.strip()
    if approve and deny:
        console.print("[red]--approve and --deny are mutually exclusive.[/red]")
        raise typer.Exit(code=2)
    if remember and not approve:
        console.print("[red]--remember requires --approve.[/red]")
        raise typer.Exit(code=2)
    if not approve and not deny:
        if remember:
            console.print("[red]--remember requires --approve.[/red]")
            raise typer.Exit(code=2)
        return None
    if approve:
        target = approve if approve.lower() != _PENDING_APPROVAL_TARGET else ""
        cmd = "/approve" + (f" {target}" if target else "")
        if remember:
            cmd += f" --remember {remember}"
        return cmd
    target = deny if deny.lower() != _PENDING_APPROVAL_TARGET else ""
    return "/deny" + (f" {target}" if target else "")


async def _run_once_host_command_turn(
    runtime: HostRuntime, command: str
) -> tuple[bool, str]:
    """Run one host-command message (no model turn), then close the pool.

    Mirrors ``_run_once_with_teardown`` without transport rendering: the
    command path returns text (or a structured result) that this helper
    prints through the shared console. The MCP pool starts in this event
    loop and closes in the same one (anyio same-loop contract). A bare
    ``/approve`` / ``/deny`` (no id) resolves to the single pending
    approval first — the "decide the thing that just stopped me" case.
    """
    from host import NullTransport
    from host.handle_message import handle_message

    command = await _resolve_pending_approval_target(runtime, command)
    try:
        await runtime.start()
        result = await handle_message(runtime, command, NullTransport())
    finally:
        await runtime.aclose()
    text = getattr(result, "text", "") or str(result)
    return runtime.session.restart_required, text


async def _resolve_pending_approval_target(
    runtime: HostRuntime, command: str
) -> str:
    """Resolve a bare ``/approve`` / ``/deny`` to the single pending id.

    The approval store is process-durable and identity-scoped, so a fresh
    once process sees the request a prior once run left pending. With
    exactly one pending record the bare verb targets it; zero or multiple
    pendings leave the command untouched so the host replies with its own
    diagnostic (usage/expired/ambiguous).
    """
    verb, _, rest = command.partition(" ")
    if verb not in {"/approve", "/deny"} or rest.strip():
        return command
    service = getattr(runtime.services, "approval_service", None)
    if service is None or not hasattr(service, "list_pending"):
        return command
    from host.approval import approval_identity

    try:
        pending = await service.list_pending(approval_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("once: pending-approval resolution failed: %s", exc)
        return command
    if len(pending) == 1:
        request_id = pending[0].invocation.request_id
        logger.info("once: resolved bare %s to pending %s", verb, request_id[:8])
        return f"{verb} {request_id}"
    return command


def _run_once_host_command(command: str, **runtime_kwargs: Any) -> None:
    """Run a host command (policy/approve/deny) through a fresh runtime.

    The CLI's runtime options are forwarded verbatim; the command string is
    normalized to its slash form (``approve X`` -> ``/approve X``) and
    dispatched via handle_message exactly as the REPL would send it, so the
    host's command handling (identity fallback, policy writes, thread
    entries) is identical in both modes. Prints the reply and applies the
    restart-required exit contract when the runtime flags one.
    """
    verb, _, rest = command.partition(" ")
    if verb in {"approve", "deny", "policy"}:
        command = f"/{command}"
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    restart_required, text = asyncio.run(
        _run_once_host_command_turn(runtime, command)
    )
    if text:
        console.print(text)
    if restart_required:
        _exit_restart_required()
    raise typer.Exit(code=0)


def _compose_once_message(instruction: str, payload: str) -> str:
    """Compose the once-mode turn message from the CLI instruction and stdin.

    When both are present the payload is wrapped in a ``<piped-input>`` block
    ahead of the instruction so the model can tell data from directive;
    otherwise the non-empty one is the message.
    """
    instruction = instruction.strip()
    payload = payload.strip()
    if payload and instruction:
        return f"<piped-input>\n{payload}\n</piped-input>\n\n{instruction}"
    return payload or instruction


@app.command("once", help="Run a single turn and print the response.")
def _once_command(
    message: str = typer.Argument(
        "",
        help=(
            "Message to route through the host. When stdin is piped (not a tty), "
            "its contents are attached as the turn payload, e.g. "
            "`cat f.conf | lisa once -r 'fix this config'`. "
            "Subcommands that skip the turn: `lisa once policy [mode M|bare on|off]`, "
            "`lisa once approve <id> [--remember <prefix>]`, "
            "`lisa once deny <id>`."
        ),
    ),
    stdin: StdinOption = True,
    raw: RawOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    session_context_refresh: SessionContextRefreshOption = False,
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
    quiet: QuietOption = False,
    approve: OnceApproveOption = "",
    deny: OnceDenyOption = "",
    remember: OnceRememberOption = "",
) -> None:
    # ── Decision subcommands (no model turn) ──────────────────────────
    # `lisa once policy ...`, `lisa once approve <id>`, and
    # `lisa once deny <id>` route through handle_message like the REPL's
    # /policy, /approve, /deny: the host owns every decision and the
    # recorded-owner identity fallback makes a cross-process decision
    # valid. These run with no turn and exit 0 (or the command's error).
    # The grammar guard keeps plain prose ("deny the claim", "approve my
    # leave request") on the model path — only genuine command shapes
    # (bare verb, verb + id/--remember args, or a policy subcommand) are
    # intercepted. The host re-validates ids and flags anyway.
    once_arg = message.strip()
    if _ONCE_HOST_COMMAND_RE.match(once_arg.lower()):
        return _run_once_host_command(
            once_arg,
            log_level=log_level,
            log_path=log,
            no_log=no_log,
            data_dir=data_dir,
            thread_dir=thread_dir,
            no_thread=no_thread,
            user_slug=session_context,
            session_context_refresh=session_context_refresh,
            timezone=timezone,
            state_snapshot_path=state_snapshot,
            streaming=streaming,
            host_log_path=host_log,
        )

    payload = ""
    if stdin and not sys.stdin.isatty():
        try:
            payload = sys.stdin.read()
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[red]Failed to read piped stdin: {exc}[/red]")
            raise typer.Exit(code=2) from exc
    # A pre-armed decision (--approve/--deny) carries no message and no
    # piped payload; it is consumed at the end of the turn.
    decision_cmd = _once_prearmed_decision(approve, deny, remember)
    if not message.strip() and not payload.strip() and decision_cmd is None:
        console.print(
            "[red]No message and no piped input. Pass a message and/or pipe "
            "data on stdin.[/red]"
        )
        raise typer.Exit(code=2)
    full_message = _compose_once_message(message, payload)
    if decision_cmd is not None:
        # Pre-arm: the decision rides ahead of the turn message so the host
        # dispatches /approve|/deny FIRST (host-command dispatch runs before
        # the model turn), then executes the turn with the approval now
        # settled. --remember persists an allow rule for the decided prefix.
        full_message = f"{decision_cmd}\n\n{full_message}" if full_message else decision_cmd
    if raw:
        full_message = f"{_RAW_CONTRACT}\n\n{full_message}"
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        session_context_refresh=session_context_refresh,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
        in_process_mcp=True,
    )
    if runtime is None:
        return
    restart_required, transport = asyncio.run(
        _run_once_with_teardown(runtime, full_message, quiet=quiet, raw=raw)
    )
    if transport.approval_unresolved:
        # The approval request id is durable; the operator must decide it in
        # an interactive session. Exit non-zero so pipelines never mistake an
        # un-decided approval for a completed turn.
        raise typer.Exit(code=_APPROVAL_REQUIRED_EXIT)
    if transport.yield_unresolved:
        # The model ended in a yield (ask-the-operator). The question was
        # printed to stderr; stdout got nothing. Exit non-zero so pipelines
        # never mistake the question for the requested artifact.
        raise typer.Exit(code=_YIELD_UNRESOLVED_EXIT)
    if restart_required:
        _exit_restart_required()


async def _run_once_with_teardown(
    runtime: HostRuntime, message: str, *, quiet: bool = False, raw: bool = False
) -> tuple[bool, OnceTransport]:
    """Run one turn, then close the MCP pool.

    Returns ``(restart_required, transport)`` — the transport is returned so
    the caller can inspect post-turn state (``approval_unresolved``).

    In raw mode the transport console moves to stderr so activity output
    (λ echo, tool lines, ⚡ footer, errors) never lands in redirected stdout;
    the response itself is written verbatim to stdout by the transport.

    A message whose FIRST line is exactly ``/approve`` or ``/deny`` (the
    ``--approve``/``--deny`` pre-arm prefix) is resolved against the
    pending-approval store before the turn runs, so a bare
    ``lisa once --approve pending 'continue'`` targets the one stopped
    action without retyping its 32-hex id.
    """
    from host.handle_message import handle_message
    from hosts.cli.transport import OnceTransport

    first_line, _, _rest = message.partition("\n")
    if first_line.strip() in {"/approve", "/deny"}:
        message = await _resolve_pending_approval_target(runtime, first_line.strip()) \
            + ("\n\n" + _rest if _rest.strip() else "")
    transport_console: Console = Console(stderr=True) if raw else console
    transport = OnceTransport(console=transport_console, quiet=quiet, raw=raw)
    transport.start_turn(message)
    try:
        await runtime.start()
        await handle_message(runtime, message, transport)
    finally:
        transport.stop_activity()
        await runtime.aclose()
    return runtime.session.restart_required, transport


# ---------------------------------------------------------------------------
# intake command
# ---------------------------------------------------------------------------


@app.command("intake", help="Run one intake tick: collect signals, process, write state.")
def _intake_command(
    data_dir: DataDirOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot intake process. Triggered by systemd timer or cron.

    Collects signals from installed pack signal connectors, runs processing
    loops, and writes the merged UnifiedState to ``<data-dir>/state.json``.
    The next turn reads that state so the model has fresh ambient data.
    """
    resolved_data_dir = Path(data_dir).expanduser() if data_dir else default_data_dir()
    try:
        from host.intake import run_intake
    except ImportError:
        console.print("[red]intake module not available in this host.[/red]")
        raise typer.Exit(code=2)

    state = asyncio.run(run_intake(data_dir=resolved_data_dir))
    console.print(
        f"[green]Intake tick complete.[/green] "
        f"{len(state.signal_snapshots)} signal(s), "
        f"{len(state.active_alerts)} alert(s). "
        f"State written to {resolved_data_dir / 'state.json'}"
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# once-unknown pre-parse routing
# ---------------------------------------------------------------------------

# Group-level callback options that consume the following token as their
# value. Used by _first_positional_index to skip option values when locating
# the subcommand position. Keep in sync with the options on _cli_callback.
_VALUE_TAKING_GROUP_FLAGS = frozenset(
    {
        "--log-level",
        "--log",
        "--data-dir",
        "--overlays-dir",
        "--thread-dir",
        "--session-context",
        "--timezone",
        "--state-snapshot",
        "--streaming",
        "--host-log",
    }
)

_TRUTHY = {"1", "true", "yes", "on"}

# Once-only flags: valid on the `once` subcommand but not at group level.
# When once-unknown routing injects `once`, these are hoisted after it so
# Typer parses them as once options (`lisa -r "fix it"` -> `lisa once -r "fix it"`).
_ONCE_ONLY_FLAGS = frozenset({"-r", "--raw", "-q", "--quiet", "--stdin", "--no-stdin"})


def _once_unknown_active(args: list[str]) -> bool:
    """True if once-unknown routing is on (env var or --once-unknown flag)."""
    if os.getenv(_ONCE_UNKNOWN_ENV, "").strip().lower() in _TRUTHY:
        return True
    return _ONCE_UNKNOWN_FLAG in args


def _known_command_names() -> set[str]:
    """Names of every registered Typer subcommand plus the auto `help`."""
    names = {cmd.name for cmd in app.registered_commands if cmd.name}
    names.add("help")
    return names


def _first_positional_index(args: list[str]) -> int:
    """Index of the first positional (non-option) token.

    Skips group-level options and their values so a value like `debug` in
    `--log-level debug` is not mistaken for the subcommand. Returns -1 when
    there is no positional token (e.g. bare `lisa` or only options).
    """
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if tok == "--":
            return i + 1 if i + 1 < n else -1
        if tok.startswith("-") and tok != "-":
            if "=" in tok:
                i += 1
                continue
            if tok in _VALUE_TAKING_GROUP_FLAGS:
                i += 2  # flag + its value
                continue
            i += 1  # bool flag or unknown flag
            continue
        return i
    return -1


def _maybe_route_to_once(args: list[str]) -> list[str]:
    """Redirect an unrecognized first positional to the `once` subcommand.

    When once-unknown routing is active (``LISA_ONCE_UNKNOWN=1`` or the
    ``--once-unknown`` flag) and the first positional token is not a known
    subcommand, inject ``once`` before it so the trailing tokens become the
    one-shot message::

        lisa continue                       -> lisa once continue
        lisa "do something else"             -> lisa once "do something else"
        lisa --log-level debug continue      -> lisa --log-level debug once continue
        lisa -r "fix it"                     -> lisa once -r "fix it"

    Once-only flags (``-r/--raw``, ``-q/--quiet``, ``--stdin/--no-stdin``)
    appearing before the positional are hoisted to just after the injected
    ``once`` token — Typer rejects them as unknown group options otherwise.
    Tokens after a bare ``--`` are positional content and are never hoisted.

    Known subcommands (e.g. `verify`, `pack`) are never rerouted, so the
    flag is safe to leave on. The ``--once-unknown`` flag is stripped before
    Typer sees argv (it is not a registered Typer option) so it never leaks
    into a subcommand's option set.
    """
    active = _once_unknown_active(args)
    # The flag is a pre-parse-only switch; strip it so Typer never sees it.
    args = [a for a in args if a != _ONCE_UNKNOWN_FLAG]
    if not active:
        return args
    idx = _first_positional_index(args)
    if idx < 0:
        return args
    if args[idx] in _known_command_names():
        return args
    head = args[:idx]
    # A bare `--` terminates option parsing: tokens after it are positional
    # content, never flags, so only hoist once-only flags before it.
    sep = head.index("--") if "--" in head else len(head)
    hoisted = [a for a in head[:sep] if a in _ONCE_ONLY_FLAGS]
    if hoisted:
        kept = [a for a in head[:sep] if a not in _ONCE_ONLY_FLAGS]
        return [*kept, *head[sep:], "once", *hoisted, *args[idx:]]
    return [*args[:idx], "once", *args[idx:]]


def main() -> None:
    """CLI entry point for the Lisa terminal host."""
    app(args=_maybe_route_to_once(list(sys.argv[1:])))


if __name__ == "__main__":
    main()
