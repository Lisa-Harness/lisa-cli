"""hosts.cli.main — Typer app and CLI entry points for the terminal host.

This is the terminal-transport entry point for the Lisa framework. It
wraps the transport-neutral :mod:`host` runtime with Typer commands
and Rich rendering. Every command is a one-shot process: turns run via
``lisa prompt`` and lifecycle/inspection commands (verify, models, policy,
pack, mcp, last, intake) run once and exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple

import typer
from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env BEFORE injecting the packs venv site-packages so that
# LISA_DATA_DIR (and LISA_PACKS_VENV) are honored when resolving the
# packs venv (``<data_dir>/venv``) location. The loader only needs framework_root() and
# default_data_dir(), both of which are env-independent, so it can run
# safely before any host import. Search order (lowest precedence first —
# every existing candidate is loaded, later ones overriding earlier ones):
#   (1) CWD — for repo-mode dev sessions
#   (2) ~/.lisa/.env — for tool-install mode (uv tool install lisa)
#   (3) project root — repo-mode back-compat
#   (4) hosts/little_lisa/.env — legacy repo-mode back-compat
from runtime.paths import (
    framework_root as _framework_root,
)

try:
    from dotenv import load_dotenv

    _FRAMEWORK_ROOT = _framework_root()
    from runtime.config import default_data_dir as _default_data_dir
    from runtime.config import profile_data_dir as _profile_data_dir

    # Honor LISA_DATA_DIR (and LISA_PROFILE) here so tool-mode installs
    # configured with a custom data dir find their .env before any
    # HostRuntime exists to do the full config resolution. Falls back to
    # the platform default (~/.lisa). Mirrors HostConfig.from_env's
    # precedence: LISA_DATA_DIR > LISA_PROFILE > default_data_dir().
    _data_dir_override = os.getenv("LISA_DATA_DIR", "").strip()
    if _data_dir_override:
        _default_env_dir = Path(_data_dir_override).expanduser()
    else:
        _profile_override = os.getenv("LISA_PROFILE", "").strip()
        try:
            _default_env_dir = (
                _profile_data_dir(_profile_override) if _profile_override else _default_data_dir()
            )
        except ValueError:
            # Invalid profile name: HostConfig.from_env raises the real
            # startup error later; the .env loader falls back to the
            # platform default so the error panel (not an import crash)
            # is what the user sees.
            _default_env_dir = _default_data_dir()

    # Module-level search order, shared with _verify_check_env (which reads
    # this constant so monkeypatched tests and the loader stay in sync).
    # Ordered lowest-precedence first; _load_env_candidates applies them in
    # this order so later (more authoritative) entries win.
    #
    # CWD first is deliberate — repo-mode dev sessions expect the checkout's
    # own .env to be the base. It is NOT an early-return: loading only the
    # first hit let an incidental CWD .env (a stale repo copy, or any
    # directory the operator happens to be in) shadow the machine-level
    # data-dir .env completely. That silently dropped every setting the
    # authoritative file carries but the shadowing one lacks — the
    # 2026-09-21 headroom incident: HEADROOM_MCP_BIN expanded to "" from the
    # repo .env, so the pack's optional BINARY server failed to compile and
    # `lisa prompt` died with a Startup Error panel.
    #
    # TEST CONTRACT: this import-time load runs during pytest collection,
    # before any fixture can intervene, so operator .env values DO land in
    # os.environ inside test processes. hosts/cli/tests/conftest.py scrubs
    # them (autouse _hermetic_env) — tests own their environment; anything
    # ambient is treated as noise, not config.
    _ENV_SEARCH_CANDIDATES = (
        Path.cwd() / ".env",
        _default_env_dir / ".env",
        _FRAMEWORK_ROOT / ".env",
        _FRAMEWORK_ROOT / "hosts" / "little_lisa" / ".env",
    )

    def _load_env_candidates(candidates: tuple[Path, ...]) -> Path | None:
        """Load every existing candidate .env; return the last one loaded.

        Later candidates are more authoritative than earlier ones: they are
        applied with ``override=True`` so a machine-level data-dir .env can
        fill in (and correct) whatever a lower-precedence file got wrong.
        Within a single file the first definition still wins, so the
        documented "``LISA_DATA_DIR`` above wins when both are set" behavior
        is preserved by ``python-dotenv``'s own no-override default.
        """
        loaded: Path | None = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            if loaded is not None:
                # More authoritative than everything loaded so far.
                load_dotenv(dotenv_path=candidate, override=True)
            else:
                load_dotenv(dotenv_path=candidate, override=False)
            loaded = candidate
        return loaded

    _load_env_candidates(_ENV_SEARCH_CANDIDATES)
except ImportError:  # pragma: no cover — python-dotenv is optional
    pass

# Inject the packs venv site-packages into sys.path BEFORE importing
# host so that importlib.metadata.entry_points discovers packs
# installed via /pack install into the packs venv. This must run
# before any host / kernel import that triggers entry-point scanning.
from runtime.paths import (
    inject_packs_venv_site_packages as _inject_packs_venv_site_packages,
)

_inject_packs_venv_site_packages()

from runtime.config import default_data_dir

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
# body runs) does not pay the ~2s ``runtime.runtime`` ->
# ``kernel.ipsar_fsm`` -> ``openai`` import chain.
if TYPE_CHECKING:
    from runtime.runtime import HostRuntime

    from hosts.cli.transport import PromptTransport

# ``handle_message`` is deferred to the turn-execution call sites
# (``_run_prompt_with_teardown``) so ``--version`` / ``modes`` / ``models`` /
# ``verify`` don't pay the ~3s ``openai`` + ``mcp`` + ``kernel.ipsar_fsm``
# import cost it drags in. See ``host/__init__.py``'s PEP 562
# ``__getattr__`` for the package-level lazy import.

from rich.traceback import install

install(show_locals=True)

logger = logging.getLogger(__name__)

console = Console()


# ---------------------------------------------------------------------------
# Group command resolution — unknown routing + dynamic pack subcommands
# ---------------------------------------------------------------------------
# Click owns the argv grammar: no pre-Typer argv rewrites. A custom group
# class resolves unknowns (see ``_LisaGroup`` below) and dynamically
# generates commands for pack-registered host commands, so `lisa stalls`
# or `lisa retry <id>` work as real subcommands with --help and completion.


_TRUTHY = {"1", "true", "yes", "on"}

# Group-level hoist flags (``-r``/``-q``/``--no-stdin`` parsed at group level
# by ``_cli_callback``) recorded here for the prompt command to OR in when
# lazy routing lands there (``lisa -r "fix it"``). One CLI process runs the
# group callback at most once, before any command body.
_GROUP_HOIST: dict[str, bool] = {}

# Env var enabling lazy routing: an unrecognized first positional is
# resolved by the custom group (``_LisaGroup.resolve_command``) to the
# `prompt` subcommand so users can run `lisa continue` or
# `lisa "do something else"` without typing `prompt`. Click owns the argv
# grammar; no pre-Typer rewrites exist.
_PROMPT_UNKNOWN_ENV = "LISA_PROMPT_UNKNOWN"


def _prompt_unknown_active() -> bool:
    """True if prompt-unknown routing is on (LISA_PROMPT_UNKNOWN)."""
    return os.getenv(_PROMPT_UNKNOWN_ENV, "").strip().lower() in _TRUTHY


_pack_command_specs: dict[str, Any] | None = None
"""Cache of pack-registered host-command specs, keyed by bare name.

Lazily discovered on first resolution: ``{name: PackCommandEntry}`` where
``name`` is the first token of the spec prefix (``/stalls`` -> ``stalls``).
``None`` means discovery has not run yet.
"""


def _pack_commands() -> dict[str, Any]:
    """Return pack-registered host commands by bare name (discovered once).

    One pack may register several specs sharing a leading verb (``/retry``,
    ``/retry-safe``, ...); the bare-name entry keeps the pack's spec list so
    the generated command can dispatch to the longest matching prefix —
    the same matching order the runtime's own dispatcher uses.
    """
    global _pack_command_specs  # noqa: PLW0603  # module-level cache seam
    if _pack_command_specs is None:
        specs: dict[str, Any] = {}
        try:
            from kernel.host_command_discovery import (  # lazy: avoids eager pack scan
                discover_pack_host_commands,
            )

            for entry in discover_pack_host_commands():
                for spec in entry.specs:
                    name = spec.prefix.strip().lstrip("/").split(None, 1)[0]
                    if name:
                        specs.setdefault(name, entry)
        except Exception:  # pylint: disable=broad-except  # packs venv may be absent
            logger.debug("pack host-command discovery failed", exc_info=True)
        _pack_command_specs = specs
    return _pack_command_specs


def _make_pack_command(name: str) -> Any:
    """Build a click command dispatching one pack-registered host command.

    The generated command takes the same shared runtime options as the
    curated one-shot commands plus a catch-all ``ARGS`` argument list; the
    tokens are joined into the pack's slash-command string so the runtime
    dispatches to the pack's own handler exactly as a persistent host
    would (the pack handler owns the sub-grammar, e.g. ``/retry <id>``).
    """
    import click  # lazy: click is a Typer dependency

    entry = _pack_commands()[name]
    help_text = ""
    for spec in entry.specs:
        if spec.help_text:
            help_text = spec.help_text
            break

    @click.command(
        name=name,
        help=help_text or f"Run the {name} pack host command.",
    )
    @click.option("--log-level", default="", help="Log threshold: debug, info, warning, error.")
    @click.option("--log", default="", help="Append JSONL turn logs to this file.")
    @click.option("--no-log", is_flag=True, default=False, help="Disable turn logging.")
    @click.option("--data-dir", default="", help="Data root (overrides LISA_DATA_DIR).")
    @click.option("--profile", default="", help="Named data-dir profile.")
    @click.option("--overlays-dir", default="", help="User-authored overlays directory.")
    @click.option("--thread-dir", default="", help="Thread log directory.")
    @click.option("--no-thread", is_flag=True, default=False, help="Disable thread logging.")
    @click.option("--session-context", default="", help="Markdown session plan file path.")
    @click.option("--timezone", default="", help="IANA timezone.")
    @click.option("--state-snapshot", default="", help="Local JSON snapshot to inject.")
    @click.option("--streaming", default="", help="Streaming mode: none, stage, text, hybrid.")
    @click.option("--host-log", default="", help="Stdlib host log file.")
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def _pack_cmd(  # Click options mirror the shared runtime options.
        args: tuple[str, ...],
        log_level: str,
        log: str,
        no_log: bool,
        data_dir: str,
        profile: str,
        overlays_dir: str,
        thread_dir: str,
        no_thread: bool,
        session_context: str,
        timezone: str,
        state_snapshot: str,
        streaming: str,
        host_log: str,
    ) -> None:
        tokens = [name, *args]
        # Longest-prefix match mirrors runtime dispatch order (a bare
        # ``/retry`` must not shadow ``/retry-safe``). Prefixes arrive with
        # their leading slash (``/stalls``); the dispatched command is
        # rebuilt from the bare form.
        best = ""
        for spec in entry.specs:
            prefix = spec.prefix.strip().rstrip("/")
            bare_prefix = prefix.lstrip("/")
            if bare_prefix and tokens[0].startswith(bare_prefix) and len(bare_prefix) > len(best):
                best = bare_prefix
        command = "/" + (best if best else name)
        for token in args:
            command += " " + token
        _run_host_command(
            command,
            log_level=log_level,
            log_path=log,
            no_log=no_log,
            data_dir=data_dir,
            profile=profile,
            overlays_dir=overlays_dir,
            thread_dir=thread_dir,
            no_thread=no_thread,
            session_context=session_context,
            timezone=timezone,
            state_snapshot_path=state_snapshot,
            streaming=streaming,
            host_log_path=host_log,
        )

    return _pack_cmd


class _LisaGroup(typer.core.TyperGroup):
    """Typer group with unknown-command routing and dynamic pack commands.

    Click itself owns the argv grammar here (no pre-Typer rewrites):

    - Unknown first positional + ``LISA_PROMPT_UNKNOWN`` on → ``prompt``
      (the remaining tokens become the one-shot message; the group-level
      ``-r/-q`` hoist options carry the prompt-only flags).
    - A first positional matching a pack-registered host command name
      (``stalls``, ``retry``, ...) → a dynamically generated command that
      dispatches to the pack's handler (one process, no model turn).
    - A first positional starting with ``/`` → a usage error pointing at
      the named command (``lisa /settings`` → ``try 'lisa settings'``).
    """

    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        if cmd_name.startswith("/"):
            bare = cmd_name.lstrip("/") or None
            if bare and (super().get_command(ctx, bare) is not None or bare in _pack_commands()):
                raise typer.BadParameter(f"Slash commands are not subcommands. Try 'lisa {bare}'.")
            raise typer.BadParameter(
                f"Slash commands are not subcommands. Try 'lisa {cmd_name}' "
                "without the leading '/'."
            )
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        if cmd_name in _pack_commands():
            return _make_pack_command(cmd_name)
        return None

    def resolve_command(self, ctx: Any, args: list[str]) -> tuple[str | None, Any, list[str]]:
        cmd_name = args[0]
        cmd = self.get_command(ctx, cmd_name)
        if cmd is None and _prompt_unknown_active():
            prompt_cmd = super().get_command(ctx, "prompt")
            if prompt_cmd is not None:
                return "prompt", prompt_cmd, list(args)
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=_LisaGroup,
    no_args_is_help=True,
    help="Run the Lisa framework terminal host.",
)
policy_app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Inspect and manage approval policy without starting a model turn.",
)
policy_rule_app = typer.Typer(no_args_is_help=True, help="Manage allow, ask, and deny rules.")
policy_app.add_typer(policy_rule_app, name="rule")
app.add_typer(policy_app, name="policy")

# Lisa palette accent for response separators (kept in sync with the shared
# palette in transport.py — the same value PromptTransport uses for the
# "Response" rule).
_STYLE_RESPONSE = "dark_sea_green4"

# Exit code returned when a `prompt` turn ends with an unresolved approval
# request. A prompt run has no interactive loop to decide an approval, so
# the request stays durable and the process exits non-zero — scripts detect
# "needs approval" instead of mistaking it for a completed turn.
_APPROVAL_REQUIRED_EXIT = 3

# Exit code returned when a `prompt` turn ends in a model yield (the model
# asked the operator a question instead of finishing). The question is
# durable on the thread and printed to stderr; stdout gets nothing. Scripts
# detect "needs input" and re-run with the missing information in the prompt.
_YIELD_UNRESOLVED_EXIT = 4

# Exit code returned when the operator aborts a prompt turn with Ctrl-C
# (SIGINT). The conventional 130 = 128 + SIGINT distinguishes an
# operator-initiated abort from a failed turn so shells and scripts can
# treat it as "cancelled, not crashed" (the 2026-10-06 marathon-hang: ^C
# during a long glm-5.3:cloud generation surfaced as a 40-line uncaught
# CancelledError traceback instead of a one-line abort notice).
_ABORT_EXIT = 130

# Sentinel for --approve/--deny meaning "the (single) pending approval":
# resolved to the concrete request id before dispatch via the approval
# service's list_pending (recorded-owner identity makes cross-process
# pending records visible to a fresh process).
_PENDING_APPROVAL_TARGET = "pending"


def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("lisa-cli")
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
ProfileOption = Annotated[
    str,
    typer.Option(
        "--profile",
        help=(
            "Named data-dir profile: resolves to ~/.lisa/profiles/<name> "
            "(overrides LISA_PROFILE; --data-dir still wins). Shares the "
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
# ---------------------------------------------------------------------------
# Runtime initialization (builds HostRuntime, not _RUNTIME_STATE)
# ---------------------------------------------------------------------------


def _resolve_profiled_data_dir(data_dir: str, profile: str) -> str:
    """Resolve the effective data dir from --data-dir / --profile.

    Precedence (matching ``HostConfig.from_env``): explicit ``--data-dir``
    wins; then ``--profile`` (CLI arg); then the framework falls back to
    ``LISA_DATA_DIR`` > ``LISA_PROFILE`` > ``~/.lisa``. Returning an empty
    string hands the decision to the framework so the env-var chain stays
    in one place. An invalid profile name raises ``ValueError``, which
    ``_initialize_runtime_or_exit`` renders as a startup-error panel.
    """
    if data_dir.strip():
        return data_dir
    name = profile.strip()
    if not name:
        return ""
    from runtime.config import profile_data_dir

    return str(profile_data_dir(name))


def _initialize_runtime(
    *,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    profile: str = "",
    overlays_dir: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    session_context: str = "",
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
    in_process_mcp: bool = False,
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Thin adapter over :func:`runtime.bootstrap.build_runtime` that
    stamps ``host_kind="cli"`` and forwards every CLI argument. The
    composition sequence (config -> runtime -> state snapshot -> framework
    wiring -> MCP config compilation) lives in the shared bootstrap so
    the CLI and web hosts cannot drift.

    ``profile`` resolves to ``~/.lisa/profiles/<name>`` here (CLI arg >
    ``LISA_PROFILE``; an explicit ``data_dir`` still wins) because
    ``--data-dir`` and ``--profile`` are both CLI-level data-dir inputs —
    the framework's ``HostConfig.from_env`` only knows the env var.

    ``session_context`` is the optional path to a Markdown session plan
    file; ``user_slug`` is left to the framework (it derives from the
    compiled plan's manifest name). The MCP pool is NOT started here —
    ``runtime.start()`` must be called inside the host's event loop (see
    the prompt/verify paths).
    """
    from runtime.bootstrap import build_runtime

    return build_runtime(
        host_kind="cli",
        log_level=log_level,
        log_path=log_path,
        no_log=no_log,
        data_dir=_resolve_profiled_data_dir(data_dir, profile),
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        timezone=timezone,
        session_context=session_context,
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
        _render_runtime_error(console, exc, title="Startup Error")
        raise typer.Exit(code=2) from exc


def _render_runtime_error(console: Console, exc: Exception, *, title: str = "Error") -> None:
    """Render a runtime/startup error in a red Rich panel.

    The exception's first line is bolded up to its first colon so the cause
    reads at a glance. (Migrated from the retired REPL module; the prompt
    and startup paths are its remaining consumers.)
    """
    lines = str(exc).splitlines()
    if lines:
        head, sep, tail = lines[0].partition(":")
        if sep:
            lines[0] = f"[bold]{head.strip()}[/bold]:{tail}"
    body = "\n".join(lines) or exc.__class__.__name__
    console.print(Panel(body, title=title, border_style="indian_red"))


# ---------------------------------------------------------------------------
# Typer callback
# ---------------------------------------------------------------------------


@app.callback()
def _cli_callback(
    ctx: typer.Context,
    version: bool = _VERSION_OPTION,
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        hidden=True,
        help="With lazy prompt routing: hoist -r/--raw onto the injected prompt turn.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        hidden=True,
        help="With lazy prompt routing: hoist -q/--quiet onto the injected prompt turn.",
    ),
    no_stdin: bool = typer.Option(
        False,
        "--no-stdin",
        hidden=True,
        help="With lazy prompt routing: hoist --no-stdin onto the injected prompt turn.",
    ),
) -> None:
    # The group-level copies of the prompt-only flags exist so Click parses
    # `lisa -r "fix it"` as [group flags, prompt] instead of dying on an
    # unknown option; the values ride on this module-level slot for the
    # prompt command to OR in (Typer commands cannot read click's
    # thread-local context — see the hoist comment in ``_prompt_command``).
    # They are hidden (not documented) — the documented home for these flags
    # is `lisa prompt`.
    _GROUP_HOIST.update(raw=raw, quiet=quiet, no_stdin=no_stdin)
    ctx.obj = {"raw": raw, "quiet": quiet, "no_stdin": no_stdin}
    if version:
        console.print(f"lisa v{_get_version()}")
        raise typer.Exit()


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
    from runtime.model_manifest import load_model_manifest

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

    for provider, variable in (
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Google Gemini", "GOOGLE_API_KEY"),
    ):
        key = os.getenv(f"LISA_{variable}", "").strip() or os.getenv(variable, "").strip()
        if key:
            parts.append(f"{provider}: {_redact_key(key)}")

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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Run verification checks of the Lisa environment."""
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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

    from rich.box import SIMPLE_HEAD

    # No zebra row_styles here either: terminal backgrounds are unknown and
    # dark-grey rows vanish on black consoles (same rationale as
    # render._styled_table).
    table = Table(
        title="lisa verify",
        box=SIMPLE_HEAD,
        header_style="bold steel_blue3",
    )
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold", no_wrap=False)

    failures = 0
    warnings = 0
    for name, status, detail in checks:
        table.add_row(name, status, detail)
        if "FAIL" in status:
            failures += 1
        elif "WARN" in status:
            warnings += 1

    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    if warnings:
        console.print(f"\n[yellow]All mandatory checks passed ({warnings} warnings).[/yellow]")
    else:
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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
        "(reload and edit only refresh transient state or print guidance; "
        "they have no durable effect as a one-shot CLI invocation.)"
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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
            "Edit models.proposed.yaml by hand, then run 'lisa models apply'."
        )
        return

    # show (fallback), discover, apply, help delegate to the
    # transport-neutral core handler (the same one the REPL /models
    # command uses), so behavior stays identical across hosts. We
    # synthesize the equivalent '/models <sub> [--force]' slash string
    # and print the returned Markdown text in a Rich panel.
    from rich.markdown import Markdown
    from runtime.commands import handle_models_command

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
    # the same task (per runtime.runtime invariants).
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    render_tools(runtime, console=console)


# ---------------------------------------------------------------------------
# last command
# ---------------------------------------------------------------------------


@app.command(
    "last",
    help=(
        "Reprint a previous response verbatim (default: the last model turn). "
        "Pass a turn number to replay that turn's response: `lisa last 3`. "
        "Reads the durable thread — no model call."
    ),
)
def _last_command(
    turn: str = typer.Argument(
        "",
        help="Optional turn number to replay that turn's response.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    from rich.markdown import Markdown  # lazy: defers startup cost
    from rich.rule import Rule  # lazy: defers startup cost
    from runtime.commands import (  # lazy: defers runtime.commands startup cost
        resolve_last_turn,
    )

    turn_index, response_text = resolve_last_turn(runtime, turn.strip())
    if turn_index is not None and not console.is_terminal:
        # Piped/redirected: a one-line header so the replayed turn is
        # addressable in captured output, then the response verbatim —
        # no Markdown re-render (the raw contract; highlight/markup off
        # so Rich never interprets the body). Diagnostics (usage,
        # not-found, friendly messages) print bare as before.
        console.print(f"[ turn {turn_index} ]", style=_STYLE_RESPONSE, markup=False)
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    if not console.is_terminal:
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    # Interactive: the same treatment the original answer gets — a
    # ``Response · turn N`` rule separator plus Rich Markdown rendering
    # (colors, headings, code fences). The number is the global thread
    # turn index the prompt output already renders, so operators address
    # a replayed turn the same way wherever it surfaces.
    console.print(Rule(title=f"Response · turn {turn_index}", style=_STYLE_RESPONSE))
    console.print(Markdown(response_text))


# ---------------------------------------------------------------------------
# settings / tokens / debug — constant host-command verbs
# ---------------------------------------------------------------------------


def _render_markdown_response(response_text: str, *, title: str) -> None:
    """Print a transport-neutral host-command reply as a Rich panel.

    Non-tty output (pipes, redirects, CI) prints the raw text verbatim —
    no Markdown re-render — so piped output stays machine-readable (the
    same contract as PromptTransport raw mode).
    """
    if not console.is_terminal:
        console.print(response_text, highlight=False, markup=False, soft_wrap=False)
        return
    from rich.markdown import Markdown  # lazy: defers startup cost

    console.print(Panel(Markdown(response_text), title=title, border_style="cyan"))


@app.command("settings", help="Show the effective runtime settings (one-shot /settings).")
def _settings_command(  # Typer exposes shared runtime options as CLI flags.
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Render the effective runtime settings without a model turn."""
    _dispatch_constant_command(
        "/settings",
        title="Settings",
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


@app.command("tokens", help="Show session token usage and budget (one-shot /tokens).")
def _tokens_command(  # Typer exposes shared runtime options as CLI flags.
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Render token accounting without a model turn."""
    _dispatch_constant_command(
        "/tokens",
        title="Tokens",
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


@app.command(
    "debug",
    help=(
        "Show the complete, unedited model input for a previous turn "
        "(default: the last turn): `lisa debug [N]`."
    ),
)
def _debug_command(  # Typer exposes shared runtime options as CLI flags.
    turn: str = typer.Argument(
        "",
        help="Optional turn number whose model input to render.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Render one turn's complete model input without a model turn."""
    _dispatch_constant_command(
        f"/debug {turn.strip()}".strip(),
        title="Debug",
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


def _dispatch_constant_command(command: str, *, title: str, **runtime_kwargs: Any) -> None:
    """Run one constant host command through a fresh runtime and render it.

    The command is one of the host's fixed admin commands (``/settings``,
    ``/tokens``, ``/debug N``) — no user-controlled grammar, so dispatching
    via ``handle_message`` costs nothing in parsing fidelity while keeping
    the runtime's own bookkeeping (thread entry, turn log) intact.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    text = asyncio.run(_run_host_command_turn(runtime, command))
    _render_markdown_response(text, title=title)


McpIntrospection = tuple[dict[str, Any], dict[str, list[MCPToolSpec]]]
"""One MCP introspection result: ``(health, tools_by_server)``.

``health`` is a ``dict[str, McpServerHealth]`` keyed by ``server_id``;
``tools_by_server`` is a ``dict[str, list[MCPToolSpec]]`` grouping the
discovered tool specs by their owning server. Both are collected by
warming the MCP pool (see :func:`_collect_mcp_tools`). The CLI render
layer reads health for the summary table and the specs for the per-server
tool detail.
"""


def _mcp_configs(runtime: Any, *, warm: bool) -> list[Any]:
    """Return the runtime's warm or cold MCP configs.

    Reads ``runtime.resources.mcp_warm_configs`` / ``mcp_cold_configs`` on a
    real :class:`HostRuntime`. Falls back to flat ``mcp_warm_configs`` /
    ``mcp_cold_configs`` attributes so test stubs need not model the
    ``resources`` indirection.
    """
    attr = "mcp_warm_configs" if warm else "mcp_cold_configs"
    resources = getattr(runtime, "resources", None)
    configs = getattr(resources, attr, None)
    if configs is None:
        configs = getattr(runtime, attr, None)
    return list(configs or [])


def _mcp_tool_pool(runtime: Any) -> Any:
    """Return the runtime's MCP tool pool (``resources`` indirection tolerated)."""
    resources = getattr(runtime, "resources", None)
    pool = getattr(resources, "mcp_tool_pool", None)
    if pool is None:
        pool = getattr(runtime, "mcp_tool_pool", None)
    return pool


def _build_mcp_health(
    pool: Any,
    is_alias: dict[str, bool],
) -> dict[str, McpServerHealth]:
    """Build a per-server health dict from the live pool's introspection accessors.

    Mirrors :func:`runtime.introspect.gather_mcp_pool_health` but runs
    in the CLI's short-lived ``asyncio.run`` event loop (the pool is started
    and torn down within ``_collect_mcp_tools``). Reads the same accessors
    (``warm_server_ids``, ``warm_failed_ids``, ``cold_failed_ids``,
    ``permanently_failed_ids``, ``restart_counts``, ``last_errors``,
    ``tools_by_server``) so the CLI summary shows identical state to any
    other host surface.
    """
    from runtime.introspect import gather_mcp_pool_health

    health = gather_mcp_pool_health(pool)
    # Annotate alias tools on the tool_names so the render layer can mark
    # them without a separate lookup. The specs themselves carry the real
    # server_id; this is purely a display hint.
    _ = is_alias  # reserved for a future render-layer annotation pass
    return health


async def _aclose_runtime_bounded(runtime: HostRuntime, *, timeout_s: float = 30.0) -> None:
    """Tear a runtime down without ever hanging the one-shot CLI command.

    ``runtime.aclose()`` closes the MCP pool's anyio contexts and terminates
    proxy subprocesses. When those teardown awaits stall (the 2026-09-21
    headroom incident: the loop sat idle in select after sequential anyio
    task-group exits), a plain ``await runtime.aclose()`` never returns and
    the CLI prints nothing. Bound the graceful path with
    :func:`asyncio.wait_for`; on timeout, fall back to terminating every
    tracked MCP/proxy subprocess directly so no orphans survive.
    """
    try:
        await asyncio.wait_for(runtime.aclose(), timeout=timeout_s)
        return
    except TimeoutError:
        logger.warning(
            "Runtime teardown exceeded %.0fs; falling back to direct subprocess termination",
            timeout_s,
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # pylint: disable=broad-except
        logger.debug("Runtime aclose raised during bounded teardown", exc_info=True)
        return
    # Forced cleanup: kill every subprocess the pools still track.
    resources = runtime.resources
    pools: list[Any] = []
    proxy_pool = getattr(resources, "_proxy_pool", None)
    if proxy_pool is not None:
        pools.append(proxy_pool)
    tool_pool = getattr(resources, "_mcp_tool_pool", None)
    if tool_pool is not None:
        pools.append(tool_pool)
    for pool in pools:
        await _force_close_pool(pool)


async def _force_close_pool(pool: Any) -> None:
    for process in list(getattr(pool, "_processes", {}).values()):
        try:
            if process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        except Exception:  # pylint: disable=broad-except
            logger.debug("Process kill failed during forced teardown", exc_info=True)
    for stack in list(getattr(pool, "_child_stacks", {}).values()):
        try:
            await stack.aclose()
        except Exception:  # pylint: disable=broad-except
            logger.debug("Child stack close failed during forced teardown", exc_info=True)


async def _collect_mcp_tools(
    runtime: HostRuntime,
    *,
    connect_server_ids: set[str] | None = None,
) -> McpIntrospection:
    """Warm the MCP pool, return per-server health + discovered tool specs.

    MCP tool names are only discoverable by calling ``tools/list`` on each
    server, which requires a live session. This starts the pool, reads the
    discovered tool specs and per-server health, and tears the pool down —
    all inside one ``asyncio.run`` event loop to honor anyio's same-loop
    contract (the pool's async context managers must be entered and exited
    in the same loop).

    By default ONLY warm servers are connected: ``list_tools`` lazily
    connects cold servers, and sequentially exiting several cold stdio
    anyio task groups in one task poisons that task — every subsequent
    await is cancelled by a stale anyio cancel scope (the 2026-09-21
    headroom incident: ``lisa mcp`` printed nothing and exited 1). Cold
    servers stay disconnected in the summary (status ``cold``); ``lisa mcp
    <server_id>`` drill-down passes the single target via
    ``connect_server_ids`` so at most one cold context is ever opened and
    closed in the CLI's task.

    Returns an empty :data:`McpIntrospection` when the runtime has no MCP
    servers configured or when pool startup fails, so the ``tools``/``mcp``
    commands degrade to showing only the internal tool catalog instead of
    crashing.
    """
    warm_ids = {c.server_id for c in _mcp_configs(runtime, warm=True) if hasattr(c, "server_id")}
    if not warm_ids and not _mcp_configs(runtime, warm=False):
        return ({}, {})
    allowed = warm_ids | (connect_server_ids or set())
    health: dict[str, Any] = {}
    tools_by_server: dict[str, list[MCPToolSpec]] = {}
    try:
        await runtime.start()
        pool = _mcp_tool_pool(runtime)
        if pool is None:
            return ({}, {})
        tools = await pool.list_tools(allowed_server_ids=allowed)
        # Capture is_alias flags BEFORE aclose() clears _synthetic_handlers.
        is_alias = {spec.name: pool.is_synthetic_tool(spec.name) for spec in tools}
        for spec in tools:
            tools_by_server.setdefault(spec.server_id, []).append(spec)
        health = _build_mcp_health(pool, is_alias)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("MCP tool collection failed: %s", exc)
        return ({}, {})
    finally:
        await _aclose_runtime_bounded(runtime)
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    server_id = target.strip()
    # Warm servers only by default — cold servers stay disconnected so the
    # one-shot command never sequentially exits multiple cold stdio task
    # groups in one task (anyio cancel-scope poison, 2026-09-21). Drill-down
    # adds exactly ONE cold target to the same pass: a single cold context
    # open/close in the CLI's task is safe (verified), while a second
    # in-process runtime init is not (stale global tool catalog breaks the
    # operation-catalog re-validation on the second boot).
    mcp_health, mcp_tools_by_server = asyncio.run(
        _collect_mcp_tools(runtime, connect_server_ids={server_id} if server_id else None)
    )
    if not mcp_health:
        console.print("[yellow]No MCP server tools discovered.[/yellow]")
        return
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
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
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
    details and exits 0 — the runtime reloads pack state on next boot.
    """
    ok, details = result
    if not ok:
        console.print(f"[red]{details}[/red]")
        raise typer.Exit(code=1)
    console.print(details)
    raise typer.Exit(code=0)


async def _run_pack_install(runtime: HostRuntime, spec: str) -> tuple[bool, str]:
    """Install a pack by spec (delegates to host pack management)."""
    from runtime.commands import run_pack_install

    return await run_pack_install(runtime, spec)


async def _run_pack_uninstall(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Uninstall a pack by package name."""
    from runtime.commands import run_pack_uninstall

    return await run_pack_uninstall(runtime, package_name)


async def _run_pack_update(runtime: HostRuntime, package_name: str) -> tuple[bool, str]:
    """Upgrade one installed pack in place from its recorded install source."""
    from runtime.commands import run_pack_update

    return await run_pack_update(runtime, package_name)


async def _run_pack_update_all(runtime: HostRuntime) -> tuple[bool, str]:
    """Upgrade every runtime-installed pack with a recorded install source."""
    from runtime.commands import run_pack_update_all

    return await run_pack_update_all(runtime)


# ---------------------------------------------------------------------------
# prompt command
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


class _PrearmedDecision(NamedTuple):
    """One pre-armed approval decision parsed from prompt CLI flags."""

    request_id: str
    approve: bool
    remember: str


def _prompt_prearmed_decision(approve: str, deny: str, remember: str) -> _PrearmedDecision | None:
    """Parse the pre-armed decision flags for a prompt turn, or None.

    ``--approve``/``--deny`` accept either a request id or, bare, the
    sentinel "pending" (an empty ``request_id`` here, resolved to the
    single pending approval via the host's recorded-owner identity).
    ``--remember <prefix>`` only applies to approve; deny + remember is a
    usage error.
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
        return None
    if approve:
        target = "" if approve.lower() == _PENDING_APPROVAL_TARGET else approve
        return _PrearmedDecision(request_id=target, approve=True, remember=remember)
    target = deny if deny.lower() != _PENDING_APPROVAL_TARGET else ""
    return _PrearmedDecision(request_id=target, approve=False, remember="")


async def _run_host_command_turn(runtime: HostRuntime, command: str) -> str:
    """Run one host-command message (no model turn), then close the pool.

    Mirrors the prompt-turn path without transport rendering: the
    command path returns text (or a structured result) that this helper
    prints through the shared console. The MCP pool starts in this event
    loop and closes in the same one (anyio same-loop contract).
    """
    from runtime import NullTransport
    from runtime.handle_message import handle_message

    try:
        await runtime.start()
        result = await handle_message(runtime, command, NullTransport())
    finally:
        await runtime.aclose()
    return getattr(result, "text", "") or str(result)


async def _resolve_pending_request_id(runtime: HostRuntime) -> str:
    """Return the single pending approval's id, or '' when ambiguous/none.

    The approval store is process-durable and identity-scoped, so a fresh
    prompt process sees the request a prior prompt run left pending. With
    exactly one pending record the bare decision targets it; zero or
    multiple pendings return '' so the typed core replies with its own
    diagnostic (usage/expired/ambiguous).
    """
    service = getattr(runtime.services, "approval_service", None)
    if service is None or not hasattr(service, "list_pending"):
        return ""
    from runtime.approval import approval_identity

    try:
        pending = await service.list_pending(approval_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("prompt: pending-approval resolution failed: %s", exc)
        return ""
    if len(pending) == 1:
        request_id = pending[0].invocation.request_id
        logger.info("prompt: resolved bare decision to pending %s", request_id[:8])
        return request_id
    return ""


async def _decide_approval_turn(
    runtime: HostRuntime,
    *,
    request_id: str,
    approve: bool,
    remember: str,
) -> str:
    """Decide one approval through the typed core, then close the pool.

    Mirrors ``_run_host_command_turn``: no transport rendering — the
    decision path returns text (or a structured continuation result) that
    this helper prints through the shared console. A bare decision (empty
    ``request_id``) resolves to the single pending approval first — the
    "decide the thing that just stopped me" case.
    """
    if not request_id:
        request_id = await _resolve_pending_request_id(runtime)
    from runtime import NullTransport
    from runtime.handle_message import decide_pending_approval

    try:
        await runtime.start()
        result = await decide_pending_approval(
            runtime,
            request_id,
            approve=approve,
            # The CLI's --remember is a value flag (explicit prefix or
            # absent): the default-prefix behavior belongs to the slash
            # grammar, not here. No --remember means no rule write.
            remember=remember or None,
            remember_default=False,
            transport=NullTransport(),
        )
    finally:
        await runtime.aclose()
    return getattr(result, "text", "") or str(result)


def _run_decision(
    request_id: str,
    *,
    approve: bool,
    remember: str,
    **runtime_kwargs: Any,
) -> None:
    """Decide one pending approval through the typed core in a fresh runtime.

    The CLI's runtime options are forwarded verbatim; the decision is a
    value (id + direction + optional remember prefix), never a command
    string, so no quoting round trip exists between the CLI and the host.
    Prints the reply and exits 0.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    text = asyncio.run(
        _decide_approval_turn(runtime, request_id=request_id, approve=approve, remember=remember)
    )
    if text:
        console.print(text)
    raise typer.Exit(code=0)


def _run_host_command(command: str, **runtime_kwargs: Any) -> None:
    """Run a host-owned admin command through a fresh runtime.

    The CLI's runtime options are forwarded verbatim; the command string is
    normalized to its slash form and dispatched via ``handle_message``
    exactly as a persistent host would send it. This keeps pack commands and
    other host-admin replies host-owned while bypassing the model-turn path.
    Prints the reply and exits 0.
    """
    if not command.startswith("/"):
        command = "/" + command
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    text = asyncio.run(_run_host_command_turn(runtime, command))
    if text:
        console.print(text)
    raise typer.Exit(code=0)


def _run_policy_action(action: Callable[[HostRuntime], str], **runtime_kwargs: Any) -> None:
    """Run one typed policy action against a fresh runtime.

    ``action`` receives the runtime and returns the reply text — the typed
    cores (:func:`set_policy_mode`, :func:`add_policy_rule`, ...) take
    values, never command strings, so no quoting round trip exists between
    the CLI flags and the policy write. The host owns the write.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return

    console.print(action(runtime))
    raise typer.Exit(code=0)


@policy_app.callback()
def _policy_group(  # Typer exposes shared runtime options as CLI flags.
    ctx: typer.Context,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Set up shared host options for policy commands; bare `policy` shows status."""
    ctx.obj = {
        "log_level": log_level,
        "log_path": log,
        "no_log": no_log,
        "data_dir": data_dir,
        "profile": profile,
        "overlays_dir": overlays_dir,
        "thread_dir": thread_dir,
        "no_thread": no_thread,
        "session_context": session_context,
        "timezone": timezone,
        "state_snapshot_path": state_snapshot,
        "streaming": streaming,
        "host_log_path": host_log,
    }
    if ctx.invoked_subcommand is None:

        def _show_document(runtime: HostRuntime) -> str:
            from runtime.commands import render_policy_document

            return render_policy_document(runtime)

        _run_policy_action(_show_document, **ctx.obj)


@policy_app.command("mode", help="Set the fallback approval mode: ask, trust, or plan.")
def _policy_mode(ctx: typer.Context, mode: Literal["ask", "trust", "plan"]) -> None:
    def _set_mode(runtime: HostRuntime) -> str:
        from runtime.commands import set_policy_mode
        from runtime.policy import PolicyMode

        return set_policy_mode(runtime, PolicyMode(mode))

    _run_policy_action(_set_mode, **ctx.obj)


@policy_app.command("bare", help="Toggle approval for unsandboxed execute actions.")
def _policy_bare(ctx: typer.Context, setting: Literal["on", "off"]) -> None:
    def _set_bare(runtime: HostRuntime) -> str:
        from runtime.commands import set_policy_bare

        return set_policy_bare(runtime, setting == "on")

    _run_policy_action(_set_bare, **ctx.obj)


@policy_rule_app.command("list", help="List allow, ask, and deny rules with remove addresses.")
def _policy_rule_list(ctx: typer.Context) -> None:
    def _list_rules(runtime: HostRuntime) -> str:
        from runtime.commands import render_policy_rules

        return render_policy_rules(runtime)

    _run_policy_action(_list_rules, **ctx.obj)


@policy_rule_app.command("remove", help="Remove a rule by its list address, e.g. allow:1.")
def _policy_rule_remove(ctx: typer.Context, address: str) -> None:
    def _remove_rule(runtime: HostRuntime) -> str:
        from runtime.commands import remove_policy_rule

        return remove_policy_rule(runtime, address)

    _run_policy_action(_remove_rule, **ctx.obj)


@policy_rule_app.command("add", help="Add an allow, ask, or deny rule.")
def _policy_rule_add(  # Typer requires each rule matcher as a CLI option.
    ctx: typer.Context,
    rule_list: Literal["allow", "ask", "deny"],
    tool: str = typer.Option("", "--tool", help="Exact capability tool ID."),
    effect: str = typer.Option("", "--effect", help="Exact effect kind."),
    pack: str = typer.Option("", "--pack", help="Exact owning pack ID."),
    prefix: str = typer.Option("", "--prefix", help="Command prefix matcher."),
    when: str = typer.Option("", "--when", help="CEL expression; quote the whole expression."),
) -> None:
    """Parse policy matchers with Typer and pass their values to the host policy writer."""

    def _add_rule(runtime: HostRuntime) -> str:
        from runtime.commands import add_policy_rule

        return add_policy_rule(
            runtime,
            rule_list,
            tool=tool,
            effect=effect,
            pack=pack,
            command_prefix=prefix,
            when=when,
        )

    _run_policy_action(_add_rule, **ctx.obj)


@app.command("approve", help="Approve one pending action without starting a model turn.")
def _approve_command(  # Match shared runtime options used by one-shot commands.
    request_id: str = typer.Argument(
        _PENDING_APPROVAL_TARGET,
        help="Approval request ID, or omit it to target the single pending request.",
    ),
    remember: str = typer.Option(
        "",
        "--remember",
        help="Persist an allow rule for the action's command prefix.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    target = "" if request_id.lower() == _PENDING_APPROVAL_TARGET else request_id
    _run_decision(
        target,
        approve=True,
        remember=remember,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


@app.command("deny", help="Deny one pending action without starting a model turn.")
def _deny_command(  # Match shared runtime options used by one-shot commands.
    request_id: str = typer.Argument(
        _PENDING_APPROVAL_TARGET,
        help="Approval request ID, or omit it to target the single pending request.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    target = "" if request_id.lower() == _PENDING_APPROVAL_TARGET else request_id
    _run_decision(
        target,
        approve=False,
        remember="",
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


@app.command(
    "run",
    hidden=True,
    help=(
        "[DEPRECATED] Run one slash command (host admin or pack-defined). "
        "Every host command now has a real subcommand (pack-registered ones "
        "like `stalls`/`retry` are generated from the pack registry); this "
        "escape hatch remains one release for compatibility. The leading '/' "
        "is optional. Quote multi-word arguments as one string; with multiple "
        "tokens they are re-quoted verbatim so the runtime sees the exact "
        "slash-command string."
    ),
)
def _run_command_command(  # Match shared runtime options used by one-shot commands.
    command: Annotated[
        list[str] | None,
        typer.Argument(help="Slash command and its arguments."),
    ] = None,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Dispatch one slash command to the runtime exactly as a persistent
    host would, via ``handle_message`` and ``_run_host_command_turn``.

    DEPRECATED: kept one release for compatibility. Every host admin verb
    now has a tailored Typer command; pack-registered commands (``stalls``,
    ``retry``, ...) resolve as dynamically generated subcommands. This
    generic string escape hatch hides from help and will be removed once
    scripts have migrated.
    """
    console.print(
        "[yellow]`lisa run` is deprecated: host commands are real subcommands "
        "now (`lisa settings`, `lisa stalls`, ...).[/yellow]"
    )
    if not command:
        console.print("[red]No slash command given. Example: lisa run '/policy mode trust'[/red]")
        raise typer.Exit(code=2)
    parts = list(command)
    command_str = parts[0] if len(parts) == 1 else " ".join(shlex.quote(part) for part in parts)
    if not command_str.startswith("/"):
        command_str = "/" + command_str
    _run_host_command(
        command_str,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


def _compose_prompt_message(instruction: str, payload: str) -> str:
    """Compose the prompt-mode turn message from the CLI instruction and stdin.

    When both are present the payload is wrapped in a ``<piped-input>`` block
    ahead of the instruction so the model can tell data from directive;
    otherwise the non-empty one is the message.
    """
    instruction = instruction.strip()
    payload = payload.strip()
    if payload and instruction:
        return f"<piped-input>\n{payload}\n</piped-input>\n\n{instruction}"
    return payload or instruction


@app.command("prompt", help="Run a single turn and print the response.")
def _prompt_command(
    message: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Message to route through the host; multiple words are joined "
                "with spaces, so short prompts work unquoted: "
                "`lisa prompt fix the config parser`. Quote only to preserve "
                "inner spaces, quote characters, or shell-expansion semantics. "
                "When stdin is piped (not a tty), its contents are attached as "
                "the turn payload, e.g. "
                "`cat f.conf | lisa prompt -r 'fix this config'`. "
                "Approval decisions that skip the turn: `lisa approve <id> "
                "[--remember <prefix>]` and `lisa deny <id>`."
            ),
        ),
    ] = None,
    stdin: StdinOption = True,
    raw: RawOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
    quiet: QuietOption = False,
    tool_detail: ToolDetailOption = 0,
    approve: PromptApproveOption = "",
    deny: PromptDenyOption = "",
    remember: PromptRememberOption = "",
) -> None:
    # Group-level hoist: `lisa -r "fix it"` parses -r at group level (see
    # _cli_callback) and lazy-routing lands here; OR the hoisted values in.
    # Typer commands cannot reliably read click's thread-local context
    # (get_current_context returns None inside a Typer command body), so
    # the group callback records the hoisted flags on this module-level
    # slot — one CLI process dispatches exactly one group callback before
    # the command body runs.
    group_obj = _GROUP_HOIST
    raw = raw or bool(group_obj.get("raw"))
    quiet = quiet or bool(group_obj.get("quiet"))
    stdin = stdin and not bool(group_obj.get("no_stdin"))
    quiet = quiet or bool(group_obj.get("quiet"))
    stdin = stdin and not bool(group_obj.get("no_stdin"))
    # Pre-armed decisions are handled below; standalone approve/deny
    # commands are registered with Typer and never enter this function.
    words = message or []
    full_instruction = " ".join(words)
    payload = ""
    if stdin and not sys.stdin.isatty():
        try:
            payload = sys.stdin.read()
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[red]Failed to read piped stdin: {exc}[/red]")
            raise typer.Exit(code=2) from exc
    # A pre-armed decision (--approve/--deny) carries no message and no
    # piped payload; it is consumed at the end of the turn.
    decision = _prompt_prearmed_decision(approve, deny, remember)
    if not full_instruction.strip() and not payload.strip() and decision is None:
        console.print(
            "[red]No message and no piped input. Pass a message and/or pipe data on stdin.[/red]"
        )
        raise typer.Exit(code=2)
    full_message = _compose_prompt_message(full_instruction, payload)
    if raw:
        full_message = f"{_RAW_CONTRACT}\n\n{full_message}"
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
        in_process_mcp=True,
    )
    if runtime is None:
        return
    try:
        transport = asyncio.run(
            _run_prompt_with_teardown(
                runtime,
                full_message,
                quiet=quiet,
                raw=raw,
                tool_detail=tool_detail,
                streaming_mode=runtime.config.streaming.mode,
                decision=decision,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        # A SIGINT mid-turn cancels the asyncio.run loop, surfacing as
        # CancelledError here (typer does not convert it). Print the
        # one-line abort notice — not the 40-line traceback the framework
        # excepthook logs — and exit with the conventional SIGINT code so
        # shells read "cancelled", not "crashed". Raw mode keeps stdout
        # clean for pipelines: the notice goes to stderr like all other
        # activity output.
        abort_console = Console(stderr=True) if raw else console
        abort_console.print("[red]⏹ Turn aborted by operator (Ctrl-C).[/red]")
        raise typer.Exit(code=_ABORT_EXIT) from None
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
    raise typer.Exit(code=0)


async def _run_prompt_with_teardown(
    runtime: HostRuntime,
    message: str,
    *,
    quiet: bool = False,
    raw: bool = False,
    tool_detail: int = 0,
    streaming_mode: str = "none",
    decision: _PrearmedDecision | None = None,
) -> PromptTransport:
    """Run one turn (and any pre-armed decision), then close the MCP pool.

    A pre-armed decision (``--approve``/``--deny``) is dispatched FIRST,
    through the typed approval core — one dispatch, one value, no command
    string — so the pending approval the prior run left behind is settled
    before the turn runs with it. ``--remember`` persists an allow rule for
    the decided prefix (default prefix when no explicit value is given).

    Returns the transport so the caller can inspect post-turn state
    (``approval_unresolved``).

    In raw mode the transport console moves to stderr so activity output
    (λ echo, tool lines, ⚡ footer, errors) never lands in redirected stdout;
    the response itself is written verbatim to stdout by the transport.
    """
    from runtime.handle_message import (  # lazy: defers turn import cost
        decide_pending_approval,
        handle_message,
    )

    from hosts.cli.transport import PromptTransport  # lazy: defers transport import

    transport_console: Console = Console(stderr=True) if raw else console
    transport = PromptTransport(
        console=transport_console,
        quiet=quiet,
        raw=raw,
        tool_detail=tool_detail,
        streaming_mode=streaming_mode,
    )
    transport.model_prices = _collect_model_prices(runtime)
    transport.model_cache_prices = _collect_model_cache_prices(runtime)
    transport.start_turn(message)
    try:
        await runtime.start()
        if decision is not None:
            request_id = decision.request_id
            if not request_id:
                request_id = await _resolve_pending_request_id(runtime)
            result = await decide_pending_approval(
                runtime,
                request_id,
                approve=decision.approve,
                # --remember is a value flag here: explicit prefix or none.
                remember=decision.remember or None,
                remember_default=False,
                transport=transport,
            )
            if not message:
                # Decision-only run: surface the decision's reply text and
                # stop — there is no turn to run.
                decision_text = getattr(result, "text", "") or str(result)
                transport.stop_activity()
                await runtime.aclose()
                await transport.on_response(decision_text)
                return transport
        if message:
            await handle_message(runtime, message, transport)
    finally:
        transport.stop_activity()
        await runtime.aclose()
    return transport


def _collect_model_prices(runtime: HostRuntime) -> dict[str, tuple[float, float]]:
    """Read per-model per-token USD prices from the model manifest.

    Returns ``{model_id: (prompt_price, completion_price)}`` — the same
    fields ``render_models``'s Cost column formats. Empty dict on any
    failure: the transport omits the cost footer rather than blocking the
    turn on an unreadable manifest.
    """
    try:
        from runtime.model_manifest import load_model_manifest

        manifest = load_model_manifest(runtime.paths.models_path)
    except Exception:  # pylint: disable=broad-except
        return {}
    prices: dict[str, tuple[float, float]] = {}
    for model in manifest.models:
        prompt_price = float(getattr(model, "prompt_price", 0.0) or 0.0)
        completion_price = float(getattr(model, "completion_price", 0.0) or 0.0)
        if prompt_price > 0.0 or completion_price > 0.0:
            prices[str(model.model_id)] = (prompt_price, completion_price)
    return prices


def _collect_model_cache_prices(runtime: HostRuntime) -> dict[str, tuple[float, float]]:
    """Read per-model per-token USD prompt-cache prices from the manifest.

    Returns ``{model_id: (cache_read_price, cache_write_price)}``. Only
    models with a nonzero read and/or write price appear; missing keys make
    the footer's estimate treat every prompt as uncached (legacy behavior).
    """
    try:
        from runtime.model_manifest import load_model_manifest

        manifest = load_model_manifest(runtime.paths.models_path)
    except Exception:  # pylint: disable=broad-except
        return {}
    prices: dict[str, tuple[float, float]] = {}
    for model in manifest.models:
        read_price = float(getattr(model, "cache_read_price", 0.0) or 0.0)
        write_price = float(getattr(model, "cache_write_price", 0.0) or 0.0)
        if read_price > 0.0 or write_price > 0.0:
            prices[str(model.model_id)] = (read_price, write_price)
    return prices


# ---------------------------------------------------------------------------
# intake command
# ---------------------------------------------------------------------------


@app.command("intake", help="Run one intake tick: collect signals, process, write state.")
def _intake_command(
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot intake process. Triggered by systemd timer or cron.

    Collects signals from installed pack signal connectors, runs processing
    loops, and writes the merged UnifiedState to ``<data-dir>/state/state.json``.
    The next turn reads that state so the model has fresh ambient data.
    """
    resolved_data_dir = (
        Path(_resolve_profiled_data_dir(data_dir, profile)).expanduser()
        if _resolve_profiled_data_dir(data_dir, profile)
        else default_data_dir()
    )
    try:
        from runtime.intake import run_intake
    except ImportError:
        console.print("[red]intake module not available in this host.[/red]")
        raise typer.Exit(code=2) from None

    state = asyncio.run(run_intake(data_dir=resolved_data_dir))
    console.print(
        f"[green]Intake tick complete.[/green] "
        f"{len(state.context_items)} context item(s). "
        f"State written to {resolved_data_dir / 'state' / 'state.json'}"
    )


def main() -> None:
    """CLI entry point for the Lisa terminal host."""
    app(args=list(sys.argv[1:]))


if __name__ == "__main__":
    main()
