"""Typer application, group routing, and root callback."""

from __future__ import annotations

import logging
import os
import sys
from types import TracebackType
from typing import Any

import typer
from rich.console import Console

from hosts.cli.bootstrap import _bootstrap_env_and_packs

logger = logging.getLogger(__name__)


def _lisa_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    """Render an uncaught exception through rich with locals suppressed.

    PROMPT_STARTUP Slice C (CLI half): this replaces the module-level
    ``rich.traceback.install(...)`` call, which imported
    ``rich.traceback`` -> ``rich.pretty`` on every ``--help`` /
    ``--version`` run. The import now happens only when an exception
    actually fires.

    show_locals=False is preserved exactly: uncaught frames here carry
    provider keys (verify's env reads), .env values, and prompt payloads —
    a locals dump prints them unredacted to the terminal. Keep the styled
    traceback, drop the values.

    Typer's ``app.__call__`` replaces this hook with its own
    pretty-exception renderer once command dispatch starts; the Typer
    app is configured with ``pretty_exceptions_show_locals=False`` (same
    contract, enforced by ``tests/test_no_locals_dump.py``). This hook
    covers everything before that point: import-time failures in modules
    loaded after this one and direct module use.
    """
    from rich.traceback import install  # lazy: crash rendering deferred (Slice C)

    install(show_locals=False, word_wrap=True)
    sys.excepthook(exc_type, exc_value, tb)


sys.excepthook = _lisa_excepthook

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
_GROUP_HOIST: dict[str, bool | None] = {}

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
        # .env + packs-venv injection must precede the entry-point scan
        # (packs live in the packs venv; LISA_PACKS_VENV may come from
        # .env). PROMPT_STARTUP Slice C: this runs on the first
        # command-resolution path that needs pack names, not at import.
        _bootstrap_env_and_packs()
        from kernel.host_command_discovery import (  # lazy: avoids eager pack scan
            discover_pack_host_commands,
        )

        specs: dict[str, Any] = {}
        for entry in discover_pack_host_commands():
            for spec in entry.specs:
                name = spec.prefix.strip().lstrip("/").split(None, 1)[0]
                if name:
                    specs.setdefault(name, entry)
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
        command = "/" + (best or name)
        for token in args:
            command += " " + token
        from hosts.cli.prompt import _run_host_command

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

    # Typer's rich help groups commands by this attribute (click commands
    # don't carry it statically; the help loop reads it with getattr). A
    # dedicated panel keeps the curated Commands box listy command rows only.
    _pack_cmd.rich_help_panel = "Pack host commands"  # type: ignore[attr-defined]  # Typer help seam
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

    def list_commands(self, ctx: Any) -> list[str]:
        """Static commands plus pack-host-command names (completions & help).

        Click's command and value completion and Typer's rich help listing
        both enumerate candidates here; without the merge, a pack command
        resolves only when the shell types it exactly, so it never shows
        up as ``lisa <TAB>`` or in the ``--help`` Commands panel.
        """
        names = list(super().list_commands(ctx))
        static = set(names)
        names.extend(pack_name for pack_name in sorted(_pack_commands()) if pack_name not in static)
        return names

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
    epilog=(
        "Scripting: `lisa prompt --raw 'task'` writes the response verbatim to "
        "stdout. Exit codes: 0 success · 1 turn/provider failure · "
        "2 usage/missing input · 3 unresolved approval · 4 model asked a "
        "question · 130 Ctrl-C."
    ),
    # Frame locals can carry provider keys and .env values — never print
    # them on an unhandled error (matches the rich.traceback install above).
    pretty_exceptions_show_locals=False,
)
policy_app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Inspect and manage approval policy without starting a model turn.",
)
policy_rule_app = typer.Typer(no_args_is_help=True, help="Manage allow, ask, and deny rules.")
policy_app.add_typer(policy_rule_app, name="rule")
app.add_typer(policy_app, name="policy")

# Help-panel names for the group listing: the flat 18-command box is split
# into role groups. Commands without a panel still land in "Commands".
_PANEL_INTROSPECTION = "Introspection"
_PANEL_LIFECYCLE = "Installation & lifecycle"
_PANEL_APPROVALS = "Approvals"

# The prompt epilog: the one-shot scripting contract, visible in --help
# without the README. Markup-safe (typer default mode is rich markup).
_PROMPT_EPILOG = (
    "Scripting contract — [bold]stdout[/bold] carries only the response in "
    "--raw mode; activity goes to stderr. [bold]Exit codes:[/bold] 0 success · "
    "1 turn/provider failure or empty raw artifact · 2 usage or missing input · "
    "3 unresolved approval (id is durable; retry with "
    "`lisa prompt --approve <id> 'continue'`) · 4 the model asked a question "
    "(printed to stderr; re-run with the answer) · 130 operator Ctrl-C."
)

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
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        return _pkg_version("lisa-cli")
    except PackageNotFoundError:
        # Source-checkout runs have no installed distribution; the declared
        # version is the only honest answer there. Other metadata failures
        # (corrupt dist-info) propagate instead of lying about the version.
        return "0.5.0"


_VERSION_OPTION = typer.Option(
    False,
    "--version",
    help="Print the lisa version and exit.",
    is_eager=True,
)

# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
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
    swap_keys: bool | None = typer.Option(
        None,
        "--swap-keys/--no-swap-keys",
        hidden=True,
        help="With lazy prompt routing: hoist the prompt turn's key-layout flag.",
    ),
) -> None:
    # The group-level copies of the prompt-only flags exist so Click parses
    # `lisa -r "fix it"` as [group flags, prompt] instead of dying on an
    # unknown option; the values ride on this module-level slot for the
    # prompt command to OR in (Typer commands cannot read click's
    # thread-local context — see the hoist comment in ``_prompt_command``).
    # They are hidden (not documented) — the documented home for these flags
    # is `lisa prompt`. swap_keys stays tri-state (None = not given) so the
    # prompt command's env fallback can still fire.
    _GROUP_HOIST.update(raw=raw, quiet=quiet, no_stdin=no_stdin, swap_keys=swap_keys)
    ctx.obj = {"raw": raw, "quiet": quiet, "no_stdin": no_stdin, "swap_keys": swap_keys}
    if version:
        console.print(f"lisa v{_get_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(code=2)
