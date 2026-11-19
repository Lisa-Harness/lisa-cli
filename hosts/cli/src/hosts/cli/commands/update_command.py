"""CLI self-update command."""

from __future__ import annotations

import platform
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markup import escape

from hosts.cli.application import _PANEL_LIFECYCLE, app, console

if TYPE_CHECKING:
    from hosts.cli.update import Selection

_UPDATE_TOOL_NAME = "lisa-cli"

#: uv cannot replace a running console-script trampoline on Windows; the
#: command reports the argv for the operator to run after exiting instead.
_UPDATE_IN_PLACE_BLOCKED = platform.system() == "Windows"


def _update_receipt_status(tool_dir: Path, selection: Selection) -> list[str]:
    from hosts.cli.update import UpdateError, policy_status, rewrite_receipt

    try:
        receipt = (tool_dir / "uv-receipt.toml").read_text(encoding="utf-8")
        status = policy_status(receipt)
        if selection.changes_receipt:
            rewrite_receipt(receipt, selection)
    except (OSError, UnicodeError, UpdateError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    return status


def _display_or_defer_update(
    tool_dir: Path,
    selection: Selection,
    uv_bin: str | None,
    *,
    force: bool,
    dry_run: bool,
    status: list[str],
) -> str:
    from hosts.cli.update import format_command, standalone_command

    uv = uv_bin or "uv"
    argv = [uv, "tool", "upgrade", *(["--reinstall"] if force else []), _UPDATE_TOOL_NAME]
    printable = shlex.join(argv)
    if not selection.changes_receipt:
        for line in status:
            console.print(escape(line))
        if status:
            console.print(
                "Use lisa update --latest to reset both, or --framework-latest "
                "to reset only the framework."
            )

    blocked = uv_bin is None or (_UPDATE_IN_PLACE_BLOCKED and not dry_run)
    if not dry_run and not blocked:
        return printable

    if selection.changes_receipt:
        command = standalone_command(tool_dir, selection, uv, force=force)
        manual = format_command(command, windows=_UPDATE_IN_PLACE_BLOCKED)
    elif _UPDATE_IN_PLACE_BLOCKED:
        tool_parent = str(tool_dir.parent).replace("'", "''")
        manual = f"$env:UV_TOOL_DIR = '{tool_parent}'; {format_command(argv, windows=True)}"
    else:
        manual = f"UV_TOOL_DIR={shlex.quote(str(tool_dir.parent))} {printable}"

    if dry_run:
        if selection.changes_receipt:
            cli = f"{selection.cli.mode} {selection.cli.value}".strip()
            framework = f"{selection.framework.mode} {selection.framework.value}".strip()
            console.print(f"Would select CLI {escape(cli)}; framework {escape(framework)}.")
        console.print(f"Run: [bold]{escape(manual)}[/bold]")
        raise typer.Exit(code=0)

    reason = "uv is not on PATH; install it first" if uv_bin is None else "Windows locks lisa.exe"
    console.print(f"[yellow]{reason}.[/yellow] Run: [bold]{escape(manual)}[/bold]")
    raise typer.Exit(code=1)


@app.command(
    "update",
    short_help="Upgrade the installed lisa CLI.",
    help="Update lisa, or select a release, minor channel, or Git ref.",
    rich_help_panel=_PANEL_LIFECYCLE,
)
def _update_command(
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Reinstall even when the recorded version is unchanged "
            "(picks up new code from a local or git checkout without a version bump)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the uv command instead of running it.",
    ),
    version: str = typer.Option("", "--version", help="Temporarily use this CLI release."),
    pin: str = typer.Option("", "--pin", help="Pin the CLI to this release."),
    channel: str = typer.Option("", "--channel", help="Track a CLI minor release, e.g. 0.3."),
    ref: str = typer.Option("", "--ref", help="Track a CLI Git tag, branch, or commit."),
    latest: bool = typer.Option(False, "--latest", help="Track the latest published releases."),
    framework_version: str = typer.Option(
        "", "--framework-version", help="Temporarily use this framework release."
    ),
    framework_pin: str = typer.Option(
        "", "--framework-pin", help="Pin the framework packages to this release."
    ),
    framework_channel: str = typer.Option(
        "", "--framework-channel", help="Track a framework minor release, e.g. 0.5."
    ),
    framework_ref: str = typer.Option(
        "", "--framework-ref", help="Track a framework Git tag, branch, or commit."
    ),
    framework_latest: bool = typer.Option(
        False, "--framework-latest", help="Track the latest published framework packages."
    ),
) -> None:
    """Upgrade the running uv tool while preserving its extras and other requirements."""
    from runtime.paths import default_packs_venv, is_tool_install

    from hosts.cli.update import (
        Policy,
        Selection,
        UpdateError,
        choose_policy,
        success_messages,
        upgrade_tool,
    )

    try:
        cli_policy = choose_policy(
            version=version, pin=pin, channel=channel, ref=ref, latest=latest
        )
        framework_policy = choose_policy(
            version=framework_version,
            pin=framework_pin,
            channel=framework_channel,
            ref=framework_ref,
            latest=framework_latest,
        )
        if latest and framework_policy.mode == "keep":
            framework_policy = Policy("latest")
        selection = Selection(cli_policy, framework_policy)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    tool_dir = _installed_tool_dir()
    if tool_dir is None:
        if not is_tool_install():
            console.print(
                "[yellow]lisa is running from a repo checkout, so there is no uv tool to "
                "upgrade. Update with:[/yellow] git pull && uv sync"
            )
        else:
            console.print(
                "[yellow]This interpreter is not a uv tool install (no uv-receipt.toml in "
                f"{sys.prefix}), so there is nothing for `uv tool upgrade` to update.[/yellow]"
            )
        raise typer.Exit(code=2)

    status = _update_receipt_status(tool_dir, selection)
    uv_bin = shutil.which("uv")
    printable = _display_or_defer_update(
        tool_dir, selection, uv_bin, force=force, dry_run=dry_run, status=status
    )

    console.print(f"Running [bold]{escape(printable)}[/bold]")
    try:
        upgrade_tool(tool_dir, selection, uv_bin or "uv", force=force)
    except UpdateError as exc:
        console.print(f"[red]{escape(str(exc))}.[/red]")
        raise typer.Exit(code=exc.code) from exc

    console.print("Updated. Restart lisa to use the new version.")
    for message in success_messages(selection):
        console.print(escape(message))
    _warn_stale_packs_venv(default_packs_venv())


def _installed_tool_dir() -> Path | None:
    """Return the uv tool directory this process is running from.

    A ``uv tool`` venv carries the receipt at its root, so ``sys.prefix``
    identifies the install unambiguously — including installs made with a
    custom ``UV_TOOL_DIR``, which uv would otherwise not find. The receipt's
    parent is the value ``UV_TOOL_DIR`` expects (it is the directory holding
    one subdirectory per tool).

    This is a positive check: a plain virtualenv or pipx install has no
    receipt and returns ``None``.
    """
    prefix = Path(sys.prefix)
    if not (prefix / "uv-receipt.toml").is_file():
        return None
    return prefix


def _warn_stale_packs_venv(packs_venv: Path) -> None:
    """Point the operator at the one follow-up packs may need.

    The packs venv is a second install with its own framework copies. Packs
    import the CLI's framework packages (the tool venv's site-packages
    precedes the packs venv on ``sys.path``), so an upgrade does not break
    them — but a pack whose *dependencies* moved can still need a re-resolve,
    and ``uv pip install`` never upgrades an already-satisfied requirement.
    ``lisa pack update --all`` re-installs each pack from its recorded source,
    which re-resolves its dependency set. Surfacing the command is cheaper
    than silently re-seeding every pack after an upgrade.
    """
    if not packs_venv.is_dir():
        return
    console.print(
        f"[yellow]Packs live in a separate venv ({packs_venv}). Packs keep working "
        "against the upgraded framework, but their own dependencies are only "
        "re-resolved by `lisa pack update --all` — run it if a pack misbehaves "
        "after an update.[/yellow]"
    )
