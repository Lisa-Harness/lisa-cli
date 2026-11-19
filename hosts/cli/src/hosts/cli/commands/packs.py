"""Pack lifecycle command."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

import typer

from hosts.cli.application import _PANEL_LIFECYCLE, app, console
from hosts.cli.cli_runtime import _initialize_runtime_or_exit
from hosts.cli.options import (
    AllPacksOption,
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    NoLogOption,
    NoThreadOption,
    PackAction,
    ProfileOption,
    SessionContextOption,
    StateSnapshotOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
)

if TYPE_CHECKING:
    from runtime.runtime import HostRuntime


@app.command(
    "pack",
    short_help="Manage runtime pack installs.",
    rich_help_panel=_PANEL_LIFECYCLE,
    help="Manage runtime pack installs (list/install/uninstall/update).",
)
def _pack_command(
    action: Annotated[
        PackAction,
        typer.Argument(..., help="Action: list, install, uninstall, update."),
    ] = "list",
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

    # Typer parsed/validated the action (Literal) — no normalization needed.
    if action == "list":
        from hosts.cli.render import (  # lazy: presentation import (Slice C)
            render_pack_list_panel as _render_pack_list_panel,
        )

        _render_pack_list_panel(runtime, console=console)
        return
    if action == "install":
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_install(runtime, target)))
        return
    if action == "uninstall":
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_uninstall(runtime, target)))
        return
    # action == "update" (Literal guarantees this; the guard below fails
    # closed for direct body callers that bypass the parser).
    if all_packs:
        _finish_pack_mutation(runtime, asyncio.run(_run_pack_update_all(runtime)))
        return
    if not isinstance(action, str) or action not in ("list", "install", "uninstall", "update"):
        console.print("[red]Unknown action. Use: list, install, uninstall, update.[/red]")
        raise typer.Exit(code=2)
    if not target.strip():
        console.print("[red]Usage: lisa pack update <pack> | lisa pack update --all.[/red]")
        raise typer.Exit(code=2)
    _finish_pack_mutation(runtime, asyncio.run(_run_pack_update(runtime, target.strip())))
    raise typer.Exit(code=0)


def _finish_pack_mutation(runtime: HostRuntime, result: tuple[bool, str]) -> None:
    """Print a pack lifecycle result and apply the exit-code contract.

    Failure prints the details in red and exits 1; success prints the
    details and exits 0 — the runtime reloads pack state on next boot.
    """
    ok, details = result
    if not ok:
        # details may embed raw uv/uv-pip output — never route it through
        # the markup parser (a bracketed token in the log would vanish).
        console.print(details, markup=False, style="red")
        raise typer.Exit(code=1)
    console.print(details, markup=False)
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
