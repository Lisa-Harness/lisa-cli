"""Runtime construction and lifecycle helpers owned by the CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

from hosts.cli.application import console
from hosts.cli.bootstrap import _bootstrap_env_and_packs, _resolve_data_dir

if TYPE_CHECKING:
    from kernel.models import McpServerHealth
    from kernel.tools.mcp_client import MCPToolSpec
    from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

McpIntrospection = tuple[dict[str, Any], dict[str, "list[MCPToolSpec]"]]


class RuntimeTeardownError(RuntimeError):
    """The one-shot CLI's bounded runtime teardown did not complete cleanly.

    Subprocesses are force-killed before this raises, so no orphans
    survive; the error exists so commands exit nonzero instead of
    reporting success over a pool that never closed.
    """


def _resolve_profiled_data_dir(data_dir: str, profile: str) -> str:
    """Resolve the effective data dir from --data-dir / --profile.

    Precedence (matching ``HostConfig.from_env``): explicit ``--data-dir``
    wins; then ``--profile`` (CLI arg); then the framework falls back to
    ``LISA_DATA_DIR`` > ``LISA_PROFILE`` > ``~/.lisa``. Returning an empty
    string hands the decision to the framework so the env-var chain stays
    in one place. An invalid profile name raises ``ValueError``, which
    ``_initialize_runtime_or_exit`` renders as a startup-error panel.
    """
    if not data_dir.strip() and not profile.strip():
        return ""
    return str(_resolve_data_dir(data_dir, profile))


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
    conversation_id: str = "",
) -> HostRuntime:
    """Build and wire a HostRuntime from CLI args + env vars.

    Thin adapter over :func:`runtime.bootstrap.build_runtime` that
    stamps ``host_kind="cli"`` and forwards every CLI argument. The
    composition sequence (config -> runtime -> state snapshot -> framework
    wiring -> MCP config compilation) lives in the shared bootstrap so
    the CLI and web hosts cannot drift.

    ``profile`` resolves to ``<profiles-root>/<name>`` here (CLI arg >
    ``LISA_PROFILE``; an explicit ``data_dir`` still wins) because
    ``--data-dir`` and ``--profile`` are both CLI-level data-dir inputs —
    the framework's ``HostConfig.from_env`` only knows the env var.

    ``session_context`` is the optional path to a Markdown session plan
    file; ``user_slug`` is left to the framework (it derives from the
    compiled plan's manifest name). ``conversation_id`` isolates the
    thread store (``<user_slug>--<conversation_id>``; the interactive
    session's resume/switch path and ``LISA_CONVERSATION_ID`` both land
    here). The MCP pool is NOT started here — ``runtime.start()`` must be
    called inside the host's event loop (see the prompt/verify paths).
    """
    # .env + packs-venv bootstrap before any framework import or
    # configuration resolution (PROMPT_STARTUP Slice C: the sweep is
    # lazy, so this is the first runtime-building path to run it).
    _bootstrap_env_and_packs()
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
        conversation_id=conversation_id,
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
    and startup paths are its remaining consumers.) Exception text is
    escaped per line: a ``[tag]`` in the message is content, not markup.
    """
    from rich.markup import escape as _escape
    from rich.panel import Panel  # lazy: presentation import (Slice C)

    lines = str(exc).splitlines()
    if lines:
        head, sep, tail = lines[0].partition(":")
        if sep:
            lines[0] = f"[bold]{_escape(head.strip())}[/bold]:{_escape(tail)}"
        else:
            lines[0] = _escape(lines[0])
    body = "\n".join(_escape(line) for line in lines) or exc.__class__.__name__
    console.print(Panel(body, title=title, border_style="indian_red"))


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
    the CLI prints nothing. Bound the graceful path with a timeout; on
    timeout, fall back to terminating every tracked MCP/proxy subprocess
    directly so no orphans survive.

    Teardown is best-effort for the process (children are killed either
    way), but an incomplete close is never silently treated as success:
    a timeout or an ``aclose()`` exception raises
    :class:`RuntimeTeardownError` so the command can exit nonzero —
    persisted work is unaffected, but the caller knows the pool did not
    close cleanly.

    The bounded close must run ``runtime.aclose()`` in the *calling* task:
    the runtime's MCP teardown exits anyio cancel scopes, and anyio forbids
    exiting a cancel scope from a different task than the one that entered
    it. :func:`asyncio.wait_for` only guarantees that on Python 3.12+ (it
    runs the wrapped coroutine in the current task there); on 3.11 it wraps
    it in a new task, so the close would trip anyio's cross-task guard and
    raise ``CancelledError``. ``asyncio.timeout`` has the in-task
    semantics on every supported version, so it is the primitive used here.
    """
    try:
        async with asyncio.timeout(timeout_s):
            await runtime.aclose()
        return
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning(
            "Runtime teardown exceeded %.0fs; falling back to direct subprocess termination",
            timeout_s,
        )
    except Exception as exc:
        logger.warning("Runtime aclose raised during bounded teardown", exc_info=True)
        _force_close_all_pools(runtime)
        raise RuntimeTeardownError(str(exc)) from exc
    # Forced cleanup: kill every subprocess the pools still track.
    _force_close_all_pools(runtime)
    raise RuntimeTeardownError(
        f"Runtime teardown did not complete within {timeout_s:.0f}s"
    ) from None


def _force_close_all_pools(runtime: HostRuntime) -> None:
    """Schedule forced cleanup of every pool the runtime still tracks."""
    resources = runtime.resources
    pools: list[Any] = []
    proxy_pool = getattr(resources, "_proxy_pool", None)
    if proxy_pool is not None:
        pools.append(proxy_pool)
    tool_pool = getattr(resources, "_mcp_tool_pool", None)
    if tool_pool is not None:
        pools.append(tool_pool)
    loop = asyncio.get_running_loop()
    pending = [loop.create_task(_force_close_pool(pool)) for pool in pools]
    for task in pending:
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)


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

    Returns an empty :data:`McpIntrospection` only when the runtime has no
    MCP servers configured — genuine absence. A configured pool that fails
    to start, yields no tool pool, or fails ``tools/list`` raises: the
    ``mcp``/``tools`` commands must report the failure, not render an
    empty catalog as if no servers existed.
    """
    warm_ids = {c.server_id for c in _mcp_configs(runtime, warm=True) if hasattr(c, "server_id")}
    cold_configs = _mcp_configs(runtime, warm=False)
    if not warm_ids and not cold_configs:
        return ({}, {})
    allowed = warm_ids | (connect_server_ids or set())
    await runtime.start()
    pool = _mcp_tool_pool(runtime)
    if pool is None:
        raise RuntimeError("MCP servers are configured but the runtime provided no tool pool")
    try:
        tools = await pool.list_tools(allowed_server_ids=allowed)
        # Capture is_alias flags BEFORE aclose() clears _synthetic_handlers.
        is_alias = {spec.name: pool.is_synthetic_tool(spec.name) for spec in tools}
        for spec in tools:
            _ = spec  # grouping below keyed by server_id, read once
        tools_by_server: dict[str, list[MCPToolSpec]] = {}
        for spec in tools:
            tools_by_server.setdefault(spec.server_id, []).append(spec)
        health = _build_mcp_health(pool, is_alias)
    except BaseException:
        with contextlib.suppress(Exception):
            await _aclose_runtime_bounded(runtime)
        raise
    await _aclose_runtime_bounded(runtime)
    return (health, tools_by_server)
