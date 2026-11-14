"""Tests for bounded runtime teardown in one-shot CLI commands."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from runtime.runtime import HostRuntime

from hosts.cli import main as cli_main


@pytest.mark.asyncio
async def test_timed_out_teardown_closes_each_pools_children_once() -> None:
    proxy_process = Mock(returncode=None)
    tool_process = Mock(returncode=None)
    exited_process = Mock(returncode=0)
    proxy_stack = SimpleNamespace(aclose=AsyncMock())
    tool_stack = SimpleNamespace(aclose=AsyncMock())
    runtime = Mock(
        spec=HostRuntime,
        aclose=AsyncMock(side_effect=TimeoutError),
        resources=SimpleNamespace(
            _proxy_pool=SimpleNamespace(
                _processes={"proxy": proxy_process},
                _child_stacks={"proxy": proxy_stack},
            ),
            _mcp_tool_pool=SimpleNamespace(
                _processes={"tool": tool_process, "exited": exited_process},
                _child_stacks={"tool": tool_stack},
            ),
        ),
    )

    await cli_main._aclose_runtime_bounded(runtime)

    proxy_process.kill.assert_called_once_with()
    tool_process.kill.assert_called_once_with()
    exited_process.kill.assert_not_called()
    proxy_stack.aclose.assert_awaited_once_with()
    tool_stack.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_bounded_teardown_closes_in_the_calling_task() -> None:
    """``runtime.aclose()`` runs in the caller's task, never a nested one.

    The runtime's MCP teardown exits anyio cancel scopes, and anyio
    forbids exiting a cancel scope from a different task than the one
    that entered it. ``asyncio.wait_for`` only guarantees the in-task
    semantics on Python 3.12+ (it runs the wrapped coroutine in the
    current task there); on 3.11 it wraps it in a new task, which trips
    anyio's cross-task guard and raises ``CancelledError``. The bounded
    close therefore uses a primitive whose semantics are the same on
    every supported version.
    """
    seen: list[asyncio.Task[object]] = []

    async def _aclose() -> None:
        seen.append(asyncio.current_task())

    runtime = Mock(
        spec=HostRuntime,
        aclose=AsyncMock(side_effect=_aclose),
        resources=SimpleNamespace(_proxy_pool=None, _mcp_tool_pool=None),
    )

    await cli_main._aclose_runtime_bounded(runtime)

    assert seen, "aclose was never called"
    assert seen[0] is asyncio.current_task()
