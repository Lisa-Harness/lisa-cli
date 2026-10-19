"""Tests for bounded runtime teardown in one-shot CLI commands."""

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
