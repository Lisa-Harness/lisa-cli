"""Tests for policy commands and prompt-mode pre-armed decisions.

Policy management uses Typer commands and never routes through a model turn.
Prompt-mode pre-armed approval/denial still closes the loop a persistent host
closes with keystrokes:

- `lisa policy ...`, `lisa approve <id>`, and `lisa deny <id>` are parsed by
    Typer and dispatched directly to host-owned handlers without a model turn.
- `--approve/--deny [--remember]` pre-arm a decision that is dispatched
  ahead of the turn message; a bare `/approve` resolves to the single
  pending approval via the durable store.
- Mutual-exclusion / misuse flags exit 2 without running anything.

The runtime builder and handle_message are stubbed; no MCP subprocess is
spawned and no model is called.
"""

from __future__ import annotations

# pylint: disable=protected-access
import io
import sys
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from hosts.cli import main as cli_main

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Any) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _invoke_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    argv: list[str],
    *,
    captured: dict[str, Any] | None = None,
) -> int | None:
    """Invoke ``_prompt_command`` with stubbed runtime + handle_message.

    Records each dispatched message in ``captured["messages"]`` (list) and
    the runtime in ``captured["runtime"]``. Returns the Typer exit code.
    """
    runtime = _make_runtime(tmp_path)

    async def _fake_handle_message(_runtime, message, _transport):
        if captured is not None:
            captured.setdefault("messages", []).append(message)
        return "ok"

    def _fake_init(**_kwargs):
        if captured is not None:
            captured["runtime"] = runtime
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", _fake_handle_message)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result: int | None = cli_main.app(
            args=["prompt", *argv], standalone_mode=False
        )
    return result


def _invoke_policy(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    captured: dict[str, Any] | None = None,
) -> int | None:
    """Invoke the Typer policy group and capture its host command dispatch."""

    def _fake_run(command: str, **_kwargs: Any) -> None:
        if captured is not None:
            captured.setdefault("commands", []).append(command)

    monkeypatch.setattr(cli_main, "_run_policy_command", _fake_run)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        return cli_main.app(args=["policy", *argv], standalone_mode=False)


def _invoke_approval_command(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    captured: dict[str, Any] | None = None,
) -> int | None:
    """Invoke a top-level approve/deny command and capture host dispatch."""

    def _fake_run(command: str, **_kwargs: Any) -> None:
        if captured is not None:
            captured.setdefault("commands", []).append(command)

    monkeypatch.setattr(cli_main, "_run_host_command", _fake_run)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        return cli_main.app(args=argv, standalone_mode=False)


# ── Direct policy command group ──────────────────────────────────────


def _exit_code(result: int | None) -> int:
    """Normalize a standalone_mode=False Typer result to an exit code.

    click converts ``Exit(0)`` into a ``None`` return; non-zero exits come
    back as their code.
    """
    return 0 if result is None else int(result)


def test_policy_without_subcommand_shows_current_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    result = _invoke_policy(monkeypatch, [], captured=captured)
    assert _exit_code(result) == 0
    assert captured["commands"] == ["/policy"]


def test_policy_mode_dispatches_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_policy(monkeypatch, ["mode", "trust"], captured=captured)
    assert captured["commands"] == ["/policy mode trust"]


def test_policy_rule_add_preserves_cel_as_one_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_policy(
        monkeypatch,
        [
            "rule",
            "add",
            "allow",
            "--tool",
            "coding_create_worktree",
            "--effect",
            "write",
            "--pack",
            "lisa-pack-coding",
            "--when",
            'identity.mode_id == "executor" && identity.workspace_id == "p920"',
        ],
        captured=captured,
    )
    assert captured["commands"] == [
        "/policy rule add allow --tool coding_create_worktree --effect write "
        "--pack lisa-pack-coding --when "
        "'identity.mode_id == \"executor\" && identity.workspace_id == \"p920\"'"
    ]


def test_policy_rule_list_and_remove_dispatch_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_policy(monkeypatch, ["rule", "list"], captured=captured)
    _invoke_policy(monkeypatch, ["rule", "remove", "allow:1"], captured=captured)
    assert captured["commands"] == ["/policy rule list", "/policy rule remove allow:1"]


def test_policy_command_calls_host_policy_api_without_message_dispatch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kwargs: runtime)
    from runtime import commands

    def handle_policy(_runtime: Any, stripped: str, message: str) -> str:
        captured["args"] = (stripped, message)
        return "policy output"

    monkeypatch.setattr(commands, "handle_policy_command", handle_policy)
    import runtime.handle_message as message_module

    def unexpected_message_dispatch(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("policy must not dispatch through handle_message")

    monkeypatch.setattr(message_module, "handle_message", unexpected_message_dispatch)
    with (
        patch.object(cli_main, "console", new=Console(quiet=True)),
        pytest.raises(cli_main.typer.Exit) as exit_info,
    ):
        cli_main._run_policy_command("/policy rule list")

    assert exit_info.value.exit_code == 0
    assert captured["args"] == ("/policy rule list", "/policy rule list")


def test_top_level_approve_dispatches_host_command_without_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["approve", "abc123def456"], captured=captured)
    assert captured["commands"] == ["/approve abc123def456"]


def test_top_level_approve_remember_dispatches_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(
        monkeypatch,
        ["approve", "abc123def456", "--remember", "git status"],
        captured=captured,
    )
    assert captured["commands"] == ["/approve abc123def456 --remember git status"]


def test_top_level_bare_approve_targets_single_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["approve"], captured=captured)
    assert captured["commands"] == ["/approve"]


def test_top_level_deny_dispatches_host_command_without_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["deny", "abc123def456"], captured=captured)
    assert captured["commands"] == ["/deny abc123def456"]


def test_prompt_prose_is_not_intercepted(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch, tmp_path, ["deny the claim in the report"], captured=captured
    )
    assert captured["messages"] == ["deny the claim in the report"]
    captured = {}
    _invoke_prompt(
        monkeypatch, tmp_path, ["approve my leave request"], captured=captured
    )
    assert captured["messages"] == ["approve my leave request"]
    captured = {}
    _invoke_prompt(monkeypatch, tmp_path, ["approve abc123def456"], captured=captured)
    assert captured["messages"] == ["approve abc123def456"]
    captured = {}
    _invoke_prompt(monkeypatch, tmp_path, ["deny abc123def456"], captured=captured)
    assert captured["messages"] == ["deny abc123def456"]


def test_policy_text_in_prompt_is_not_treated_as_admin_command(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prose mentioning `policy rule` stays on the model path.

    The grammar guard must not grow prose false-positives: "policy rule of
    thumb" and friends route to the model turn like any other message.
    """
    for prose in (
        'policy rule add allow --tool bash_run --when action.argv[0] == "git"',
        "policy rule of thumb says less is more",
        "policy rules should be simple to audit",
        "tell me about policy rule engines and their design",
    ):
        captured: dict[str, Any] = {}
        _invoke_prompt(monkeypatch, tmp_path, [prose], captured=captured)
        assert captured["messages"] == [prose]


# ── Pre-armed decision flags ──────────────────────────────────────────


def test_prearm_approve_prepends_decision_command(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--approve <id> dispatches /approve first, then runs the turn."""
    captured: dict[str, Any] = {}
    result = _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456", "continue the task"],
        captured=captured,
    )
    assert _exit_code(result) == 0
    msgs = captured["messages"]
    # handle_message is called once with the combined message; the decision
    # must be its first line.
    assert len(msgs) == 1
    assert msgs[0].startswith("/approve abc123def456\n\n")
    assert "continue the task" in msgs[0]


def test_prearm_deny_with_message_prepends_denial(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--deny", "abc123def456", "what now"],
        captured=captured,
    )
    msg = captured["messages"][0]
    assert msg.startswith("/deny abc123def456\n\n")
    assert msg.endswith("what now")


def test_prearm_approve_remember_rides_along(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456", "--remember", "systemctl status", "go"],
        captured=captured,
    )
    msg = captured["messages"][0]
    assert msg.startswith("/approve abc123def456 --remember systemctl status\n\n")


def test_prearm_approve_and_deny_mutually_exclusive(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc", "--deny", "def", "x"],
    )
    assert _exit_code(result) == 2


def test_prearm_remember_requires_approve(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--deny", "abc123def456", "--remember", "git", "x"],
    )
    assert _exit_code(result) == 2


def test_prearm_decision_only_no_message_is_valid(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--approve with no message/payload is a valid decision-only run."""
    captured: dict[str, Any] = {}
    result = _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456"],
        captured=captured,
    )
    assert _exit_code(result) == 0
    assert captured["messages"][0] == "/approve abc123def456"


def test_no_message_no_stdin_no_decision_still_exits_2(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke_prompt(monkeypatch, tmp_path, [])
    assert _exit_code(result) == 2


# ── Sentinel resolution ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bare_decision_resolves_single_pending_id(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare /approve against one pending record resolves to its id."""
    import kernel.approval as k_approval

    runtime = _make_runtime(tmp_path)
    identity = k_approval.ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id=runtime.session.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
    )
    invocation = k_approval.build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "ls"},
        identity=identity,
        pack_config={},
        safe_preview="run a shell command",
    )
    await runtime.services.approval_service.request(invocation)
    resolved = await cli_main._resolve_pending_approval_target(runtime, "/approve")
    assert resolved.startswith("/approve ")
    assert resolved != "/approve"


def test_bare_decision_with_explicit_id_is_untouched(tmp_path: Any) -> None:
    async def _check() -> None:
        runtime = _make_runtime(tmp_path)
        cmd = "/approve abc123def456"
        resolved = await cli_main._resolve_pending_approval_target(runtime, cmd)
        assert resolved == cmd

    import asyncio as _asyncio

    _asyncio.run(_check())


# ── Decision-subcommand run path is turn-free ─────────────────────────


def test_policy_command_does_not_read_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy command never reads prompt input or starts a model turn."""
    captured: dict[str, Any] = {}
    stdin_obj = io.StringIO("should-be-ignored\n")
    stdin_obj.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    result = _invoke_policy(monkeypatch, ["rule", "list"], captured=captured)
    assert _exit_code(result) == 0
    assert captured["commands"] == ["/policy rule list"]