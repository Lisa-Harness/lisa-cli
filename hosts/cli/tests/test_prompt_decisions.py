"""Tests for policy commands and prompt-mode pre-armed decisions.

Policy management uses Typer commands and never routes through a model turn.
Prompt-mode pre-armed approval/denial still closes the loop a persistent host
closes with keystrokes:

- `lisa policy ...`, `lisa approve <id>`, and `lisa deny <id>` are parsed by
    Typer and dispatched to host-owned typed cores (values, never command
    strings) without a model turn.
- `--approve/--deny [--remember]` pre-arm a decision that is dispatched
    FIRST as its own typed dispatch through ``decide_pending_approval``,
    then the turn message runs as a second dispatch on the same runtime; a
    bare `--approve` resolves to the single pending approval via the
    durable store.
- Mutual-exclusion / misuse flags exit 2 without running anything.

The runtime builder and the approval core are stubbed; no MCP subprocess is
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
    """Invoke ``_prompt_command`` with stubbed runtime + approval dispatch.

    Records each dispatched decision in ``captured["decisions"]`` and each
    turn message in ``captured["messages"]``. Returns the Typer exit code.
    """
    runtime = _make_runtime(tmp_path)

    async def _fake_decide(runtime_arg, request_id, *, approve, transport, **kwargs):
        if captured is not None:
            captured.setdefault("decisions", []).append((request_id, approve))
            captured.setdefault("decision_kwargs", []).append(kwargs)
        return f"decided {request_id}"

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
    monkeypatch.setattr(hm_module, "decide_pending_approval", _fake_decide)
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
    """Invoke the Typer policy group and capture the typed policy action."""

    def _fake_run_action(action, **_kwargs: Any) -> None:
        if captured is not None:
            captured.setdefault("actions", []).append(action)

    monkeypatch.setattr(cli_main, "_run_policy_action", _fake_run_action)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        return cli_main.app(args=["policy", *argv], standalone_mode=False)


def _invoke_approval_command(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    captured: dict[str, Any] | None = None,
) -> int | None:
    """Invoke a top-level approve/deny command and capture the typed decision."""

    def _fake_run(request_id, *, approve, remember, **_kwargs: Any) -> None:
        if captured is not None:
            captured.setdefault("decisions", []).append((request_id, approve, remember))

    monkeypatch.setattr(cli_main, "_run_decision", _fake_run)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        return cli_main.app(args=argv, standalone_mode=False)


# ── Direct policy command group ──────────────────────────────────────


def _exit_code(result: int | None) -> int:
    """Normalize a standalone_mode=False Typer result to an exit code."""
    return 0 if result is None else int(result)


def test_policy_without_subcommand_shows_current_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare `lisa policy` runs the document-render action."""
    captured: dict[str, Any] = {}
    result = _invoke_policy(monkeypatch, [], captured=captured)
    assert _exit_code(result) == 0
    assert len(captured["actions"]) == 1


def test_policy_mode_dispatches_typed_core(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa policy mode trust` calls set_policy_mode with the typed value."""
    from runtime.policy import PolicyMode

    runtime = _make_runtime(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    from runtime import commands

    def _fake_set_mode(_runtime, mode):
        seen["mode"] = mode
        return "ok"

    monkeypatch.setattr(commands, "set_policy_mode", _fake_set_mode)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result = cli_main.app(args=["policy", "mode", "trust"], standalone_mode=False)
    assert _exit_code(result) == 0
    assert seen["mode"] == PolicyMode.TRUST


def test_policy_rule_add_passes_typed_matchers(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rule add` forwards each matcher as a value — no quoting round trip."""
    runtime = _make_runtime(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    from runtime import commands

    def _fake_add(_runtime, list_name, *, tool, effect, pack, command_prefix, when):
        seen.update(
            list=list_name,
            tool=tool,
            effect=effect,
            pack=pack,
            prefix=command_prefix,
            when=when,
        )
        return "ok"

    monkeypatch.setattr(commands, "add_policy_rule", _fake_add)
    cel = 'identity.mode_id == "executor" && action.argv.size() <= 2'
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result = cli_main.app(
            args=[
                "policy",
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
                cel,
            ],
            standalone_mode=False,
        )
    assert _exit_code(result) == 0
    assert seen["list"] == "allow"
    assert seen["tool"] == "coding_create_worktree"
    assert seen["effect"] == "write"
    assert seen["pack"] == "lisa-pack-coding"
    # The CEL expression arrives verbatim — exactly what the user typed.
    assert seen["when"] == cel


def test_policy_rule_remove_and_list_dispatch_typed_cores(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    from runtime import commands

    def _fake_remove(_runtime, address):
        seen["address"] = address
        return "ok"

    def _fake_list(_runtime):
        seen["listed"] = True
        return "ok"

    monkeypatch.setattr(commands, "remove_policy_rule", _fake_remove)
    monkeypatch.setattr(commands, "render_policy_rules", _fake_list)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result = cli_main.app(args=["policy", "rule", "remove", "allow:1"], standalone_mode=False)
        assert _exit_code(result) == 0
        result = cli_main.app(args=["policy", "rule", "list"], standalone_mode=False)
        assert _exit_code(result) == 0
    assert seen["address"] == "allow:1"
    assert seen["listed"] is True


def test_top_level_approve_dispatches_typed_decision_without_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["approve", "abc123def456"], captured=captured)
    assert captured["decisions"] == [("abc123def456", True, "")]


def test_top_level_approve_remember_dispatches_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(
        monkeypatch,
        ["approve", "abc123def456", "--remember", "git status"],
        captured=captured,
    )
    assert captured["decisions"] == [("abc123def456", True, "git status")]


def test_top_level_bare_approve_targets_single_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["approve"], captured=captured)
    # The "pending" sentinel resolves to an empty request id; the async turn
    # resolves it against the durable store.
    assert captured["decisions"] == [("", True, "")]


def test_top_level_deny_dispatches_typed_decision_without_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _invoke_approval_command(monkeypatch, ["deny", "abc123def456"], captured=captured)
    assert captured["decisions"] == [("abc123def456", False, "")]


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
    """Prose mentioning `policy rule` stays on the model path."""
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


def test_prearm_approve_dispatches_decision_then_turn(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--approve <id> dispatches the typed decision, then runs the turn."""
    captured: dict[str, Any] = {}
    result = _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456", "continue the task"],
        captured=captured,
    )
    assert _exit_code(result) == 0
    assert captured["decisions"] == [("abc123def456", True)]
    assert captured["messages"] == ["continue the task"]


def test_prearm_deny_with_message_dispatches_then_turns(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--deny", "abc123def456", "what now"],
        captured=captured,
    )
    assert captured["decisions"] == [("abc123def456", False)]
    assert captured["messages"] == ["what now"]


def test_prearm_approve_remember_passes_prefix(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--remember rides along as a value on the typed decision."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456", "--remember", "systemctl status", "go"],
        captured=captured,
    )
    kwargs = captured["decision_kwargs"][0]
    assert kwargs["remember"] == "systemctl status"
    assert kwargs["remember_default"] is False


def test_prearm_approve_without_remember_defaults_cleanly(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --remember: neither remember value nor default-prefix behavior."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--approve", "abc123def456", "go"],
        captured=captured,
    )
    kwargs = captured["decision_kwargs"][0]
    assert kwargs["remember"] is None
    assert kwargs["remember_default"] is False


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
    assert captured["decisions"] == [("abc123def456", True)]
    assert captured.get("messages", []) == []


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
    """A bare decision against one pending record resolves to its id."""
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
    resolved = await cli_main._resolve_pending_request_id(runtime)
    assert resolved == invocation.request_id


def test_bare_decision_with_no_pending_returns_empty(tmp_path: Any) -> None:
    async def _check() -> None:
        runtime = _make_runtime(tmp_path)
        resolved = await cli_main._resolve_pending_request_id(runtime)
        assert resolved == ""

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
    assert len(captured["actions"]) == 1