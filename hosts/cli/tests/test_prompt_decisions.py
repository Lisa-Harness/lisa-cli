"""Tests for prompt-mode decision subcommands and pre-armed decisions.

Covers the prompt-mode approval/policy surface added so `lisa prompt` can
close the loop a persistent host closes with keystrokes:

- `lisa prompt policy ...` / `lisa prompt approve <id>` / `lisa prompt deny
  <id>` route through handle_message (the host's own command dispatch) with
  no model turn; the grammar guard keeps plain prose ("deny the claim") on
  the model path.
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


# ── Subcommand routing (grammar guard) ────────────────────────────────


def _exit_code(result: int | None) -> int:
    """Normalize a standalone_mode=False Typer result to an exit code.

    click converts ``Exit(0)`` into a ``None`` return; non-zero exits come
    back as their code.
    """
    return 0 if result is None else int(result)


def test_prompt_policy_routes_to_host_command(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt policy` dispatches /policy through handle_message."""
    captured: dict[str, Any] = {}
    result = _invoke_prompt(monkeypatch, tmp_path, ["policy"], captured=captured)
    assert _exit_code(result) == 0
    assert captured["messages"] == ["/policy"]


def test_prompt_policy_mode_trust_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt "policy mode trust"` sends the policy subcommand verbatim."""
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["policy mode trust"], captured=captured)
    assert captured["messages"] == ["/policy mode trust"]


def test_prompt_approve_subcommand_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt "approve <id>"` dispatches /approve <id>."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch, tmp_path, ["approve abc123def456"], captured=captured
    )
    assert captured["messages"] == ["/approve abc123def456"]


def test_prompt_approve_remember_dispatches_flag(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--remember rides along on the dispatched /approve."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["approve abc123def456 --remember git status"],
        captured=captured,
    )
    assert captured["messages"] == ["/approve abc123def456 --remember git status"]


def test_prompt_deny_subcommand_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch, tmp_path, ["deny abc123def456"], captured=captured
    )
    assert captured["messages"] == ["/deny abc123def456"]


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


# ── `policy rule` management subcommands ──────────────────────────────


def test_prompt_policy_rule_add_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt "policy rule add ..."` sends the rule command verbatim.

    The free-form `--when` tail (a CEL expression with spaces and quoted
    string literals) must survive the grammar guard intact.
    """
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ['policy rule add allow --tool bash_run --when action.argv[0] == "git"'],
        captured=captured,
    )
    assert captured["messages"] == [
        '/policy rule add allow --tool bash_run --when action.argv[0] == "git"'
    ]


def test_prompt_policy_rule_add_multiple_flags_with_when(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several matchers before the --when expression all pass through."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        [
            "policy rule add ask --effect external_write --pack lisa-pack-github "
            "--when action.parameters.base_branch == 'main'"
        ],
        captured=captured,
    )
    assert captured["messages"] == [
        "/policy rule add ask --effect external_write --pack lisa-pack-github "
        "--when action.parameters.base_branch == 'main'"
    ]


def test_prompt_policy_rule_remove_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch, tmp_path, ["policy rule remove allow:2"], captured=captured
    )
    assert captured["messages"] == ["/policy rule remove allow:2"]


def test_prompt_policy_rule_list_dispatches_slash_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["policy rule list"], captured=captured)
    assert captured["messages"] == ["/policy rule list"]


def test_prompt_policy_rule_prose_is_not_intercepted(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prose mentioning `policy rule` stays on the model path.

    The grammar guard must not grow prose false-positives: "policy rule of
    thumb" and friends route to the model turn like any other message.
    """
    for prose in (
        "policy rule of thumb says less is more",
        "policy rules should be simple to audit",
        "tell me about policy rule engines and their design",
    ):
        captured: dict[str, Any] = {}
        _invoke_prompt(monkeypatch, tmp_path, [prose], captured=captured)
        assert captured["messages"] == [prose]


# ── Bare verb resolves the single pending approval ────────────────────


def test_prompt_approve_bare_resolves_single_pending(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt approve` targets the one pending record in the store."""
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["approve"], captured=captured)
    # The pending store is empty here so the bare verb passes through
    # unresolved and the host replies with usage — no crash, exit 0. The
    # dispatched command is normalized to the slash form.
    assert captured["messages"] == ["/approve"]


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


def test_prompt_policy_does_not_read_stdin(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subcommand path never consumes piped stdin (nothing to attach)."""
    captured: dict[str, Any] = {}
    stdin_text = "should-be-ignored\n"

    runtime = _make_runtime(tmp_path)

    async def _fake_handle_message(_runtime, message, _transport):
        captured["messages"] = [message]
        return "ok"

    def _fake_init(**_kwargs):
        captured["runtime"] = runtime
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", _fake_handle_message)
    stdin_obj = io.StringIO(stdin_text)
    stdin_obj.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result = cli_main.app(args=["prompt", "policy"], standalone_mode=False)
    assert _exit_code(result) == 0
    assert captured["messages"] == ["/policy"]