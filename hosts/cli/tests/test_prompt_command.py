"""Tests for prompt-mode piped stdin, ``--raw/-r``, and approval fail-fast.

Covers the three prompt-mode pipeline surfaces:

- stdin attachment: piped stdin (non-tty) becomes a ``<piped-input>`` block
  ahead of the instruction; empty stdin + no message exits 2.
- ``--raw`` output contract: the raw-mode directive is prepended to the turn
  message and the transport is constructed in raw mode with a stderr console.
- approval fail-fast: an unresolved approval sets ``approval_unresolved`` and
  the command exits with ``_APPROVAL_REQUIRED_EXIT`` instead of rendering an
  interactive panel into redirected output.

The runtime builder and the turn runner are stubbed; no MCP subprocess is
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
from hosts.cli.transport import PromptTransport

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Any) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _invoke_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    argv: list[str],
    *,
    stdin_text: str | None = None,
    captured: dict[str, Any] | None = None,
    console: Console | None = None,
) -> int | None:
    """Invoke ``_prompt_command`` via the Typer app with a stubbed runtime.

    ``handle_message`` is stubbed so no model is called; the captured message
    is recorded in ``captured["message"]``. ``stdin_text`` (when given) is
    fed through a patched ``sys.stdin`` so the command reads it as a pipe.
    Returns the click exit code when the command raises ``typer.Exit``
    (standalone_mode=False turns Exit into a return value), else ``None``.
    The patched-in console is recorded via ``captured["console"]``.
    """
    runtime = _make_runtime(tmp_path)

    async def _fake_handle_message(_runtime, message, _transport):
        if captured is not None:
            captured["message"] = message
        return "ok"

    def _fake_init(**_kwargs):
        return runtime

    if stdin_text is None:
        stdin_obj: Any = io.StringIO()
        stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    else:
        stdin_obj = io.StringIO(stdin_text)
        stdin_obj.isatty = lambda: False  # type: ignore[method-assign]

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    # Patch the module object (not the string path): the host package's lazy
    # __getattr__ resolves `runtime.handle_message` to the function itself.
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", _fake_handle_message)
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    if console is None:
        console = Console(quiet=True)
    if captured is not None:
        captured["console"] = console
    with patch.object(cli_main, "console", new=console):
        result: int | None = cli_main.app(args=["prompt", *argv], standalone_mode=False)
    return result


# ── Stdin attachment ──────────────────────────────────────────────────


def test_piped_stdin_wrapped_as_piped_input_block(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped stdin becomes a <piped-input> block ahead of the instruction."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["recommend improvements"],
        stdin_text="key=value\n",
        captured=captured,
    )
    msg = captured["message"]
    assert msg.startswith("<piped-input>\nkey=value\n</piped-input>")
    assert msg.endswith("recommend improvements")


def test_stdin_only_message_is_payload(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no instruction, the stdin payload is the whole message."""
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, [], stdin_text="just data\n", captured=captured)
    assert captured["message"] == "just data"


def test_no_stdin_tty_uses_message_only(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tty stdin is not read; the message passes through unchanged."""
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["plain message"], captured=captured)
    assert captured["message"] == "plain message"


def test_multi_word_arguments_are_joined_with_spaces(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unquoted multi-word prompts arrive as one space-joined message."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["fix", "the", "config", "parser"],
        captured=captured,
    )
    assert captured["message"] == "fix the config parser"


def test_multi_word_empty_token_still_exits_2(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tokens that join to whitespace only are not a message (exit 2)."""
    result = _invoke_prompt(monkeypatch, tmp_path, ["", " "])
    assert result == 2


def test_no_message_and_no_stdin_exits_2(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither message nor piped input is a usage error (exit 2)."""
    result = _invoke_prompt(monkeypatch, tmp_path, [])
    assert result == 2


def test_no_stdin_flag_skips_pipe(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-stdin leaves piped data unread; the message stands alone."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["--no-stdin", "instruction only"],
        stdin_text="ignored\n",
        captured=captured,
    )
    assert captured["message"] == "instruction only"


# ── Raw mode ──────────────────────────────────────────────────────────


def test_raw_prepends_output_contract(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """-r prepends the raw-mode output contract to the turn message."""
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["-r", "fix it"], captured=captured)
    assert captured["message"].startswith("OUTPUT CONTRACT (raw mode):")
    assert "fix it" in captured["message"]


def test_raw_wraps_stdin_payload_too(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """-r + piped stdin: contract first, then payload block, then instruction."""
    captured: dict[str, Any] = {}
    _invoke_prompt(
        monkeypatch,
        tmp_path,
        ["-r", "fix it"],
        stdin_text="config line\n",
        captured=captured,
    )
    msg = captured["message"]
    assert msg.startswith("OUTPUT CONTRACT (raw mode):")
    assert "<piped-input>\nconfig line\n</piped-input>" in msg
    assert msg.rstrip().endswith("fix it")


def test_raw_transport_gets_stderr_console(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """In raw mode the transport console is NOT the shared console; else it is."""
    from hosts.cli.transport import PromptTransport as _OT

    created: dict[str, Any] = {}
    real_init = _OT.__init__

    def _spy_init(self: Any, **kwargs: Any) -> None:
        created["console"] = kwargs.get("console")
        real_init(self, **kwargs)

    monkeypatch.setattr(_OT, "__init__", _spy_init)
    captured: dict[str, Any] = {}
    _invoke_prompt(monkeypatch, tmp_path, ["-r", "fix it"], captured=captured)
    assert created["console"] is not captured["console"]
    # The non-raw path keeps the shared (patched-in) console.
    created.clear()
    _invoke_prompt(monkeypatch, tmp_path, ["no raw"], captured=captured)
    assert created["console"] is captured["console"]


# ── Yield fail-fast ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_transport_yield_unresolved_skips_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A yielded response writes nothing to stdout; the question goes to stderr."""
    transport = PromptTransport(
        console=Console(stderr=True, force_terminal=False, width=120), raw=True
    )
    await transport.on_response(
        "I need the box's current network prefixes.", {"yielded": "Which networks?"}
    )
    captured = capsys.readouterr()
    assert transport.yield_unresolved is True
    assert transport.yield_question == "Which networks?"
    # stdout received NOTHING — the question must never land in a redirect.
    assert captured.out == ""
    assert "asked a question" in captured.err
    assert "Which networks?" in captured.err


@pytest.mark.asyncio
async def test_prompt_transport_yield_tty_shows_question_inline() -> None:
    """A tty transport shows the question inline (no stderr fail-fast text)."""
    record = Console(record=True, force_terminal=True, width=100)
    transport = PromptTransport(console=record)
    await transport.on_response("I need more detail.", {"yielded": "What detail do you need?"})
    assert transport.yield_unresolved is True
    assert "What detail do you need?" in record.export_text()


@pytest.mark.asyncio
async def test_prompt_transport_yield_panel_formats_context_mode_remediation() -> None:
    """The yield panel shows the question, why, the resume mode, and commands."""
    record = Console(record=True, force_terminal=True, width=90)
    transport = PromptTransport(console=record)
    await transport.on_response(
        "Preparation now fails at the pack's import-origin verification.",
        {
            "yielded": (
                "Preparation now fails at the pack's import-origin verification. "
                "Can you fix the pack source so the verification accepts namespace packages?"
            ),
            "yield_context": (
                "pydantic validation error is gone; preparation fails at a new stage "
                "(RuntimeError: module has no import origin: sdk)"
            ),
            "yield_mode_id": "coding_execute",
            "remediation": ["/stall item-1", "/skip item-1"],
        },
    )
    rendered = record.export_text()
    assert transport.yield_unresolved is True
    assert transport.yield_question.startswith("Preparation now fails")
    # The question reads first, in a dedicated panel.
    assert "Asked you a question" in rendered
    assert "import-origin verification" in rendered
    # Kernel-supplied why / where-to-reply / remediation lines are present.
    assert "Why:" in rendered and "import origin: sdk" in rendered
    assert "Reply resumes in mode: coding_execute" in rendered
    assert "/stall item-1" in rendered and "/skip item-1" in rendered
    # The plain answer path is NOT taken: no Response rule title.
    assert "Response" not in rendered.split("Asked you a question")[0]


@pytest.mark.asyncio
async def test_prompt_transport_yield_bare_question_has_no_empty_labels() -> None:
    """A plain model ask (no context/mode/remediation) renders only the question."""
    record = Console(record=True, force_terminal=True, width=90)
    transport = PromptTransport(console=record)
    await transport.on_response("answer text", {"yielded": "Which networks?"})
    rendered = record.export_text()
    assert "Which networks?" in rendered
    assert "Asked you a question" in rendered
    # No fabricated labels for fields the runtime did not supply.
    assert "Why:" not in rendered
    assert "Reply resumes in mode" not in rendered
    assert "Remediation:" not in rendered


def test_yielded_prompt_exits_4(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn that ends in a yield exits _YIELD_UNRESOLVED_EXIT (4)."""
    runtime = _make_runtime(tmp_path)

    async def _fake_handle_message(_runtime, _message, transport):
        await transport.on_response("I need the current networks.", {"yielded": "Which networks?"})
        return "ok"

    def _fake_init(**_kwargs):
        return runtime

    import runtime.handle_message as hm_module

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    monkeypatch.setattr(hm_module, "handle_message", _fake_handle_message)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result = cli_main.app(args=["prompt", "-r", "update the config"], standalone_mode=False)
    assert result == cli_main._YIELD_UNRESOLVED_EXIT


def _approval_payload() -> dict[str, Any]:
    return {
        "schema": "lisa.approval.v1",
        "request_id": "abc123def456",
        "target_kind": "tool",
        "target_id": "bash_run",
        "effect": "execute",
        "preview": "run a shell command",
        "command": "ls -la",
        "containment": "contained",
        "policy_verdict": "ask",
    }


@pytest.mark.asyncio
async def test_prompt_transport_flags_unresolved_approval() -> None:
    """A response carrying an approval payload sets approval_unresolved."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    transport = PromptTransport(console=console)
    await transport.on_response("Approval required.", {"approval": _approval_payload()})
    assert transport.approval_unresolved is True
    assert transport.pending_approval is not None
    # Nothing interactive leaked into the captured (stdout) console.
    assert "CONTAINED" not in buffer.getvalue()


@pytest.mark.asyncio
async def test_prompt_transport_approval_stderr_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-tty transport console: the approval notice goes to stderr.

    The notice advertises the top-level decision commands (no interactive
    loop needed) rather than the bare /approve slash form.
    """
    transport = PromptTransport(console=Console(force_terminal=False, width=120))
    await transport.on_response("Approval required.", {"approval": _approval_payload()})
    captured = capsys.readouterr()
    assert "lisa approve abc123def456" in captured.err
    assert "Approval required" in captured.err
    assert "lisa deny abc123def456" in captured.err


@pytest.mark.asyncio
async def test_prompt_transport_approval_panel_on_tty() -> None:
    """A tty transport console still renders the interactive panel."""
    record = Console(record=True, force_terminal=True, width=100)
    transport = PromptTransport(console=record)
    await transport.on_response("Approval required.", {"approval": _approval_payload()})
    assert "CONTAINED" in record.export_text()
    assert transport.approval_unresolved is True


@pytest.mark.asyncio
async def test_prompt_transport_raw_writes_verbatim_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Raw on_response writes the text verbatim; footer goes to stderr."""
    # Mirror production raw wiring: the transport console is stderr.
    transport = PromptTransport(
        console=Console(stderr=True, force_terminal=False, width=120), raw=True
    )
    transport._status_turn_tokens = 4200
    await transport.on_response("server {\n  listen 80;\n}")
    captured = capsys.readouterr()
    assert captured.out == "server {\n  listen 80;\n}\n"
    assert "⚡" in captured.err
    # No markdown re-render artifacts.
    assert "─" not in captured.out


def test_ctrl_c_abort_prints_notice_and_exits_130(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGINT-style cancellation mid-turn prints one abort line, exit 130.

    asyncio.run surfaces a mid-turn cancel as CancelledError from the
    coroutine; the command must convert it to a one-line abort notice
    (never the framework excepthook's 40-line traceback) and exit with
    the conventional SIGINT code so shells read cancelled-not-crashed.
    """
    import asyncio

    runtime = _make_runtime(tmp_path)

    async def _cancel_mid_turn(_runtime, _message, _transport):
        raise asyncio.CancelledError

    def _fake_init(**_kwargs):
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", _cancel_mid_turn)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    rec = Console(record=True, force_terminal=True, width=100)
    with patch.object(cli_main, "console", new=rec):
        result: int | None = cli_main.app(args=["prompt", "long think"], standalone_mode=False)
    assert result == cli_main._ABORT_EXIT
    out = rec.export_text()
    assert "aborted" in out
    assert "Traceback" not in out


def test_ctrl_c_abort_raw_mode_notice_on_stderr(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In raw mode the abort notice goes to stderr, keeping stdout clean."""
    import asyncio

    runtime = _make_runtime(tmp_path)

    async def _cancel_mid_turn(_runtime, _message, _transport):
        raise asyncio.CancelledError

    def _fake_init(**_kwargs):
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    import runtime.handle_message as hm_module

    monkeypatch.setattr(hm_module, "handle_message", _cancel_mid_turn)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    # The raw-mode abort notice is printed to a fresh stderr console, so
    # assert on capsys: stdout must stay empty (the raw contract), stderr
    # carries the abort line.
    with patch.object(cli_main, "console", new=Console(quiet=True)):
        result: int | None = cli_main.app(
            args=["prompt", "-r", "long think"], standalone_mode=False
        )
    assert result == cli_main._ABORT_EXIT
    captured = capsys.readouterr()
    assert "aborted" in captured.err
    assert captured.out == ""
