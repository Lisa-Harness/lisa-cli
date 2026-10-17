"""Tests for the `lisa last` CLI subcommand.

`lisa last` reprints a previous response verbatim from the durable thread —
the shell-side twin of the REPL's `/last`. Covers:

- default replay of the previous model response (byte-for-byte),
- explicit turn addressing (`lisa last 1`),
- usage / not-found diagnostics and their exit codes,
- the empty-thread friendly message.

The runtime builder is stubbed (no MCP subprocess, no model call); the
rendering goes through the real `runtime.commands.render_last_turn` so the
CLI and REPL share one code path.
"""

from __future__ import annotations

import re

# pylint: disable=protected-access
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from hosts.cli import main as cli_main

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Any) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _invoke_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    argv: list[str],
    *,
    force_terminal: bool = False,
) -> tuple[int | None, str]:
    """Invoke ``_last_command`` with a stubbed runtime and capture stdout.

    Returns ``(exit_code, captured_output)``. The runtime is a real test
    runtime (thread paths resolve against tmp_path) so `render_last_turn`
    reads the entries the test appends. ``force_terminal`` exercises the
    interactive rendering path (Rich Markdown + Response rule).
    """
    runtime = _make_runtime(tmp_path)

    def _fake_init(**_kwargs):
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _fake_init)
    out_console = Console(
        file=__import__("io").StringIO(),
        width=200,
        legacy_windows=False,
        force_terminal=force_terminal,
    )
    with patch.object(cli_main, "console", new=out_console):
        result: int | None = cli_main.app(args=["last", *argv], standalone_mode=False)
    output = out_console.file.getvalue()  # type: ignore[attr-defined]
    code = 0 if result is None else int(result)
    return code, output


# ── Replay behavior ───────────────────────────────────────────────────


def test_last_replays_previous_model_response(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa last` prints the previous model turn's response verbatim."""
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="model answer",
        turn=0,
    )
    code, output = _invoke_last(monkeypatch, tmp_path, [])

    assert code == 0
    assert "model answer" in output


def test_last_explicit_turn_index(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """`lisa last 1` replays turn 1's response."""
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="first question",
        response_text="first answer",
        turn=0,
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="second question",
        response_text="second answer",
        turn=1,
    )
    code, output = _invoke_last(monkeypatch, tmp_path, ["1"])

    assert code == 0
    assert "second answer" in output
    assert "first answer" not in output


def test_last_empty_thread_friendly_message(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """`lisa last` with no completed turns prints the friendly message."""
    code, output = _invoke_last(monkeypatch, tmp_path, [])

    assert code == 0
    assert "No completed turns yet." in output


# ── Turn-number surfacing ─────────────────────────────────────────────


def test_last_piped_output_carries_turn_header(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped replay prepends ``[ turn N ]`` so captured output is addressable."""
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    for i, (question, answer) in enumerate(
        [("first question", "first answer"), ("second question", "second answer")]
    ):
        append_thread_entry(
            thread_path=runtime.paths.thread_path,
            user_slug=runtime.config.user_slug,
            mode_id="default",
            query=question,
            response_text=answer,
            turn=i,
        )
    code, output = _invoke_last(monkeypatch, tmp_path, [])

    assert code == 0
    assert "[ turn 1 ]" in output
    assert "second answer" in output
    assert "turn 0" not in output


def test_last_explicit_piped_output_carries_that_turn_header(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lisa last 0`` piped labels the replay with turn 0, not the newest."""
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    for i, (question, answer) in enumerate(
        [("first question", "first answer"), ("second question", "second answer")]
    ):
        append_thread_entry(
            thread_path=runtime.paths.thread_path,
            user_slug=runtime.config.user_slug,
            mode_id="default",
            query=question,
            response_text=answer,
            turn=i,
        )
    code, output = _invoke_last(monkeypatch, tmp_path, ["0"])

    assert code == 0
    assert "[ turn 0 ]" in output
    assert "first answer" in output


def test_last_diagnostics_bare_without_turn_header(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostics (usage / not-found / empty) print bare, no header line."""
    code, output = _invoke_last(monkeypatch, tmp_path, [])

    assert code == 0
    assert "No completed turns yet." in output
    assert "[ turn " not in output
    code, output = _invoke_last(monkeypatch, tmp_path, ["42"])
    assert "Turn 42 not found in the thread log." in output
    assert "[ turn " not in output


def test_last_tty_rule_carries_turn_number(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive replay renders ``Response · turn N`` like the prompt output."""
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="tty answer",
        turn=7,
    )
    code, output = _invoke_last(monkeypatch, tmp_path, [], force_terminal=True)

    assert code == 0
    # Rich highlights the digits inside the rule title, so assert on the
    # ANSI-stripped text.
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    assert "Response · turn 7" in clean
    assert "tty answer" in output


# ── Diagnostics ───────────────────────────────────────────────────────


def test_last_bad_argument_usage_error(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric turn argument renders the usage message."""
    code, output = _invoke_last(monkeypatch, tmp_path, ["abc"])

    assert code == 0
    assert "Usage: /last [turn_number]" in output


def test_last_unknown_turn_index(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An out-of-range turn index renders the not-found message."""
    code, output = _invoke_last(monkeypatch, tmp_path, ["42"])

    assert code == 0
    assert "Turn 42 not found in the thread log." in output


# ── Rendering parity (tty) ────────────────────────────────────────────


def test_last_tty_renders_markdown_like_the_original_answer(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a tty, the replay gets the same treatment as the original answer:
    a ``Response`` rule separator plus Rich Markdown rendering (headings,
    bold, code fences) — not the plain unstyled print.
    """
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="formatting question",
        response_text="## Heading\n\n**bold point**\n\n```python\nprint('hi')\n```",
        turn=0,
    )
    code, output = _invoke_last(monkeypatch, tmp_path, [], force_terminal=True)

    assert code == 0
    assert "Response" in output  # the Rule title
    assert "Heading" in output
    assert "bold point" in output
