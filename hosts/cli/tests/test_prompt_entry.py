"""Tests for the no-parameter interactive prompt mode.

When ``lisa prompt`` runs on a tty with no argv message, no piped
payload, and no pre-armed decision, it drops into lisa-managed input
(hosts.cli.entry) so the message bypasses the shell's quoting rules:

- The editor is only offered on the interactive path (stdin AND stdout
  are ttys); redirection keeps the exit-2 usage error.
- A submitted message becomes the one-shot turn verbatim (multi-line
  text joined as-is; quote characters pass literally — the reader seam
  bypasses both the shell and the $() expander by design in tests).
- Ctrl-C / no input aborts with the SIGINT exit code (130); whitespace
  input is a usage error (exit 2).

The reader seam (``cli_main._prompt_reader``) is monkeypatched so no
test touches a real terminal.
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


def _make_runtime(tmp_path: Any) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── Fixtures ──────────────────────────────────────────────────────────


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    argv: list[str],
    *,
    reader_result: Any,
    stdout_tty: bool = True,
    captured: dict[str, Any] | None = None,
) -> int | None:
    """Invoke the prompt command with scripted terminal input.

    ``reader_result`` is what the patched-in interactive reader returns:
    a string (submitted message), ``None`` (operator abort), or a list
    of ``KeyboardInterrupt``/string sentinel behaviors are not needed —
    the seam returns directly. When ``stdout_tty`` is False stdout reads
    as redirected (a pipe/file), keeping the editor away from pipelines.
    """
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kw: runtime)

    import runtime.handle_message as hm_module

    async def _fake_handle_message(_runtime, message, _transport):
        if captured is not None:
            captured["message"] = message
        return "ok"

    monkeypatch.setattr(hm_module, "handle_message", _fake_handle_message)

    reader_calls: list[tuple[str, str]] = []

    def _fake_reader(console: Any) -> str | None:
        reader_calls.append(("reader", "called"))
        return reader_result

    monkeypatch.setattr(cli_main, "_prompt_reader", _fake_reader)

    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)

    stdout_obj: Any = io.StringIO()
    stdout_obj.isatty = lambda: stdout_tty  # type: ignore[method-assign]
    real_stdout = sys.stdout
    monkeypatch.setattr(sys, "stdout", stdout_obj)

    console = Console(quiet=True, file=stdout_obj)
    try:
        with patch.object(cli_main, "console", new=console):
            result: int | None = cli_main.app(args=["prompt", *argv], standalone_mode=False)
    finally:
        monkeypatch.setattr(sys, "stdout", real_stdout)
    if captured is not None:
        captured["reader_called"] = bool(reader_calls)
        captured["stdout"] = stdout_obj.getvalue()
    return result


# ── Editor offered on the interactive path ────────────────────────────


def test_no_args_tty_opens_editor_and_sends_text(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lisa prompt` with no args collects the message from the editor."""
    captured: dict[str, Any] = {}
    result = _invoke(
        monkeypatch,
        tmp_path,
        [],
        reader_result='say "hello" now\nsecond line',
        captured=captured,
    )
    assert result == 0
    assert captured["reader_called"] is True
    # Quote characters pass literally — the whole point of lisa-managed
    # input. (The seam bypasses the $() expander; its path is tested in
    # test_prompt_expand.py.)
    assert captured["message"] == 'say "hello" now\nsecond line'


def test_editor_message_composes_with_flags(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw mode still prepends the contract to the editor-collected text."""
    captured: dict[str, Any] = {}
    result = _invoke(
        monkeypatch,
        tmp_path,
        ["-r"],
        reader_result="produce the artifact",
        captured=captured,
    )
    assert result == 0
    assert cli_main._RAW_CONTRACT in captured["message"]
    assert captured["message"].rstrip().endswith("produce the artifact")


# ── Abort and empty-submission contracts ──────────────────────────────


def test_editor_abort_exits_sigint(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader returning None (Ctrl-C/no text) is the SIGINT exit."""
    result = _invoke(monkeypatch, tmp_path, [], reader_result=None)
    assert result == cli_main._ABORT_EXIT


def test_editor_whitespace_text_exits_2(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only editor output is treated as no message (exit 2)."""
    result = _invoke(monkeypatch, tmp_path, [], reader_result="   \n")
    assert result == 2


def test_editor_not_offered_when_stdout_is_pipe(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirected stdout keeps the usage error — never block a pipeline."""
    captured: dict[str, Any] = {}
    result = _invoke(
        monkeypatch,
        tmp_path,
        [],
        reader_result="never reached",
        stdout_tty=False,
        captured=captured,
    )
    assert result == 2
    assert captured["reader_called"] is False


# ── Plain reader behavior ─────────────────────────────────────────────


def test_plain_reader_joins_lines_until_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plain reader accumulates lines; EOF terminates the loop."""
    from hosts.cli.entry import read_prompt_message

    lines = iter(['say "hi"', "second line"])

    def _fake_input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)
    console = Console(quiet=True)
    result = read_prompt_message(console)
    assert result == 'say "hi"\nsecond line'


def test_plain_reader_keyboard_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt (Ctrl-C) maps to the abort sentinel."""

    from hosts.cli.entry import read_prompt_message

    def _raise() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)
    console = Console(quiet=True)
    result = read_prompt_message(console)
    assert result is None


def test_plain_reader_eof_immediately_treated_as_no_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate EOF produces an empty string — the call site exits 2."""

    from hosts.cli.entry import read_prompt_message

    def _raise() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    console = Console(quiet=True)
    result = read_prompt_message(console)
    assert result == ""


# ── History path resolution ───────────────────────────────────────────


def test_history_path_explicit_data_dir(tmp_path: Any) -> None:
    """An explicit data dir pins the history file under its state dir."""
    from hosts.cli.entry import prompt_history_path

    path = prompt_history_path(tmp_path, "")
    assert path == tmp_path / "state" / "prompt_history.txt"


def test_history_path_profile_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No explicit dir: the profile dir is used when a profile is named."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setattr(
        "runtime.config.profile_data_dir", lambda name: tmp_path / "profiles" / name
    )
    path = entry_mod.prompt_history_path("", "work")
    assert path == tmp_path / "profiles" / "work" / "state" / "prompt_history.txt"


def test_history_path_framework_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No dir, no profile: LISA_DATA_DIR then ~/.lisa decide."""
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path / "envdata"))
    from hosts.cli.entry import prompt_history_path

    path = prompt_history_path("", "")
    assert path == tmp_path / "envdata" / "state" / "prompt_history.txt"
    monkeypatch.delenv("LISA_DATA_DIR")
    path = prompt_history_path("", "")
    assert path.name == "prompt_history.txt"
    assert path.parent.name == "state"


# ── LISA_PROMPT_PS1 hint contract ─────────────────────────────────────


def test_ps1_unset_keeps_default_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LISA_PROMPT_PS1: prompt_ps1 is None and readers print the default."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv(entry_mod._PROMPT_PS1_ENV, raising=False)
    assert entry_mod.prompt_ps1() is None
    out = io.StringIO()
    plain_console = Console(file=out, record=True, width=200)

    lines = iter(["hi"])

    def _fake_input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)
    entry_mod.read_prompt_message(plain_console)
    text = plain_console.export_text()
    assert entry_mod._HINT_PLAIN in text


def test_ps1_empty_suppresses_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """LISA_PROMPT_PS1='' suppresses the plain reader's banner entirely."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setenv(entry_mod._PROMPT_PS1_ENV, "")
    assert entry_mod.prompt_ps1() is None
    out = io.StringIO()
    console = Console(file=out, record=True, width=200)
    lines = iter([])

    def _fake_input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)
    entry_mod.read_prompt_message(console)
    assert entry_mod._HINT_PLAIN not in console.export_text()


def test_ps1_custom_text_replaces_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """LISA_PROMPT_PS1=<text> replaces the plain reader's banner verbatim."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setenv(entry_mod._PROMPT_PS1_ENV, "chat> ")
    assert entry_mod.prompt_ps1() == "chat> "
    out = io.StringIO()
    console = Console(file=out, record=True, width=200)
    lines = iter([])

    def _fake_input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)
    entry_mod.read_prompt_message(console)
    export = console.export_text()
    assert "chat> " in export
    assert entry_mod._HINT_PLAIN not in export


def test_ps1_whitespace_only_suppresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only PS1 value suppresses like empty — no blank banner."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setenv(entry_mod._PROMPT_PS1_ENV, "   ")
    assert entry_mod.prompt_ps1() is None


# ── Grapheme-cluster deletion (emoji-safe backspace/delete) ───────────


def test_grapheme_backspace_deletes_zwj_family_whole() -> None:
    """Backspace removes a ZWJ family emoji as one cluster — no tearing."""
    from hosts.cli import entry as entry_mod

    before = "prefix 👨‍👩‍👧‍👦!"
    assert entry_mod._delete_last_grapheme(before) == 1  # '!'
    torn = before[: len(before) - 1]
    assert entry_mod._delete_last_grapheme(torn) == 7  # the whole family
    rest = torn[: len(torn) - 7]
    assert rest == "prefix "


def test_grapheme_backspace_never_leaves_dangling_zwj() -> None:
    """No delete path leaves a dangling U+200D behind the cursor."""
    from hosts.cli import entry as entry_mod

    text = "x 👨‍👩‍👧‍👦"
    for _ in range(len(text)):
        cut = entry_mod._delete_last_grapheme(text)
        text = text[: len(text) - cut]
        if text and "\u200d" in text:
            assert not text.endswith("\u200d")
    assert text in {"x ", ""}


def test_grapheme_delete_forward_removes_flag_pair() -> None:
    """Forward delete removes a complete flag pair, never one half."""
    from hosts.cli import entry as entry_mod

    assert entry_mod._delete_first_grapheme("🇺🇸 tail") == 2
    assert entry_mod._delete_first_grapheme("👍🏽 x") == 2  # base + skin tone
    assert entry_mod._delete_first_grapheme("é") == 1  # precomposed é
    assert entry_mod._delete_first_grapheme("e\u0301") == 2  # decomposed
    half = "🇺"
    assert entry_mod._delete_first_grapheme(half) == 1


def test_grapheme_backspace_empty_and_noise() -> None:
    """Empty text and pure combining noise delete without error."""
    from hosts.cli import entry as entry_mod

    assert entry_mod._delete_last_grapheme("") == 0
    noise = "\u200d\u0301"
    assert entry_mod._delete_last_grapheme(noise) == len(noise)
    assert entry_mod._delete_last_grapheme("a\u200d") == 2  # dangling ZWJ rides with 'a'


def test_grapheme_backspace_keep_word_motion_scenarios() -> None:
    """Plain ASCII and CJK delete one codepoint (no over-reach)."""
    from hosts.cli import entry as entry_mod

    assert entry_mod._delete_last_grapheme("hello") == 1
    assert entry_mod._delete_last_grapheme("世界") == 1
    assert entry_mod._delete_last_grapheme("🙂") == 1
