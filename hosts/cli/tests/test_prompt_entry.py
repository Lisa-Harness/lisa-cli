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
- Key layout: swapped by default (Enter = newline, Ctrl-J =
  send); ``--swap-keys`` (hoisted at group level too) or truthy
  ``LISA_PROMPT_KEYS`` selects chat-box (Enter sends). The TUI binding
  builder is exercised directly for both layouts, plus the env
  resolver's truthy set.

The reader seam (``cli_main._prompt_reader``) is monkeypatched so no
test touches a real terminal.
"""

from __future__ import annotations

import asyncio

# pylint: disable=protected-access
import io
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.live import Live

from hosts.cli import main as cli_main
from hosts.cli.transport import PromptTransport


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
    console: Console | None = None,
    on_initialize: Callable[[], None] | None = None,
) -> int | None:
    """Invoke the prompt command with scripted terminal input.

    ``reader_result`` is what the patched-in interactive reader returns:
    a string (submitted message), ``None`` (operator abort), or a list
    of ``KeyboardInterrupt``/string sentinel behaviors are not needed —
    the seam returns directly. When ``stdout_tty`` is False stdout reads
    as redirected (a pipe/file), keeping the editor away from pipelines.
    """
    runtime = _make_runtime(tmp_path)

    def _initialize(**_kw: Any) -> Any:
        if on_initialize is not None:
            on_initialize()
        return runtime

    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", _initialize)

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

    if console is None:
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


@pytest.fixture
def prompt_transports(monkeypatch: pytest.MonkeyPatch) -> list[PromptTransport]:
    transports: list[PromptTransport] = []
    original_init = PromptTransport.__init__

    def _initialize(transport: PromptTransport, **kwargs: Any) -> None:
        original_init(transport, **kwargs)
        transports.append(transport)

    monkeypatch.setattr(PromptTransport, "__init__", _initialize)
    return transports


def test_send_shows_activity_before_expansion_and_runtime_setup(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    prompt_transports: list[PromptTransport],
) -> None:
    output = io.StringIO()
    terminal = Console(file=output, force_terminal=True)
    activities: list[Live] = []

    def _check_activity() -> None:
        assert len(prompt_transports) == 1
        activity = prompt_transports[0]._activity
        assert activity is not None
        assert "starting runtime" in output.getvalue()
        activities.append(activity)

    def _expand(text: str) -> str:
        _check_activity()
        assert text == "submitted prompt"
        return "expanded prompt"

    monkeypatch.setattr(cli_main, "_expand_editor_message", _expand)
    original_start_turn = PromptTransport.start_turn

    def _start_turn(transport: PromptTransport, message: str) -> None:
        _check_activity()
        assert message == "expanded prompt"
        original_start_turn(transport, message)

    monkeypatch.setattr(PromptTransport, "start_turn", _start_turn)
    assert (
        _invoke(
            monkeypatch,
            tmp_path,
            [],
            reader_result="submitted prompt",
            console=terminal,
            on_initialize=_check_activity,
        )
        == 0
    )
    assert len(activities) == 3
    assert all(activity is activities[0] for activity in activities)
    assert prompt_transports[0]._activity is None


@pytest.mark.parametrize("stage", ["expansion", "initialization"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_startup_failure_cleans_up_activity(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    prompt_transports: list[PromptTransport],
    stage: str,
    error: type[BaseException],
) -> None:
    terminal = Console(file=io.StringIO(), force_terminal=True)

    def _fail(*_args: Any) -> None:
        assert prompt_transports[0]._activity is not None
        raise error("startup interrupted")

    if stage == "expansion":
        monkeypatch.setattr(cli_main, "_expand_editor_message", _fail)

    def _run() -> int | None:
        return _invoke(
            monkeypatch,
            tmp_path,
            [],
            reader_result="submitted prompt",
            console=terminal,
            on_initialize=_fail if stage == "initialization" else None,
        )

    if error is KeyboardInterrupt:
        assert _run() == cli_main._ABORT_EXIT
    else:
        with pytest.raises(error, match="startup interrupted"):
            _run()
    assert prompt_transports[0]._activity is None


@pytest.mark.parametrize(
    ("argv", "terminal", "expected_activity"),
    [([], True, True), (["-q"], True, False), ([], False, False)],
)
def test_startup_feedback_respects_quiet_and_redirection(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    prompt_transports: list[PromptTransport],
    argv: list[str],
    *,
    terminal: bool,
    expected_activity: bool,
) -> None:
    output = io.StringIO()

    def _check_activity() -> None:
        assert (prompt_transports[0]._activity is not None) is expected_activity
        assert ("starting runtime" in output.getvalue()) is expected_activity

    assert (
        _invoke(
            monkeypatch,
            tmp_path,
            [*argv, "argument prompt"],
            reader_result="unused",
            console=Console(file=output, force_terminal=terminal),
            on_initialize=_check_activity,
        )
        == 0
    )


@pytest.mark.parametrize("quiet", [False, True])
def test_raw_startup_feedback_uses_stderr(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    prompt_transports: list[PromptTransport],
    quiet: bool,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    terminal = Console(file=stderr, force_terminal=True)

    def _console(*, stderr: bool) -> Console:
        assert stderr is True
        return terminal

    def _check_activity() -> None:
        assert (prompt_transports[0]._activity is not None) is not quiet
        assert ("starting runtime" in stderr.getvalue()) is not quiet
        assert stdout.getvalue() == ""

    monkeypatch.setattr(cli_main, "Console", _console)
    assert (
        _invoke(
            monkeypatch,
            tmp_path,
            ["-q", "-r"] if quiet else ["-r"],
            reader_result="submitted prompt",
            console=Console(file=stdout, force_terminal=True),
            on_initialize=_check_activity,
        )
        == 0
    )
    assert stdout.getvalue() == ""


@pytest.mark.parametrize(("message", "exit_code"), [(None, 130), ("   \n", 2)])
def test_editor_abort_or_empty_submission_does_not_start_activity(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    prompt_transports: list[PromptTransport],
    message: str | None,
    exit_code: int,
) -> None:
    output = io.StringIO()
    assert (
        _invoke(
            monkeypatch,
            tmp_path,
            [],
            reader_result=message,
            console=Console(file=output, force_terminal=True),
        )
        == exit_code
    )
    assert prompt_transports[0]._activity is None
    assert "starting runtime" not in output.getvalue()


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


def test_ps1_unset_keeps_default_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No LISA_PROMPT_PS1 no settings file: the built-in default hint."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv(entry_mod._PROMPT_PS1_ENV, raising=False)
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))  # no prompt_hints.yaml here
    for frag in entry_mod._FRAGMENT_ENV.values():
        monkeypatch.delenv(frag, raising=False)
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
    assert entry_mod._DEFAULT_BANNER in text
    assert entry_mod._DEFAULT_SEND["plain"] in text


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
    export = console.export_text()
    assert entry_mod._DEFAULT_BANNER not in export
    assert entry_mod._DEFAULT_SEND["plain"] not in export


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
    assert entry_mod._DEFAULT_BANNER not in export


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


# ── Key layout: swapped default, --swap-keys / LISA_PROMPT_KEYS opt-in ─


def test_prompt_keys_default_is_swapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LISA_PROMPT_KEYS: the shipped swapped layout (Enter = newline)."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv(entry_mod._PROMPT_KEYS_ENV, raising=False)
    assert entry_mod.prompt_keys_env_wants_enter_send() is False


@pytest.mark.parametrize("val", ["0", "false", "no", "off", " "])
def test_prompt_keys_falsy_env_keeps_swap(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    """Falsy LISA_PROMPT_KEYS is still the shipped swapped default."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setenv(entry_mod._PROMPT_KEYS_ENV, val)
    assert entry_mod.prompt_keys_env_wants_enter_send() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "YES"])
def test_prompt_keys_truthy_env_wants_enter_send(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    """Truthy LISA_PROMPT_KEYS selects chat-box (Enter sends)."""
    from hosts.cli import entry as entry_mod

    monkeypatch.setenv(entry_mod._PROMPT_KEYS_ENV, val)
    assert entry_mod.prompt_keys_env_wants_enter_send() is True


def test_binding_layouts_swap_send_and_newline_keys() -> None:
    """Enter and Ctrl-J land on opposite actions in the two layouts."""
    from hosts.cli import entry as entry_mod

    def _layout(swapped: bool) -> dict[tuple, str]:
        actions: dict[tuple, str] = {}

        class _FakeBindings:
            def add(self, *keys: Any, **_kw: Any):
                def _decorator(fn: Any) -> Any:
                    actions[keys] = {
                        "_send": "send",
                        "_newline": "newline",
                    }.get(fn.__name__, "other")
                    return fn

                return _decorator

        entry_mod._add_navigate_bindings(_FakeBindings(), swapped=swapped)
        return actions

    swapped = _layout(True)
    chat = _layout(False)
    assert swapped[("enter",)] == "newline"
    assert swapped[("c-j",)] == "send"
    assert swapped[("escape", "enter")] == "other"
    assert chat[("enter",)] == "send"
    assert chat[("c-j",)] == "newline"
    assert chat[("escape", "enter")] == "other"


def test_tui_bindings_eager_send_keys_in_both_layouts() -> None:
    """Single-key send bindings are eager; sequence bindings are not."""
    from hosts.cli import entry as entry_mod

    def _captured(swapped: bool) -> dict[tuple, bool]:
        calls: dict[tuple, bool] = {}

        class _FakeBindings:
            def add(self, *keys: Any, eager: Any = False, **_kw: Any):
                def _decorator(fn: Any) -> Any:
                    calls[keys] = bool(eager)
                    return fn

                return _decorator

        entry_mod._add_navigate_bindings(_FakeBindings(), swapped=swapped)
        return calls

    swapped = _captured(True)
    chat = _captured(False)
    assert swapped[("c-j",)] is True  # single-key send swallowed eagerly
    assert swapped[("escape", "enter")] is False  # sequence: not eager
    assert chat[("enter",)] is True  # chat-box Enter-send is eager too
    assert chat[("escape", "enter")] is False


def test_tui_reader_hint_follows_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The toolbar hint names the active layout: swapped vs chat-box."""
    from prompt_toolkit.key_binding import KeyBindings as _RealKB

    from hosts.cli import entry as entry_mod

    seen: dict[str, Any] = {}

    class _FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            seen["toolbar"] = kwargs.get("bottom_toolbar")
            seen["bindings"] = kwargs.get("key_bindings")

        def prompt(self, *_a: Any, **_kw: Any) -> str:
            return "hello"

    real_pt = entry_mod._PT
    fake_pt = dict(real_pt)
    fake_pt["PromptSession"] = _FakeSession
    fake_pt["KeyBindings"] = _RealKB
    monkeypatch.setattr(entry_mod, "_PT", fake_pt)
    hist = tmp_path / "state" / "prompt_history.txt"
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path / "no-settings"))  # no yaml here

    for frag in entry_mod._FRAGMENT_ENV.values():
        monkeypatch.delenv(frag, raising=False)
    # swapped (shipped default: no env, no flag)
    monkeypatch.delenv(entry_mod._PROMPT_KEYS_ENV, raising=False)
    entry_mod.read_prompt_message_tui(Console(quiet=True, file=io.StringIO()), history_path=hist)
    toolbar = str(seen["toolbar"])
    assert entry_mod._DEFAULT_SEND["swap"] in toolbar
    assert entry_mod._DEFAULT_NEWLINE["swap"] in toolbar
    # chat-box via truthy env (Enter sends hint)
    monkeypatch.setenv(entry_mod._PROMPT_KEYS_ENV, "1")
    entry_mod.read_prompt_message_tui(Console(quiet=True, file=io.StringIO()), history_path=hist)
    toolbar = str(seen["toolbar"])
    assert entry_mod._DEFAULT_SEND["chat"] in toolbar
    assert entry_mod._DEFAULT_NEWLINE["chat"] in toolbar
    assert entry_mod._DEFAULT_SEND["swap"] not in toolbar
    # explicit swapped_keys=True pins the shipped layout against the env
    entry_mod.read_prompt_message_tui(
        Console(quiet=True, file=io.StringIO()), history_path=hist, swapped_keys=True
    )
    assert entry_mod._DEFAULT_SEND["swap"] in str(seen["toolbar"])


def test_seam_threads_swap_keys_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The flag passes through tri-state: absent -> env, True -> chat-box."""
    calls: list[dict[str, Any]] = []

    def _fake_tui(console: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "msg"

    monkeypatch.setattr("hosts.cli.entry.read_prompt_message_tui", _fake_tui)
    monkeypatch.delenv("LISA_PROMPT_KEYS", raising=False)
    stdin_obj: Any = io.StringIO()
    stdin_obj.isatty = lambda: True  # type: ignore[method-assign]
    stdout_obj: Any = io.StringIO()
    stdout_obj.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    monkeypatch.setattr(sys, "stdout", stdout_obj)

    # Flag absent: the env decides inside the reader (swapped_keys=None).
    cli_main._collect_interactive_or_error("", "", "/tmp/d", "", swap_keys=None)
    assert calls[-1].get("swapped_keys") is None
    # --no-swap-keys pins the shipped swapped layout.
    cli_main._collect_interactive_or_error("", "", "/tmp/d", "", swap_keys=False)
    assert calls[-1].get("swapped_keys") is True
    # --swap-keys selects chat-box (Enter sends).
    cli_main._collect_interactive_or_error("", "", "/tmp/d", "", swap_keys=True)
    assert calls[-1].get("swapped_keys") is False


# ── Hint-fragment settings (prompt_hints.yaml + env backstops) ────────


def test_settings_file_path_resolution(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_hints.yaml sits under <data-dir>/config/, profile-aware."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    assert entry_mod.settings_file_path(tmp_path, "") == (tmp_path / "config" / "prompt_hints.yaml")


def test_settings_file_absent_means_no_fragments(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No prompt_hints.yaml: the loader returns an empty mapping."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    assert entry_mod.load_hint_fragments(tmp_path, "") == {}


def test_settings_file_fragments_load(tmp_path: Any) -> None:
    """A valid YAML blob's recognized string keys surface as fragments."""
    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text(
        'send: "Ctrl-K sends"\nnewline: "Tab adds a line"\njunk: ignored\n',
        encoding="utf-8",
    )
    fragments = entry_mod.load_hint_fragments(tmp_path, "")
    assert fragments == {"send": "Ctrl-K sends", "newline": "Tab adds a line"}


def test_settings_fragment_values_suppress(
    tmp_path: Any,
) -> None:
    """Empty/null fragment values are treated as absent, not suppression."""
    import yaml as yaml_mod

    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text(
        "send: ''\nnewline: null\nabort: 'Abort with Ctrl-C'\n", encoding="utf-8"
    )
    fragments = entry_mod.load_hint_fragments(tmp_path, "")
    assert fragments == {"abort": "Abort with Ctrl-C"}
    del yaml_mod


def test_settings_file_malformed_is_loud_but_nonfatal(tmp_path: Any, capsys: Any) -> None:
    """A broken YAML file warns on stderr and falls back to built-ins."""
    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text("send: [unclosed\n", encoding="utf-8")
    assert entry_mod.load_hint_fragments(tmp_path, "") == {}
    err = capsys.readouterr().err
    assert "ignoring unreadable" in err


def test_resolve_hints_precedence_env_over_file_over_default(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env backstop > settings file > built-in, per fragment."""
    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text('send: "file says ctrl-k"\n', encoding="utf-8")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    # file wins over built-in
    hints = entry_mod.resolve_hints(tmp_path, "", layout="swap")
    assert hints["send"] == "file says ctrl-k"
    assert hints["abort"] == entry_mod._DEFAULT_ABORT
    # env wins over file
    monkeypatch.setenv(entry_mod._FRAGMENT_ENV["send"], "env says f4")
    hints = entry_mod.resolve_hints(tmp_path, "", layout="swap")
    assert hints["send"] == "env says f4"


def test_resolve_hints_layout_selects_key_facts(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without settings, the layout determines the send/newline facts."""
    from hosts.cli import entry as entry_mod

    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    for frag in entry_mod._FRAGMENT_ENV.values():
        monkeypatch.delenv(frag, raising=False)
    swap = entry_mod.resolve_hints(tmp_path, "", layout="swap")
    chat = entry_mod.resolve_hints(tmp_path, "", layout="chat")
    plain = entry_mod.resolve_hints(tmp_path, "", layout="plain")
    assert swap["send"] == entry_mod._DEFAULT_SEND["swap"]
    assert chat["send"] == entry_mod._DEFAULT_SEND["chat"]
    assert plain["send"] == entry_mod._DEFAULT_SEND["plain"]
    assert swap["newline"] == entry_mod._DEFAULT_NEWLINE["swap"]
    assert chat["newline"] == entry_mod._DEFAULT_NEWLINE["chat"]


def test_plain_reader_assembles_resolved_fragments(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain banner renders the settings-file fragments."""
    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text('send: "Ctrl-K on blank line sends"\n', encoding="utf-8")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv(entry_mod._PROMPT_PS1_ENV, raising=False)
    for frag in entry_mod._FRAGMENT_ENV.values():
        monkeypatch.delenv(frag, raising=False)
    out = io.StringIO()
    console = Console(file=out, record=True, width=200)
    lines = iter([])

    def _fake_input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)
    entry_mod.read_prompt_message(console, data_dir=tmp_path)
    text = console.export_text()
    assert "Ctrl-K on blank line sends" in text
    assert entry_mod._DEFAULT_BANNER in text  # untouched fragments stay built-in


def test_tui_toolbar_assembles_resolved_fragments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The TUI toolbar renders settings fragments (swap layout here)."""
    import yaml as yaml_mod
    from prompt_toolkit.formatted_text import HTML as _HTML

    from hosts.cli import entry as entry_mod

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "prompt_hints.yaml").write_text('send: "F5 transmits"\n', encoding="utf-8")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)

    seen: dict[str, Any] = {}

    class _FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            seen["toolbar"] = kwargs.get("bottom_toolbar")

        def prompt(self, *_a: Any, **_kw: Any) -> str:
            return "x"

    class _FakeHTML:
        def __init__(self, text: str) -> None:
            self.text = text

        def __repr__(self) -> str:
            return f"HTML({self.text!r})"

    real_pt = entry_mod._PT
    fake_pt = dict(real_pt)
    fake_pt["PromptSession"] = _FakeSession
    fake_pt["HTML"] = _FakeHTML
    monkeypatch.setattr(entry_mod, "_PT", fake_pt)
    entry_mod.read_prompt_message_tui(
        Console(quiet=True, file=io.StringIO()),
        history_path=tmp_path / "state" / "h.txt",
        swapped_keys=True,
        data_dir=tmp_path,
    )
    assert "F5 transmits" in repr(seen["toolbar"])
    import prompt_toolkit.formatted_text

    del yaml_mod, _HTML, prompt_toolkit


# ── Real terminal input through prompt_toolkit ───────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("swapped", "keys"),
    [(True, "first\rsecond\n"), (False, "first\nsecond\r")],
)
async def test_editor_sends_multiline_text_with_real_key_processing(
    swapped: bool, keys: str
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from hosts.cli import entry as entry_mod

    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            multiline=True,
            key_bindings=entry_mod._tui_key_bindings(swapped=swapped),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text(keys)
        assert await asyncio.wait_for(session.prompt_async(""), timeout=2) == "first\nsecond"


@pytest.mark.asyncio
@pytest.mark.parametrize("swapped", [True, False])
@pytest.mark.parametrize("alt_enter", ["\x1b\r", "\x1b[27;3;13~", "\x1b[13;3u"])
async def test_alt_enter_does_not_submit_or_modify_text(swapped: bool, alt_enter: str) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from hosts.cli import entry as entry_mod

    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            multiline=True,
            key_bindings=entry_mod._tui_key_bindings(swapped=swapped),
            input=pipe,
            output=DummyOutput(),
        )
        send = "\n" if swapped else "\r"
        pipe.send_text(f"before{alt_enter}after{send}")
        assert await asyncio.wait_for(session.prompt_async(""), timeout=2) == "beforeafter"


@pytest.mark.parametrize("layout", ["plain", "chat", "swap"])
def test_legacy_installer_hints_follow_active_reader(tmp_path: Any, layout: str) -> None:
    from hosts.cli import entry as entry_mod

    config = tmp_path / "config"
    config.mkdir()
    (config / "prompt_hints.yaml").write_text(
        'send: "Ctrl-J or Alt+Enter sends"\nnewline: "Enter adds a line"\n', encoding="utf-8"
    )
    hints = entry_mod.resolve_hints(tmp_path, "", layout=layout)
    assert hints["send"] == entry_mod._DEFAULT_SEND[layout]
    assert hints["newline"] == entry_mod._DEFAULT_NEWLINE[layout]


def test_missing_editor_dependency_warns_and_uses_correct_hints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from hosts.cli import entry as entry_mod

    config = tmp_path / "config"
    config.mkdir()
    (config / "prompt_hints.yaml").write_text(
        'send: "Ctrl-J or Alt+Enter sends"\n', encoding="utf-8"
    )
    monkeypatch.setattr(entry_mod, "_PT", None)
    lines = iter(["hello"])

    def _input() -> str:
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _input)
    output = io.StringIO()
    text = entry_mod.read_prompt_message_tui(
        Console(file=output, width=200),
        history_path=tmp_path / "history",
        data_dir=tmp_path,
    )
    assert text == "hello"
    assert "prompt-toolkit is not installed" in output.getvalue()
    assert "Ctrl-D on an empty line sends" in output.getvalue()
    assert "Ctrl-J" not in output.getvalue()
    assert "Alt+Enter" not in output.getvalue()
