"""Interactive input surfaces for one-shot commands.

``lisa prompt`` invoked with no positional reads its message from the
terminal instead of argv so the message is never subject to the shell's
argument grammar: quotes, backticks, and apostrophes arrive literally
without escaping. The collected text becomes the one-shot turn — no
REPL loop, no transcript, one invocation = one turn.

Two readers, same contract:

- :func:`read_prompt_message` — plain ``input()`` loop (no extra
  dependency). Lines accumulate until EOF on an empty line (Ctrl-D, or
  Ctrl-Z+Enter on Windows).
- :func:`read_prompt_message_tui` — prompt_toolkit single-box editor
  (when the optional ``lisa-cli[prompt]`` extra is installed, with
  graceful fallback to the plain reader). Enter/numpad-Enter submits;
  Ctrl-J and Alt+Enter (``escape, enter``) start a new line — the
  reverse of prompt_toolkit's ``multiline=True`` defaults, matching the
  chat-box semantics the pre-REPL CLI settled on, portable across
  kitty/CSI-u terminals where Shift+Enter cannot be distinguished.

Both return ``None`` to mean "operator aborted with no message"; the
caller turns that into the conventional SIGINT exit. Whitespace-only
text means "nothing entered" and is a usage error at the call site.
The history file under ``state/`` persists submitted prompts for
``Up``-arrow reuse in the TUI reader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console

#: Marker printed before the plain reader so the operator knows the
#: terminal, not argv, owns the input now.
_HINT_PLAIN = (
    "Enter your message (quotes and $() pass literally). "
    "Ctrl-D — or Ctrl-Z then Enter on Windows — on the last line to send."
)
_HINT_SUBMIT = "Enter sends · Alt+Enter or Ctrl-J adds a line · Ctrl-C aborts"

#: prompt_toolkit pieces when the optional extra is installed; ``None``
#: means the plain reader is the only reader. The import lives here
#: rather than in main.py so ``--help`` and piped runs never even import
#: this module, and the extra is genuinely optional.
_PT: dict[str, Any] | None = None
try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.formatted_text import HTML as _HTML
    from prompt_toolkit.history import FileHistory as _FileHistory
    from prompt_toolkit.key_binding import KeyBindings as _KeyBindings

    _PT = {
        "PromptSession": _PromptSession,
        "HTML": _HTML,
        "FileHistory": _FileHistory,
        "KeyBindings": _KeyBindings,
    }
except ImportError:  # optional extra lisa-cli[prompt] absent
    _PT = None


def prompt_history_path(data_dir: str | Path, profile: str) -> Path:
    """History file for prompt-mode input under the profile's state dir.

    Lives in ``<data-dir>/state/prompt_history.txt`` — the same category
    the REPL-era history used (machine-written, small). Profile resolution
    mirrors ``_resolve_profiled_data_dir``: explicit data dir wins, then
    the profile dir; empty hands the decision to the framework's default
    (``LISA_DATA_DIR`` > ``~/.lisa``).
    """
    base = str(data_dir).strip() if str(data_dir).strip() else ""
    if not base and profile.strip():
        from runtime.config import profile_data_dir  # lazy: import cost

        base = str(profile_data_dir(profile.strip()))
    if not base:
        base = os.getenv("LISA_DATA_DIR", "") or str(Path.home() / ".lisa")
    return Path(base).expanduser() / "state" / "prompt_history.txt"


def read_prompt_message(console: Console) -> str | None:
    """Read a multiline message from the plain terminal (no dependencies).

    Lines accumulate until EOF on an empty line (Ctrl-D on POSIX,
    Ctrl-Z+Enter on Windows). Returns ``None`` when the operator aborts
    with Ctrl-C — a one-line notice is printed so pipelines read
    "cancelled, not crashed"; the caller exits with ``_ABORT_EXIT``.

    Pasting is safe: a paste of N lines adds N lines of buffer and does
    not submit, so quotes, backticks, and ``$()`` arrive literally.
    """
    console.print(f"[dim]{_HINT_PLAIN}[/dim]")
    lines: list[str] = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("[red]⏹ Entry aborted by operator (Ctrl-C).[/red]")
            return None
    return "\n".join(lines)


def read_prompt_message_tui(console: Console, *, history_path: Path) -> str | None:
    """Read a single-box message with prompt_toolkit (extra installed).

    Chat-box semantics: Enter and numpad-Enter submit; Ctrl-J and
    Alt+Enter / Esc+Enter (the ``escape, enter`` sequence) start a new
    line; Ctrl-C aborts (returns ``None``); bare Ctrl-D or an empty
    submit also returns ``None`` (nothing to send). Up-arrow history
    comes from ``history_path``.

    Falls back to :func:`read_prompt_message` when the extra is absent.
    """
    if _PT is None:
        return read_prompt_message(console)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = _PT["FileHistory"](str(history_path))
        bindings = _PT["KeyBindings"]()

        @bindings.add("enter", eager=True)
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @bindings.add("c-j")
        def _newline_ctrl_j(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "enter")
        def _newline_alt_enter(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        session: Any = _PT["PromptSession"](
            history=history,
            multiline=True,
            key_bindings=bindings,
            bottom_toolbar=_PT["HTML"](f"<style fg='ansibrightblack'>{_HINT_SUBMIT}</style>"),
        )
        text: str | None = session.prompt("❯ ", enable_suspend=True)
    except KeyboardInterrupt:
        console.print("[red]⏹ Entry aborted by operator (Ctrl-C).[/red]")
        return None
    except (EOFError, OSError) as exc:
        console.print(f"[red]Failed to read prompt input: {exc}[/red]")
        return None
    if text is None or not text.strip():
        return None
    return text.strip("\n")


__all__ = ["prompt_history_path", "read_prompt_message", "read_prompt_message_tui"]
