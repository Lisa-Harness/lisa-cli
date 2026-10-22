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
  graceful fallback to the plain reader). Send and newline are
  swappable between Enter and Ctrl-J (Alt+Enter follows Ctrl-J in both
  layouts); the shipped default is swapped — Enter inserts a newline
  and Ctrl-J / Alt+Enter sends — so a hurried Enter never fires the
  turn (finish with Ctrl-J, or Ctrl-D on an empty line in the plain
  reader) and no trailing keystroke can reach the caller's shell.
  ``--swap-keys`` / truthy ``LISA_PROMPT_KEYS`` restores chat-box
  semantics (Enter sends).

The TUI reader owns its navigation model rather than trusting
prompt_toolkit's defaults: Ctrl+Up/Ctrl+Down are inert (prompt_toolkit
binds them to history navigation that replaces the whole multiline
buffer — the corruption that forced Ctrl-C), Alt+Up/Alt+Down walk
submitted-prompt history instead, and backspace deletes whole grapheme
clusters so multi-codepoint emoji (ZWJ families, flags, skin tones)
never tear. Word motion (Ctrl+Left/Right) keeps the prompt_toolkit
defaults.

``LISA_PROMPT_PS1`` customizes or suppresses the editor hints, PS1-style:
unset keeps the defaults, empty suppresses the hint text entirely, any
other value replaces it verbatim (plain text, no templates). The plain
reader prints it as its banner; the TUI reader puts it in the bottom
toolbar and drops the toolbar when suppressed.

Both return ``None`` to mean "operator aborted with no message"; the
caller turns that into the conventional SIGINT exit. Whitespace-only
text means "nothing entered" and is a usage error at the call site.
Quote characters survive because no shell ever tokenized this input;
``$(cmd)`` does NOT pass literally — the editor path expands it
(:mod:`hosts.cli.expand`). The history file under ``state/`` persists
submitted prompts for ``Alt+Up`` reuse in the TUI reader.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any

from rich.console import Console

#: Unicode general categories that extend a grapheme cluster: combining
#: marks, enclosing marks, format characters (variation selectors, ZWSP),
#: and modifier symbols (skin-tone modifiers). ZWJ itself is ``Cf`` but is
#: spelled out for clarity at the fold sites.
_EXTEND_CATEGORIES = ("Mn", "Me", "Mc", "Cf", "Sk")

#: Regional Indicator codepoint range (U+1F1E6 - U+1F1FF): the two halves of
#: an emoji flag. A flag is exactly one pair; a lone indicator is torn.
_RI_FIRST = 0x1F1E6
_RI_LAST = 0x1F1FF
_RI_PAIR = 2  # one flag = exactly two regional indicators

#: Env var customizing the editor hints, PS1-style. Empty value suppresses
#: the hints entirely; any other value replaces the default hint text
#: verbatim (plain text — no template expansion); unset defers to the
#: settings file, then built-in defaults.
_PROMPT_PS1_ENV = "LISA_PROMPT_PS1"

#: Settings file for the editor hints: ``<data-dir>/config/prompt_hints.yaml``.
#: A hand-editable YAML blob; each recognized key overrides that fragment of
#: the default hint line (or ``null``/``""`` suppresses it):
#:
#:     send: "Ctrl-J sends"            # how the editor sends
#:     newline: "Enter adds a line"    # what plain Enter does
#:     abort: "Ctrl-C aborts"          # how to back out
#:     banner: "…quoted are literal…"  # plain reader's first line
#:
#: Precedence for the whole-line text: ``LISA_PROMPT_PS1`` (verbatim, or
#: empty = suppressed) > the settings file (fragments assembled) > the
#: built-in defaults. Per-fragment env backstops
#: (``LISA_PROMPT_HINT_SEND`` etc.) sit between the file and the default:
#: env > file > built-in, per fragment.
_PROMPT_HINTS_FILE = "prompt_hints.yaml"
_PROMPT_HINT_ENV_PREFIX = "LISA_PROMPT_HINT_"
_FRAGMENT_ENV = {
    "send": "LISA_PROMPT_HINT_SEND",
    "newline": "LISA_PROMPT_HINT_NEWLINE",
    "abort": "LISA_PROMPT_HINT_ABORT",
    "banner": "LISA_PROMPT_HINT_BANNER",
}

#: Env var selecting the editor key model, PS1-style. Unset or falsy
#: (``0/false/no/off``): swapped default — Enter inserts a newline, Ctrl-J
#: or Alt+Enter sends (newline-first, so an impatient finisher cannot
#: send a half-typed message, and a double-tap cannot reach the shell).
#: Truthy (``1/true/yes/on``): chat-box — Enter sends, Ctrl-J or
#: Alt+Enter adds a newline. ``--swap-keys`` wins over the env var.
_PROMPT_KEYS_ENV = "LISA_PROMPT_KEYS"
_TRUTHY = {"1", "true", "yes", "on"}

#: Marker printed before the plain reader so the operator knows the
#: terminal, not argv, owns the input now. $() expands (expand.py runs
#: over editor text); quote characters survive because no shell ever
#: tokenized this input. The three strings below are the built-in
#: fallbacks; the settings file (prompt_hints.yaml) and env backstops
#: override fragments — see :func:`resolve_hints`.
_DEFAULT_BANNER = "Lisa prompt - quotes are literal; $(cmd) is expanded. ✨"
_DEFAULT_SEND = {
    "plain": "Ctrl-D on an empty line sends",
    "chat": "Enter sends",
    "swap": "Ctrl-J or Alt+Enter sends",
}
_DEFAULT_NEWLINE = {
    "plain": "each line is kept",
    "chat": "Alt+Enter or Ctrl-J adds a line",
    "swap": "Enter adds a line",
}
_DEFAULT_ABORT = "Ctrl-C aborts"
_PLAIN_SUSPEND_NOTE = "(Ctrl-Z suspends to your shell, it does not send.)"

_HINT_PLAIN = f"{_DEFAULT_BANNER} {_DEFAULT_SEND['plain']}; {_DEFAULT_ABORT}. {_PLAIN_SUSPEND_NOTE}"
_HINT_SUBMIT_CHAT = f"{_DEFAULT_SEND['chat']} · {_DEFAULT_NEWLINE['chat']} · {_DEFAULT_ABORT}"
_HINT_SUBMIT_SWAP = f"{_DEFAULT_SEND['swap']} · {_DEFAULT_NEWLINE['swap']} · {_DEFAULT_ABORT}"


def prompt_keys_env_wants_enter_send() -> bool:
    """Resolve the key model from ``LISA_PROMPT_KEYS`` (env back for the flag).

    The reader calls this only when neither ``--swap-keys`` nor
    ``--no-swap-keys`` was given. Returns ``True`` when the env asks for
    chat-box (Enter sends — the exceptional model); ``False`` otherwise
    (unset/env falsy = the shipped swapped layout, Enter = newline).
    """
    value = os.getenv(_PROMPT_KEYS_ENV)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


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


def prompt_ps1() -> str | None:
    """Resolve the editor hint text from ``LISA_PROMPT_PS1``.

    - Unset: built-in hint (plain reader's banner, TUI's toolbar).
    - Set to empty: ``None`` — hints suppressed entirely.
    - Any other value: that text verbatim (plain text, no templates).

    A whitespace-only value suppresses too: an operator asking for a
    quieter editor should get it, not a near-blank banner.
    """
    value = os.getenv(_PROMPT_PS1_ENV)
    if value is None:
        return None
    if not value.strip():
        return None
    return value  # verbatim — no strip, the operator owns the spacing


def settings_file_path(data_dir: str | Path, profile: str) -> Path:
    """The prompt-hints settings file under the profile's config dir.

    ``<data-dir>/config/prompt_hints.yaml`` — hand-editable, alongside the
    other authored config (models.yaml, policy.yaml). Profile resolution
    mirrors :func:`prompt_history_path`.
    """
    from runtime.config import profile_data_dir  # lazy: import cost

    base = str(data_dir).strip() if str(data_dir).strip() else ""
    if not base and profile.strip():
        base = str(profile_data_dir(profile.strip()))
    if not base:
        base = os.getenv("LISA_DATA_DIR", "") or str(Path.home() / ".lisa")
    return Path(base).expanduser() / "config" / _PROMPT_HINTS_FILE


def load_hint_fragments(data_dir: str | Path, profile: str) -> dict[str, str | None]:
    """Read the settings-file hint fragments (missing file: all absent).

    Recognized keys (send, newline, abort, banner) map to their string
    value, or ``None`` when the file does not exist, is unreadable, is
    not valid YAML, or the key is missing/an empty string (which the
    resolver treats as "fragment not set" — to suppress a fragment use
    the key's YAML null... which is the same as absent: both mean the
    backstops decide; there is no per-fragment suppression by design,
    a hint line with a hole is worse than no hole).

    A malformed file is a loud stderr warning once per process, then the
    resolver falls back to built-ins — settings never block a turn.
    """
    path = settings_file_path(data_dir, profile)
    if not path.is_file():
        return {}
    try:
        import yaml  # pyyaml is a hard dependency of the CLI host

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:  # pylint: disable=broad-except  # settings must not block
        import sys

        print(f"lisa: ignoring unreadable {path}", file=sys.stderr, flush=True)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str | None] = {}
    for key in ("send", "newline", "abort", "banner"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value
    return out


def resolve_hints(
    data_dir: str | Path,
    profile: str,
    *,
    layout: str,
) -> dict[str, str | None]:
    """Assemble the effective hint fragments for one reader invocation.

    ``layout`` selects the key-model facts: ``"plain"`` (EOF-send), or
    ``"chat"`` / ``"swap"`` (which of Enter/Ctrl-J sends). Precedence
    per fragment: env backstop > settings file > built-in default.
    Returns ``{send, newline, abort, banner}`` — a ``None`` value means
    the fragment is suppressed (only possible when the env backstop or
    settings file sets it explicitly empty... which :func:`load_hint_
    fragments` treats as absent today; the assembly below keeps the
    None slot so callers can render "suppressed" cleanly later).
    """
    file_fragments = load_hint_fragments(data_dir, profile)
    resolved: dict[str, str | None] = {}
    for key in ("send", "newline", "abort", "banner"):
        env_value = os.getenv(_FRAGMENT_ENV[key])
        if env_value is not None:
            resolved[key] = env_value.strip() or None
            continue
        if key in file_fragments:
            resolved[key] = file_fragments[key]
            continue
        resolved[key] = None
    # Facts no settings file should be able to contradict: the layout
    # determines the send/newline text when the operator has not spoken.
    if resolved["send"] is None:
        resolved["send"] = _DEFAULT_SEND[layout]
    if resolved["newline"] is None:
        resolved["newline"] = _DEFAULT_NEWLINE[layout]
    if resolved["abort"] is None:
        resolved["abort"] = _DEFAULT_ABORT
    return resolved


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


def read_prompt_message(
    console: Console,
    *,
    data_dir: str | Path = "",
    profile: str = "",
) -> str | None:
    """Read a multiline message from the plain terminal (no dependencies).

    Lines accumulate until EOF on an empty line (Ctrl-D on POSIX,
    Ctrl-Z+Enter on Windows). Returns ``None`` when the operator aborts
    with Ctrl-C — a one-line notice is printed so pipelines read
    "cancelled, not crashed"; the caller exits with ``_ABORT_EXIT``.

    Pasting is safe: a paste of N lines adds N lines of buffer and does
    not submit, so quotes, backticks, and apostrophes arrive literally
    (no shell ever tokenizes this input; ``$(cmd)`` is expanded by the
    call site — see :mod:`hosts.cli.expand`).

    The banner respects ``LISA_PROMPT_PS1`` (whole-line replace/suppress)
    and the hint-fragment settings (prompt_hints.yaml / env backstops)
    for banner, send, and abort fragments; the Ctrl-Z note stays (a
    settings file cannot claim Ctrl-Z sends — it suspends, factually).
    """
    hint = prompt_ps1() if _PROMPT_PS1_ENV in os.environ else None
    if _PROMPT_PS1_ENV in os.environ:
        # PS1 present: an empty/whitespace value means explicitly
        # suppressed — no fragments, no banner. A value is a verbatim
        # whole-line replace. The settings file only participates when
        # PS1 is unset.
        if hint:
            console.print(f"[dim]{hint}[/dim]")
    else:
        fragments = resolve_hints(data_dir, profile, layout="plain")
        if any(fragments.values()):
            parts = [
                fragments.get("banner") or _DEFAULT_BANNER,
                f"{fragments['send']}; {fragments['abort']}.",
                _PLAIN_SUSPEND_NOTE,
            ]
            console.print(f"[dim]{' '.join(parts)}[/dim]")
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


def _delete_last_grapheme(text: str) -> int:
    """Length (in codepoints) of the last extended grapheme cluster.

    prompt_toolkit's backspace deletes one codepoint, which tears
    multi-codepoint emoji: a ZWJ family loses its last member and leaves
    a dangling ``U+200D`` (rendered as broken halves), a flag loses one
    regional indicator, a skin-tone modifier splits off its base. The
    returned length deletes the whole cluster with a pragmatic
    UAX #29-like rule:

    - A trailing regional indicator: a complete pair is one cluster; an
      odd trailing run means the pair is already torn — the single
      dangling indicator counts alone so the next delete can re-pair.
    - Otherwise trailing combining marks / format chars / modifier
      symbols (variation selectors, skin tones) fold onto their base;
      a ``U+200D`` directly before that base folds the base's own
      cluster in (👨ZWJ👩ZWJ👧ZWJ👦 deletes as one).
    - A whitespace base never folds onto an emoji cluster.
    - An empty result means the whole text was combining/format noise.

    This is a heuristic, not full UAX #29 — tuned for what chat inputs
    actually hit (emoji families, flags, tones, VS16, CJK) and
    conservative elsewhere.
    """
    n = len(text)
    if not n:
        return 0
    last = text[-1]
    if _RI_FIRST <= ord(last) <= _RI_LAST:  # regional indicator — pairs only
        i = n
        while i > 0 and _RI_FIRST <= ord(text[i - 1]) <= _RI_LAST:
            i -= 1
        run = n - i
        return 1 if run % 2 == 1 else _RI_PAIR
    i = n
    while i > 0 and (
        unicodedata.category(text[i - 1]) in _EXTEND_CATEGORIES or text[i - 1] == "\u200d"
    ):
        i -= 1
    if i == 0:
        return n  # everything was combining/format noise; it is one cluster
    i -= 1  # the base codepoint
    # ZWJ chain: a U+200D directly before this cluster's base folds that
    # base's cluster in (👨+ZWJ+👩+ZWJ+👧+ZWJ+👦 is one cluster). A
    # whitespace base never folds: 'a <ZWJ>' is a torn input, not a family.
    while i > 1 and text[i - 1] == "\u200d" and not text[i - 2].isspace():
        i -= 2
    return n - i


def _delete_first_grapheme(text: str) -> int:
    """Length (in codepoints) of the first extended grapheme cluster.

    The forward-delete counterpart of :func:`_delete_last_grapheme`:
    same rule, walked from the front. Leading regional indicators delete
    as a pair when two or more are present (an odd leading run deletes
    two first, so a torn flag re-pairs on subsequent deletes).
    """
    n = len(text)
    if not n:
        return 0
    if _RI_FIRST <= ord(text[0]) <= _RI_LAST:
        i = 0
        while i < n and _RI_FIRST <= ord(text[i]) <= _RI_LAST:
            i += 1
        return _RI_PAIR if i > 1 else 1
    i = 1
    while i < n:
        ch = text[i]
        if unicodedata.category(ch) in _EXTEND_CATEGORIES or ch == "\u200d":
            i += 1
            if i < n and text[i - 1] == "\u200d" and not text[i].isspace():
                i += 1  # the base after the ZWJ folds in
                continue
            continue
        break
    return i


def _add_navigate_bindings(bindings: Any, *, swapped: bool = True) -> None:
    """Bind submit/newline and navigation keys.

    ``swapped`` (the default) puts newline on Enter and send on
    Ctrl-J / Alt+Enter / Esc-then-Enter — the impatient-finisher-safe
    model: Enter never sends, so a hurried double-tap adds a blank line
    instead of firing the turn, and no buffered keystroke can reach the
    caller's shell after the process exits. ``swapped=False`` is
    chat-box semantics (Enter sends), opt-in via ``--swap-keys`` /
    ``LISA_PROMPT_KEYS``. Either way Ctrl+Up/Down stay inert
    (prompt_toolkit's history navigation replaces the whole multiline
    buffer) and Alt+Up/Down walk prompt history.
    """
    send_seqs, newline_seqs = (
        ((("c-j",), ("escape", "enter")), (("enter",),))
        if swapped
        else ((("enter",),), (("c-j",), ("escape", "enter")))
    )

    def _send(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    for seq in send_seqs:
        # eager on the single Enter key: swallow it before PT's
        # buffer-complete machinery; the escape-enter sequence is a
        # two-key prefix match and must not be eager.
        bindings.add(*seq, eager=len(seq) == 1)(_send)
    for seq in newline_seqs:
        bindings.add(*seq)(_newline)

    @bindings.add("c-up", eager=True)
    def _inert_c_up(event: Any) -> None:
        pass

    @bindings.add("c-down", eager=True)
    def _inert_c_down(event: Any) -> None:
        pass

    @bindings.add("escape", "up")
    def _history_prev(event: Any) -> None:
        event.current_buffer.history_backward()

    @bindings.add("escape", "down")
    def _history_next(event: Any) -> None:
        event.current_buffer.history_forward()


def _add_grapheme_delete_bindings(bindings: Any) -> None:
    """Bind backspace (Ctrl-H) and forward Delete to cluster-aware cuts.

    prompt_toolkit deletes one codepoint, tearing multi-codepoint emoji
    (ZWJ families, flags, skin tones) into broken halves.
    """

    @bindings.add("c-h", eager=True)
    def _grapheme_backspace(event: Any) -> None:
        buffer = event.current_buffer
        pos = buffer.cursor_position
        if pos == 0:
            return
        cut = _delete_last_grapheme(buffer.text[:pos])
        buffer.text = buffer.text[: pos - cut] + buffer.text[pos:]
        buffer.cursor_position = pos - cut

    @bindings.add("delete", eager=True)
    def _grapheme_delete(event: Any) -> None:
        buffer = event.current_buffer
        pos = buffer.cursor_position
        if pos >= len(buffer.text):
            return
        cut = _delete_first_grapheme(buffer.text[pos:])
        buffer.text = buffer.text[:pos] + buffer.text[pos + cut :]
        buffer.cursor_position = pos


def _tui_key_bindings(swapped: bool = True) -> Any:
    """Build the owned key model for the TUI reader.

    ``swapped`` selects the send/newline layout (see
    :func:`_add_navigate_bindings`); this exists so the reader body
    stays under its complexity budget.
    """
    bindings = _PT["KeyBindings"]()
    _add_navigate_bindings(bindings, swapped=swapped)
    _add_grapheme_delete_bindings(bindings)
    return bindings


def read_prompt_message_tui(
    console: Console,
    *,
    history_path: Path,
    swapped_keys: bool | None = None,
    data_dir: str | Path = "",
    profile: str = "",
) -> str | None:
    """Read a single-box message with prompt_toolkit (extra installed).

    Owned key model (prompt_toolkit's defaults are not trusted for a
    multiline chat box); ``swapped_keys`` picks the layout, tri-state:

    - ``True`` (the shipped default; also when the flag is absent and
      ``LISA_PROMPT_KEYS`` is unset/falsy): Enter inserts a newline;
      Ctrl-J, Alt+Enter, or Esc-then-Enter sends. Enter never sends — a
      hurried double-tap adds a blank line instead of firing the turn,
      and no buffered keystroke can reach the caller's shell after the
      process exits.
    - ``False`` (chat-box, opt-in via ``--swap-keys`` or truthy
      ``LISA_PROMPT_KEYS``): Enter sends; Ctrl-J and Alt+Enter add a
      newline.
    - Ctrl+Up / Ctrl+Down are inert: prompt_toolkit binds them to
      previous/next-history, which replaces the whole multiline buffer
      with a history entry — the corruption that forced Ctrl-C.
    - Alt+Up / Alt+Down walk submitted-prompt history (from
      ``history_path``).
    - Backspace and Delete remove whole grapheme clusters so
      multi-codepoint emoji (ZWJ families, flags, skin tones) never tear.
    - Ctrl+Left / Ctrl+Right keep the default word motion; Ctrl-C aborts
      (returns ``None``); bare Ctrl-D or an empty submit also returns
      ``None`` (nothing to send).

    Falls back to :func:`read_prompt_message` when the extra is absent.
    The bottom toolbar respects ``LISA_PROMPT_PS1`` (suppressed when
    empty, custom text when set).
    """
    if _PT is None:
        return read_prompt_message(console)
    if swapped_keys is None:
        # Neither flag nor parameter picked a layout: the env decides.
        # Truthy LISA_PROMPT_KEYS = chat-box (Enter sends); unset or
        # falsy = the shipped swapped layout (Enter = newline).
        swapped_keys = not prompt_keys_env_wants_enter_send()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = _PT["FileHistory"](str(history_path))
        bindings = _tui_key_bindings(swapped=swapped_keys)
        if _PROMPT_PS1_ENV in os.environ:
            hint = prompt_ps1()
        else:
            fragments = resolve_hints(data_dir, profile, layout="swap" if swapped_keys else "chat")
            hint = (
                f"{fragments['send']} · {fragments['newline']} · {fragments['abort']}"
                if any(fragments[k] for k in ("send", "newline", "abort"))
                else None
            )
        hint_html: Any | None = (
            _PT["HTML"](f"<style fg='ansibrightblack'>{hint}</style>") if hint else None
        )
        session: Any = _PT["PromptSession"](
            history=history,
            multiline=True,
            key_bindings=bindings,
            bottom_toolbar=hint_html,
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


__all__ = [
    "prompt_history_path",
    "prompt_keys_env_wants_enter_send",
    "read_prompt_message",
    "read_prompt_message_tui",
]
