"""Shell command substitution for editor-collected prompt text.

When the message comes from the no-args interactive editor (not argv),
the shell never ran, so ``$(cmd)`` would reach the model literally. This
module restores that one piece of shell immersion for editor text:
``$(cmd)`` (nesting honored) is replaced by the command's captured
standard output; ``\\$(`` passes through as the literal text ``$(``.

Scope discipline:

- Only editor-collected text passes through here. Shell-arg text was
  already expanded by the shell; expanding again would double-substitute.
- Backticks are NOT substitution delimiters — they stay literal. Nested
  backticks need escaping and ``$()`` covers the need.
- Double-quoted spans still expand (a user writing "run $(date)" inside
  a sentence expects it); single-quoted spans do NOT expand, matching
  POSIX single-quote semantics: 'no $(expansion) here'.

Each substitution runs through ``$SHELL -c`` (fallback ``/bin/sh``)
with a per-command timeout. A failed or empty substitute is loud:
stderr notice + exit code 2 at the call site — never silently literal,
never silently empty.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable

#: Substitution timeout per command (seconds) — prompts should not hang.
EXPAND_TIMEOUT_S = 10.0

#: Runner seam for tests (signature: command -> (stdout, ok)).
_runner: Callable[[str], tuple[str, bool]] | None = None


def _default_runner(command: str) -> tuple[str, bool]:
    """Run one command via the user's shell; return (stripped stdout, ok)."""
    shell = os.environ.get("SHELL") or "/bin/sh"
    try:
        # check=False by design: failures are reported through the
        # returncode, not raised — the expander's contract is (stdout, ok).
        proc = subprocess.run(
            [shell, "-c", command],
            capture_output=True,
            text=True,
            timeout=EXPAND_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", False
    except OSError:
        return "", False
    if proc.returncode != 0:
        return "", False
    return proc.stdout.rstrip("\n"), True


def _find_span_end(text: str, start: int) -> int:
    """End index (past the ``)``) of the ``$(...`` span at ``start``, or -1.

    ``start`` points at ``$``; nesting is honored. An unterminated span
    returns -1 (the caller leaves it literal — prose, not a typo).
    """
    n = len(text)
    j = start + 2
    depth = 1
    while j < n and depth:
        cj = text[j]
        if cj == "(":
            depth += 1
        elif cj == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return j + 1 if depth == 0 else -1


class _ScanState:
    """Mutable scanner state threaded through :func:`_scan_step`."""

    __slots__ = ("in_double", "in_single")

    def __init__(self) -> None:
        self.in_single = False
        self.in_double = False


def expand_shell_substitutions(text: str) -> tuple[str, list[str]]:
    """Expand ``$(cmd)`` spans in editor text; report failed command strings.

    Returns ``(expanded_text, failed_commands)``. The caller turns a
    non-empty ``failed_commands`` into a loud usage error (exit 2), so a
    typo'd command can never mutate the message silently.

    Scanning rules:
    - ``$(...)`` spans nest; depth tracking handles inner ``$()``.
    - ``\\$(`` emits a literal ``$(`` and is not scanned further.
    - Single-quoted spans (``'...'``) are skipped: no expansion inside.
    - Double-quoted spans still expand, per POSIX double-quote semantics.
    - An unterminated ``$(`` is left literal (likely prose, not a typo).
    """
    runner = _runner or _default_runner
    out: list[str] = []
    failures: list[str] = []
    state = _ScanState()
    i = 0
    n = len(text)
    while i < n:
        i = _scan_step(text, i, out, failures, runner=runner, state=state)
    return "".join(out), failures


def _scan_step(
    text: str,
    i: int,
    out: list[str],
    failures: list[str],
    *,
    runner: Callable[[str], tuple[str, bool]],
    state: _ScanState,
) -> int:
    """Consume one scanner decision at ``i``; return the next index (or len).

    A single step of the expand loop, split out so the loop stays
    readable and the McCabe budget owns the grammar, not the plumbing.
    """
    ch = text[i]
    if state.in_single:
        if ch == "'":
            state.in_single = False
    elif ch == "\\" and text[i + 1 : i + 2] == "$" and text[i + 2 : i + 3] == "(":
        out.append("$(")
        return i + 3
    elif ch == "'" and not state.in_double:
        state.in_single = True
    elif ch == '"':
        state.in_double = not state.in_double
    elif ch == "$" and text[i + 1 : i + 2] == "(":
        end = _find_span_end(text, i)
        if end < 0:  # unterminated: leave the rest literal
            out.append(text[i:])
            return len(text)
        command = text[i + 2 : end - 1]
        captured, ok = runner(command)
        if ok:
            out.append(captured)
        else:
            failures.append(command)
            out.append(text[i:end])  # keep the literal span for debuggability
        return end
    out.append(ch)
    return i + 1


__all__ = ["EXPAND_TIMEOUT_S", "expand_shell_substitutions"]
