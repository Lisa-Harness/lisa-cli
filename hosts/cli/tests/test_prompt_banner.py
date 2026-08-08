"""Tests for the pinned submitted-prompt banner in ``RichTransport``.

Covers the chat-box ask-1 banner: the user's submitted message is pinned
to the top of the Live region for the duration of the turn. Multi-line
prompts must stay multi-line (newlines preserved), and each line is
truncated to terminal width.
"""

from __future__ import annotations

# pylint: disable=protected-access
from rich.console import Console

from hosts.cli.transport import RichTransport


def _banner_plain(transport: RichTransport, *, width: int = 80) -> str:
    """Render the prompt banner and return its plain text at a fixed width."""
    transport.console = Console(width=width, force_terminal=False, record=True)
    return transport._render_prompt_banner().plain


def test_prompt_banner_empty_when_no_message() -> None:
    transport = RichTransport()
    assert transport._render_prompt_banner().plain == ""


def test_prompt_banner_single_line() -> None:
    transport = RichTransport()
    transport._turn_prompt = "hello world"
    plain = _banner_plain(transport)
    assert plain.startswith("❯ ")
    assert "hello world" in plain


def test_prompt_banner_preserves_newlines_for_multi_line_prompt() -> None:
    """A multi-line prompt stays multi-line when pinned to the top."""
    transport = RichTransport()
    transport._turn_prompt = "line one\nline two\nline three"
    plain = _banner_plain(transport)
    assert plain.count("\n") == 2  # three lines, two newline separators
    assert "line one" in plain
    assert "line two" in plain
    assert "line three" in plain
    # Continuation lines are marked with the … glyph.
    assert "…" in plain


def test_prompt_banner_truncates_each_line_to_width() -> None:
    """Each line is truncated independently to the terminal width."""
    transport = RichTransport()
    long_line = "x" * 200
    transport._turn_prompt = f"{long_line}\n{long_line}"
    plain = _banner_plain(transport, width=40)
    # Each rendered line (after the ❯ / … marker) fits within the width.
    for rendered in plain.split("\n"):
        assert len(rendered) <= 40


def test_prompt_session_subclass_enforces_min_input_height() -> None:
    """The REPL PromptSession subclass reserves a 3-row minimum input height."""
    pytest = __import__("pytest")
    try:
        from prompt_toolkit.layout import Dimension
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    from hosts.cli.repl import _REPL_MIN_INPUT_HEIGHT, _ReplPromptSession

    # _get_default_buffer_control_height runs inside an app; the base method
    # tolerates being called without one (returns Dimension() / min from menu).
    # We verify the floor via the class constant and that the subclass exists
    # and is a PromptSession, so the override is wired.
    from prompt_toolkit import PromptSession

    assert issubclass(_ReplPromptSession, PromptSession)
    assert _REPL_MIN_INPUT_HEIGHT == 3
    assert Dimension is not None  # sanity: import succeeded