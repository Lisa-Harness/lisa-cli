"""Tests for the conversation catalog (:mod:`hosts.cli.conversations`).

Synthetic thread stores pin the catalog's read contracts without a runtime:

- id parsing (``<slug>`` → default; ``<slug>--<conv>`` → conv; foreign
  stems are skipped);
- turn counts span sealed history (index ``turn_hi``) plus the active
  segment's lines;
- summaries come from the last active entry; sealed-only conversations
  fall back to the index timestamp with no query;
- ordering is last-activity desc; corrupt/mid-append files degrade to
  what is readable, never raising;
- resolve/latest helpers pick by id and recency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hosts.cli.conversations import (
    ConversationRecord,
    latest_conversation,
    resolve_conversation,
    scan_conversations,
)

_SLUG = "cli-user"


def _write_thread(
    thread_dir: Path,
    stem: str,
    *,
    entries: list[dict],
    index_lines: list[dict] | None = None,
) -> None:
    """Write one synthetic thread store (active segment + sparse index)."""
    thread_dir.mkdir(parents=True, exist_ok=True)
    current = thread_dir / f"{stem}.current.jsonl"
    with current.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if index_lines:
        with (thread_dir / f"{stem}.index.jsonl").open("w", encoding="utf-8") as handle:
            for record in index_lines:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _entry(
    turn: int, query: str, *, ts: str = "2026-11-23T10:00:00", mode: str = "default"
) -> dict:
    return {
        "timestamp_utc": ts,
        "user_slug": _SLUG,
        "mode_id": mode,
        "query": query,
        "response_text": "ok",
        "turn": turn,
    }


def test_scan_lists_default_and_isolated_conversations(tmp_path: Path) -> None:
    thread_dir = tmp_path / "data" / "threads"
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, "hello there")])
    _write_thread(thread_dir, f"{_SLUG}--alpha", entries=[_entry(0, "alpha task")])
    # A foreign user's store and a stray file are skipped.
    _write_thread(thread_dir, "other-user", entries=[_entry(0, "not mine")])

    records = scan_conversations(thread_dir, user_slug=_SLUG)
    ids = [record.conversation_id for record in records]
    assert "default" in ids
    assert "alpha" in ids
    assert all(record.conversation_id != "other-user" for record in records)
    assert len(records) == 2


def test_scan_counts_sealed_history_plus_active(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(
        thread_dir,
        f"{_SLUG}--beta",
        entries=[_entry(200, "recent work")],
        index_lines=[
            {
                "seg_id": "0001",
                "turn_lo": 0,
                "turn_hi": 199,
                "ts_lo": "2026-11-01T00:00:00",
                "ts_hi": "2026-11-20T00:00:00",
                "bytes_raw": 10,
                "bytes_zst": 5,
                "tail_sha": "",
            }
        ],
    )
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert len(records) == 1
    assert records[0].turn_count == 200  # 199 sealed hi + 1 active line
    assert records[0].summary == "recent work"
    assert records[0].last_mode == "default"


def test_scan_sealed_only_falls_back_to_index_timestamp(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(
        thread_dir,
        _SLUG,
        entries=[],
        index_lines=[
            {
                "seg_id": "0001",
                "turn_lo": 0,
                "turn_hi": 42,
                "ts_lo": "2026-10-01T00:00:00",
                "ts_hi": "2026-10-15T12:30:00",
                "bytes_raw": 10,
                "bytes_zst": 5,
                "tail_sha": "",
            }
        ],
    )
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert len(records) == 1
    record = records[0]
    assert record.turn_count == 42
    assert record.summary == ""
    assert record.last_activity_utc == "2026-10-15T12:30:00"


def test_scan_orders_by_last_activity(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, "older", ts="2026-11-01T00:00:00")])
    _write_thread(
        thread_dir, f"{_SLUG}--newer", entries=[_entry(0, "newer", ts="2026-11-22T00:00:00")]
    )
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert records[0].conversation_id == "newer"
    assert records[1].conversation_id == "default"


def test_scan_tolerates_corrupt_files(tmp_path: Path) -> None:
    """Malformed lines and unreadable files read as what parses; never raise."""
    thread_dir = tmp_path / "threads"
    thread_dir.mkdir(parents=True)
    current = thread_dir / f"{_SLUG}.current.jsonl"
    current.write_text(
        "{not json}\n" + json.dumps(_entry(0, "good line")) + "\n\n", encoding="utf-8"
    )
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert len(records) == 1
    # Line count is the store's own turn-count convention (the active
    # segment's non-empty lines), even when a line is mid-append garbage;
    # the *summary* reads only the last parseable entry.
    assert records[0].turn_count == 2
    assert records[0].summary == "good line"


def test_scan_tolerates_unreadable_and_corrupt_index(tmp_path: Path) -> None:
    """An unparseable index line degrades to zero sealed turns; missing files read empty."""
    thread_dir = tmp_path / "threads"
    thread_dir.mkdir(parents=True)
    # Index with malformed lines only: no usable turn_hi, active still counts.
    (thread_dir / f"{_SLUG}.index.jsonl").write_text(
        "not json\n"
        + json.dumps({"seg_id": "0001", "no_turn_hi": True})
        + "\n"
        + json.dumps({"seg_id": "0002", "turn_hi": "not-a-number", "ts_hi": "x"})
        + "\n",
        encoding="utf-8",
    )
    (thread_dir / f"{_SLUG}.current.jsonl").write_text(
        json.dumps(_entry(0, "active only")) + "\n", encoding="utf-8"
    )
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert len(records) == 1
    assert records[0].turn_count == 1
    assert records[0].summary == "active only"

    # A directory that cannot be listed (missing) reads as no conversations.
    assert scan_conversations(tmp_path / "nope", user_slug=_SLUG) == []


def test_missing_files_read_empty(tmp_path: Path) -> None:
    """Current-segment readers tolerate absent files with zero turns/empty entry."""
    from hosts.cli.conversations import _count_active, _last_entry, _sealed_turn_hi

    missing = tmp_path / "absent.current.jsonl"
    assert _count_active(missing) == 0
    assert _last_entry(missing) == {}
    assert _sealed_turn_hi(tmp_path / "absent.index.jsonl") == (0, "")


def test_scan_empty_dir(tmp_path: Path) -> None:
    records = scan_conversations(tmp_path / "threads", user_slug=_SLUG)
    assert records == []


def test_resolve_conversation_by_id(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(thread_dir, f"{_SLUG}--alpha", entries=[_entry(0, "alpha")])
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, "default store")])

    found = resolve_conversation(thread_dir, user_slug=_SLUG, conversation_id="alpha")
    assert found is not None and found.conversation_id == "alpha"
    default = resolve_conversation(thread_dir, user_slug=_SLUG, conversation_id="default")
    assert default is not None and default.conversation_id == "default"
    missing = resolve_conversation(thread_dir, user_slug=_SLUG, conversation_id="nope")
    assert missing is None


def test_latest_conversation_picks_newest(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, "old", ts="2026-11-01T00:00:00")])
    _write_thread(
        thread_dir, f"{_SLUG}--fresh", entries=[_entry(0, "fresh", ts="2026-11-23T09:00:00")]
    )
    latest = latest_conversation(thread_dir, user_slug=_SLUG)
    assert latest is not None
    assert latest.conversation_id == "fresh"


def test_latest_conversation_empty_store(tmp_path: Path) -> None:
    assert latest_conversation(tmp_path / "threads", user_slug=_SLUG) is None


def test_summary_truncates_long_queries(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    long_query = (
        "this is a very long operator question that should be truncated "
        "down to the catalog's summary budget for table display purposes"
    )
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, long_query)])
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert len(records[0].summary) <= 60
    assert records[0].summary.endswith("…")


def test_multiline_query_summarizes_first_line(tmp_path: Path) -> None:
    thread_dir = tmp_path / "threads"
    _write_thread(thread_dir, _SLUG, entries=[_entry(0, "first line\nsecond line")])
    records = scan_conversations(thread_dir, user_slug=_SLUG)
    assert records[0].summary == "first line"


def test_record_sort_key(tmp_path: Path) -> None:
    early = ConversationRecord("a", "a", 1, "2026-11-01", "x", "default")
    late = ConversationRecord("b", "b", 2, "2026-11-22", "y", "default")
    assert late.sort_key > early.sort_key


def test_sessions_command_registered() -> None:
    """The Typer app exposes ``lisa sessions`` in the help listing."""
    import hosts.cli.main  # noqa: F401  (imports register every command module)
    from hosts.cli.application import app

    names = {getattr(cmd, "name", "") for cmd in app.registered_commands}
    assert "sessions" in names, f"sessions not registered: {sorted(names)[:20]}"


def test_resume_step_raises_session_switch(tmp_path: Path) -> None:
    """``/resume <id>`` unwinds the loop with the target id; unknown ids report."""
    import json

    import pytest
    from runtime.testing import build_test_runtime

    from hosts.cli import interactive as interactive_mod
    from hosts.cli.session import InteractiveSession

    runtime = build_test_runtime(tmp_path, models_yaml="")
    thread_dir = runtime.paths.thread_dir
    thread_dir.mkdir(parents=True, exist_ok=True)
    slug = runtime.config.user_slug
    current = thread_dir / f"{slug}--alpha.current.jsonl"
    current.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-11-23T10:00:00",
                "user_slug": slug,
                "mode_id": "default",
                "query": "alpha work",
                "response_text": "ok",
                "turn": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session = InteractiveSession(runtime=runtime, transport=_NullTransport())  # type: ignore[arg-type]

    with pytest.raises(interactive_mod._SessionSwitch) as excinfo:
        interactive_mod._handle_local_step(session, _console_stub(), "resume", "alpha")
    assert excinfo.value.conversation_id == "alpha"

    # An unknown id reports and continues instead of switching.
    assert (
        interactive_mod._handle_local_step(session, _console_stub(), "resume", "nope") == "continue"
    )

    # Bare ``/resume`` picks the most recent conversation (here: alpha —
    # the current default store is empty/older than the seeded one).
    with pytest.raises(interactive_mod._SessionSwitch) as bare:
        interactive_mod._handle_local_step(session, _console_stub(), "resume", "")
    assert bare.value.conversation_id == "alpha"


def test_new_step_switches_to_default(tmp_path: Path) -> None:
    """``/new`` unwinds with the default conversation id."""
    import pytest
    from runtime.testing import build_test_runtime

    from hosts.cli import interactive as interactive_mod
    from hosts.cli.session import InteractiveSession

    runtime = build_test_runtime(tmp_path, models_yaml="")
    session = InteractiveSession(runtime=runtime, transport=_NullTransport())  # type: ignore[arg-type]
    with pytest.raises(interactive_mod._SessionSwitch) as excinfo:
        interactive_mod._handle_local_step(session, _console_stub(), "new", "")
    assert excinfo.value.conversation_id == "default"


def test_sessions_table_renders_records() -> None:
    """The tty table lists id, turns, activity, mode, and summary."""
    from rich.console import Console

    from hosts.cli.commands.sessions import _sessions_table, _short_timestamp

    console = Console(file=_string_io(), width=120, force_terminal=True)
    records = [
        ConversationRecord(
            "alpha", "cli-user--alpha", 12, "2026-11-23T10:00:00", "fix the parser", "default"
        ),
        ConversationRecord("default", "cli-user", 3, "2026-11-22T09:00:00", "hello", "default"),
    ]
    console.print(_sessions_table(records, user_slug="cli-user", thread_dir=Path("/t")))
    rendered = console.file.getvalue()  # type: ignore[attr-defined]
    assert "alpha" in rendered
    assert "fix the parser" in rendered

    console2 = Console(file=_string_io(), width=120, force_terminal=True)
    console2.print(_sessions_table([], user_slug="cli-user", thread_dir=Path("/t")))
    assert "no conversations" in console2.file.getvalue()  # type: ignore[attr-defined]

    assert _short_timestamp("2026-11-23T10:00:00+00:00") == "2026-11-23 10:00"
    assert _short_timestamp("") == "—"


def _string_io() -> Any:
    import io

    return io.StringIO()


def test_sessions_command_scans_without_a_pool(monkeypatch: Any, tmp_path: Path) -> None:
    """``lisa sessions`` builds the runtime, scans the catalog, renders — no start()."""
    from types import SimpleNamespace

    from rich.console import Console

    from hosts.cli.commands import sessions as sessions_mod

    runtime = SimpleNamespace(
        paths=SimpleNamespace(thread_dir=tmp_path / "threads"),
        config=SimpleNamespace(user_slug="cli-user"),
    )

    def _fake_init(**_kwargs: Any) -> Any:
        return runtime

    monkeypatch.setattr(sessions_mod, "_initialize_runtime_or_exit", _fake_init)

    records = [
        ConversationRecord("alpha", "cli-user--alpha", 5, "2026-11-23T10:00:00", "sum", "default")
    ]
    from hosts.cli import conversations as catalog_mod

    monkeypatch.setattr(catalog_mod, "scan_conversations", lambda d, *, user_slug: records)

    # Non-tty console: plain, tab-separated lines (machine-greppable).
    import io

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    monkeypatch.setattr(sessions_mod, "console", console)
    sessions_mod._sessions_command()
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "alpha" in out
    assert "5" in out
    assert "sum" in out

    # Empty catalog prints the no-conversations marker.
    monkeypatch.setattr(catalog_mod, "scan_conversations", lambda d, *, user_slug: [])
    console2 = Console(file=io.StringIO(), width=100, force_terminal=False)
    monkeypatch.setattr(sessions_mod, "console", console2)
    sessions_mod._sessions_command()
    assert "no conversations" in console2.file.getvalue()  # type: ignore[attr-defined]

    # Tty console: the rendered table path (rich table, not plain lines).
    console3 = Console(file=io.StringIO(), width=120, force_terminal=True, record=True)
    monkeypatch.setattr(catalog_mod, "scan_conversations", lambda d, *, user_slug: records)
    monkeypatch.setattr(sessions_mod, "console", console3)
    sessions_mod._sessions_command()
    assert "alpha" in console3.file.getvalue()  # type: ignore[attr-defined]


def test_short_timestamp_without_t_separator() -> None:
    """A timestamp without the ISO ``T`` renders its first 16 chars."""
    from hosts.cli.commands.sessions import _short_timestamp

    assert _short_timestamp("2026-11-23 10:00") == "2026-11-23 10:00"


def _console_stub() -> Any:
    import io

    from rich.console import Console

    return Console(file=io.StringIO(), width=100, force_terminal=False)


class _NullTransport:
    """Transport stub for the surface-path tests (dispatch is stubbed)."""

    def start_turn(self, message: str) -> None:
        return None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        return None
