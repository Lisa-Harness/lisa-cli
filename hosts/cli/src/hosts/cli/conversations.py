"""Conversation catalog over the durable thread stores.

A "conversation" is one thread store: ``<user_slug>.current.jsonl`` (the
default) or ``<user_slug>--<conversation_id>.current.jsonl`` (an isolated
conversation, via ``--conversation-id`` / ``LISA_CONVERSATION_ID``). The
catalog lists them for ``lisa sessions``, picks the latest in a workspace
for ``--continue``, and resolves ids for ``--resume <id>``.

This module is pure: it reads thread files and their sparse seal indexes,
builds records, and never imports Rich, Typer, or the runtime — the
command layer owns presentation and the interactive surface owns
rebuilding a session around a selected conversation.

Turn counts include sealed history: the sparse index's ``turn_hi`` of the
newest segment plus the active segment's line count. Summaries come from
the active segment's last entry (its query and timestamp); a conversation
whose history is fully sealed (empty active segment) falls back to the
index's newest ``ts_hi`` with no query. Corrupt or partially written
files degrade to what is readable — the catalog must never crash on a
store mid-append, so malformed lines are skipped and unreadable files
read as empty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The default conversation's file suffix (no ``--<id>`` isolation).
_DEFAULT_CONVERSATION_ID = "default"

#: Catalog title length for the summary column.
_SUMMARY_MAX = 60


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """One resumable conversation from the thread stores."""

    #: The conversation id to pass to ``--resume`` / ``--conversation-id``;
    #: ``default`` for the un-isolated store.
    conversation_id: str
    #: The thread base name as it appears on disk (``<slug>`` or
    #: ``<slug>--<conv>``), for the Sessions table's File column.
    base_name: str
    #: Total turns including sealed segments.
    turn_count: int
    #: Last turn's timestamp (ISO-8601 UTC) or ``""`` when unknown.
    last_activity_utc: str
    #: Last operator query (single line, truncated) or ``""``.
    summary: str
    #: Mode of the last entry, or ``""``.
    last_mode: str

    @property
    def sort_key(self) -> tuple[str, int]:
        """Activity sort: by timestamp desc, then turn count desc."""
        return (self.last_activity_utc or "", self.turn_count)


def _parse_conversation_id(base_stem: str, user_slug: str) -> str | None:
    """Extract the conversation id from one ``*.current.jsonl`` stem.

    ``<slug>`` → ``default``; ``<slug>--<conv>`` → ``<conv>``. Any stem
    that is not one of those two shapes (another user's store the slug
    does not prefix, or a foreign file) yields ``None``.
    """
    if base_stem == user_slug:
        return _DEFAULT_CONVERSATION_ID
    prefix = f"{user_slug}--"
    if base_stem.startswith(prefix) and len(base_stem) > len(prefix):
        return base_stem[len(prefix) :]
    return None


def _last_entry(current_path: Path) -> dict[str, Any]:
    """The last parseable entry of the active segment, or ``{}``."""
    last: dict[str, Any] = {}
    try:
        with current_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    last = decoded
    except OSError:
        return {}
    return last


def _sealed_turn_hi(index_path: Path) -> tuple[int, str]:
    """The newest sealed segment's turn_hi and ts_hi, or ``(0, "")``."""
    hi_turn = 0
    hi_ts = ""
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, ""
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        try:
            turn_hi = int(decoded["turn_hi"])
            ts_hi = str(decoded.get("ts_hi") or "")
        except (KeyError, TypeError, ValueError):
            continue
        if turn_hi > hi_turn:
            hi_turn = turn_hi
            hi_ts = ts_hi
    return hi_turn, hi_ts


def _count_active(current_path: Path) -> int:
    """Count non-empty lines in the active segment (best-effort)."""
    count = 0
    try:
        with current_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _summarize(query: str) -> str:
    """One-line, truncated summary of an operator query."""
    line = query.strip().splitlines()[0] if query.strip() else ""
    if len(line) > _SUMMARY_MAX:
        return line[: _SUMMARY_MAX - 1] + "…"
    return line


def scan_conversations(
    thread_dir: Path,
    *,
    user_slug: str,
) -> list[ConversationRecord]:
    """List every conversation for ``user_slug`` under ``thread_dir``.

    Sorted by last activity (newest first). Sealed-only conversations
    (an index but an empty/absent active segment) still appear, with
    their newest sealed timestamp. Files that are mid-append or corrupt
    contribute what is parseable; the scan never raises on store state.
    """
    records: list[ConversationRecord] = []
    try:
        candidates = list(thread_dir.glob("*.current.jsonl"))
    except OSError:
        return records
    for current_path in candidates:
        stem = current_path.name[: -len(".current.jsonl")]
        conversation_id = _parse_conversation_id(stem, user_slug)
        if conversation_id is None:
            continue
        base = thread_dir / stem
        index_path = Path(f"{base}.index.jsonl")
        sealed_hi, sealed_ts = _sealed_turn_hi(index_path)
        active_count = _count_active(current_path)
        last = _last_entry(current_path)
        timestamp = str(last.get("timestamp_utc") or "") or sealed_ts
        records.append(
            ConversationRecord(
                conversation_id=conversation_id,
                base_name=stem,
                turn_count=sealed_hi + active_count,
                last_activity_utc=timestamp,
                summary=_summarize(str(last.get("query") or "")),
                last_mode=str(last.get("mode_id") or ""),
            )
        )
    records.sort(key=lambda r: r.sort_key, reverse=True)
    return records


def resolve_conversation(
    thread_dir: Path,
    *,
    user_slug: str,
    conversation_id: str,
) -> ConversationRecord | None:
    """Resolve one conversation id to its record, or ``None`` when absent."""
    wanted = (
        _DEFAULT_CONVERSATION_ID if conversation_id == _DEFAULT_CONVERSATION_ID else conversation_id
    )
    for record in scan_conversations(thread_dir, user_slug=user_slug):
        if record.conversation_id == wanted:
            return record
    return None


def latest_conversation(
    thread_dir: Path,
    *,
    user_slug: str,
) -> ConversationRecord | None:
    """The most recently active conversation, or ``None`` for an empty store."""
    records = scan_conversations(thread_dir, user_slug=user_slug)
    return records[0] if records else None


__all__ = [
    "ConversationRecord",
    "latest_conversation",
    "resolve_conversation",
    "scan_conversations",
]
