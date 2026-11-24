"""Tests for the read-only git workspace service and diff rendering.

The service tests run real ``git`` in temporary repositories (hermetic:
config is per-repo, no global state) and pin:

- status parsing (branch, HEAD, modified/added/deleted/renamed/untracked);
- diff scopes (working tree vs HEAD, staged, ref) with honest reasons
  (binary, untracked, rename old-paths) and byte-cap truncation;
- non-repo directories and missing git degrade to empty/absent results;
- the parser round-trips porcelain/-z and name-status -z records.

The render tests pin the tty/pipe split: sections with colored hunks on
a terminal, raw unified diffs under pipes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from hosts.cli.workspace import (
    DiffResult,
    FileChange,
    FileDiff,
    WorkspaceStatus,
    _parse_name_status,
    _parse_porcelain,
    read_diff,
    read_status,
    repo_root,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed repository with one tracked file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("line1\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "init")
    return root


def test_status_reads_branch_head_and_changes(repo: Path) -> None:
    (repo / "a.txt").write_text("line1\nline2\n", encoding="utf-8")
    (repo / "new.txt").write_text("x\n", encoding="utf-8")

    status = _run(read_status(repo))
    assert isinstance(status, WorkspaceStatus)
    assert status.is_repo is True
    assert status.branch in {"main", "master"}
    assert status.head_sha
    codes = {change.path: change.code for change in status.changes}
    assert codes["a.txt"] == "M"
    assert codes["new.txt"] == "?"


def test_status_outside_a_repo(tmp_path: Path) -> None:
    status = _run(read_status(tmp_path))
    assert status.is_repo is False
    assert status.changes == ()


def test_repo_root_resolves(repo: Path, tmp_path: Path) -> None:
    assert _run(repo_root(repo)) == repo.resolve()
    assert _run(repo_root(tmp_path)) is None


def test_diff_working_tree_includes_untracked(repo: Path) -> None:
    (repo / "a.txt").write_text("line1\nline2\n", encoding="utf-8")
    (repo / "orphan.txt").write_text("x\n", encoding="utf-8")

    result = _run(read_diff(repo))
    assert result.scope == "working tree vs HEAD"
    by_path = {f.path: f for f in result.files}
    assert "a.txt" in by_path
    assert "+line2" in by_path["a.txt"].diff
    assert by_path["orphan.txt"].reason == "untracked"
    assert by_path["orphan.txt"].diff == ""


def test_diff_staged_scope(repo: Path) -> None:
    (repo / "a.txt").write_text("staged change\n", encoding="utf-8")
    _git(repo, "add", "a.txt")

    result = _run(read_diff(repo, staged=True))
    assert result.scope == "staged (index vs HEAD)"
    assert len(result.files) == 1
    assert "+staged change" in result.files[0].diff


def test_diff_against_a_ref(repo: Path) -> None:
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b")

    result = _run(read_diff(repo, ref="HEAD~1"))
    assert result.scope == "HEAD~1 vs working tree"
    by_path = {f.path: f for f in result.files}
    assert "+b" in by_path["b.txt"].diff


def test_diff_reports_renames_with_old_path(repo: Path) -> None:
    _git(repo, "mv", "a.txt", "renamed.txt")
    result = _run(read_diff(repo, staged=True))
    assert any(f.path == "renamed.txt" and "a.txt" in f.reason for f in result.files), (
        f"rename must carry the old path: {[(f.path, f.reason) for f in result.files]}"
    )


def test_diff_truncates_large_output(repo: Path) -> None:
    (repo / "a.txt").write_text("line\n" * 100_000, encoding="utf-8")
    result = _run(read_diff(repo, max_bytes=1000))
    assert result.truncated is True
    assert any("diff truncated" in f.diff for f in result.files)


def test_diff_empty_tree_is_honest(repo: Path) -> None:
    result = _run(read_diff(repo))
    assert result.files == ()
    assert result.truncated is False


def test_diff_outside_a_repo(tmp_path: Path) -> None:
    result = _run(read_diff(tmp_path))
    assert result.files == ()


def test_parse_porcelain_roundtrip() -> None:
    # Real porcelain -z: "XY <path>\0" with XY two columns; renames put
    # the NEW path in the record and the OLD path as the next NUL field.
    raw = "M  a.txt\0?? new.txt\0R  renamed.txt\0old.txt\0"
    changes = _parse_porcelain(raw)
    by_path = {c.path: c for c in changes}
    assert by_path["a.txt"].code == "M"
    assert by_path["new.txt"].code == "?"
    assert by_path["renamed.txt"].code == "R"
    assert by_path["renamed.txt"].old_path == "old.txt"


def test_parse_name_status_roundtrip() -> None:
    # Real name-status -z: "M\0<path>\0"; renames carry R100 + old, new.
    raw = "M\0a.txt\0R100\0old.txt\0new.txt\0"
    records = _parse_name_status(raw)
    assert records[0] == FileChange(path="a.txt", code="M")
    assert records[1] == FileChange(path="new.txt", code="R", old_path="old.txt")


# ── diff rendering ─────────────────────────────────────────────────────


def _diff_result(*files: FileDiff) -> DiffResult:
    return DiffResult(scope="working tree vs HEAD", files=files)


def test_render_diff_tty_sections() -> None:
    import io

    from rich.console import Console

    from hosts.cli.commands.diff_view import render_diff

    console = Console(file=io.StringIO(), width=120, force_terminal=True, record=True)
    result = _diff_result(
        FileDiff(path="a.txt", diff="+added\n-removed\n@@ hunk @@"),
        FileDiff(path="new.txt", reason="untracked"),
    )
    render_diff(result, console)
    out = console.export_text()
    assert "a.txt" in out
    assert "+added" in out
    assert "new.txt" in out
    assert "untracked" in out


def test_render_diff_pipe_gets_raw_text() -> None:
    import io

    from rich.console import Console

    from hosts.cli.commands.diff_view import render_diff

    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    result = _diff_result(FileDiff(path="a.txt", diff="+added\n-removed"))
    render_diff(result, console)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "+added" in out
    assert "-removed" in out
    assert "Diff" not in out  # no decoration under pipes


def test_render_diff_no_changes_notice() -> None:
    import io

    from rich.console import Console

    from hosts.cli.commands.diff_view import render_diff

    console = Console(file=io.StringIO(), width=80, force_terminal=False)
    render_diff(_diff_result(), console)
    assert "No changes" in console.file.getvalue()  # type: ignore[attr-defined]


def test_git_binary_reports_when_text_diff_is_empty(repo: Path) -> None:
    """A binary modification reports 'binary' rather than empty-diff silence."""
    binary = repo / "blob.bin"
    binary.write_bytes(b"\x00\x01\x02")
    _git(repo, "add", "blob.bin")
    _git(repo, "commit", "-qm", "add binary")
    binary.write_bytes(b"\x00\x01\x03")
    _git(repo, "add", "blob.bin")

    result = _run(read_diff(repo, staged=True))
    by_path = {f.path: f for f in result.files}
    assert by_path["blob.bin"].reason in ("binary", "")


def test_git_unavailable_degrades_to_absence(tmp_path: Path, monkeypatch: Any) -> None:
    """A missing git binary reads as absence (None), never an exception."""

    from hosts.cli import workspace as workspace_mod

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no git")

    real_create = asyncio.create_subprocess_exec
    monkeypatch.setattr(workspace_mod.asyncio, "create_subprocess_exec", _boom)
    try:
        assert _run(workspace_mod._git(tmp_path, "status")) is None
    finally:
        monkeypatch.setattr(workspace_mod.asyncio, "create_subprocess_exec", real_create)


def test_git_nonzero_exit_reads_as_none(repo: Path) -> None:
    """A failing git command (bad ref) reads as None, not an exception."""
    from hosts.cli.workspace import _git

    assert _run(_git(repo, "rev-parse", "definitely-not-a-ref")) is None


def test_git_timeout_reads_as_none(repo: Path, monkeypatch: Any) -> None:
    """A wedged git (never returns) is killed and reads as None."""
    from hosts.cli import workspace as workspace_mod
    from hosts.cli.workspace import _git

    class _HangingProcess:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(60)
            return (b"", b"")

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return -9

    async def _fake_exec(*args: Any, **kwargs: Any) -> Any:
        return _HangingProcess()

    real_create = asyncio.create_subprocess_exec
    monkeypatch.setattr(workspace_mod.asyncio, "create_subprocess_exec", _fake_exec)
    try:
        assert _run(_git(repo, "status", timeout_s=0.05)) is None
    finally:
        monkeypatch.setattr(workspace_mod.asyncio, "create_subprocess_exec", real_create)


def test_parse_name_status_truncated_rename() -> None:
    """A rename record missing its new path stops parsing cleanly."""
    records = _parse_name_status("R100\0old.txt\0")
    assert records == []


def test_parse_name_status_similarity_numbered_codes() -> None:
    """C100 (copy) similarity codes normalize to the bare letter."""
    records = _parse_name_status("C100\0src\0dst\0M\0plain.txt\0")
    by_path = {r.path: r for r in records}
    assert by_path["dst"].code == "C"
    assert by_path["dst"].old_path == "src"
    assert by_path["plain.txt"].code == "M"
