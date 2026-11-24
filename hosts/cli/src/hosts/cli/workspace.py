"""Read-only git workspace service for the CLI host.

Every query runs ``git`` via :func:`asyncio.create_subprocess_exec` — no
shell, no string interpolation — inside the host's event loop, with a
bounded timeout, and treats every failure as honest data (``None`` or an
empty result), never an exception across the surface. The service is
strictly read-only: ``status``, ``diff`` families only; it never mutates
the repository.

Scope (Milestone 5): the working tree the operator owns. Lisa's own edits
are not attributed here — pre-existing user edits are shown as they are,
and the diff command documents that. Worktrees/agents/checkpointing are
later milestones.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Bounded git timeout — a wedged git must never hang the session.
_GIT_TIMEOUT_S = 10.0

#: Maximum unified-diff bytes fetched per file; larger diffs are truncated
#: with an honest marker (the operator can open the file for the rest).
_MAX_DIFF_BYTES = 200_000


@dataclass(frozen=True, slots=True)
class FileChange:
    """One changed path from ``git status --porcelain``."""

    path: str
    #: ``M`` modified, ``A`` added, ``D`` deleted, ``R`` renamed, ``?`` untracked.
    code: str
    #: New path for renames (``R``); empty otherwise.
    old_path: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceStatus:
    """A snapshot of the working tree."""

    #: True when the directory is inside a git repository.
    is_repo: bool
    #: Current branch name (``HEAD`` in detached state, ``""`` when unknown).
    branch: str
    changes: tuple[FileChange, ...] = ()
    #: Short ``HEAD`` sha, ``""`` when unknown.
    head_sha: str = ""


async def _git(root: Path, *args: str, timeout_s: float = _GIT_TIMEOUT_S) -> str | None:
    """Run one read-only git command; ``None`` on any failure or timeout.

    Failures are logged at debug level — an absent repo or a timeout is
    data (``None``), not an error the caller must handle specially.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return None  # git not installed — honest absence, not a crash
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        logger.debug("workspace: git %s timed out", args[0])
        return None
    if process.returncode != 0:
        return None
    return stdout.decode("utf-8", errors="replace")


async def repo_root(start: Path) -> Path | None:
    """The enclosing repository root for ``start``, or ``None`` outside a repo."""
    out = await _git(start, "rev-parse", "--show-toplevel")
    if not out:
        return None
    return Path(out.strip())


async def read_status(root: Path) -> WorkspaceStatus:
    """Snapshot the working tree: branch, HEAD, and changed paths."""
    is_repo = await repo_root(root) is not None
    if not is_repo:
        return WorkspaceStatus(is_repo=False, branch="", changes=())
    branch_out = await _git(root, "branch", "--show-current")
    branch = (branch_out or "").strip() or "HEAD"  # detached HEAD renders honestly
    head_out = await _git(root, "rev-parse", "--short", "HEAD")
    head_sha = (head_out or "").strip()
    status_out = await _git(root, "status", "--porcelain", "-z")
    return WorkspaceStatus(
        is_repo=True,
        branch=branch,
        head_sha=head_sha,
        changes=tuple(_parse_porcelain(status_out or "")),
    )


def _parse_porcelain(raw: str) -> list[FileChange]:
    """Parse ``status --porcelain -z`` output (NUL-separated records)."""
    changes: list[FileChange] = []
    fields = raw.split("\0")
    i = 0
    while i < len(fields):
        field = fields[i]
        if not field:
            i += 1
            continue
        code = field[:2].strip()
        path = field[3:]
        old_path = ""
        # Rename records carry the new path first, then the old path.
        if code.startswith("R") and i + 1 < len(fields):
            old_path = fields[i + 1]
            i += 1
        if code == "??":
            code = "?"
        changes.append(FileChange(path=path, code=code[:1], old_path=old_path))
        i += 1
    return changes


@dataclass(frozen=True, slots=True)
class FileDiff:
    """One file's unified diff (or its honest absence)."""

    path: str
    diff: str = ""
    #: Why no diff text: "binary", "untracked", "submodule", or "" (present).
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DiffResult:
    """A diff query's outcome over the requested range."""

    #: The compared range label (e.g. "working tree", "HEAD", "refs").
    scope: str
    files: tuple[FileDiff, ...] = ()
    truncated: bool = False


async def read_diff(
    root: Path,
    *,
    staged: bool = False,
    ref: str = "",
    max_bytes: int = _MAX_DIFF_BYTES,
) -> DiffResult:
    """Unified diffs for the requested comparison.

    ``staged`` compares the index against HEAD; ``ref`` compares ``ref``
    against the working tree (mutually exclusive with ``staged``); the
    default compares the working tree against HEAD. Untracked files are
    listed with their honest reason (they have no diff hunk), binaries
    report ``binary``, and the total output is bounded with a marker.
    """
    if staged:
        git_args: tuple[str, ...] = ("--cached", "HEAD")
        label = "staged (index vs HEAD)"
    elif ref:
        git_args = (ref,)
        label = f"{ref} vs working tree"
    else:
        git_args = ("HEAD",)
        label = "working tree vs HEAD"

    name_out = await _git(root, "diff", "--name-status", "-z", *git_args)
    if name_out is None:
        return DiffResult(scope=label, files=())
    files: list[FileDiff] = []
    truncated = False
    for record in _parse_name_status(name_out):
        file_diff = await _diff_one_file(root, git_args, record)
        if record.code in {"R", "C"}:
            # Rename/copy: the old path rides the reason for the header.
            file_diff = FileDiff(
                path=record.path,
                diff=file_diff.diff,
                reason=f"{record.code} {record.old_path}",
            )
        if len(file_diff.diff) > max_bytes:
            file_diff = FileDiff(
                path=file_diff.path,
                diff=file_diff.diff[:max_bytes] + "\n… diff truncated",
                reason=file_diff.reason,
            )
            truncated = True
        files.append(file_diff)
    # Untracked files never appear in a git diff (they have no tracked
    # counterpart); list them from the status snapshot with their honest
    # reason so the operator sees the whole working tree.
    if not staged and not ref:
        status = await read_status(root)
        tracked = {record.path for record in files}
        files.extend(
            FileDiff(path=change.path, diff="", reason="untracked")
            for change in status.changes
            if change.code == "?" and change.path not in tracked
        )
    return DiffResult(scope=label, files=tuple(files), truncated=truncated)


async def _diff_one_file(root: Path, git_args: tuple[str, ...], record: FileChange) -> FileDiff:
    """Read one file's unified diff; empty output reads as binary."""
    diff_out = await _git(root, "diff", "--no-color", *git_args, "--", record.path, timeout_s=5.0)
    text = (diff_out or "").strip()
    if not text:
        return FileDiff(path=record.path, diff="", reason="binary")
    return FileDiff(path=record.path, diff=text)


def _parse_name_status(raw: str) -> list[FileChange]:
    """Parse ``diff --name-status -z`` output into change records.

    The ``-z`` format separates the status and path with NUL; the status
    letter carries a similarity number for renames/copies (``R100``).
    Rename/copy records list the OLD path first, then the NEW path
    (``R100\\0<old>\\0<new>\\0``).
    """
    fields = [field for field in raw.split("\0") if field]
    records: list[FileChange] = []
    i = 0
    while i < len(fields):
        code_field = fields[i]
        code = code_field[0] if code_field else ""
        if i + 1 >= len(fields):
            break
        path = fields[i + 1]
        old_path = ""
        i += 2
        if code in {"R", "C"}:
            if i >= len(fields):
                # A truncated rename/copy record cannot be attributed —
                # skip it rather than report a wrong path.
                continue
            # name-status lists old-then-new: report the NEW path as the
            # record's path (the diffable one) and keep the OLD alongside.
            old_path = path
            path = fields[i]
            i += 1
        records.append(FileChange(path=path, code=code, old_path=old_path))
    return records
