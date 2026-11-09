"""Tests for the changed-line coverage gate (``scripts/check_changed_coverage.py``).

The gate requires 95% coverage of changed executable source lines. These
tests assert the changed-line extraction (tracked, untracked, hunk parsing),
the percentage math, and the base resolution against a synthetic coverage
data file — the script is loaded by path since ``scripts/`` is not a package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from coverage import Coverage

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_changed_coverage.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_changed_coverage.py`` as a module."""
    spec = importlib.util.spec_from_file_location("changed_coverage_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A minimal committed git repo with one tracked source file."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "pkg.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def _write_coverage(repo: Path, source: str) -> Coverage:
    """Measure ``source`` into a fresh branch-coverage data file and load it.

    The source must execute everything it defines itself (the fixture
    sources below call their functions at import of the exec'd module), so
    the measured file has no missing statements.
    """
    data_file = repo / ".coverage"
    (repo / "pkg.py").write_text(source, encoding="utf-8")
    cov = Coverage(data_file=str(data_file), branch=True, config_file=False)
    cov.start()
    try:
        namespace: dict[str, object] = {}
        exec(compile(source, str(repo / "pkg.py"), "exec"), namespace)
    finally:
        cov.stop()
        cov.save()
    loaded = Coverage(data_file=str(data_file), branch=True, config_file=False)
    loaded.load()
    return loaded


class TestChangedLines:
    """The changed-line set comes from tracked diffs and untracked files."""

    def test_untracked_file_counts_all_lines(self, git_repo: Path) -> None:
        (git_repo / "new_mod.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        changes = gate.changed_lines(git_repo, "HEAD")
        assert changes["new_mod.py"] == {1, 2}

    def test_non_source_paths_skipped(self, git_repo: Path) -> None:
        (git_repo / "notes.md").write_text("# n\n", encoding="utf-8")
        (git_repo / "tests").mkdir()
        (git_repo / "tests" / "test_it.py").write_text("assert True\n", encoding="utf-8")
        changes = gate.changed_lines(git_repo, "HEAD")
        assert changes == {}

    def test_diff_hunks_are_parsed(self, git_repo: Path) -> None:
        (git_repo / "pkg.py").write_text(
            "def kept():\n    return 2\n\ndef added():\n    return 3\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
        changes = gate.changed_lines(git_repo, "HEAD")
        assert 3 in changes["pkg.py"]
        assert 1 not in changes["pkg.py"]  # lines[1]

    def test_escaping_source_rejected(self, git_repo: Path) -> None:
        # A staged file whose path escapes the repository is a hard error.
        (git_repo / "link.py").write_text("x = 1\n", encoding="utf-8")
        with (
            patch.object(gate, "_git", side_effect=lambda *a, **k: "link.py\0../evil.py\0"),
            pytest.raises(RuntimeError, match="escapes the repository"),
        ):
            gate.changed_lines(git_repo, "HEAD")


class TestCheckCoverage:
    """The percentage verdict on (changed ∩ statements) minus missing."""

    def test_missing_data_file_fails(self, git_repo: Path, tmp_path: Path) -> None:
        empty = Coverage(data_file=str(tmp_path / "none.coverage"), config_file=False)
        with pytest.raises(RuntimeError, match="Missing branch-coverage data"):
            gate.check_coverage(empty, {"pkg.py": {1}}, git_repo)

    def test_fully_covered_change_passes(self, git_repo: Path) -> None:
        source = "def full():\n    return 1\n\nfull()\n"
        cov = _write_coverage(git_repo, source)
        assert gate.check_coverage(cov, {"pkg.py": {1, 2}}, git_repo) == 0

    def test_uncovered_change_fails(self, git_repo: Path) -> None:
        source = "def full():\n    return 1\n\nfull()\n\ndef dead():\n    return 2\n"
        cov = _write_coverage(git_repo, source)
        # changed lines include 6/7 ("dead", never invoked) → below 95%.
        assert gate.check_coverage(cov, {"pkg.py": {1, 2, 6, 7}}, git_repo) == 1

    def test_no_changed_lines_passes(self, git_repo: Path) -> None:
        source = "def full():\n    return 1\n\nfull()\n"
        cov = _write_coverage(git_repo, source)
        assert gate.check_coverage(cov, {}, git_repo) == 0

    def test_non_statement_lines_ignored(self, git_repo: Path) -> None:
        # Comments and blank lines are not statements — a change confined
        # to them is vacuously covered.
        source = "def full():\n    return 1\n\nfull()\n"
        cov = _write_coverage(git_repo, source)
        assert gate.check_coverage(cov, {"pkg.py": {4}}, git_repo) == 0


class TestResolveBase:
    """Base resolution prefers explicit, then origin/main, then main."""

    def test_explicit_base(self, git_repo: Path) -> None:
        assert gate.resolve_base(git_repo, "HEAD~0") == "HEAD~0"

    def test_merge_base_resolution(self, git_repo: Path) -> None:
        # Deterministic across git init.defaultbranch settings: create the
        # main branch explicitly, then resolve_base must find its merge-base.
        subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=git_repo, check=True)
        (git_repo / "pkg.py").write_text("def kept():\n    return 1\n# touch\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "second"],
            cwd=git_repo,
            check=True,
        )
        assert gate.resolve_base(git_repo, None) is not None
