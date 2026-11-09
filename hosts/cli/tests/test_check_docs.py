"""Tests for the documentation quality gate (``scripts/check_docs.py``).

The gate must not fire on the corpus as it stands, and it must fire on the
defects it claims to catch. Both halves are asserted here so a change to
the checker cannot silently widen or narrow the corpus.

The script is loaded by path rather than imported as a package: it lives
in ``scripts/``, which is not an importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_docs.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_docs.py`` as a module."""
    spec = importlib.util.spec_from_file_location("check_docs_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_docs = _load_module()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return a minimal repository root with one documentation file."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    return tmp_path


def _rules(repo: Path) -> set[str]:
    """Return the rules reported for the fixture's documentation file."""
    findings = check_docs._check_file("docs/guide.md", repo, {})
    return {finding.rule for finding in findings}


class TestCorpusSelection:
    """The corpus is the maintained documentation, not runtime content."""

    def test_guides_and_root_docs_are_included(self) -> None:
        assert check_docs._is_documentation("docs/README.md")
        assert check_docs._is_documentation("README.md")
        assert check_docs._is_documentation("CHANGELOG.md")
        assert check_docs._is_documentation("CODE_OF_CONDUCT.md")
        assert check_docs._is_documentation("CONTRIBUTING.md")
        assert check_docs._is_documentation("SECURITY.md")

    def test_runtime_source_and_unrelated_files_are_excluded(self) -> None:
        # Package-source Markdown is runtime input, not documentation.
        assert not check_docs._is_documentation("hosts/cli/src/hosts/cli/_mode.md")
        assert not check_docs._is_documentation("hosts/cli/tests/fixtures/prompt.md")
        # Other package files are outside the checker's interest.
        assert not check_docs._is_documentation("hosts/cli/NOTES.md")
        assert not check_docs._is_documentation("scripts/pack_assets.md")


class TestDetectedDefects:
    """Each check fires on the defect it claims to catch."""

    def test_broken_link_target_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [missing](missing.md).\n", encoding="utf-8"
        )
        assert "broken-link" in _rules(repo)

    def test_broken_anchor_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [section](#no-such-heading).\n", encoding="utf-8"
        )
        assert "broken-anchor" in _rules(repo)

    def test_valid_link_and_anchor_are_accepted(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n## Real Section\n\nSee [section](#real-section) and [other](other.md).\n",
            encoding="utf-8",
        )
        (repo / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")
        anchors = {
            "docs/guide.md": check_docs._heading_anchors(
                (repo / "docs" / "guide.md").read_text(encoding="utf-8").splitlines()
            ),
            "docs/other.md": {"other": 1},
        }
        findings = check_docs._check_file("docs/guide.md", repo, anchors)
        assert findings == []

    def test_unlabelled_code_fence_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n\n```\ncode\n```\n", encoding="utf-8")
        assert "fence-language" in _rules(repo)

    def test_labelled_code_fence_is_accepted(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n```python\ncode\n```\n", encoding="utf-8"
        )
        assert _rules(repo) == set()

    def test_unclosed_code_fence_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n\n```python\ncode\n", encoding="utf-8")
        assert "fence-unclosed" in _rules(repo)

    def test_missing_final_newline_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide", encoding="utf-8")
        assert "final-newline" in _rules(repo)

    def test_trailing_blank_lines_are_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n\n\n", encoding="utf-8")
        assert "final-newline" in _rules(repo)

    def test_trailing_whitespace_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\ntext with trailing space  \n", encoding="utf-8"
        )
        assert "trailing-whitespace" in _rules(repo)

    def test_tab_indentation_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n\tindented with a tab\n", encoding="utf-8"
        )
        assert "tab-indentation" in _rules(repo)

    def test_heading_without_a_leading_blank_line_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n## Right After\n", encoding="utf-8")
        assert "heading-spacing" in _rules(repo)

    def test_list_without_a_leading_blank_line_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nA paragraph.\n- first item\n", encoding="utf-8"
        )
        assert "list-spacing" in _rules(repo)

    def test_wrapped_list_items_are_not_reported(self, repo: Path) -> None:
        # A list item's continuation lines are indented and must not be
        # mistaken for a new list that lacks a leading blank line.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n1. first item that wraps\n   onto a second line.\n2. second item\n",
            encoding="utf-8",
        )
        assert "list-spacing" not in _rules(repo)

    def test_code_span_heading_keeps_its_anchor(self, repo: Path) -> None:
        # GitHub strips the period out of a code-span heading, so the
        # anchor is ``contextlanes``. The gate must agree with that.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n### `context.lanes`\n\nSee [lanes](#contextlanes).\n", encoding="utf-8"
        )
        anchors = {
            "docs/guide.md": check_docs._heading_anchors(
                (repo / "docs" / "guide.md").read_text(encoding="utf-8").splitlines()
            ),
        }
        findings = check_docs._check_file("docs/guide.md", repo, anchors)
        assert findings == []

    def test_code_span_heading_rejects_a_wrong_anchor(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n### `context.lanes`\n\nSee [lanes](#context-lanes).\n", encoding="utf-8"
        )
        anchors = {
            "docs/guide.md": check_docs._heading_anchors(
                (repo / "docs" / "guide.md").read_text(encoding="utf-8").splitlines()
            ),
        }
        findings = check_docs._check_file("docs/guide.md", repo, anchors)
        assert [f.rule for f in findings] == ["broken-anchor"]

    def test_read_error_is_reported(self, repo: Path) -> None:
        # A path that cannot be read surfaces as a read-error finding
        # rather than silently passing.
        findings = check_docs._check_file("docs/gone.md", repo, {})
        assert [f.rule for f in findings] == ["read-error"]


class TestEscapedLinks:
    """Links the checker deliberately ignores (schemes, out-of-tree)."""

    def test_absolute_urls_are_ignored(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [pydantic](https://docs.pydantic.dev) and"
            " [mail](mailto:someone@example.com).\n",
            encoding="utf-8",
        )
        assert _rules(repo) == set()

    def test_out_of_tree_relative_link_is_ignored(self, repo: Path) -> None:
        (repo / ".." / "escape.md").write_text("# Escape\n", encoding="utf-8")
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [out](../../escape.md).\n", encoding="utf-8"
        )
        assert "broken-link" not in _rules(repo)


class TestMain:
    """The CLI surface returns 0 on the standing corpus, 1 on defects."""

    def _git_init(self, repo: Path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)

    def test_clean_repo_passes(self, repo: Path) -> None:
        (repo / "README.md").write_text("# Root\n", encoding="utf-8")
        self._git_init(repo)
        assert check_docs.main(["--repo-root", str(repo)]) == 0

    def test_defect_fails(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n\n",
            encoding="utf-8",
        )
        self._git_init(repo)
        assert check_docs.main(["--repo-root", str(repo)]) == 1
