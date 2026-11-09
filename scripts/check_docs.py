#!/usr/bin/env python3
"""Check the maintained documentation corpus for structural defects for lisa-cli.

DESIGN — The repository keeps one documentation corpus (the guides under
``docs/``, the package READMEs, and the root-level Markdown). A doc-only
change must not be able to break a link, a heading anchor, or a code fence
without a job noticing, so this script checks the properties that the
corpus already satisfies today:

- **Relative links resolve.** Every relative link target must exist in the
  working tree, and every ``#anchor`` fragment must match a heading in the
  file it points at. Absolute URLs, ``mailto:`` links, and pure fragments
  are resolved against the current file.
- **Code fences are labelled and balanced.** Every fenced block declares a
  language and every fence is closed.
- **Files end with exactly one newline.** No trailing blank lines and no
  missing final newline.
- **No trailing whitespace** on any line, and no tab indentation.
- **Headings and lists are spaced.** A heading needs a blank line above
  it, and a list needs a blank line before it (unless it continues a list).

Checks are deliberately structural rather than stylistic. Line length,
heading duplication, and table spacing vary across the corpus today, and
a gate that flags them would fail on the existing corpus instead of on
regressions. Those remain authoring conventions enforced by review.

Markdown that is *not* documentation is out of scope: pack content
(``_mode.md``, ``*.operation.md``, prompt and session templates), test
fixtures, and Copilot customization files are inputs to the runtime or
the agent host, and are checked by the tests that load them.

Usage:
    uv run python scripts/check_docs.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------

#: Directories whose Markdown is documentation.
_DOC_DIRS = ("docs/",)

#: File names that are documentation wherever they live.
_DOC_NAMES = ("README.md", "AUTHORING.md")

#: Root-level Markdown files that are documentation.
_ROOT_DOC_FILES = (
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "LICENSE",
)

#: Package source trees whose Markdown is runtime or agent input, never
#: documentation.
_RUNTIME_SOURCE_PREFIXES = (
    "hosts/cli/src/",
    "hosts/cli/tests/",
)


def _git_tracked_markdown(repo_root: Path) -> list[str]:
    """Return the repository's tracked Markdown files as POSIX-relative paths."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _is_documentation(rel_path: str) -> bool:
    """Report whether a repo-relative Markdown path belongs to the doc corpus.

    Documentation is the guides under ``docs/``, the root-level policy
    files, and any package README or authoring guide. Everything else is
    an input to the runtime or the agent host: pack content (modes,
    operations, prompts, sessions), test fixtures, and Copilot
    customization files. Those are covered by the tests that load them.

    The rule mirrors the CI path filter, so a file is never documentation
    for one and behavior for the other. A README inside a package source
    tree is a runtime asset, not a guide.
    """
    if rel_path.startswith(_DOC_DIRS):
        return True
    if "/" not in rel_path:
        return rel_path in _ROOT_DOC_FILES
    if rel_path.startswith(_RUNTIME_SOURCE_PREFIXES):
        return False
    return rel_path.rsplit("/", 1)[1] in _DOC_NAMES


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One defect in one documentation file."""

    path: str
    line: int
    rule: str
    message: str

    def render(self) -> str:
        """Format the finding as a one-line report."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# ---------------------------------------------------------------------------
# Markdown primitives
# ---------------------------------------------------------------------------

#: A fenced code block opener or closer.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*)$")

#: An ATX heading.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

#: A setext heading underline (``Title`` followed by ``===`` or ``---``).
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")

#: A list item at any indentation.
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")

#: An inline link or image.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]*)\)")

#: A scheme-bearing URL that is never resolved against the working tree.
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

#: Inline HTML tags.
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")


def _iter_code_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` line index of each fenced block.

    ``start`` is the opener line and ``end`` is the line after the closer, so
    ``lines[start:end]`` is the block's content. Unbalanced fences extend to
    the end of the file.
    """
    regions: list[tuple[int, int]] = []
    open_at: int | None = None
    open_marker = ""
    for index, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if open_at is None:
            open_at = index
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not info:
            regions.append((open_at, index + 1))
            open_at = None
    if open_at is not None:
        regions.append((open_at, len(lines)))
    return regions


def _blank_fences(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks, leaving everything else untouched.

    Inline code spans are preserved: a heading may contain a code span that
    is part of its title (``### `context.lanes```), and stripping it would
    change the anchor the heading produces.
    """
    blanked = list(lines)
    for start, end in _iter_code_regions(lines):
        for index in range(start, end):
            blanked[index] = ""
    return blanked


def _strip_code(lines: list[str]) -> list[str]:
    """Blank out fenced blocks and inline code spans.

    Returns one string per input line, with code content removed so that
    link, heading, and spacing checks do not fire on examples.
    """
    return [re.sub(r"`[^`]*`", "", line) for line in _blank_fences(lines)]


#: Characters GitHub removes from a heading when building its anchor.
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")


def _slugify(text: str) -> str:
    """GitHub-compatible heading anchor."""
    text = _SLUG_STRIP_RE.sub("", text)
    text = re.sub(r"\s+", "-", text.strip())
    return text.lower()


def _heading_anchors(lines: list[str]) -> dict[str, int]:
    """Return a map of anchor -> line number for all headings in the file."""
    anchors: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            heading_text = match.group(2)
            anchor = _slugify(heading_text)
            anchors[anchor] = index + 1
        elif index > 0 and _SETEXT_RE.match(line):
            prev = lines[index - 1].strip()
            if prev and not _HEADING_RE.match(prev):
                anchor = _slugify(prev)
                anchors[anchor] = index + 1
    return anchors


def _resolve_link(link: str, current_file: Path, repo_root: Path) -> tuple[Path | None, str | None]:
    """Resolve a Markdown link target to an absolute path and optional anchor.

    Returns ``(target_path, anchor)`` where ``target_path`` is the resolved
    file (or ``None`` if the link is external) and ``anchor`` is the
    fragment (or ``None`` if no fragment).
    """
    if _URL_SCHEME_RE.match(link) or link.startswith("mailto:"):
        return None, None
    if link.startswith("#"):
        return current_file, link[1:]
    if "#" in link:
        path_part, anchor = link.split("#", 1)
    else:
        path_part, anchor = link, None
    if not path_part:
        return current_file, anchor
    target = (current_file.parent / path_part).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError:
        return None, None
    return target, anchor


def _check_whitespace(rel_path: str, findings: list[Finding], lines: list[str]) -> None:
    """Check trailing whitespace and tab indentation on each line."""
    for index, line in enumerate(lines):
        if line.rstrip("\n") != line.rstrip():
            findings.append(
                Finding(rel_path, index + 1, "trailing-whitespace", "trailing whitespace")
            )
        if "\t" in line:
            findings.append(Finding(rel_path, index + 1, "tab-indentation", "tab indentation"))


def _check_fences(rel_path: str, findings: list[Finding], lines: list[str]) -> None:
    """Check that fenced code blocks open with a language and are closed."""
    open_at: int | None = None
    open_marker = ""
    for index, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if open_at is None:
            if not info:
                findings.append(
                    Finding(rel_path, index + 1, "fence-language", "fenced block missing language")
                )
            open_at = index
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not info:
            open_at = None
    if open_at is not None:
        findings.append(Finding(rel_path, open_at + 1, "fence-unclosed", "unclosed fenced block"))


def _check_heading_spacing(rel_path: str, findings: list[Finding], lines: list[str]) -> None:
    """Check that headings and lists have a blank line above them."""
    previous = ""  # The previous non-blank line, or "" at the start of the file.
    separated = True  # Whether a blank line follows ``previous``.
    for index, line in enumerate(lines):
        if not line.strip():
            separated = True
            continue
        if _HEADING_RE.match(line):
            if not separated:
                findings.append(
                    Finding(
                        rel_path, index + 1, "heading-spacing", "heading needs blank line above"
                    )
                )
        elif _LIST_RE.match(line) and not separated:
            # A list item directly after prose needs a blank line — but an
            # indented continuation line is not prose: it wraps the previous
            # item, so the next item following it is fine.
            follows_prose = (
                _LIST_RE.match(previous) is None
                and _HEADING_RE.match(previous) is None
                and previous[:1] not in (" ", "\t")
            )
            if follows_prose:
                findings.append(
                    Finding(rel_path, index + 1, "list-spacing", "list needs blank line above")
                )
        previous = line
        separated = False


def _check_links(
    rel_path: str,
    findings: list[Finding],
    lines: list[str],
    repo_root: Path,
    all_docs: dict[str, dict[str, int]],
) -> None:
    """Check relative links resolve to real files and anchors exist."""
    for index, line in enumerate(lines):
        for match in _LINK_RE.finditer(line):
            target, anchor = _resolve_link(match.group(1), repo_root / rel_path, repo_root)
            if target is None:
                continue
            if not target.is_file():
                findings.append(
                    Finding(
                        rel_path,
                        index + 1,
                        "broken-link",
                        f"target does not exist: {match.group(1)}",
                    )
                )
            elif anchor and anchor not in all_docs.get(
                target.relative_to(repo_root).as_posix(), {}
            ):
                findings.append(
                    Finding(
                        rel_path,
                        index + 1,
                        "broken-anchor",
                        f"anchor not found in target: #{anchor}",
                    )
                )


def _check_file(
    rel_path: str, repo_root: Path, all_docs: dict[str, dict[str, int]]
) -> list[Finding]:
    """Check a single documentation file."""
    findings: list[Finding] = []
    path = repo_root / rel_path
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        findings.append(Finding(rel_path, 0, "read-error", f"cannot read file: {exc}"))
        return findings

    lines = content.splitlines(keepends=True)

    # 1. Final newline
    if not content.endswith("\n"):
        findings.append(Finding(rel_path, len(lines), "final-newline", "missing final newline"))
    elif content.endswith("\n\n"):
        findings.append(Finding(rel_path, len(lines), "final-newline", "trailing blank lines"))

    _check_whitespace(rel_path, findings, lines)
    _check_fences(rel_path, findings, lines)

    # 4./5. Heading/list spacing and links (on code-stripped lines)
    stripped = _strip_code(lines)
    _check_heading_spacing(rel_path, findings, stripped)
    _check_links(rel_path, findings, stripped, repo_root, all_docs)

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    tracked = _git_tracked_markdown(repo_root)
    docs = [p for p in tracked if _is_documentation(p)]

    # Pre-compute heading anchors for all doc files
    all_anchors: dict[str, dict[str, int]] = {}
    for rel in docs:
        path = repo_root / rel
        try:
            content = path.read_text(encoding="utf-8")
            all_anchors[rel] = _heading_anchors(content.splitlines())
        except OSError:
            all_anchors[rel] = {}

    all_findings: list[Finding] = []
    for rel in docs:
        all_findings.extend(_check_file(rel, repo_root, all_anchors))

    if all_findings:
        for finding in all_findings:
            print(finding.render())
        print(f"\nDocumentation check failed: {len(all_findings)} defect(s).")
        return 1

    print("Documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
