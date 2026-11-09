#!/usr/bin/env python3
"""Enforce direct file I/O policy with an explicit allowlist for lisa-cli.

Scans runtime Python code for filesystem primitives that bypass higher-level
storage routing. Any detected callsite must be explicitly allowlisted by file
+ snippet, or migrated behind the storage seam.

Scope
-----
In-tree packages (``hosts/cli/``): scans write/mutation primitives only.
Reads are a separate, larger follow-up because the codebase has legitimate
read callsites.

The scanner uses ``ast`` for reliable detection (catches aliased names,
multiline calls, and attribute chains that regex misses).
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Banned primitives
# ---------------------------------------------------------------------------

# Attribute-method calls on any object (e.g. ``path.write_text(...)``).
# Matched by attribute name only -- the receiver type is resolved at runtime.
_BANNED_METHODS_WRITE: frozenset[str] = frozenset(
    {
        "write_text",
        "write_bytes",
    }
)

_BANNED_METHODS_MUTATE: frozenset[str] = frozenset(
    {
        "unlink",
        "mkdir",
        "rmdir",
    }
)

# Module-level function calls (e.g. ``os.replace(...)``, ``sqlite3.connect(...)``).
# Keyed by ``(module_name, function_name)``.
_BANNED_MODULE_FUNCS_WRITE: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "replace"),
        ("os", "remove"),
        ("os", "rmdir"),
        ("shutil", "copy"),
        ("shutil", "copy2"),
        ("shutil", "copyfile"),
        ("shutil", "rmtree"),
        ("sqlite3", "connect"),
    }
)

# ``open()`` mode argument prefixes that indicate write/mutation.
_WRITE_OPEN_MODES: frozenset[str] = frozenset({"w", "a", "x", "wb", "ab", "xb"})

# ---------------------------------------------------------------------------
# Allowlist -- in-tree carve-outs
# ---------------------------------------------------------------------------

# Allowed direct-write callsites, keyed by repo-relative path.
# Each value is a tuple of line substrings that are permitted in that file.
_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "hosts/cli/src/hosts/cli/entry.py": ("history_path.parent.mkdir(parents=True, exist_ok=True)",),
    # uv owns the tool receipt; update must atomically change its requirements
    # without routing it through Lisa's runtime data storage.
    "hosts/cli/src/hosts/cli/update.py": (
        "os.replace(temp_path, receipt_path)",
        "temp_path.unlink(missing_ok=True)",
    ),
}


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_runtime_python(path: Path) -> bool:
    rel = path.as_posix()
    if not rel.endswith(".py"):
        return False
    if "/tests/" in rel:
        return False
    if path.name.startswith("test_"):
        return False
    if path.name == "conftest.py":
        return False
    # Exclude build artifacts
    return "/build/" not in rel


# In-tree scan roots (relative to repo root).
_IN_TREE_ROOTS: tuple[str, ...] = ("hosts/cli",)


def _iter_in_tree_files(root: Path) -> list[Path]:
    """Runtime .py files in hosts/cli (excl. tests)."""
    files: list[Path] = [
        path
        for top in _IN_TREE_ROOTS
        if (root / top).is_dir()
        for path in (root / top).rglob("*.py")
        if _is_runtime_python(path)
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# AST-based detection
# ---------------------------------------------------------------------------


class _IOLocator(ast.NodeVisitor):
    """Walk an AST tree and record lines containing banned I/O calls."""

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def _check_method(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return
        attr = func.attr
        if attr in _BANNED_METHODS_WRITE or attr in _BANNED_METHODS_MUTATE:
            self.violations.append((node.lineno, ast.unparse(node)))

    def _check_module_func(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return
        # e.g. os.replace, sqlite3.connect, shutil.copy
        if not isinstance(func.value, ast.Name):
            return
        key = (func.value.id, func.attr)
        if key in _BANNED_MODULE_FUNCS_WRITE:
            self.violations.append((node.lineno, ast.unparse(node)))

    def _check_open(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "open":
            return
        # The mode is the second positional argument or the ``mode`` keyword;
        # the checker classifies reads/writes by the mode's first character.
        mode_arg: ast.expr | None = None
        if len(node.args) > 1:
            mode_arg = node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_arg = keyword.value
                break
        if not isinstance(mode_arg, ast.Constant) or not isinstance(mode_arg.value, str):
            return
        mode = mode_arg.value
        if mode[:1] in _WRITE_OPEN_MODES:
            self.violations.append((node.lineno, ast.unparse(node)))

    def visit_Call(self, node: ast.Call) -> None:
        self._check_method(node)
        self._check_module_func(node)
        self._check_open(node)
        self.generic_visit(node)


def _find_io_violations(source: str) -> list[tuple[int, str]]:
    """Parse source and return (line_number, source_snippet) for banned calls."""
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError:
        return []
    locator = _IOLocator()
    locator.visit(tree)
    return locator.violations


# ---------------------------------------------------------------------------
# Allowlist matching
# ---------------------------------------------------------------------------


def _is_allowlisted(rel_path: str, line_text: str) -> bool:
    allowed_snippets = _ALLOWLIST.get(rel_path)
    if not allowed_snippets:
        return False
    return any(snippet in line_text for snippet in allowed_snippets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _check_file(
    path: Path,
    rel_path: str,
    *,
    allowlist_enabled: bool,
) -> list[tuple[str, int, str]]:
    """Return violations for a single file as (rel_path, line_no, snippet)."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [(rel_path, 0, f"read error: {exc}")]

    lines = source.splitlines()
    violations: list[tuple[str, int, str]] = []

    for line_no, _snippet in _find_io_violations(source):
        line_text = lines[line_no - 1].strip() if line_no <= len(lines) else ""
        if allowlist_enabled and _is_allowlisted(rel_path, line_text):
            continue
        violations.append((rel_path, line_no, line_text))

    return violations


def main() -> int:
    root = _repo_root()

    # In-tree: writes only, allowlist applies. These are enforced (exit 1).
    in_tree_violations: list[tuple[str, int, str]] = []
    for path in _iter_in_tree_files(root):
        rel_path = path.relative_to(root).as_posix()
        in_tree_violations.extend(
            _check_file(
                path,
                rel_path,
                allowlist_enabled=True,
            )
        )

    if in_tree_violations:
        print("Direct I/O policy violations detected (in-tree, enforced):")
        for rel_path, line_no, line in in_tree_violations:
            print(f"- {rel_path}:{line_no}: {line}")
        print()
        print("Every in-tree direct-write callsite must be allowlisted in")
        print("scripts/check_direct_io_policy.py (_ALLOWLIST) or migrated behind")
        print("the storage seam / a documented host-internal carve-out.")
        return 1

    print("Direct I/O policy check passed (no unallowlisted runtime I/O callsites).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
