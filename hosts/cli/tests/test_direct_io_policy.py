"""Tests for the direct-I/O policy gate (``scripts/check_direct_io_policy.py``).

The gate must scan the real hosts/cli tree green, must flag the write
primitives it claims to ban, and must honor its allowlist. The script is
loaded by path since ``scripts/`` is not an importable package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_direct_io_policy.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_direct_io_policy.py`` as a module."""
    spec = importlib.util.spec_from_file_location("direct_io_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = _load_module()


class TestScopeSelection:
    """Only runtime, non-test source is scanned."""

    def test_runtime_files_scanned(self, tmp_path: Path) -> None:
        (tmp_path / "hosts" / "cli" / "src").mkdir(parents=True)
        src = tmp_path / "hosts" / "cli" / "src" / "x.py"
        src.write_text("x = 1\n", encoding="utf-8")
        assert policy._iter_in_tree_files(tmp_path) == [src]

    def test_tests_and_builds_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "hosts" / "cli").mkdir(parents=True)
        for name in ("tests/t.py", "build/x.py"):
            p = tmp_path / "hosts" / "cli" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x = 1\n", encoding="utf-8")
        assert policy._iter_in_tree_files(tmp_path) == []


class TestDetection:
    """Each banned primitive is found; safe code is not."""

    def test_write_method_flagged(self) -> None:
        ((line, snippet),) = policy._find_io_violations("p.write_text('x')\n")
        assert line == 1
        assert "write_text" in snippet

    def test_mutate_method_flagged(self) -> None:
        found = policy._find_io_violations("p.rmdir()\nq.unlink()\np.mkdir()\n")
        assert [ln for ln, _ in found] == [1, 2, 3]

    def test_module_func_flagged(self) -> None:
        found = policy._find_io_violations(
            "import os\nimport shutil\nimport sqlite3\n"
            "os.replace('a', 'b')\n"
            "shutil.copy('a', 'b')\n"
            "sqlite3.connect('db')\n"
        )
        assert [ln for ln, _ in found] == [4, 5, 6]

    def test_write_open_mode_flagged(self) -> None:
        # The checker reads the positional mode argument and classifies by
        # its first character: 'wb', 'w+b', 'ab' all reduce to their head.
        for mode in ("w", "wb", "w+b", "a", "ab", "a+b", "x"):
            found = policy._find_io_violations(f"open('f', {mode!r})\n")
            assert found, mode

    def test_read_open_mode_not_flagged(self) -> None:
        assert policy._find_io_violations("open('f', mode='r')\n") == []
        assert policy._find_io_violations("open('f')\n") == []

    def test_safe_calls_not_flagged(self) -> None:
        source = "import json\nvalue = json.dumps({'k': 1})\np.read_text()\nhandle.flush()\n"
        assert policy._find_io_violations(source) == []

    def test_syntax_error_yields_no_findings(self) -> None:
        assert policy._find_io_violations("def broken(:\n") == []


class TestAllowlist:
    """An allowlisted line is exempt; other writes in the file are not."""

    def test_allowlisted_callsite_exempt(self) -> None:
        line = "history_path.parent.mkdir(parents=True, exist_ok=True)"
        assert policy._is_allowlisted("hosts/cli/src/hosts/cli/entry.py", line)

    def test_other_file_not_exempt(self) -> None:
        line = "history_path.parent.mkdir(parents=True, exist_ok=True)"
        assert not policy._is_allowlisted("hosts/cli/src/hosts/cli/main.py", line)

    def test_check_file_respects_allowlist(self, tmp_path: Path) -> None:
        src = tmp_path / "entry.py"
        src.write_text(
            "# noqa seam\nhistory_path.parent.mkdir(parents=True, exist_ok=True)\n",
            encoding="utf-8",
        )
        violations = policy._check_file(
            src, "hosts/cli/src/hosts/cli/entry.py", allowlist_enabled=True
        )
        assert violations == []

    def test_check_file_flags_unallowed_write(self, tmp_path: Path) -> None:
        src = tmp_path / "entry.py"
        src.write_text("Path('x').write_text('hi')\n", encoding="utf-8")
        violations = policy._check_file(
            src, "hosts/cli/src/hosts/cli/entry.py", allowlist_enabled=True
        )
        assert [(v[0], v[1]) for v in violations] == [("hosts/cli/src/hosts/cli/entry.py", 1)]

    def test_check_file_read_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.py"
        violations = policy._check_file(missing, "gone.py", allowlist_enabled=False)
        assert len(violations) == 1 and "read error" in violations[0][2]


class TestMain:
    """The repo as it stands passes; a bad allowlist state fails."""

    def test_repo_is_green(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert policy.main() == 0
        assert "check passed" in capsys.readouterr().out
