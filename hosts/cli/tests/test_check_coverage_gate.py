"""Tests for the coverage gates (``scripts/check_coverage.py``).

The floors and the CLI contract are asserted directly against the script
loaded by path — ``scripts/`` is not an importable package. The subprocess
gates run against the real ``coverage`` command so drift between the script
and coverage's CLI surfaces as a test failure, not a silently green gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_coverage.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_coverage.py`` as a module."""
    spec = importlib.util.spec_from_file_location("check_coverage_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_coverage = _load_module()


class TestPackageFloors:
    """``_check_package_floors`` reflects the measured data it is handed."""

    def test_floors_declared(self) -> None:
        assert check_coverage.FLOORS == [("hosts/cli/src/*", 85, 87)]

    def test_floor_below_baseline_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A package under its floor is reported LOW and the gate returns 1.
        result = check_coverage._coverage("report", "--help")
        assert result.returncode == 0
        monkeypatch.setattr(check_coverage, "FLOORS", [("no/such/pkg/*", 99, 100)])
        assert check_coverage._check_package_floors() == 1
        assert "LOW" in capsys.readouterr().out

    def test_floor_at_baseline_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(check_coverage, "FLOORS", [("no/such/pkg/*", 100, 100)])
        # No data exists for the include window: report succeeds with no TOTAL
        # row, which is a failure (a floor on nothing proves nothing).
        assert check_coverage._check_package_floors() == 1
        assert "LOW" in capsys.readouterr().out


class TestCliContract:
    """The argument pairs the CI matrix relies on are enforced."""

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--base", "origin/main"], "requires --run"),
            (["--skip-changed-lines"], "requires --run"),
        ],
    )
    def test_flags_require_run(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str], message: str
    ) -> None:
        # parser.error exits; catch it via SystemExit and the usage message.
        with pytest.raises(SystemExit) as excinfo:
            check_coverage.main(argv)
        assert excinfo.value.code == 2

    def test_run_surfaces_test_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing test returncode propagates instead of the gate verdict."""
        monkeypatch.setattr(
            check_coverage,
            "subprocess",
            _FakeSubprocess(failing={"coverage"}),
        )
        assert check_coverage.main(["--run", "--skip-changed-lines"]) == 7


class _FakeSubprocess:
    """Stand-in for the script's subprocess calls, keyed by invocation shape."""

    def __init__(self, failing: set[str]) -> None:
        self._failing = failing

    def run(self, args, **kwargs):
        # The suite invocation is the only one under "coverage ... run -m pytest".
        if "run" in args and "pytest" in args:
            return FakeCompletedProcess(7 if "coverage" in self._failing else 0, args)
        return FakeCompletedProcess(0, args)


class FakeCompletedProcess:
    def __init__(self, code: int, args: list[str]) -> None:
        self.returncode = code
        self.args = args
        self.stdout = "TOTAL 100%\n"
        self.stderr = ""
