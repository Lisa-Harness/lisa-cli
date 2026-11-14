"""Tests for the ``--version`` output path and the metadata fallback.

``lisa --version`` prints ``lisa v<installed>`` from importlib metadata;
when the package is not installed as a distribution (checked out without
``uv sync``), the literal version constant is the documented fallback.
The version constant itself is enforced against the package manifest — a
release that bumps one and not the other fails here.
"""

from __future__ import annotations

# pylint: disable=protected-access
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from hosts.cli import main as cli_main

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_version_prints_installed_distribution_version() -> None:
    result: Any = runner.invoke(cli_main.app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "lisa v" in result.output


def test_version_with_subcommand_prints_then_dispatches() -> None:
    result: Any = runner.invoke(cli_main.app, ["--version", "verify"])
    assert result.exit_code == 0, result.output
    assert "lisa v" in result.output


def test_bare_invocation_prints_help_and_exits_two() -> None:
    """Flag-only invocation (no subcommand): help + exit 2."""
    result: Any = runner.invoke(cli_main.app, ["-r"])
    assert result.exit_code == 2, result.output
    assert "COMMAND [ARGS]" in result.output


def test_metadata_failure_falls_back_to_declared_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing distribution falls back to the manifest's version string."""
    import importlib.metadata as md

    def _missing(name: Any) -> str:
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", _missing)
    assert cli_main._get_version() == "0.5.0"


def test_version_constant_matches_package_manifest() -> None:
    """The hardcoded fallback never drifts from hosts/cli/pyproject.toml."""
    manifest = _REPO_ROOT / "hosts/cli/pyproject.toml"
    declared = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["version"]

    # The fallback literal lives in the module source; extract and compare.
    import inspect
    import re

    source = inspect.getsource(cli_main._get_version)
    match = re.compile(r'return "([^"]+)"').search(source)
    assert match, source
    assert match.group(1) == declared
