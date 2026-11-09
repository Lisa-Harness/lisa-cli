"""Import-time .env loader: profile candidate + base dir fallback order.

Covers _load_env_candidates on the full CWD → profile → base → framework
order the startup loader assembles when LISA_PROFILE points at a profile
that does not carry its own .env.
"""

import os
from pathlib import Path

import pytest

from hosts.cli import main as cli_main


def _patch_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Point the loader's dir resolvers at a scratch tree; return cwd."""
    base_dir = tmp_path / "lisa_home"
    cwd = tmp_path / "cwd"
    base_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(cli_main, "_default_data_dir", lambda: base_dir)
    monkeypatch.setattr(
        cli_main,
        "_profile_data_dir",
        lambda profile, base_dir=None: tmp_path / "profiles" / profile,
    )
    return cwd


def _rebuild_candidates() -> tuple[Path, ...]:
    """Re-resolve the candidate order against the patched dirs."""
    return cli_main._env_candidates()[1]


def test_profile_without_env_falls_back_to_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LISA_PROFILE set, no profile .env → base dir .env loads and verify."""
    cwd = _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    base_dir = tmp_path / "lisa_home"
    (base_dir / ".env").write_text("PROFILE_BASE_VAR=1\n", encoding="utf-8")
    candidates = _rebuild_candidates()
    try:
        loaded = cli_main._load_env_candidates(candidates)
        profile_env = tmp_path / "profiles" / "taxes" / ".env"
        assert loaded == base_dir / ".env"
        assert os.environ.get("PROFILE_BASE_VAR") == "1"
        # The base fallback is a distinct slot after the profile candidate.
        assert candidates.index(profile_env) < candidates.index(base_dir / ".env")
        assert candidates[0] == (cwd / ".env")
    finally:
        os.environ.pop("PROFILE_BASE_VAR", None)


def test_profile_env_wins_over_base_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile .env overrides the base dir fallback for shared keys."""
    _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    base_dir = tmp_path / "lisa_home"
    profile_dir = tmp_path / "profiles" / "taxes"
    profile_dir.mkdir(parents=True)
    (profile_dir / ".env").write_text("SHARED=profile\n", encoding="utf-8")
    candidates = _rebuild_candidates()
    try:
        # The base candidate doesn't exist yet: the profile is the only hit,
        # so it is loaded no-override. Then the base file appears (a later
        # candidate is "more authoritative") and must NOT clobber SHARED.
        loaded = cli_main._load_env_candidates(candidates)
        assert loaded == profile_dir / ".env"
        assert os.environ.get("SHARED") == "profile"
        assert os.environ.get("BASE_ONLY") != "1"
        (base_dir / ".env").write_text("SHARED=base\nBASE_ONLY=1\n", encoding="utf-8")
        loaded_again = cli_main._load_env_candidates(candidates)
        assert loaded_again == base_dir / ".env"
        assert os.environ.get("SHARED") == "base"  # override=True wins
    finally:
        for var in ("SHARED", "BASE_ONLY"):
            os.environ.pop(var, None)


def test_no_profile_keeps_single_data_dir_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No LISA_DATA_DIR / LISA_PROFILE → the base dir entry is not duplicated."""
    _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PROFILE", raising=False)
    candidates = _rebuild_candidates()
    base_dir = tmp_path / "lisa_home"
    env_dir_paths = [str(p.parent) for p in candidates[1:] if p.parent == base_dir]
    assert len(env_dir_paths) == 1
