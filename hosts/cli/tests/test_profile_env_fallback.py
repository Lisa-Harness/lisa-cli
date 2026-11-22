"""Trusted dotenv candidates, precedence, and executable-path trust.

The working directory is never an implicit configuration source. Trusted
base/profile/framework files load in that order, and the original process
environment wins over every file.
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
    monkeypatch.setattr(cli_main, "_mirror_default_data_dir", lambda: base_dir)
    monkeypatch.setattr(
        cli_main,
        "_mirror_profile_data_dir",
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
        # Base configuration is lower precedence than profile configuration.
        assert candidates.index(base_dir / ".env") < candidates.index(profile_env)
        assert (cwd / ".env") not in candidates
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
        loaded = cli_main._load_env_candidates(candidates)
        assert loaded == profile_dir / ".env"
        assert os.environ.get("SHARED") == "profile"
        assert os.environ.get("BASE_ONLY") != "1"
        (base_dir / ".env").write_text("SHARED=base\nBASE_ONLY=1\n", encoding="utf-8")
        loaded_again = cli_main._load_env_candidates(candidates)
        assert loaded_again == profile_dir / ".env"
        assert os.environ.get("SHARED") == "profile"
        assert os.environ.get("BASE_ONLY") == "1"
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
    env_dir_paths = [str(p.parent) for p in candidates if p.parent == base_dir]
    assert len(env_dir_paths) == 1


def test_cwd_env_cannot_select_packs_venv_or_execute_pth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile checkout has no influence on executable search paths."""
    import subprocess
    import sys

    cwd = tmp_path / "hostile-checkout"
    data_dir = tmp_path / "trusted-data"
    venv = cwd / "attacker-venv"
    site_packages = cli_main._mirror_packs_venv_site_packages(venv)
    marker = tmp_path / "pth-executed"
    site_packages.mkdir(parents=True)
    data_dir.mkdir()
    (cwd / ".env").write_text(f"LISA_PACKS_VENV={venv}\n", encoding="utf-8")
    (site_packages / "hostile.pth").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    src_root = Path(cli_main.__file__).resolve().parents[2]
    monkeypatch.setattr(cli_main, "_PACKS_VENV_INJECTED", False)
    monkeypatch.setattr(cli_main, "_ENV_LOADED", False)
    monkeypatch.chdir(cwd)
    completed = subprocess.run(
        [sys.executable, "-c", "from hosts.cli.application import app; app(args=['--help'])"],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(src_root),
            "LISA_DATA_DIR": str(data_dir),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert str(site_packages) not in sys.path


def test_original_process_environment_wins_over_all_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dotenv files cannot overwrite values supplied by the calling shell."""
    monkeypatch.setattr(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework")
    monkeypatch.setenv("SHARED", "shell")
    monkeypatch.delenv("BASE_ONLY", raising=False)
    candidates = (tmp_path / "base.env", tmp_path / "profile.env")
    candidates[0].write_text("SHARED=base\nBASE_ONLY=base\n", encoding="utf-8")
    candidates[1].write_text("SHARED=profile\nBASE_ONLY=profile\n", encoding="utf-8")
    loaded = cli_main._load_env_candidates(candidates)
    assert loaded == candidates[1]
    assert os.environ["SHARED"] == "shell"
    assert os.environ["BASE_ONLY"] == "profile"


def test_lower_precedence_file_cannot_influence_later_file_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Files merge in memory: base values never expand inside profile values."""
    monkeypatch.delenv("ENDPOINT", raising=False)
    first = tmp_path / "base.env"
    second = tmp_path / "profile.env"
    first.write_text("HOST=base\n", encoding="utf-8")
    second.write_text("ENDPOINT=${HOST}\n", encoding="utf-8")
    assert cli_main._load_env_candidates((first, second)) == second
    assert os.environ["ENDPOINT"] == "${HOST}"


def test_shared_resolver_honors_explicit_and_environment_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI arguments, environment data dir, and environment profile compose."""
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path / "environment-data"))
    monkeypatch.setenv("LISA_PROFILE", "environment-profile")
    assert cli_main._resolve_data_dir() == tmp_path / "environment-data"
    assert cli_main._resolve_data_dir("", "cli-profile").name == "cli-profile"
    assert cli_main._resolve_data_dir(tmp_path / "explicit", "cli-profile") == (
        tmp_path / "explicit"
    )
    monkeypatch.delenv("LISA_DATA_DIR")
    assert cli_main._resolve_data_dir().name == "environment-profile"
