"""Exercise policies against real uv using only locally generated fixture wheels."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from hosts.cli import update

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required")


def _wheel(directory: Path, name: str, version: str) -> Path:
    normalized = name.replace("-", "_")
    dist = f"{normalized}-{version}.dist-info"
    path = directory / f"{normalized}-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if name == "lisa-cli":
        metadata += "Provides-Extra: prompt\n"
        metadata += "".join(
            f"Requires-Dist: {member}>=0.5.2\n" for member, _ in update.FRAMEWORK_MEMBERS
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{normalized}.py", "def main():\n    print('fixture')\n")
        archive.writestr(f"{dist}/METADATA", metadata + "\n")
        archive.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist}/RECORD", "")
        if name == "lisa-cli":
            archive.writestr(
                f"{dist}/entry_points.txt", "[console_scripts]\nlisa = lisa_cli:main\n"
            )
    return path


@pytest.fixture
def uv_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for version in ("1.0.0", "1.1.0", "1.1.2", "1.2.0"):
        _wheel(wheels, "lisa-cli", version)
    for name, _ in update.FRAMEWORK_MEMBERS:
        for version in ("0.5.2", "0.5.3", "0.6.0"):
            _wheel(wheels, name, version)
    _wheel(wheels, "fixture-extra", "1.0.0")

    for name in list(os.environ):
        if name.startswith("UV_"):
            monkeypatch.delenv(name)
    values = {
        "UV_TOOL_DIR": str(tmp_path / "tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "UV_NO_CONFIG": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(wheels),
        "UV_PYTHON": sys.executable,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(update, "INSTALL_PYTHON", sys.executable)
    return tmp_path / "tools" / "lisa-cli", wheels


def _uv(*args: str) -> None:
    result = subprocess.run(["uv", *args], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _version(tool_dir: Path, package: str) -> str:
    interpreter = tool_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import importlib.metadata, sys; print(importlib.metadata.version(sys.argv[1]))",
            package,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _requirements(tool_dir: Path) -> list[dict]:
    receipt = tomllib.loads((tool_dir / "uv-receipt.toml").read_text(encoding="utf-8"))
    return receipt["tool"]["requirements"]


def _run(
    tool_dir: Path, cli: update.Policy | None = None, framework: update.Policy | None = None
) -> None:
    update.upgrade_tool(
        tool_dir, update.Selection(cli or update.Policy(), framework or update.Policy()), "uv"
    )


def test_real_uv_version_and_framework_policies(uv_home: tuple[Path, Path]) -> None:
    tool_dir, wheels = uv_home
    _uv("tool", "install", "--no-sources", "lisa-cli[prompt]", "--with", "fixture-extra==1.0.0")
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.6.0"

    _run(tool_dir, update.Policy("version", "1.0.0"))
    assert _version(tool_dir, "lisa-cli") == "1.0.0"
    assert next(item for item in _requirements(tool_dir) if item["name"] == "lisa-cli") == {
        "name": "lisa-cli",
        "extras": ["prompt"],
    }
    assert any(item["name"] == "fixture-extra" for item in _requirements(tool_dir))
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.2.0"

    _run(tool_dir, update.Policy("pin", "1.1.0"))
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.1.0"
    _run(tool_dir, update.Policy("latest"))
    assert _version(tool_dir, "lisa-cli") == "1.2.0"

    _run(tool_dir, update.Policy("channel", "1.1"))
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.1.2"
    _run(tool_dir, update.Policy("latest"))
    assert _version(tool_dir, "lisa-cli") == "1.2.0"

    _run(tool_dir, framework=update.Policy("version", "0.5.2"))
    assert all(_version(tool_dir, name) == "0.5.2" for name, _ in update.FRAMEWORK_MEMBERS)
    assert not any(item["name"] in update._FRAMEWORK_NAMES for item in _requirements(tool_dir))
    _uv("tool", "upgrade", "lisa-cli")
    assert all(_version(tool_dir, name) == "0.6.0" for name, _ in update.FRAMEWORK_MEMBERS)

    _run(tool_dir, framework=update.Policy("pin", "0.5.3"))
    _uv("tool", "upgrade", "lisa-cli")
    assert all(_version(tool_dir, name) == "0.5.3" for name, _ in update.FRAMEWORK_MEMBERS)
    _run(tool_dir, framework=update.Policy("latest"))
    assert all(_version(tool_dir, name) == "0.6.0" for name, _ in update.FRAMEWORK_MEMBERS)

    receipt = tool_dir / "uv-receipt.toml"
    original = receipt.read_text(encoding="utf-8")
    with pytest.raises(update.UpdateError):
        _run(tool_dir, update.Policy("pin", "99.0.0"))
    assert receipt.read_text(encoding="utf-8") == original
    assert _version(tool_dir, "lisa-cli") == "1.2.0"

    cli_wheel = (wheels / "lisa_cli-1.1.0-py3-none-any.whl").as_uri()
    framework_wheel = (wheels / "lisa_harness_runtime-0.5.3-py3-none-any.whl").as_uri()
    _uv(
        "tool",
        "install",
        "--force",
        "--no-sources",
        f"lisa-cli[prompt] @ {cli_wheel}",
        "--with",
        f"lisa-harness-runtime @ {framework_wheel}",
        "--with",
        "fixture-extra==1.0.0",
    )
    _run(tool_dir, update.Policy("version", "1.0.0"))
    assert _version(tool_dir, "lisa-cli") == "1.0.0"
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.5.3"
    assert any(item["name"] == "fixture-extra" for item in _requirements(tool_dir))


def test_real_uv_installer_selections(uv_home: tuple[Path, Path]) -> None:
    tool_dir, _ = uv_home
    selection = update.Selection(
        update.Policy("version", "1.0.0"), update.Policy("version", "0.5.2")
    )
    update.install_tool("pypi", "", "", selection, "uv")
    assert _version(tool_dir, "lisa-cli") == "1.0.0"
    assert all(_version(tool_dir, name) == "0.5.2" for name, _ in update.FRAMEWORK_MEMBERS)
    requirements = _requirements(tool_dir)
    assert requirements == [{"name": "lisa-cli", "extras": ["prompt"]}]
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.6.0"

    update.install_tool(
        "pypi",
        "",
        "",
        update.Selection(update.Policy("channel", "1.1"), update.Policy("pin", "0.5.3")),
        "uv",
    )
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.1.2"
    assert all(_version(tool_dir, name) == "0.5.3" for name, _ in update.FRAMEWORK_MEMBERS)
    original = (tool_dir / "uv-receipt.toml").read_text(encoding="utf-8")
    with pytest.raises(update.UpdateError, match="install failed"):
        update.install_tool("pypi", "", "", update.Selection(update.Policy("pin", "99.0.0")), "uv")
    assert (tool_dir / "uv-receipt.toml").read_text(encoding="utf-8") == original
    assert _version(tool_dir, "lisa-cli") == "1.1.2"

    update.install_tool(
        "pypi",
        "",
        "",
        update.Selection(update.Policy("latest"), update.Policy("latest")),
        "uv",
    )
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert all(_version(tool_dir, name) == "0.6.0" for name, _ in update.FRAMEWORK_MEMBERS)
    assert _requirements(tool_dir) == [{"name": "lisa-cli", "extras": ["prompt"]}]


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="installer selects Python 3.12")
@pytest.mark.parametrize("installer", ["bash", "powershell"])
def test_installers_apply_selected_versions_offline(
    uv_home: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installer: str
) -> None:
    if installer == "powershell" and shutil.which("pwsh") is None:
        pytest.skip("PowerShell is required")
    if installer == "bash" and (os.name == "nt" or shutil.which("bash") is None):
        pytest.skip("Bash installer test requires POSIX")
    tool_dir, _ = uv_home
    repo = Path(__file__).resolve().parents[3]
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LISA_DATA_DIR", str(home / ".lisa"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    if installer == "bash":
        command = [
            "bash",
            str(repo / "install.sh"),
            "--source",
            "pypi",
            "--version",
            "1.0.0",
            "--framework-version",
            "0.5.2",
            "--no-verify",
        ]
    else:
        command = [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(repo / "install.ps1"),
            "-Source",
            "pypi",
            "-Version",
            "1.0.0",
            "-FrameworkVersion",
            "0.5.2",
            "-SkipDiagnostics",
        ]
    result = subprocess.run(command, cwd=repo, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _version(tool_dir, "lisa-cli") == "1.0.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.5.2"
    assert _requirements(tool_dir) == [{"name": "lisa-cli", "extras": ["prompt"]}]
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.6.0"


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="installer selects Python 3.12")
@pytest.mark.parametrize("installer", ["bash", "powershell"])
def test_piped_installer_defaults_to_pypi_and_persists_pin(
    uv_home: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installer: str
) -> None:
    if installer == "powershell" and shutil.which("pwsh") is None:
        pytest.skip("PowerShell is required")
    if installer == "bash" and (os.name == "nt" or shutil.which("bash") is None):
        pytest.skip("Bash installer test requires POSIX")
    tool_dir, _ = uv_home
    repo = Path(__file__).resolve().parents[3]
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LISA_DATA_DIR", str(home / ".lisa"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("LISA_TEST_HELPER", str(repo / "hosts/cli/src/hosts/cli/update.py"))
    for key in ("LISA_SOURCE", "LISA_GIT_REPO", "LISA_CLI_REPO"):
        monkeypatch.delenv(key, raising=False)

    if installer == "bash":
        source = (repo / "install.sh").read_text(encoding="utf-8")
        curl_stub = (
            'curl() { [[ "$1" == "-fsSL" && "$3" == "-o" ]] || return 1; '
            'cp "$LISA_TEST_HELPER" "$4"; }\n'
        )
        command = ["bash", "-s", "--", "--pin", "1.1.0", "--no-verify"]
        stdin = curl_stub + source
    else:
        monkeypatch.setenv("LISA_TEST_INSTALLER", str(repo / "install.ps1"))
        powershell = (
            "function Invoke-WebRequest { "
            "param($Uri, $OutFile, [switch]$UseBasicParsing) "
            "Copy-Item -LiteralPath $env:LISA_TEST_HELPER -Destination $OutFile }; "
            "$source = Get-Content -Raw -LiteralPath $env:LISA_TEST_INSTALLER; "
            'Invoke-Expression "& { $source } -Pin 1.1.0 -SkipDiagnostics"'
        )
        command = ["pwsh", "-NoLogo", "-NoProfile", "-Command", powershell]
        stdin = None

    result = subprocess.run(
        command, input=stdin, cwd=repo, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _version(tool_dir, "lisa-cli") == "1.1.0"
    assert _version(tool_dir, "lisa-harness-runtime") == "0.6.0"
    _uv("tool", "upgrade", "lisa-cli")
    assert _version(tool_dir, "lisa-cli") == "1.1.0"


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="installer selects Python 3.12")
@pytest.mark.parametrize("installer", ["bash", "powershell"])
def test_invalid_release_is_rejected_before_clean_or_reset(
    uv_home: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installer: str
) -> None:
    if installer == "powershell" and shutil.which("pwsh") is None:
        pytest.skip("PowerShell is required")
    if installer == "bash" and (os.name == "nt" or shutil.which("bash") is None):
        pytest.skip("Bash installer test requires POSIX")
    del uv_home
    repo = Path(__file__).resolve().parents[3]
    home = tmp_path / "home"
    marker = home / ".lisa" / "venv" / "keep"
    marker.parent.mkdir(parents=True)
    marker.write_text("existing state", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LISA_DATA_DIR", str(home / ".lisa"))
    if installer == "bash":
        command = [
            "bash",
            str(repo / "install.sh"),
            "--source",
            "pypi",
            "--pin",
            "1.0bogus",
            "--clean",
            "--reset",
        ]
    else:
        command = [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(repo / "install.ps1"),
            "-Source",
            "pypi",
            "-Pin",
            "1.0bogus",
            "-Clean",
            "-Reset",
        ]
    result = subprocess.run(command, cwd=repo, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Invalid release version" in result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8") == "existing state"


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Bash required")
def test_saved_bash_installer_defaults_to_local_without_a_tty(
    uv_home: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del uv_home
    repo = Path(__file__).resolve().parents[3]
    working = tmp_path / "work"
    working.mkdir()
    for key in ("LISA_SOURCE", "LISA_GIT_REPO", "LISA_CLI_REPO"):
        monkeypatch.delenv(key, raising=False)
    result = subprocess.run(
        ["bash", str(repo / "install.sh"), "--no-verify"],
        cwd=working,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "framework repo not found: ../lisa" in result.stdout + result.stderr


def _git_project(path: Path, name: str, version: str) -> None:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        f'[project]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    dependencies = (
        "".join(f"Requires-Dist: {member}>=0.5.2\n" for member, _ in update.FRAMEWORK_MEMBERS)
        if name == "lisa-cli"
        else ""
    )
    normalized = name.replace("-", "_")
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{dependencies}"
    wheel_metadata = (
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    entrypoints = "[console_scripts]\nlisa = lisa_cli:main\n"
    backend = f"""\
from pathlib import Path
import zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = {normalized!r}
    version = {version!r}
    dist = f"{{name}}-{{version}}.dist-info"
    filename = f"{{name}}-{{version}}-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w") as archive:
        archive.writestr(f"{{name}}.py", "def main(): print('fixture')\\n")
        archive.writestr(f"{{dist}}/METADATA", {metadata!r})
        archive.writestr(f"{{dist}}/WHEEL", {wheel_metadata!r})
        archive.writestr(f"{{dist}}/RECORD", "")
        if name == "lisa_cli":
            archive.writestr(f"{{dist}}/entry_points.txt", {entrypoints!r})
    return filename
"""
    (path / "backend.py").write_text(backend, encoding="utf-8")


def test_real_uv_local_source_in_path_with_spaces(
    uv_home: tuple[Path, Path], tmp_path: Path
) -> None:
    tool_dir, _ = uv_home
    repo = tmp_path / "checkout with spaces"
    _git_project(repo / "hosts" / "cli", "lisa-cli", "1.1.0")
    for name, subdirectory in update.FRAMEWORK_MEMBERS:
        _git_project(repo / subdirectory, name, "0.5.3")
    update.install_tool("local", str(repo), str(repo), update.Selection(), "uv")
    assert _version(tool_dir, "lisa-cli") == "1.1.0"
    assert all(_version(tool_dir, name) == "0.5.3" for name, _ in update.FRAMEWORK_MEMBERS)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_real_uv_git_refs_and_return_to_releases(
    uv_home: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_dir, _ = uv_home
    repo = tmp_path / "repo"
    _git_project(repo / "hosts" / "cli", "lisa-cli", "1.1.0")
    for name, subdirectory in update.FRAMEWORK_MEMBERS:
        _git_project(repo / subdirectory, name, "0.5.3")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "Fixture\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "branch", "feature/fix#1"], check=True)
    monkeypatch.setattr(update, "CLI_REPO_URL", repo.as_uri())
    monkeypatch.setattr(update, "FRAMEWORK_REPO_URL", repo.as_uri())
    _uv("tool", "install", "--no-sources", "lisa-cli[prompt]")

    _run(tool_dir, update.Policy("ref", "feature/fix#1"), update.Policy("ref", "feature/fix#1"))
    assert _version(tool_dir, "lisa-cli") == "1.1.0"
    assert all(_version(tool_dir, name) == "0.5.3" for name, _ in update.FRAMEWORK_MEMBERS)
    assert all(
        "rev=feature%2Ffix%231" in entry["git"]
        for entry in _requirements(tool_dir)
        if entry["name"] in update._FRAMEWORK_NAMES | {"lisa-cli"}
    )
    _run(tool_dir, update.Policy("latest"), update.Policy("latest"))
    assert _version(tool_dir, "lisa-cli") == "1.2.0"
    assert all(_version(tool_dir, name) == "0.6.0" for name, _ in update.FRAMEWORK_MEMBERS)

    update.install_tool(
        "pypi",
        "",
        "",
        update.Selection(
            update.Policy("ref", "feature/fix#1"), update.Policy("ref", "feature/fix#1")
        ),
        "uv",
    )
    assert _version(tool_dir, "lisa-cli") == "1.1.0"
    assert all(_version(tool_dir, name) == "0.5.3" for name, _ in update.FRAMEWORK_MEMBERS)
