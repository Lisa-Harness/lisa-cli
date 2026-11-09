"""Version policy and uv receipt regression tests."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from hosts.cli import update

RECEIPT = """\
[tool]
requirements = [
    { name = "lisa-cli", extras = ["prompt"], directory = "/old/cli" },
    { name = "lisa-harness-runtime", git = "https://example.test/repo?rev=old" },
    { name = "unrelated", extras = ["extra"], specifier = ">=1" },
]
constraints = [
    { name = "lisa-cli", specifier = "==0.2.0" },
    { name = "lisa-harness-sdk", specifier = "<0.6" },
    { name = "unrelated", specifier = "<3" },
]
overrides = [
    { name = "lisa-pack-core", specifier = "==0.5.0" },
    { name = "other", specifier = "==2" },
]
python = "3.12"
entrypoints = [{ name = "lisa", install-path = "/bin/lisa", from = "lisa-cli" }]

[tool.options]
no-sources = true
find-links = ["https://example.test/packages"]
"""


def _tool(text: str) -> dict:
    return tomllib.loads(text)["tool"]


def _requirement(tool: dict, name: str) -> dict:
    return next(entry for entry in tool["requirements"] if entry["name"] == name)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (update.Policy("version", "0.3.0"), "==0.3.0"),
        (update.Policy("pin", "1.0rc1"), "==1.0rc1"),
        (update.Policy("channel", "0.5"), ">=0.5,<0.6"),
        (update.Policy("latest"), ""),
    ],
)
def test_specifier(policy: update.Policy, expected: str) -> None:
    assert update.specifier(policy) == expected


@pytest.mark.parametrize(
    ("mode", "value"),
    [
        ("bad", ""),
        ("keep", "0.3.0"),
        ("latest", "0.3.0"),
        ("pin", ">=0.3"),
        ("version", "0.3,!=0.3.1"),
        ("pin", "0.3.0garbage"),
        ("pin", "0.3..1"),
        ("version", "1.0+build..1"),
        ("channel", "0.3.1"),
        ("channel", "abc"),
        ("ref", ""),
        ("ref", "branch name"),
    ],
)
def test_invalid_policies(mode: str, value: str) -> None:
    with pytest.raises(ValueError):
        update.Policy(mode, value)


def test_rejects_multiple_selections() -> None:
    with pytest.raises(ValueError, match="only one"):
        update.choose_policy(version="0.3.0", pin="0.3.0")
    assert update.choose_policy(latest=True) == update.Policy("latest")


@pytest.mark.parametrize(
    "version",
    ["0", "v1.2.3", "1!2.0rc1", "1.0-2", "1.0.post1", "1.0.dev2+build.5"],
)
def test_valid_release_versions(version: str) -> None:
    assert update.Policy("pin", version).value == version


def test_rewrite_preserves_unrelated_requirements_and_options() -> None:
    selection = update.Selection(update.Policy("pin", "0.3.1"), update.Policy("channel", "0.5"))
    tool = _tool(update.rewrite_receipt(RECEIPT, selection))

    assert _requirement(tool, "lisa-cli") == {
        "name": "lisa-cli",
        "extras": ["prompt"],
        "specifier": "==0.3.1",
    }
    assert _requirement(tool, "unrelated") == {
        "name": "unrelated",
        "extras": ["extra"],
        "specifier": ">=1",
    }
    for name, _ in update.FRAMEWORK_MEMBERS:
        assert _requirement(tool, name)["specifier"] == ">=0.5,<0.6"
    assert tool["constraints"] == [{"name": "unrelated", "specifier": "<3"}]
    assert tool["overrides"] == [{"name": "other", "specifier": "==2"}]
    assert tool["options"] == {
        "no-sources": True,
        "find-links": ["https://example.test/packages"],
    }
    assert tool["python"] == "3.12"
    assert tool["entrypoints"][0]["install-path"] == "/bin/lisa"


def test_git_refs_are_escaped_and_applied_to_every_framework_member() -> None:
    ref = "feature/fix#1"
    selection = update.Selection(update.Policy("ref", ref), update.Policy("ref", ref))
    tool = _tool(update.rewrite_receipt(RECEIPT, selection))
    cli_url = _requirement(tool, "lisa-cli")["git"]
    assert "subdirectory=hosts%2Fcli" in cli_url
    assert "rev=feature%2Ffix%231" in cli_url
    for name, subdirectory in update.FRAMEWORK_MEMBERS:
        url = _requirement(tool, name)["git"]
        assert url.startswith(update.FRAMEWORK_REPO_URL + "?")
        assert f"subdirectory={subdirectory.replace('/', '%2F')}" in url
        assert "rev=feature%2Ffix%231" in url


def test_latest_removes_selected_sources_and_constraints_only() -> None:
    tool = _tool(
        update.rewrite_receipt(
            RECEIPT, update.Selection(update.Policy("latest"), update.Policy("latest"))
        )
    )
    assert _requirement(tool, "lisa-cli") == {"name": "lisa-cli", "extras": ["prompt"]}
    assert _requirement(tool, "lisa-harness-runtime") == {"name": "lisa-harness-runtime"}
    assert len(tool["requirements"]) == 3
    assert tool["constraints"] == [{"name": "unrelated", "specifier": "<3"}]
    assert tool["overrides"] == [{"name": "other", "specifier": "==2"}]


def test_changing_only_framework_preserves_cli_and_its_constraints() -> None:
    tool = _tool(
        update.rewrite_receipt(RECEIPT, update.Selection(framework=update.Policy("pin", "0.5.2")))
    )
    assert _requirement(tool, "lisa-cli")["directory"] == "/old/cli"
    assert {"name": "lisa-cli", "specifier": "==0.2.0"} in tool["constraints"]
    assert _requirement(tool, "lisa-harness-runtime")["specifier"] == "==0.5.2"


@pytest.mark.parametrize(
    "text",
    [
        "not toml",
        "[other]\nrequirements=[]\n",
        "[tool]\nrequirements=[]\n",
        '[tool]\nrequirements=[{name="lisa-cli"},{name="lisa_cli"}]\n',
        '[tool]\nrequirements=[{name="lisa-cli"}]\nconstraints="bad"\n',
        '[tool]\nrequirements=["lisa-cli"]\n',
        '[tool]\nrequirements=[{name="lisa-cli"},{name="lisa-pack-core"},{name="lisa_pack_core"}]\n',
    ],
)
def test_invalid_receipts_are_rejected(text: str) -> None:
    with pytest.raises(update.UpdateError):
        update.rewrite_receipt(
            text, update.Selection(update.Policy("latest"), update.Policy("latest"))
        )


def test_rewrite_handles_quotes_comments_and_inline_arrays() -> None:
    text = """\
[tool]
requirements = [{name="lisa-cli", extras=["x]", "y"]}, # comment with ]
{name="unrelated", url="https://example.test/#part"}]
python = "3.12"
"""
    tool = _tool(update.rewrite_receipt(text, update.Selection(update.Policy("pin", "0.3.1"))))
    assert _requirement(tool, "lisa-cli")["extras"] == ["x]", "y"]
    assert _requirement(tool, "unrelated")["url"] == "https://example.test/#part"


def test_receipt_serialization_round_trips_supported_values() -> None:
    value = {
        "enabled": True,
        "count": 12,
        "ratio": 1.5,
        "extras": ["a", "b"],
        "other": False,
        "key\U0001f98a": "/Users/\U0001f98a/pack",
    }
    assert tomllib.loads("value = " + update._toml_value(value))["value"] == value
    with pytest.raises(update.UpdateError, match="non-string"):
        update._toml_value({1: "bad"})
    with pytest.raises(update.UpdateError, match="Unsupported"):
        update._toml_value(None)
    with pytest.raises(update.UpdateError, match="without a name"):
        update._entry_name({"name": 1})


def test_receipt_array_parser_handles_escapes_and_errors() -> None:
    text = r"""[{name="a\"b", extras=["x]"]}, {name='c'}]"""
    assert update._array_end(text, 0) == len(text)
    assert update.rewrite_receipt(RECEIPT, update.Selection()) == RECEIPT
    with pytest.raises(update.UpdateError, match="end of an array"):
        update._array_end('["incomplete', 0)
    with pytest.raises(update.UpdateError, match=r"\[tool\]"):
        update._replace_array("[other]\nrequirements=[]\n", "requirements", [])
    with pytest.raises(update.UpdateError, match="Cannot locate requirements"):
        update._replace_array("[tool]\nother=[]\n", "requirements", [])
    with pytest.raises(update.UpdateError, match="unsupported"):
        update._replace_array("[tool]\nrequirements=1\n", "requirements", [])


def test_receipt_rewrite_rejects_changes_to_unrelated_settings() -> None:
    original = update._replace_array

    def alter_other_setting(text: str, key: str, entries: list[dict[str, object]]) -> str:
        return original(text, key, entries).replace('python = "3.12"', 'python = "3.11"')

    with (
        patch.object(update, "_replace_array", side_effect=alter_other_setting),
        pytest.raises(update.UpdateError, match="without changing other settings"),
    ):
        update.rewrite_receipt(RECEIPT, update.Selection(update.Policy("latest")))


def test_receipt_rewrite_preserves_unicode_paths_and_reports_invalid_output() -> None:
    original = RECEIPT.replace(
        'name = "unrelated", extras', 'name = "unrelated", path = "/Users/\U0001f98a", extras'
    )
    rewritten = update.rewrite_receipt(original, update.Selection(update.Policy("latest")))
    assert _requirement(_tool(rewritten), "unrelated")["path"] == "/Users/\U0001f98a"
    with (
        patch.object(update, "_replace_array", return_value="[tool]\nrequirements = ["),
        pytest.raises(update.UpdateError, match="valid TOML"),
    ):
        update.rewrite_receipt(original, update.Selection(update.Policy("latest")))


def test_temporary_selection_restores_latest_tracking_after_success(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    tool_dir.mkdir(parents=True)
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    selection = update.Selection(
        update.Policy("version", "0.3.0"), update.Policy("version", "0.5.2")
    )

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        staged = _tool(receipt.read_text(encoding="utf-8"))
        assert _requirement(staged, "lisa-cli")["specifier"] == "==0.3.0"
        assert _requirement(staged, "lisa-pack-core")["specifier"] == "==0.5.2"
        assert kwargs["env"]["UV_TOOL_DIR"] == str(tool_dir.parent)
        return subprocess.CompletedProcess(argv, 0)

    with patch.object(update.subprocess, "run", side_effect=run):
        update.upgrade_tool(tool_dir, selection, "/bin/uv")

    final = _tool(receipt.read_text(encoding="utf-8"))
    assert _requirement(final, "lisa-cli") == {"name": "lisa-cli", "extras": ["prompt"]}
    assert _requirement(final, "lisa-harness-runtime") == {"name": "lisa-harness-runtime"}
    assert {entry["name"] for entry in final["requirements"]} == {
        "lisa-cli",
        "lisa-harness-runtime",
        "unrelated",
    }
    assert final["constraints"] == [{"name": "unrelated", "specifier": "<3"}]


def test_failure_restores_original_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    result = subprocess.CompletedProcess(["uv"], 7)
    with (
        patch.object(update.subprocess, "run", return_value=result),
        pytest.raises(update.UpdateError, match="exit 7") as exc,
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("pin", "0.3.0")), "uv")
    assert exc.value.code == 7
    assert receipt.read_text(encoding="utf-8") == RECEIPT


def test_os_error_restores_original_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    with (
        patch.object(update.subprocess, "run", side_effect=OSError("cannot launch")),
        pytest.raises(update.UpdateError, match="cannot launch"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("latest")), "uv")
    assert receipt.read_text(encoding="utf-8") == RECEIPT


def test_receipt_io_failures_are_reported(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    with pytest.raises(update.UpdateError, match="Cannot read"):
        update.upgrade_tool(tool_dir, update.Selection(), "uv")

    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_bytes(b"\xff")
    with pytest.raises(update.UpdateError, match=r"Cannot read.*utf-8"):
        update.upgrade_tool(tool_dir, update.Selection(), "uv")
    receipt.write_text(RECEIPT, encoding="utf-8")
    with (
        patch.object(update, "_write_receipt", side_effect=OSError("read only")),
        pytest.raises(update.UpdateError, match=r"Cannot update.*read only"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("latest")), "uv")

    write = update._write_receipt
    writes = 0

    def fail_restore(path: Path, text: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("restore failed")
        write(path, text)

    result = subprocess.CompletedProcess(["uv"], 1)
    with (
        patch.object(update, "_write_receipt", side_effect=fail_restore),
        patch.object(update.subprocess, "run", return_value=result),
        pytest.raises(update.UpdateError, match=r"could not be restored.*restore failed"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("latest")), "uv")


def test_failed_update_does_not_overwrite_a_concurrently_changed_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    other_receipt = '[tool]\nrequirements = [{name="lisa-cli", specifier="==0.4.0"}]\n'

    def concurrent_change(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        receipt.write_text(other_receipt, encoding="utf-8")
        return subprocess.CompletedProcess(argv, 1)

    with (
        patch.object(update.subprocess, "run", side_effect=concurrent_change),
        pytest.raises(update.UpdateError, match="receipt changed"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("pin", "0.3.0")), "uv")
    assert receipt.read_text(encoding="utf-8") == other_receipt


def test_failed_update_reports_receipt_that_disappeared(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")

    def remove_receipt(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        receipt.unlink()
        return subprocess.CompletedProcess(argv, 1)

    with (
        patch.object(update.subprocess, "run", side_effect=remove_receipt),
        pytest.raises(update.UpdateError, match=r"could not be restored.*Cannot read"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("pin", "0.3.0")), "uv")


def test_keyboard_interrupt_restores_the_original_receipt(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    with (
        patch.object(update.subprocess, "run", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("pin", "0.3.0")), "uv")
    assert receipt.read_text(encoding="utf-8") == RECEIPT


def test_temporary_cleanup_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    tool_dir = tmp_path / "lisa-cli"
    tool_dir.mkdir()
    (tool_dir / "uv-receipt.toml").write_text(RECEIPT, encoding="utf-8")
    write = update._write_receipt
    calls = 0

    def fail_cleanup(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        write(path, text)

    with (
        patch.object(update, "_write_receipt", side_effect=fail_cleanup),
        patch.object(update.subprocess, "run", return_value=subprocess.CompletedProcess(["uv"], 0)),
        pytest.raises(update.UpdateError, match="latest tracking could not be restored"),
    ):
        update.upgrade_tool(tool_dir, update.Selection(update.Policy("version", "0.3.0")), "uv")


def test_status_describes_pins_without_exposing_other_requirements() -> None:
    status = update.policy_status(RECEIPT)
    assert "CLI tracking: ==0.2.0" in status
    assert any("Framework tracking:" in line and "Git ref old" in line for line in status)
    assert all("unrelated" not in line for line in status)


def test_framework_pin_suggests_reset_without_clearing_cli_selection() -> None:
    selection = update.Selection(update.Policy("pin", "0.3.0"), update.Policy("pin", "0.5.2"))
    messages = update.success_messages(selection)
    assert "lisa update --latest" in messages[0]
    assert "lisa update --framework-latest" in messages[1]


def test_atomic_write_preserves_permissions(tmp_path: Path) -> None:
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text(RECEIPT, encoding="utf-8")
    receipt.chmod(0o600)
    updated = update.rewrite_receipt(RECEIPT, update.Selection(update.Policy("latest")))
    update._write_receipt(receipt, updated)
    assert receipt.read_text(encoding="utf-8") == updated
    if os.name != "nt":
        assert receipt.stat().st_mode & 0o777 == 0o600


def test_windows_command_is_copyable_and_quotes_paths_and_refs(tmp_path: Path) -> None:
    selection = update.Selection(update.Policy("ref", "feature/jane's-fix"))
    argv = update.standalone_command(
        tmp_path / "tools space" / "lisa-cli", selection, "uv", force=True
    )
    command = update.format_command(argv, windows=True)
    assert command.startswith("& 'uv' 'run'")
    assert "'feature/jane''s-fix'" in command
    assert "'--force'" in command


def test_installer_source_selection_and_local_paths(tmp_path: Path) -> None:
    framework = tmp_path / "lisa"
    cli = tmp_path / "lisa-cli"
    framework.mkdir()
    cli.mkdir()
    assert update._install_cli("pypi", "", "", update.Policy()) == "lisa-cli[prompt]"
    assert "git+https://" in update._install_cli("git", "", "", update.Policy())
    assert str(cli / "hosts" / "cli") in update._install_cli(
        "local", "", str(framework), update.Policy()
    )
    local = update._install_framework("local", str(framework), update.Policy())
    git = update._install_framework("git", "", update.Policy())
    assert len(local) == len(git) == 16
    assert all(str(framework) in item for item in local[1::2])
    assert all("git+https://" in item for item in git[1::2])
    assert update._install_framework("git", "", update.Policy("latest")) == []
    with pytest.raises(update.UpdateError, match="checkout not found"):
        update._install_cli("local", "", "", update.Policy())
    with pytest.raises(update.UpdateError, match="Unknown install source"):
        update.install_tool("invalid", "", "", update.Selection(), "uv")


def test_installer_retry_and_failure_paths() -> None:
    fail = subprocess.CompletedProcess(["uv"], 4)
    success = subprocess.CompletedProcess(["uv"], 0)
    with patch.object(update.subprocess, "run", side_effect=[fail, success]) as run:
        update.install_tool("pypi", "", "", update.Selection(update.Policy("pin", "0.3.0")), "uv")
    assert "--no-cache" in run.call_args_list[1].args[0]
    with (
        patch.object(update.subprocess, "run", side_effect=[fail, fail]),
        pytest.raises(update.UpdateError, match=r"install failed.*4"),
    ):
        update.install_tool("pypi", "", "", update.Selection(), "uv")
    missing_dir = subprocess.CompletedProcess(["uv"], 1, stdout="", stderr="failed")
    with (
        patch.object(update.subprocess, "run", side_effect=[success, missing_dir]),
        pytest.raises(update.UpdateError, match="uv tool dir failed"),
    ):
        update.install_tool(
            "pypi", "", "", update.Selection(update.Policy("version", "0.3.0")), "uv"
        )


def test_standalone_update_and_install_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool_dir = tmp_path / "tools" / "lisa-cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update.py",
            "update",
            "--tool-dir",
            str(tool_dir),
            "--cli-mode",
            "pin",
            "--cli-value",
            "0.3.0",
            "--force",
        ],
    )
    with patch.object(update, "upgrade_tool") as upgrade:
        assert update._standalone_main() == 0
    upgrade.assert_called_once_with(
        tool_dir, update.Selection(update.Policy("pin", "0.3.0")), "uv", force=True
    )
    assert "tracking pin 0.3.0" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["update.py", "install", "--framework-mode", "latest"])
    with patch.object(update, "install_tool") as install:
        assert update._standalone_main() == 0
    install.assert_called_once_with(
        "pypi", "", "", update.Selection(framework=update.Policy("latest")), "uv"
    )
    monkeypatch.setattr(sys, "argv", ["update.py", "validate"])
    assert update._standalone_main() == 0


def test_standalone_reports_validation_and_uv_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["update.py", "validate", "--cli-mode", "pin"])
    assert update._standalone_main() == 1
    assert "Invalid release version" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["update.py", "update"])
    with pytest.raises(SystemExit) as exc:
        update._standalone_main()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", ["update.py", "update", "--tool-dir", str(tmp_path)])
    with patch.object(update, "upgrade_tool", side_effect=update.UpdateError("failed", 7)):
        assert update._standalone_main() == 7
    assert "failed" in capsys.readouterr().err


def test_standalone_script_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(update.__file__), "validate"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(update.__file__), run_name="__main__")
    assert exc.value.code == 0
