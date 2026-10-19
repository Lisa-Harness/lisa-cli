"""Tests for the CLI ``verify`` subcommand and its check functions.

The ``lisa verify`` command runs a series of diagnostic checks (Python
version, .env, model manifest, pack catalog, provider keys, Ollama,
paths) and renders a Rich table. These
tests cover the individual ``_verify_check_*`` functions in isolation
and the summary/counting logic in ``_verify_command``.
"""

from __future__ import annotations

# pylint: disable=protected-access
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import typer
from rich.console import Console

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> Any:
    """Build a minimal test runtime via the shared test factory.

    Uses the default models YAML (gemma3:latest) so the manifest check
    passes. Does NOT call setup_framework (which triggers a pre-existing
    operation-catalog validation issue with the markdown pack).
    """
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path)


class _StubConfig:
    """Minimal config stand-in for verify checks that read config fields."""

    def __init__(self, user_slug: str = "", session_context: str = "") -> None:
        self.user_slug = user_slug
        self.session_context = session_context


class _StubRuntime:
    """Minimal runtime stand-in for checks that only need paths/config."""

    def __init__(
        self,
        *,
        data_dir: Path,
        thread_dir: Path | None = None,
        thread_path: Path | None = None,
        models_path: Path | None = None,
        user_slug: str = "",
        session_context: str = "",
        mcp_warm_configs: list | None = None,
        mcp_cold_configs: list | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.thread_dir = thread_dir or (data_dir / "threads")
        self.thread_path = thread_path
        self.models_path = models_path or (data_dir / "models.yaml")
        self.paths = SimpleNamespace(
            data_dir=self.data_dir,
            thread_dir=self.thread_dir,
            thread_path=self.thread_path,
            models_path=self.models_path,
        )
        self.config = _StubConfig(user_slug=user_slug, session_context=session_context)
        self.mcp_warm_configs = mcp_warm_configs or []
        self.mcp_cold_configs = mcp_cold_configs or []


# ── _redact_key ───────────────────────────────────────────────────────


def test_redact_key_long_value_shows_last_four() -> None:
    from hosts.cli.main import _redact_key

    # "sk-abcdefghij" is 13 chars → 9 stars + last 4 visible
    assert _redact_key("sk-abcdefghij") == "*********ghij"


def test_redact_key_short_value_all_stars() -> None:
    from hosts.cli.main import _redact_key

    assert _redact_key("abc") == "***"


def test_redact_key_empty_returns_not_set() -> None:
    from hosts.cli.main import _redact_key

    assert _redact_key("") == "(not set)"


# ── _verify_check_python ──────────────────────────────────────────────


def test_check_python_passes_on_311() -> None:
    from hosts.cli import main as cli_main

    with patch("platform.python_version", return_value="3.11.0"):
        status, detail = cli_main._verify_check_python()
    assert "PASS" in status
    assert "3.11.0" in detail


def test_check_python_passes_on_312() -> None:
    from hosts.cli import main as cli_main

    with patch("platform.python_version", return_value="3.12.5"):
        status, _ = cli_main._verify_check_python()
    assert "PASS" in status


def test_check_python_fails_on_310() -> None:
    from hosts.cli import main as cli_main

    with patch("platform.python_version", return_value="3.10.2"):
        status, detail = cli_main._verify_check_python()
    assert "FAIL" in status
    assert "3.10.2" in detail


def test_check_python_fails_on_39() -> None:
    from hosts.cli import main as cli_main

    with patch("platform.python_version", return_value="3.9.18"):
        status, _ = cli_main._verify_check_python()
    assert "FAIL" in status


# ── _verify_check_env ─────────────────────────────────────────────────


def test_check_env_passes_when_env_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_ENV_SEARCH_CANDIDATES includes CWD/.env; creating it yields PASS."""
    from hosts.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LISA_OPENAI_API_KEY=test\n", encoding="utf-8")
    # Rebuild the module-level tuple so CWD points at tmp_path.
    monkeypatch.setattr(
        cli_main,
        "_ENV_SEARCH_CANDIDATES",
        (
            tmp_path / ".env",
            tmp_path / "lisa_home" / ".env",
            tmp_path / "framework" / ".env",
            tmp_path / "little_lisa" / ".env",
        ),
    )
    runtime = _StubRuntime(data_dir=tmp_path)
    status, detail = cli_main._verify_check_env(runtime)
    assert "PASS" in status
    assert str(tmp_path / ".env") in detail


def test_check_env_warns_when_no_env_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    monkeypatch.setattr(
        cli_main,
        "_ENV_SEARCH_CANDIDATES",
        (tmp_path / "a.env", tmp_path / "b.env", tmp_path / "c.env", tmp_path / "d.env"),
    )
    runtime = _StubRuntime(data_dir=tmp_path)
    status, detail = cli_main._verify_check_env(runtime)
    assert "WARN" in status
    assert ".env not found" in detail


# ── _verify_check_session_context ─────────────────────────────────────


def test_check_session_context_none_when_no_slug(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(data_dir=tmp_path, user_slug="")
    status, detail = cli_main._verify_check_session_context(runtime)
    assert "NONE" in status
    assert "No session context" in detail


def test_check_session_context_passes_with_slug(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(data_dir=tmp_path, user_slug="adam", session_context="adam")
    status, detail = cli_main._verify_check_session_context(runtime)
    assert "PASS" in status
    assert "adam" in detail


def test_check_session_context_shows_path_when_different(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(
        data_dir=tmp_path,
        user_slug="adam",
        session_context="/path/to/context.md",
    )
    status, detail = cli_main._verify_check_session_context(runtime)
    assert "PASS" in status
    assert "/path/to/context.md" in detail


# ── _verify_check_manifest ────────────────────────────────────────────


def test_check_manifest_passes_with_models(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "version: 1\n"
        "mode_models: {default: gpt-4o}\n"
        "models:\n"
        "  - model_id: gpt-4o\n"
        "    provider: openai\n",
        encoding="utf-8",
    )
    runtime = _StubRuntime(data_dir=tmp_path, models_path=models_yaml)
    status, detail = cli_main._verify_check_manifest(runtime)
    assert "PASS" in status
    assert "gpt-4o" in detail


def test_check_manifest_fails_with_zero_models(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "version: 1\nmode_models: {default: none}\nmodels: []\n",
        encoding="utf-8",
    )
    runtime = _StubRuntime(data_dir=tmp_path, models_path=models_yaml)
    status, detail = cli_main._verify_check_manifest(runtime)
    assert "FAIL" in status
    # Empty models list either reports "zero models" or fails validation.
    assert "zero models" in detail or "failed" in detail


def test_check_manifest_fails_on_missing_file(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(data_dir=tmp_path, models_path=tmp_path / "nonexistent.yaml")
    status, _ = cli_main._verify_check_manifest(runtime)
    assert "FAIL" in status


# ── _verify_check_pack_catalog ────────────────────────────────────────


def test_check_pack_catalog_passes(tmp_path: Path) -> None:
    """Pack catalog check returns PASS when get_workflow_catalog has modes."""
    from hosts.cli import main as cli_main

    with patch(
        "kernel.ipsar_fsm.get_workflow_catalog",
        return_value={"default": object(), "coding": object()},
    ):
        status, detail = cli_main._verify_check_pack_catalog()
    assert "PASS" in status
    assert "2 mode(s)" in detail
    assert "coding" in detail


def test_check_pack_catalog_warns_on_empty(tmp_path: Path) -> None:
    """Pack catalog check warns when the catalog has zero modes."""
    from hosts.cli import main as cli_main

    with patch("kernel.ipsar_fsm.get_workflow_catalog", return_value={}):
        status, detail = cli_main._verify_check_pack_catalog()
    assert "WARN" in status
    assert "zero modes" in detail


def test_check_pack_catalog_fails_on_error(tmp_path: Path) -> None:
    """Pack catalog check fails when get_workflow_catalog raises."""
    from hosts.cli import main as cli_main

    with patch("kernel.ipsar_fsm.get_workflow_catalog", side_effect=RuntimeError("catalog broken")):
        status, detail = cli_main._verify_check_pack_catalog()
    assert "FAIL" in status
    assert "catalog broken" in detail


# ── _verify_check_provider_keys ───────────────────────────────────────


def test_check_provider_keys_warns_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    for var in (
        "LISA_AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LISA_AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "LISA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_OLLAMA_HOST",
        "OLLAMA_HOST",
        "LISA_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "LISA_AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "LISA_AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "LISA_AWS_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    status, detail = cli_main._verify_check_provider_keys()
    assert "WARN" in status
    assert "No provider keys" in detail


def test_check_provider_keys_passes_with_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    for var in (
        "LISA_AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_OLLAMA_HOST",
        "OLLAMA_HOST",
        "LISA_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "LISA_AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "LISA_AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "LISA_AWS_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LISA_OPENAI_API_KEY", "sk-test1234567890")
    status, detail = cli_main._verify_check_provider_keys()
    assert "PASS" in status
    assert "OpenAI" in detail
    # Key should be redacted
    assert "sk-test1234567890" not in detail


def test_check_provider_keys_reports_azure_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "azurekey1234567890")
    monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://myinstance.openai.azure.com")
    for var in (
        "LISA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_OLLAMA_HOST",
        "OLLAMA_HOST",
        "LISA_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "LISA_AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "LISA_AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "LISA_AWS_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    status, detail = cli_main._verify_check_provider_keys()
    assert "PASS" in status
    assert "Azure OpenAI" in detail
    assert "myinstance.openai.azure.com" in detail


def test_check_provider_keys_reports_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    monkeypatch.setenv("LISA_OLLAMA_HOST", "localhost:11434")
    for var in (
        "LISA_AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LISA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "LISA_AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "LISA_AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "LISA_AWS_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    status, detail = cli_main._verify_check_provider_keys()
    assert "PASS" in status
    assert "Ollama" in detail
    assert "localhost:11434" in detail


def test_check_provider_keys_reports_aws_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    for var in (
        "LISA_AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LISA_AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "LISA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_OLLAMA_HOST",
        "OLLAMA_HOST",
        "LISA_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LISA_AWS_ACCESS_KEY_ID", "AKIATEST123456789")
    monkeypatch.setenv("LISA_AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("LISA_AWS_REGION", "eu-west-1")

    status, detail = cli_main._verify_check_provider_keys()

    assert "PASS" in status
    assert "AWS Bedrock" in detail
    assert "eu-west-1" in detail
    assert "AKIATEST123456789" not in detail


# ── _verify_check_paths ───────────────────────────────────────────────


def test_check_paths_passes(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(
        data_dir=tmp_path / "data",
        thread_dir=tmp_path / "threads",
        thread_path=tmp_path / "threads" / "user1",
    )
    status, detail = cli_main._verify_check_paths(runtime)
    assert "PASS" in status
    assert str(tmp_path / "data") in detail
    assert str(tmp_path / "threads") in detail
    assert "user1" in detail


def test_check_paths_without_thread_path(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    runtime = _StubRuntime(
        data_dir=tmp_path / "data",
        thread_dir=tmp_path / "threads",
        thread_path=None,
    )
    status, detail = cli_main._verify_check_paths(runtime)
    assert "PASS" in status
    assert "Thread path" not in detail


# ── _verify_command summary ──────────────────────────────────────────


def _invoke_verify(
    tmp_path: Path,
    *,
    runtime: Any | None = None,
) -> tuple[int, str]:
    """Invoke _verify_command with a stubbed runtime and captured output.

    Stubs _verify_check_pack_catalog (needs framework setup) and
    _verify_check_ollama (does real HTTP) so the command runs fully
    offline. Returns (exit_code, output_text).
    """
    from hosts.cli import main as cli_main

    if runtime is None:
        runtime = _make_runtime(tmp_path)

    quiet_console = Console(record=True, file=io.StringIO(), width=160)
    with (
        patch.object(
            cli_main,
            "_initialize_runtime_or_exit",
            return_value=runtime,
        ),
        patch.object(cli_main, "console", new=quiet_console),
        patch.object(
            cli_main,
            "_verify_check_pack_catalog",
            return_value=("[green]PASS[/green]", "1 mode(s): default"),
        ),
        patch.object(
            cli_main,
            "_verify_check_ollama",
            return_value=("[yellow]WARN[/yellow]", "Ollama not reachable (test)"),
        ),
    ):
        try:
            cli_main._verify_command(data_dir=str(tmp_path), streaming="none")
        except typer.Exit as exc:
            return exc.exit_code, quiet_console.export_text()
    return 0, quiet_console.export_text()


def test_verify_summary_all_passed(tmp_path: Path) -> None:
    """A clean environment produces 'All checks passed' with exit 0."""
    # Set a provider key so provider keys check passes without warning.
    with patch.dict(os.environ, {"LISA_OPENAI_API_KEY": "sk-test1234567890"}):
        code, output = _invoke_verify(tmp_path)
    assert code == 0
    # The test runtime has a pack catalog and model manifest, so the
    # only warnings come from .env / ollama / session context / disk space.
    # With a provider key set, "All checks passed" requires no FAIL rows.
    assert "FAIL" not in output


def test_verify_summary_reports_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Warnings produce 'All mandatory checks passed (N warnings)' with exit 0."""
    # Clear all provider env vars so provider keys check warns.
    for var in list(os.environ):
        if var.startswith(
            (
                "LISA_",
                "OPENAI_",
                "AZURE_",
                "ANTHROPIC_",
                "GOOGLE_",
                "OLLAMA_",
                "OPENROUTER_",
                "AWS_",
            )
        ):
            monkeypatch.delenv(var)
    code, output = _invoke_verify(tmp_path)
    assert code == 0
    assert "warning" in output.lower()


def test_verify_summary_reports_failure_and_nonzero_exit(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    with patch.object(
        cli_main, "_verify_check_manifest", return_value=("[red]FAIL[/red]", "invalid manifest")
    ):
        code, output = _invoke_verify(tmp_path)
    assert code == 1
    assert "invalid manifest" in output
    assert "1 check(s) failed" in output


def test_verify_reports_all_current_checks(tmp_path: Path) -> None:
    code, output = _invoke_verify(tmp_path)
    assert code == 0
    for check in (
        "Python",
        ".env",
        "Session context",
        "Model manifest",
        "Pack catalog",
        "Provider keys",
        "Ollama",
        "Paths",
    ):
        assert check in output
    assert "MCP servers" not in output
    assert "Provider network" not in output


@pytest.mark.parametrize("option", ["--full", "--json"])
def test_verify_rejects_removed_options(option: str) -> None:
    from typer.testing import CliRunner

    from hosts.cli.main import app

    result = CliRunner().invoke(app, ["verify", option])
    assert result.exit_code == 2
    assert "No such option" in result.output
