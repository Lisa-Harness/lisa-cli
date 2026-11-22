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
    """Build a manifest-backed runtime without starting framework services."""
    from runtime.testing import build_test_runtime

    return build_test_runtime(
        tmp_path,
        models_yaml=(
            "version: 1\n"
            "mode_models: {default: gpt-4o}\n"
            "models:\n"
            "  - model_id: gpt-4o\n"
            "    provider: openai\n"
        ),
    )


def _make_manifest(provider: str, model_id: str = "selected-model") -> Any:
    """Build a typed selected-model manifest for provider-specific checks."""
    from runtime.model_manifest import load_model_manifest_text

    return load_model_manifest_text(
        "version: 1\n"
        f"mode_models: {{default: {model_id}}}\n"
        "models:\n"
        f"  - model_id: {model_id}\n"
        f"    provider: {provider}\n"
    )


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


def test_check_python_passes_on_future_major() -> None:
    from hosts.cli import main as cli_main

    with patch("platform.python_version", return_value="4.0.0"):
        status, _ = cli_main._verify_check_python()
    assert "PASS" in status


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


def test_check_env_ignores_env_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CWD/.env is untrusted input, not an implicit configuration source."""
    from hosts.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LISA_OPENAI_API_KEY=test\n", encoding="utf-8")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PROFILE", raising=False)
    monkeypatch.delenv("LISA_PROFILES_DIR", raising=False)
    with (
        patch.object(cli_main, "_mirror_default_data_dir", lambda: tmp_path / "lisa_home"),
        patch.object(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework"),
    ):
        runtime = _StubRuntime(data_dir=tmp_path)
        status, detail = cli_main._verify_check_env(runtime)
    assert "SKIP" in status
    assert str(tmp_path / ".env") not in detail


def test_check_env_warns_when_no_env_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    # Point every dynamic candidate at empty scratch dirs: CWD via chdir,
    # the data-dir/env-dir and framework slots via _resolve_env_candidates
    # patches (the framework .env path must not exist for WARN).
    monkeypatch.chdir(tmp_path)
    fake_profiles_dir = tmp_path / "profiles"
    with (
        patch.object(cli_main, "_mirror_default_data_dir", lambda: tmp_path / "lisa_home"),
        patch.object(
            cli_main,
            "_mirror_profile_data_dir",
            lambda profile, base_dir=None: fake_profiles_dir / profile,
        ),
        patch.object(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework"),
    ):
        monkeypatch.delenv("LISA_DATA_DIR", raising=False)
        monkeypatch.delenv("LISA_PROFILE", raising=False)
        monkeypatch.delenv("LISA_PROFILES_DIR", raising=False)
        runtime = _StubRuntime(data_dir=tmp_path)
        status, detail = cli_main._verify_check_env(runtime)
    assert "SKIP" in status
    assert "process configuration" in detail


def test_check_env_falls_back_to_base_env_for_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile without its own .env still finds ~/.lisa/.env → PASS."""
    from hosts.cli import main as cli_main

    base_dir = tmp_path / "lisa_home"
    base_dir.mkdir()
    (base_dir / ".env").write_text("LISA_OPENAI_API_KEY=test\n", encoding="utf-8")
    # Route the module's dir resolvers at the scratch tree (env vars alone
    # would still touch the real ~/.lisa; the loader ignores LISA_PROFILES_DIR
    # because _mirror_profile_data_dir is patched to live under tmp_path).
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(cli_main, "_mirror_default_data_dir", lambda: base_dir),
        patch.object(
            cli_main,
            "_mirror_profile_data_dir",
            lambda profile, base_dir=None: tmp_path / "profiles" / profile,
        ),
        patch.object(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework"),
    ):
        runtime = _StubRuntime(data_dir=base_dir)
        status, detail = cli_main._verify_check_env(runtime)
    assert "PASS" in status
    assert str(base_dir / ".env") in detail


def test_check_env_profile_env_beats_base_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile's own .env wins over the base dir fallback in check order."""
    from hosts.cli import main as cli_main

    base_dir = tmp_path / "lisa_home"
    base_dir.mkdir()
    (base_dir / ".env").write_text("BASE=1\n", encoding="utf-8")
    profile_dir = tmp_path / "profiles" / "taxes"
    profile_dir.mkdir(parents=True)
    (profile_dir / ".env").write_text("SHARED=profile\n", encoding="utf-8")
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(cli_main, "_mirror_default_data_dir", lambda: base_dir),
        patch.object(
            cli_main,
            "_mirror_profile_data_dir",
            lambda profile, base_dir=None: tmp_path / "profiles" / profile,
        ),
        patch.object(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework"),
    ):
        runtime = _StubRuntime(data_dir=base_dir)
        loaded = cli_main._load_env_candidates(cli_main._resolve_env_candidates())
        status, detail = cli_main._verify_check_env(runtime)
    assert loaded == profile_dir / ".env"
    assert os.environ["SHARED"] == "profile"
    assert os.environ["BASE"] == "1"
    assert "PASS" in status
    assert str(profile_dir / ".env") in detail


def test_check_env_data_dir_override_skips_redundant_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LISA_DATA_DIR set: the base dir fallback stays in the order.

    The override replaces the profile slot but the base dir's .env (the
    profile fallback slot) remains, pointing at the real platform default
    (~/.lisa) — distinct from the override dir, so it is a real fallback.
    """
    from hosts.cli import main as cli_main

    data_dir = tmp_path / "pinned"
    monkeypatch.setenv("LISA_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)
    fake_base_dir = tmp_path / "lisa_home"

    with (
        patch.object(cli_main, "_mirror_default_data_dir", lambda: fake_base_dir),
        patch.object(cli_main, "_FRAMEWORK_ROOT", tmp_path / "framework"),
    ):
        candidates = cli_main._resolve_env_candidates()
    assert candidates == (
        fake_base_dir / ".env",
        data_dir / ".env",
        tmp_path / "framework" / ".env",
    )


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


# ── selected-provider configuration and Ollama ───────────────────────


def _runtime_for_manifest(manifest: Any) -> Any:
    """Return a stub runtime whose active manifest is already resolved."""
    runtime = _StubRuntime(data_dir=Path("/tmp/lisa-verify"))
    runtime.session = SimpleNamespace(manifest=manifest)
    return runtime


def test_selected_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    runtime = _runtime_for_manifest(_make_manifest("openai", "gpt-4o"))
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "FAIL" in status
    assert "LISA_OPENAI_API_KEY" in detail

    monkeypatch.setenv("LISA_OPENAI_API_KEY", "sk-test1234567890")
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "PASS" in status
    assert "sk-test1234567890" not in detail


def test_selected_azure_missing_endpoint_is_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hosts.cli import main as cli_main

    monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "azure-key")
    runtime = _runtime_for_manifest(_make_manifest("azure_openai", "gpt-4o"))
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "FAIL" in status
    assert "LISA_AZURE_OPENAI_ENDPOINT" in detail


def test_unused_partial_azure_does_not_fail_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hosts.cli import main as cli_main

    monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "azure-key")
    runtime = _runtime_for_manifest(_make_manifest("openai", "gpt-4o"))
    monkeypatch.setenv("LISA_OPENAI_API_KEY", "openai-key")
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "PASS" in status
    assert "partially configured" not in detail


def test_aws_standard_credential_chain_is_supported() -> None:
    from hosts.cli import main as cli_main

    runtime = _runtime_for_manifest(_make_manifest("aws", "bedrock-model"))
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "PASS" in status
    assert "AWS_ACCESS_KEY" not in detail


def test_model_declared_env_names_are_required() -> None:
    from runtime.model_manifest import load_model_manifest_text

    from hosts.cli import main as cli_main

    manifest = load_model_manifest_text(
        "version: 1\n"
        "mode_models: {default: custom}\n"
        "models:\n"
        "  - model_id: custom\n"
        "    provider: ollama\n"
        "    api_key_env: CUSTOM_REQUIRED_KEY\n"
        "    base_url_env: CUSTOM_REQUIRED_URL\n"
    )
    runtime = _runtime_for_manifest(manifest)
    status, detail = cli_main._verify_check_provider_keys(runtime)
    assert "FAIL" in status
    assert "CUSTOM_REQUIRED_KEY" in detail
    assert "CUSTOM_REQUIRED_URL" in detail


def test_unselected_ollama_is_skip_without_any_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hosts.cli import main as cli_main

    runtime = _runtime_for_manifest(_make_manifest("openai", "gpt-4o"))
    monkeypatch.delenv("LISA_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    status, detail = cli_main._verify_check_ollama(runtime)
    assert "SKIP" in status
    assert "not selected" in detail


def test_required_ollama_unreachable_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, _url: str) -> Any:
            raise OSError("connection refused")

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    runtime = _runtime_for_manifest(_make_manifest("ollama", "gemma3:latest"))
    status, detail = cli_main._verify_check_ollama(runtime)
    assert "FAIL" in status
    assert "Required Ollama" in detail


def test_ollama_scheme_is_not_double_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hosts.cli import main as cli_main

    seen: list[str] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "gemma3:latest"}]}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, url: str) -> Any:
            seen.append(url)
            return _Response()

    monkeypatch.setenv("LISA_OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setattr("httpx.AsyncClient", _Client)
    runtime = _runtime_for_manifest(_make_manifest("ollama", "gemma3:latest"))
    status, _ = cli_main._verify_check_ollama(runtime)
    assert "PASS" in status
    assert seen == ["http://127.0.0.1:11434/api/tags"]


def test_ollama_manifest_endpoint_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.model_manifest import load_model_manifest_text

    from hosts.cli import main as cli_main

    seen: list[str] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "gemma3:latest"}]}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, url: str) -> Any:
            seen.append(url)
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    manifest = load_model_manifest_text(
        "version: 1\n"
        "mode_models: {default: gemma3:latest}\n"
        "models:\n"
        "  - model_id: gemma3:latest\n"
        "    provider: ollama\n"
        "    base_url: http://manifest-host:11434/v1\n"
    )
    runtime = _runtime_for_manifest(manifest)
    status, _ = cli_main._verify_check_ollama(runtime)
    assert "PASS" in status
    assert seen == ["http://manifest-host:11434/api/tags"]


def test_ollama_selected_model_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosts.cli import main as cli_main

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "other-model"}]}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, _url: str) -> Any:
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    runtime = _runtime_for_manifest(_make_manifest("ollama", "gemma3:latest"))
    status, detail = cli_main._verify_check_ollama(runtime)
    assert "FAIL" in status
    assert "gemma3:latest" in detail


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

    Stubs _verify_check_pack_catalog (needs framework setup). Provider checks
    use a manifest-backed OpenAI model and a test credential; Ollama is not
    selected and therefore must be skipped. Returns (exit_code, output_text).
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


def test_verify_without_provider_keys_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing optional dotenv/Ollama and unused credentials stay neutral."""
    monkeypatch.delenv("LISA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from runtime.testing import build_test_runtime

    from hosts.cli import main as cli_main

    runtime = build_test_runtime(tmp_path)
    with patch.object(
        cli_main,
        "_verify_check_ollama",
        return_value=("[dim]SKIP[/dim]", "Ollama is not selected by the active manifest"),
    ):
        code, output = _invoke_verify(tmp_path, runtime=runtime)
    assert code == 0
    assert "WARN" not in output
    assert "No provider keys configured" not in output
    assert "SKIP" in output


def test_verify_summary_reports_failure_and_nonzero_exit(tmp_path: Path) -> None:
    from hosts.cli import main as cli_main

    with (
        patch.dict(os.environ, {"LISA_OPENAI_API_KEY": "test-key"}),
        patch.object(
            cli_main, "_verify_check_manifest", return_value=("[red]FAIL[/red]", "invalid manifest")
        ),
    ):
        code, output = _invoke_verify(tmp_path)
    assert code == 1
    assert "invalid manifest" in output
    assert "1 check(s) failed" in output


def test_verify_reports_all_current_checks(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"LISA_OPENAI_API_KEY": "test-key"}):
        code, output = _invoke_verify(tmp_path)
    assert code == 0
    for check in (
        "Python",
        ".env",
        "Session context",
        "Model manifest",
        "Pack catalog",
        "Provider configuration",
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
