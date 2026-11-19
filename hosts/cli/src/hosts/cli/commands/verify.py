"""Environment verification command."""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from hosts.cli.application import _PANEL_LIFECYCLE, app, console
from hosts.cli.bootstrap import _env_candidates
from hosts.cli.cli_runtime import _initialize_runtime_or_exit
from hosts.cli.options import (
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    NoLogOption,
    NoThreadOption,
    ProfileOption,
    SessionContextOption,
    StateSnapshotOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
)

if TYPE_CHECKING:
    from runtime.runtime import HostRuntime

# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------

_VERIFY_PASS = "[green]PASS[/green]"
_VERIFY_WARN = "[yellow]WARN[/yellow]"
_VERIFY_FAIL = "[red]FAIL[/red]"
_KEY_REDACT_VISIBLE_CHARS = 4
_MIN_PYTHON_MAJOR = 3
_MIN_PYTHON_MINOR = 11
_HTTP_OK = 200
_OLLAMA_MODEL_PREVIEW_LIMIT = 5


def _redact_key(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= _KEY_REDACT_VISIBLE_CHARS:
        return "*" * len(value)
    return "*" * (len(value) - _KEY_REDACT_VISIBLE_CHARS) + value[-_KEY_REDACT_VISIBLE_CHARS:]


def _verify_check_python() -> tuple[str, str]:
    version = platform.python_version()
    major, minor, *_ = version.split(".")
    if int(major) >= _MIN_PYTHON_MAJOR and int(minor) >= _MIN_PYTHON_MINOR:
        return _VERIFY_PASS, f"Python {version}"
    return _VERIFY_FAIL, f"Python {version} (need >= 3.11)"


def _resolve_env_candidates() -> tuple[Path, ...]:
    """Recompute the .env search order for the *current* environment.

    Thin wrapper over the import-time :func:`_env_candidates` so ``lisa
    verify`` reports what startup will actually load even when LISA_
    PROFILE / LISA_DATA_DIR / LISA_PROFILES_DIR changed after import or
    under test monkeypatching. Returns the same tuple when the
    environment is unchanged.
    """
    return _env_candidates()[1]


def _verify_check_env(runtime: HostRuntime) -> tuple[str, str]:
    # Resolve dynamically (not from the module-level constant) so LISA_DATA_DIR
    # / LISA_PROFILE set after import — and under test monkeypatching — are
    # honored; the order mirrors the import-time construction including the
    # profile→base .env fallback, so verify reports what startup really loads.
    for candidate in _resolve_env_candidates():
        if candidate.exists():
            return _VERIFY_PASS, f".env found ({candidate})"
    return _VERIFY_WARN, ".env not found. Create one at ~/.lisa/.env with your model provider keys."


def _verify_check_manifest(runtime: HostRuntime) -> tuple[str, str]:
    from runtime.model_manifest import load_model_manifest

    try:
        path = runtime.paths.models_path
        manifest = load_model_manifest(path)
        model_count = len(manifest.models)
        if model_count == 0:
            return _VERIFY_FAIL, "Model manifest loaded but contains zero models"
        model_ids = ", ".join(m.model_id for m in manifest.models)
        return _VERIFY_PASS, f"{model_count} model(s): {model_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Model manifest failed to load: {exc}"


def _verify_check_pack_catalog() -> tuple[str, str]:
    try:
        from kernel.ipsar_fsm import get_workflow_catalog

        catalog = get_workflow_catalog()
        mode_count = len(catalog)
        if mode_count == 0:
            return _VERIFY_WARN, "Pack catalog loaded but contains zero modes"
        mode_ids = ", ".join(sorted(catalog.keys()))
        return _VERIFY_PASS, f"{mode_count} mode(s): {mode_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Pack catalog failed to load: {exc}"


def _verify_check_ollama() -> tuple[str, str]:
    import os

    import httpx

    configured_host = (
        os.getenv("LISA_OLLAMA_HOST", "").strip()
        or os.getenv("OLLAMA_HOST", "").strip()
        or "127.0.0.1:11434"
    )

    async def _ping() -> tuple[str, str]:
        try:
            base_url = f"http://{configured_host}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == _HTTP_OK:
                    data = resp.json()
                    models = [m.get("name", "?") for m in data.get("models", [])]
                    if models:
                        preview = ", ".join(models[:_OLLAMA_MODEL_PREVIEW_LIMIT])
                        suffix = "..." if len(models) > _OLLAMA_MODEL_PREVIEW_LIMIT else ""
                        return _VERIFY_PASS, (
                            f"Ollama reachable, {len(models)} model(s) pulled: {preview}{suffix}"
                        )
                    return _VERIFY_WARN, "Ollama reachable but no models pulled"
                return _VERIFY_WARN, f"Ollama returned HTTP {resp.status_code}"
        except Exception as exc:
            return _VERIFY_WARN, f"Ollama not reachable at {configured_host}: {exc}"

    try:
        return asyncio.run(_ping())
    except Exception as exc:
        return _VERIFY_WARN, f"Ollama check failed: {exc}"


def _verify_check_provider_keys() -> tuple[str, str]:
    import os

    parts: list[str] = []
    # Check LISA_* vars first (the new host's env vars), then fall back to
    # the bare provider variables accepted by the host gateway provider.
    azure_key = (
        os.getenv("LISA_AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    azure_endpoint = (
        os.getenv("LISA_AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    if azure_key and azure_endpoint:
        parts.append(f"Azure OpenAI: {_redact_key(azure_key)} @ {azure_endpoint}")
    elif azure_key or azure_endpoint:
        parts.append("Azure OpenAI: partially configured")

    for provider, variable in (
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Google Gemini", "GOOGLE_API_KEY"),
    ):
        key = os.getenv(f"LISA_{variable}", "").strip() or os.getenv(variable, "").strip()
        if key:
            parts.append(f"{provider}: {_redact_key(key)}")

    ollama_host = os.getenv("LISA_OLLAMA_HOST", "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        parts.append(f"Ollama: {ollama_host}")

    openrouter_key = (
        os.getenv("LISA_OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    if openrouter_key:
        parts.append(f"OpenRouter: {_redact_key(openrouter_key)}")

    aws_access_key = (
        os.getenv("LISA_AWS_ACCESS_KEY_ID", "").strip()
        or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    )
    aws_secret_key = (
        os.getenv("LISA_AWS_SECRET_ACCESS_KEY", "").strip()
        or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    aws_region = (
        os.getenv("LISA_AWS_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )
    if aws_access_key and aws_secret_key and aws_region:
        parts.append(f"AWS Bedrock: {_redact_key(aws_access_key)} @ {aws_region}")
    elif aws_access_key or aws_secret_key or aws_region:
        parts.append("AWS Bedrock: partially configured")

    if not parts:
        return _VERIFY_WARN, "No provider keys configured"
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_paths(runtime: HostRuntime) -> tuple[str, str]:
    parts: list[str] = []
    parts.append(f"Data dir: {runtime.paths.data_dir}")
    parts.append(f"Thread logs: {runtime.paths.thread_dir}")
    tp = runtime.paths.thread_path
    if tp:
        parts.append(f"Thread path: {tp}")
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_session_context(runtime: HostRuntime) -> tuple[str, str]:
    if not runtime.config.user_slug:
        return "[yellow]NONE[/yellow]", "No session context active"
    parts = [f"Context: {runtime.config.user_slug}"]
    ref = runtime.config.session_context.strip()
    if ref and ref != runtime.config.user_slug:
        parts.append(f"Path: {ref}")
    return _VERIFY_PASS, "; ".join(parts)


@app.command(
    "verify",
    rich_help_panel=_PANEL_LIFECYCLE,
    help="Run verification checks of the Lisa environment.",
)
def _verify_command(
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Run verification checks of the Lisa environment."""
    runtime = _initialize_runtime_or_exit(
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    if runtime is None:
        return
    checks: list[tuple[str, str, str]] = [
        ("Python", *_verify_check_python()),
        (".env", *_verify_check_env(runtime)),
        ("Session context", *_verify_check_session_context(runtime)),
        ("Model manifest", *_verify_check_manifest(runtime)),
        ("Pack catalog", *_verify_check_pack_catalog()),
        ("Provider keys", *_verify_check_provider_keys()),
        ("Ollama", *_verify_check_ollama()),
        ("Paths", *_verify_check_paths(runtime)),
    ]

    from rich.box import SIMPLE_HEAD
    from rich.table import Table  # lazy: presentation import (Slice C)

    # No zebra row_styles here either: terminal backgrounds are unknown and
    # dark-grey rows vanish on black consoles (same rationale as
    # render._styled_table).
    table = Table(
        title="lisa verify",
        box=SIMPLE_HEAD,
        header_style="bold steel_blue3",
    )
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold", no_wrap=False)

    failures = 0
    warnings = 0
    for name, status, detail in checks:
        table.add_row(name, status, detail)
        if "FAIL" in status:
            failures += 1
        elif "WARN" in status:
            warnings += 1

    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    if warnings:
        console.print(f"\n[yellow]All mandatory checks passed ({warnings} warnings).[/yellow]")
    else:
        console.print("\n[green]All checks passed.[/green]")
