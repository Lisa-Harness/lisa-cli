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
_VERIFY_SKIP = "[dim]SKIP[/dim]"
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
    if (int(major), int(minor)) >= (_MIN_PYTHON_MAJOR, _MIN_PYTHON_MINOR):
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
    # Candidates are ordered lowest→highest precedence: report the file whose
    # values actually win, not the lowest-precedence file that merely exists.
    for candidate in reversed(_resolve_env_candidates()):
        if candidate.exists():
            return _VERIFY_PASS, f".env found ({candidate})"
    return _VERIFY_SKIP, "No .env file; using the existing process configuration"


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


def _load_verified_manifest(runtime: HostRuntime) -> object:
    """Load the active manifest once for all selected-provider checks."""
    loaded = getattr(getattr(runtime, "session", None), "manifest", None)
    if loaded is not None:
        return loaded
    from runtime.model_manifest import load_model_manifest

    return load_model_manifest(runtime.paths.models_path)


def _selected_models(runtime: HostRuntime) -> tuple[object, ...]:
    """Return models required by active mode pins, falling back to all models."""
    manifest = _load_verified_manifest(runtime)
    models = tuple(getattr(manifest, "models", ()) or ())
    mode_models = getattr(manifest, "mode_models", {}) or {}
    if not mode_models:
        return models
    selected: list[object] = []
    for mode_id, ref in mode_models.items():
        model = getattr(manifest, "resolve", lambda _ref: None)(ref)
        if model is None:
            raise ValueError(f"Mode '{mode_id}' references unknown model '{ref}'")
        if model not in selected:
            selected.append(model)
    return tuple(selected)


def _ollama_selected_host(models: tuple[object, ...]) -> str:
    """Resolve an endpoint declared by selected Ollama models, if any."""
    import os

    endpoints: list[str] = []
    for model in models:
        endpoint = str(getattr(model, "base_url", "")).strip()
        env_name = str(getattr(model, "base_url_env", "")).strip()
        if not endpoint and env_name:
            endpoint = os.getenv(env_name, "").strip()
        if endpoint and endpoint not in endpoints:
            endpoints.append(endpoint)
    if len(endpoints) > 1:
        raise ValueError("Selected Ollama models declare conflicting endpoints")
    return endpoints[0] if endpoints else ""


def _verify_check_ollama(runtime: HostRuntime | None = None) -> tuple[str, str]:
    """Check Ollama only when the active manifest selects Ollama models."""
    import os

    import httpx
    from runtime.provider_endpoints import normalize_ollama_host_to_base_url

    if runtime is None:
        return _VERIFY_SKIP, "Ollama is not selected by the active manifest"
    try:
        models = tuple(
            model
            for model in _selected_models(runtime)
            if str(getattr(model, "provider", "")).strip().lower() == "ollama"
        )
    except Exception as exc:
        return _VERIFY_FAIL, f"Cannot resolve selected Ollama models: {exc}"
    if not models:
        return _VERIFY_SKIP, "Ollama is not selected by the active manifest"
    configured_host = _ollama_selected_host(models)
    if not configured_host:
        configured_host = (
            os.getenv("LISA_OLLAMA_HOST", "").strip()
            or os.getenv("OLLAMA_HOST", "").strip()
            or "127.0.0.1:11434"
        )
    try:
        configured_host = _ollama_selected_host(models) or configured_host
        base_url = normalize_ollama_host_to_base_url(configured_host)
        tags_url = f"{base_url.removesuffix('/v1/').rstrip('/')}/api/tags"
    except (TypeError, ValueError) as exc:
        return _VERIFY_FAIL, f"Invalid Ollama endpoint {configured_host!r}: {exc}"

    async def _ping() -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(tags_url)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("Ollama /api/tags returned no models list")
            available = {
                str(item.get("name") or item.get("model") or "").strip()
                for item in rows
                if isinstance(item, dict)
            }
            required = {str(getattr(model, "model_id", "")).strip() for model in models}
            missing = sorted(ref for ref in required if ref and ref not in available)
            if missing:
                return _VERIFY_FAIL, (
                    "Ollama is reachable but selected model(s) are missing: " + ", ".join(missing)
                )
            preview = ", ".join(sorted(available)[:_OLLAMA_MODEL_PREVIEW_LIMIT])
            return _VERIFY_PASS, (
                f"Ollama reachable at {configured_host}; selected model(s) available. "
                f"Catalog: {preview or 'empty'}"
            )

    try:
        return asyncio.run(_ping())
    except Exception as exc:
        return _VERIFY_FAIL, f"Required Ollama is unavailable at {configured_host}: {exc}"


def _env_value(*names: str) -> str:
    import os

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _provider_requirements(provider: str) -> tuple[tuple[str, ...], ...]:
    """Required environment alternatives for a selected built-in provider."""
    requirements = {
        "openai": (("LISA_OPENAI_API_KEY", "OPENAI_API_KEY"),),
        "anthropic": (("LISA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),),
        "google": (("LISA_GOOGLE_API_KEY", "GOOGLE_API_KEY"),),
        "openrouter": (("LISA_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),),
        "azure_openai": (
            ("LISA_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
            ("LISA_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
        ),
        # AWS deliberately has no mandatory env credentials: boto3's standard
        # credential chain is supported, and its region has a provider default.
        "aws": (),
        "ollama": (),
    }
    return requirements.get(provider, ())


def _verify_check_provider_keys(runtime: HostRuntime | None = None) -> tuple[str, str]:
    """Validate only configuration required by selected manifest models."""
    if runtime is None:
        return _VERIFY_SKIP, "No active manifest selects provider credentials"
    try:
        selected = _selected_models(runtime)
    except Exception as exc:
        return _VERIFY_FAIL, f"Cannot resolve selected provider configuration: {exc}"
    failures: list[str] = []
    checked: list[str] = []
    for model in selected:
        provider = str(getattr(model, "provider", "")).strip().lower()
        checked.append(f"{getattr(model, 'model_id', '?')} ({provider})")
        api_key_env = str(getattr(model, "api_key_env", "")).strip()
        if api_key_env and not _env_value(api_key_env):
            failures.append(f"{getattr(model, 'model_id', '?')}: {api_key_env} is not set")
        base_url_env = str(getattr(model, "base_url_env", "")).strip()
        if base_url_env and not _env_value(base_url_env):
            failures.append(f"{getattr(model, 'model_id', '?')}: {base_url_env} is not set")
        failures.extend(
            f"{provider}: set {alternatives[0]} (or {alternatives[1]})"
            for alternatives in _provider_requirements(provider)
            if not api_key_env and not _env_value(*alternatives)
        )
    if failures:
        return _VERIFY_FAIL, "; ".join(dict.fromkeys(failures))
    if not selected:
        return _VERIFY_SKIP, "No selected models require provider configuration"
    return _VERIFY_PASS, f"Selected provider requirements present for {len(checked)} model(s)"


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
        ("Provider configuration", *_verify_check_provider_keys(runtime)),
        ("Ollama", *_verify_check_ollama(runtime)),
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
