"""Fixtures for the scripted-model integration suite.

Layer contract (``docs/SPEC/CLI_INTEGRATION_LAYER.md``): the CLI's seam
surface is small and load-bearing — ``_initialize_runtime`` (runtime
construction), ``session.provider_gateways_override`` (gateway scripting),
``_run_prompt_with_teardown`` (the turn), and the durable approval store. The
scenarios call exactly these, so a decomposition that changes their semantics
fails here even when every unit-suite monkeypatch keeps passing.

Fixtures are offline and hermetic: no provider credential is read (the
gateway seam means no provider client is ever constructed), the runtime's
data-dir is per-test ``tmp_path``, and the pack catalog is supplied by
``build_test_runtime``'s in-process sources.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from e2etest.scripted import Reply, ScriptedGatewayProvider, ScriptTurn

from hosts.cli import main as cli_main

# Harness base-data overlay: the web_search contract stub flipped ambient, so
# the default mode's surface carries it without a provider pack (same files
# the harness smoke scenario uses, copied into the test data dir).
_OVERLAY_SRC = Path(__file__).resolve().parents[5] / ("lisa/e2etest/base_data/config/overlays")

_OFFLINE_MODELS_YAML = (
    "version: 1\n"
    "mode_models: {default: offline-stub-model}\n"
    "models:\n"
    "- model_id: offline-stub-model\n"
    "  provider: openai\n"
    "  display_name: Offline stub (never called)\n"
    "  capabilities: [text, tools]\n"
)


def _session_plan(name: str, comment: str) -> str:
    """The front matter the scenarios need: routing + core pack + budgets."""
    return (
        "---\n"
        "schema_version: 2\n"
        f"name: {name}\n"
        "surface:\n"
        "  type: personal_dm\n"
        "  id: host-console\n"
        "  memory_scope: personal\n"
        "resources:\n"
        "  packs:\n"
        "    required:\n"
        "      - lisa-pack-core\n"
        "providers:\n"
        "  modes:\n"
        "    default: lisa-pack-core\n"
        "  tools:\n"
        "    web_search: e2etest-offline\n"
        "routing: { from: { pack: lisa-pack-core, session: default } }\n"
        "execution:\n"
        "  max_session_tokens: 200000\n"
        "  max_completion_tokens: 50000\n"
        "  max_capability_calls: 50\n"
        "---\n"
        f"# {comment}\n"
    )


def _write_overlay(data_dir: Path) -> None | None:
    """Copy the harness web_search overlay into ``<data_dir>/config/overlays``."""
    src = _OVERLAY_SRC
    if not src.is_dir():
        return None
    dst_root = data_dir / "config" / "overlays"
    if dst_root.is_dir():
        return dst_root / "tools" / "web_search" / "web_search.tool.yml"
    shutil.copytree(src / "tools", dst_root / "tools")
    return dst_root / "tools" / "web_search" / "web_search.tool.yml"


def _scripted(
    respond: Callable[[ScriptTurn], Reply],
    *,
    auxiliary: Callable[[ScriptTurn], Reply] | None = None,
) -> ScriptedGatewayProvider:
    """Build the scripted gateway the kernel's override seam accepts."""
    if auxiliary is None:

        def auxiliary(turn: ScriptTurn) -> Reply:
            if turn.request.response_format is not None:
                return Reply.text("{}")
            return Reply.text("")

    return ScriptedGatewayProvider(respond=respond, auxiliary=auxiliary)


class RecordingTransport:
    """Captures the stage/response events ``PromptTransport`` would render.

    Implements the ``TransportCallbacks`` protocol plus the lifecycle
    methods ``_run_prompt_with_teardown`` drives (``start_turn``,
    ``start_activity``, ``stop_activity``, price/streaming fields), so the
    CLI's turn path can drive the recorder directly. The runtime cannot tell
    the difference, and the tests assert on structured events (the CLI
    transport's richer rendering is unit-tested elsewhere).
    """

    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[str] = []
        self.errors: list[str] = []
        self.model_prices: dict[str, tuple[float, float]] = {}
        self.model_cache_prices: dict[str, tuple[float, float]] = {}
        self.streaming_mode: str = "none"

    def start_turn(self, message: str) -> None:
        self.started_message = message

    def start_activity(self) -> None:
        return None

    def stop_activity(self) -> None:
        return None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        self.stages.append((label, dict(data or {})))

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        self.responses.append(text)

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self.errors.append(message)


_SENTINEL_PACKS_VENV = "/nonexistent-lisa-cli-integration-packs-venv"


@pytest.fixture(autouse=True)
def _hermetic_packs_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI's packs-venv injection at a nonexistent (empty) venv.

    The CLI's startup bootstrap injects the operator packs venv
    (``~/.lisa/venv`` or ``LISA_PACKS_VENV``) into ``sys.path`` before the
    first entry-point scan; on this machine that venv holds third-party packs
    (serper, brave, ...) which would leak into every content-root discovery.
    The scenarios' runtimes install their own packs into the test data dir's
    ``venv`` (the CLI's data-dir-anchored resolution), never the operator's.
    """
    monkeypatch.setattr(cli_main, "_mirror_default_packs_venv", lambda: Path(_SENTINEL_PACKS_VENV))


@pytest.fixture
def offline_runtime_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Build runtimes through the CLI's own constructor, per test data dir.

    Calls ``hosts.cli.main._initialize_runtime`` (not
    ``runtime.bootstrap.build_runtime``) so the scenarios exercise the seam the
    CLI actually owns, including its packs-venv bootstrap. The catalog
    sources are in-process (``in_process_mcp=True``) so no MCP subprocess is
    started; the pack venv the CLI would install stays unused because every
    scenario's session plan requires only bundled packs.
    """

    def factory(
        data_dir: Path,
        *,
        session_plan: str | None = None,
        **kwargs: Any,
    ) -> Any:
        (data_dir / "config").mkdir(parents=True, exist_ok=True)
        (data_dir / "config" / "models.yaml").write_text(_OFFLINE_MODELS_YAML, encoding="utf-8")
        _write_overlay(data_dir)
        # The kernel boot that auto-installs a pack must be followed by a
        # second discovery pass (the same contract the CLI's restart flow
        # implements); in-process, rerunning the build gives the fixed view.
        # The factory stays a thin wrapper over the CLI constructor so the
        # scenarios exercise the seam the CLI owns.
        return cli_main._initialize_runtime(
            data_dir=str(data_dir),
            session_context=session_plan or "",
            streaming="none",
            no_log=True,
            in_process_mcp=True,
            state_snapshot_path="",
            **kwargs,
        )

    return factory


@pytest.fixture
def scripted_gateway() -> Callable[..., ScriptedGatewayProvider]:
    """Build a scripted model; wire/clear the override seam around the test."""

    def factory(respond: Callable[[ScriptTurn], Reply], **kwargs: Any) -> Any:
        return _scripted(respond, **kwargs)

    return factory


@pytest.fixture
def recording_transport() -> RecordingTransport:
    return RecordingTransport()


@pytest.fixture
def scenario_plan(tmp_path: Path) -> Path:
    """Write the default session plan (core pack, offline web_search)."""
    plan = tmp_path / "session_context.md"
    plan.write_text(
        _session_plan(
            "cli-integration-smoke",
            "CLI integration smoke session (offline scripted)",
        ),
        encoding="utf-8",
    )
    return plan


@pytest.fixture
def pack_autoinstall_plan(tmp_path: Path) -> Path:
    """Session plan whose required bash pack auto-installs from PyPI.

    The CLI's own startup auto-install seeds the packs venv, exercising the
    production install path offline: the scripted model stands in for the
    LLM, but the pack install, venv seeding, content-root discovery, and
    approval orchestration are all real. The bare-name auto-install spec
    resolves against the trusted index (``lisa-pack-bash`` is published);
    ``LISA_CLI_INTEGRATION_BASH_PACK`` may point at a local checkout instead.
    """
    plan = tmp_path / "session_context.md"
    bash_source = _bash_pack_source()
    if bash_source is None:
        pytest.skip("lisa-pack-bash not available for the approval scenario")
    plan.write_text(
        _session_plan("cli-integration-approval", "CLI approval scenario (offline)").replace(
            "    required:\n      - lisa-pack-core\n",
            "    required:\n      - lisa-pack-core\n"
            "      - lisa-pack-bash\n"
            "    auto_install:\n"
            "      lisa-pack-bash:\n"
            f"        source: {bash_source}\n",
        ),
        encoding="utf-8",
    )
    return plan


def _bash_pack_source() -> str | None:
    """Resolve the bash pack auto-install source: env override, else PyPI name."""
    override = os.environ.get("LISA_CLI_INTEGRATION_BASH_PACK", "").strip()
    if override:
        override_path = Path(override)
        return str(override_path) if override_path.is_dir() else None
    # The published distribution installs from the trusted index. (Bare
    # names are a git-root-template feature; the plain PyPI name installs
    # directly and passes the ``lisa-pack-`` prefix validation.)
    return "lisa-pack-bash"
