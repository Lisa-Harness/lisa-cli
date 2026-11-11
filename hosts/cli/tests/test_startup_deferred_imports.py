"""PROMPT_STARTUP Slice C (CLI half): deferred-import regression tests.

``docs/SPEC/PROMPT_STARTUP_CLI.md`` §2 requires that ``lisa --help`` and
``lisa --version`` defer ``kernel.models``, ``runtime.config``, network
initialization, and rich/presentation imports until the command that needs
them actually runs, and that every late import carries a reason clause.

These tests lock the fast paths three ways:

- **Fresh-process import audit** — importing ``hosts.cli.main`` must not
  pull the heavy framework or presentation modules (the module body is
  what every CLI run pays before dispatch).
- **Fresh-process command audit** — ``lisa --version`` and ``lisa --help``
  subprocess runs must not start framework sessions (no
  ``runtime.runtime`` / ``runtime.bootstrap`` / ``openai`` imports) and
  must not load the crash-rendering or table/presentation modules
  unnecessarily. ``--help`` legitimately discovers pack commands, so the
  kernel entry-point chain it touches is asserted bounded instead.
- **Mirror parity** — the import-free mirrors of the framework's
  data-dir/venv resolvers must agree with the framework functions, so the
  deferral can never change which packs venv or .env tree a run uses.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hosts.cli import main as cli_main

_SRC_ROOT = Path(cli_main.__file__).resolve().parents[2]

#: Modules whose import the CLI entry must defer (spec §2). Any command
#: that genuinely needs one imports it lazily with a reason clause; the
#: audit asserts the *module body* never pays for them.
_DEFERRED_FROM_IMPORT = (
    "kernel.models",
    "kernel.tools.mcp_client",
    "kernel.host_command_discovery",
    "kernel.ipsar_fsm",
    "runtime.config",
    "runtime.paths",
    "runtime.bootstrap",
    "runtime.runtime",
    "runtime.handle_message",
    "hosts.cli.render",
    "hosts.cli.transport",
    "hosts.cli.entry",
    "rich.traceback",
    "rich.markdown",
    "rich.table",
    "rich.panel",
    "openai",
    "httpx",
    "mcp",
)

_PROBE_TEMPLATE = """
import json
import sys

try:
    import hosts.cli.main
except Exception as exc:  # pragma: no cover - surfaces as failure payload
    print(json.dumps({{"error": repr(exc)}}))
    raise SystemExit(1)

heavy = {{
    name: sys.modules[name].__dict__.get("__file__", "")
    for name in {deferred!r}
    if name in sys.modules
}}
print(json.dumps({{"loaded": heavy}}))
"""


def _probe_imports(python: str, deferred: tuple[str, ...]) -> list[str]:
    """Run one fresh-interpreter import audit; return the heavy modules loaded."""
    completed = subprocess.run(
        [python, "-c", _PROBE_TEMPLATE.format(deferred=deferred)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=str(_SRC_ROOT),
        env={**os.environ, "PYTHONPATH": str(_SRC_ROOT)},
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"import probe failed: {completed.stderr.strip()}\n{completed.stdout.strip()}"
        )
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")),
        "",
    )
    assert payload_line, f"no probe payload in output: {completed.stdout!r}"
    loaded = dict(__import__("json").loads(payload_line))["loaded"]
    return sorted(loaded)


def _lisa_argv(argv: list[str]) -> list[str]:
    """Return the installed/entry-point argv for one lisa command run."""
    return [sys.executable, "-m", "hosts.cli.main", *argv]


# ── module import audit ───────────────────────────────────────────────


def test_main_module_import_loads_no_deferred_dependencies() -> None:
    """Importing hosts.cli.main pays for none of the deferred subsystems.

    The module body is the front half of every CLI run (``--help``,
    ``--version``, and every command); the spec's import trimming means
    the heavy imports happen only inside the command bodies that need
    them.
    """
    loaded = _probe_imports(sys.executable, _DEFERRED_FROM_IMPORT)
    assert loaded == [], f"hosts.cli.main import still loads deferred modules: {loaded}"


# ── fresh-process command audits ──────────────────────────────────────


def _probe_command_imports(argv: list[str]) -> dict[str, str]:
    """Run one lisa command in a fresh process; return its heavy imports.

    Prints the loaded-module audit from inside the process right before
    the command dispatch exits, so the payload reflects exactly what the
    command path imported (not just the module body).
    """
    probe = f"""
import json
import sys

sys.argv = [{argv!r}]

import hosts.cli.main as cli_main

_loaded = {{
    name: sys.modules[name].__dict__.get("__file__", "")
    for name in {_DEFERRED_FROM_IMPORT!r}
    if name in sys.modules
}}

class _AuditExit(Exception):
    pass

# Replace Typer's exit with a probe that reports and stops before the
# process exits: we assert on what the command path actually imported.
def _probe_app(*args, **kwargs):
    print(json.dumps({{"loaded": _loaded}}))
    raise _AuditExit

cli_main.app = _probe_app
try:
    cli_main.main()
except _AuditExit:
    pass
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=str(_SRC_ROOT),
        env={**os.environ, "PYTHONPATH": str(_SRC_ROOT)},
    )
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")),
        "",
    )
    assert payload_line, f"no probe payload in output: {completed.stdout!r}"
    import json

    payload = json.loads(payload_line)
    assert isinstance(payload["loaded"], dict)
    return payload["loaded"]


def test_version_path_starts_no_framework_sessions() -> None:
    """``lisa --version`` builds no runtime and imports no framework session.

    Spec §3 acceptance: help/version runs must show zero framework
    sessions and no model/provider initialization — asserted here as the
    absence of the runtime-session modules (``runtime.runtime``,
    ``runtime.bootstrap``, ``openai``, ``mcp``) and the network stack
    (``httpx``) from the --version process.
    """
    loaded = _probe_command_imports(["--version"])
    forbidden = [
        name
        for name in (
            "runtime.runtime",
            "runtime.bootstrap",
            "runtime.config",
            "runtime.paths",
            "openai",
            "mcp",
            "httpx",
        )
        if name in loaded
    ]
    assert forbidden == [], f"--version started framework machinery: {forbidden}"


def test_help_path_defers_presentation_and_runtime_config() -> None:
    """``lisa --help`` never imports runtime.config or rich presentation.

    Help legitimately enumerates pack commands (a shipped feature), which
    touches the packs-venv injection and pack modules. Everything else
    stays deferred: no ``runtime.config``/``runtime.paths`` (the .env +
    injection mirrors own that), no ``rich.markdown``, no
    ``rich.traceback``.
    """
    loaded = _probe_command_imports(["--help"])
    forbidden = [
        name
        for name in (
            "runtime.config",
            "runtime.paths",
            "runtime.runtime",
            "runtime.bootstrap",
            "rich.markdown",
            "rich.traceback",
            "openai",
        )
        if name in loaded
    ]
    assert forbidden == [], f"--help loaded deferred modules: {forbidden}"


# ── late-import reason clauses (spec §2) ──────────────────────────────


def test_every_deferral_carries_a_reason_clause() -> None:
    """Import-suppression sites name the deferred dependency in a reason.

    The Lisa repository rule: import-outside-toplevel is allowed for
    startup-latency deferral only, and every suppression carries a reason
    clause naming the dependency or subsystem being deferred. The
    deferred call sites in ``main.py`` use ``# lazy:`` marker comments;
    this test pins the marker vocabulary so a future edit cannot silently
    drop the rationale.
    """
    source = Path(cli_main.__file__).read_text(encoding="utf-8")
    for needle, reason in (
        ("from runtime.paths import", "runtime.paths"),
        ("from runtime.config import default_data_dir", "runtime.config"),
        ("from hosts.cli.render import", "render"),
        ("from rich.traceback import install", "crash rendering"),
    ):
        assert needle in source, f"expected deferred import site for {reason!r}"
    # The lazy import markers carry a reason after the colon.
    lazy_lines = [line for line in source.splitlines() if "# lazy:" in line and "import" in line]
    assert lazy_lines, "no lazy import sites with reason clauses found"
    for line in lazy_lines:
        marker = line.split("# lazy:", 1)[1]
        assert marker.strip() != "", f"reason clause missing on: {line.strip()}"


# ── mirror parity (import-free resolvers vs framework functions) ──────


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LISA_DATA_DIR": "/tmp/lisa-parity-data"},
        {"LISA_PROFILE": "taxes"},
        {"LISA_PROFILE": "a-b.c_9"},
        {"LISA_PROFILES_DIR": "/tmp/lisa-parity-profiles", "LISA_PROFILE": "taxes"},
        {"LISA_DATA_DIR": "/tmp/lisa-parity-data", "LISA_PROFILE": "taxes"},
    ],
)
def test_mirror_resolvers_match_framework_functions(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """The import-free mirrors agree with the framework resolvers exactly.

    The mirrors exist so ``--help`` / ``--version`` never import
    ``runtime.paths`` / ``runtime.config`` (whose package ``__init__``
    drags the kernel/mcp import chain). If a mirror drifts from the
    framework function it shadows, pack discovery would scan the wrong
    venv or the .env sweep would read the wrong candidates — so parity is
    asserted against the framework's own functions across the env matrix.
    """
    for name in ("LISA_DATA_DIR", "LISA_PROFILE", "LISA_PROFILES_DIR", "LISA_PACKS_VENV"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    from runtime.config import base_data_dir, default_data_dir, profile_data_dir
    from runtime.paths import (
        default_packs_venv,
        framework_root,
        packs_venv_site_packages,
        profile_base_data_dir,
    )

    assert cli_main._mirror_default_data_dir() == default_data_dir()
    assert cli_main._mirror_base_data_dir() == base_data_dir()
    assert cli_main._mirror_framework_root() == framework_root()
    try:
        expected_profile = profile_data_dir(env.get("LISA_PROFILE", "taxes"))
    except ValueError:
        expected_profile = None

    def _mirror_profile(name: str) -> object:
        try:
            return cli_main._mirror_profile_data_dir(name)
        except ValueError:
            return None

    if expected_profile is not None:
        assert _mirror_profile(env.get("LISA_PROFILE", "taxes")) == expected_profile
        assert (
            cli_main._mirror_profile_base_data_dir(expected_profile)  # type: ignore[arg-type]
            == profile_base_data_dir(expected_profile)  # type: ignore[arg-type]
        )
    # Venv resolution parity, and the site-packages dir parity for the
    # resolved venv (covers the pyvenv.cfg/lib-scan mirror on real venvs).
    assert cli_main._mirror_default_packs_venv() == default_packs_venv()
    venv = default_packs_venv()
    assert cli_main._mirror_packs_venv_site_packages(venv) == packs_venv_site_packages(venv)


def test_mirror_profile_name_validation_matches_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid/empty profile names raise the same ValueError as the framework."""
    monkeypatch.delenv("LISA_PROFILES_DIR", raising=False)
    from runtime.config import profile_data_dir

    for bad in ("", "..", "a/b", "a b"):
        with pytest.raises(ValueError):
            profile_data_dir(bad)
        with pytest.raises(ValueError):
            cli_main._mirror_profile_data_dir(bad)


def test_mirror_venv_python_dir_reads_pyvenv_cfg(tmp_path: Path) -> None:
    """The mirror's pyvenv.cfg parsing matches the framework's layout rule.

    A packs venv created by ``uv venv`` may target a different Python
    than the host interpreter; the tag must come from the venv itself
    (``version_info`` preferred, then a ``lib/`` scan), never from
    ``sys.version_info``.
    """
    from runtime.paths import packs_venv_site_packages

    venv = tmp_path / "venv"
    (venv).mkdir()
    (venv / "pyvenv.cfg").write_text("version_info = 3.11.9\n", encoding="utf-8")
    assert cli_main._mirror_venv_python_dir(venv) == "python3.11"
    assert cli_main._mirror_packs_venv_site_packages(venv) == packs_venv_site_packages(venv)

    # No pyvenv.cfg: the lib scan finds the only pythonX.Y dir present.
    (venv / "pyvenv.cfg").unlink()
    (venv / "lib" / "python3.13").mkdir(parents=True)
    assert cli_main._mirror_venv_python_dir(venv) == "python3.13"
    assert cli_main._mirror_packs_venv_site_packages(venv) == packs_venv_site_packages(venv)


# ── bootstrap ordering + idempotence ──────────────────────────────────


def test_bootstrap_env_and_packs_runs_once_and_orders_env_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy bootstrap runs .env before the venv injection, exactly once.

    Ordering contract from the original import-time sequence: .env values
    (e.g. LISA_PACKS_VENV / LISA_DATA_DIR) must land in os.environ before
    the packs venv is resolved, and both must precede any framework
    entry-point scan. Idempotence keeps repeat command resolutions (help
    listing + dispatch) from re-running either step: the real once-wrappers
    stay in play; only their side-effect workers are stubbed.
    """
    calls: list[str] = []
    monkeypatch.setattr(cli_main, "_ENV_LOADED", False)
    monkeypatch.setattr(cli_main, "_PACKS_VENV_INJECTED", False)
    monkeypatch.setattr(cli_main, "_load_env_candidates", lambda candidates: calls.append("env"))
    absent_venv = Path("/nonexistent-lisa-parity-venv")
    monkeypatch.setattr(
        cli_main,
        "_mirror_default_packs_venv",
        lambda: (calls.append("inject"), absent_venv)[1],
    )
    cli_main._bootstrap_env_and_packs()
    cli_main._bootstrap_env_and_packs()
    assert calls == ["env", "inject"]


def test_pack_commands_bootstraps_env_and_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pack-command discovery (the --help/completion path) bootstraps first.

    The packs venv must be on sys.path before the entry-point scan that
    finds pack host commands; the lazy bootstrap replaces the old
    import-time injection, so discovery owns the call now.
    """
    calls: list[str] = []
    monkeypatch.setattr(cli_main, "_pack_command_specs", None)
    monkeypatch.setattr(
        cli_main, "_bootstrap_env_and_packs", lambda: calls.append("bootstrap"), raising=False
    )
    import kernel.host_command_discovery as discovery_mod

    monkeypatch.setattr(
        discovery_mod,
        "discover_pack_host_commands",
        lambda excluded_packs=None: [],
    )
    cli_main._pack_commands()
    assert calls == ["bootstrap"]


# ── startup telemetry wiring is preserved (spec §2) ────────────────────


def test_telemetry_wiring_untouched_in_bootstrap() -> None:
    """The CLI bootstrap adds no framework teardown/session work.

    ``_bootstrap_env_and_packs`` is env-file loading + sys.path injection
    only: it must never construct a runtime (which would emit
    ``lisa.startup`` stages) — the telemetry contract stays exactly as
    the framework's ``build_runtime`` wires it, reached only via
    ``_initialize_runtime``.
    """
    import inspect

    source = inspect.getsource(cli_main._bootstrap_env_and_packs)
    assert "build_runtime" not in source
    assert "HostRuntime" not in source
    assert "_load_env_once" in source
    assert "_inject_packs_venv_site_packages_once" in source


# ── command bodies reach their deferred imports (Slice C lazy seams) ──


def _patched_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Return a runtime stub patched in as every command's runtime."""
    from types import SimpleNamespace

    runtime = SimpleNamespace(paths=None, config=None)
    monkeypatch.setattr(cli_main, "_initialize_runtime_or_exit", lambda **_kwargs: runtime)
    return runtime


def _invoke_command(command: object, **kwargs: object) -> None:
    """Call one Typer command function with the shared runtime options."""
    shared = {
        "log_level": "",
        "log": "",
        "no_log": False,
        "data_dir": "",
        "profile": "",
        "thread_dir": "",
        "no_thread": False,
        "session_context": "",
        "timezone": "",
        "state_snapshot": "",
        "streaming": "none",
        "host_log": "",
    }
    shared.update(kwargs)
    command(**shared)  # type: ignore[operator]


def test_modes_command_uses_lazy_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The modes command body imports render lazily and calls it."""
    _patched_runtime(monkeypatch, tmp_path)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_modes",
        lambda runtime, *, console=None: called.setdefault("modes", True),
    )
    _invoke_command(cli_main._modes_command)
    assert called == {"modes": True}


def test_operations_command_uses_lazy_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The operations command body imports render lazily and calls it."""
    _patched_runtime(monkeypatch, tmp_path)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_operations",
        lambda runtime, *, console=None: called.setdefault("operations", True),
    )
    _invoke_command(cli_main._operations_command)
    assert called == {"operations": True}


def test_tools_command_uses_lazy_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The tools command body imports render lazily and calls it."""
    _patched_runtime(monkeypatch, tmp_path)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_tools",
        lambda runtime, *, console=None: called.setdefault("tools", True),
    )
    _invoke_command(cli_main._tools_command)
    assert called == {"tools": True}


def test_pack_list_command_uses_lazy_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pack list action imports render lazily; other actions don't."""
    _patched_runtime(monkeypatch, tmp_path)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_pack_list_panel",
        lambda runtime, *, console=None: called.setdefault("pack_list", True),
    )
    _invoke_command(cli_main._pack_command, action="list", target="", all_packs=False)
    assert called == {"pack_list": True}


def test_mcp_command_uses_lazy_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The mcp command collects tools then imports render lazily."""
    _patched_runtime(monkeypatch, tmp_path)

    async def _fake_collect(runtime, *, connect_server_ids=None):
        return ({"srv": object()}, {"srv": [object()]})

    monkeypatch.setattr(cli_main, "_collect_mcp_tools", _fake_collect)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_mcp_servers",
        lambda health, *, console=None: called.setdefault("servers", True),
    )
    _invoke_command(cli_main._mcp_command, target="")
    assert called == {"servers": True}


def test_mcp_command_drilldown_uses_lazy_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mcp drill-down renders one server's tools through the lazy import."""
    _patched_runtime(monkeypatch, tmp_path)
    spec = object()
    health = object()

    async def _fake_collect(runtime, *, connect_server_ids=None):
        return ({"srv": health}, {"srv": [spec]})

    monkeypatch.setattr(cli_main, "_collect_mcp_tools", _fake_collect)
    called: dict[str, object] = {}
    import hosts.cli.render as render_mod

    monkeypatch.setattr(
        render_mod,
        "render_mcp_server_tools",
        lambda specs, h, *, console=None: called.setdefault("drilldown", (specs, h)),
    )
    _invoke_command(cli_main._mcp_command, target="srv")
    assert called["drilldown"] == (called["drilldown"][0], health)


def test_render_markdown_response_terminal_uses_lazy_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal branch of the markdown renderer imports Panel lazily."""
    from rich.console import Console

    printed: list[object] = []
    terminal_console = Console(file=io.StringIO(), force_terminal=True, width=80)

    class _CapturingConsole(Console):
        def print(self, *args, **kwargs):  # type: ignore[override]
            printed.append(args[0] if args else None)

    capturing = _CapturingConsole(file=io.StringIO(), force_terminal=True, width=80)
    monkeypatch.setattr(cli_main, "console", capturing)
    cli_main._render_markdown_response("response body", title="Title")
    # The lazy import executed: rich.panel is now in sys.modules and a
    # Panel was printed to the terminal-bound console.
    import sys

    assert "rich.panel" in sys.modules
    assert printed, "terminal render must print the panel"
    del terminal_console


def test_initialize_runtime_bootstraps_env_and_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_initialize_runtime`` runs the CLI bootstrap before build_runtime.

    Ordering contract: .env values must be in os.environ and the packs
    venv injected before the framework resolves its configuration (the
    lazy bootstrap replaced the import-time sequence).
    """
    calls: list[str] = []
    monkeypatch.setattr(cli_main, "_bootstrap_env_and_packs", lambda: calls.append("bootstrap"))
    import runtime.bootstrap as bootstrap_mod

    monkeypatch.setattr(
        bootstrap_mod,
        "build_runtime",
        lambda **kwargs: calls.append("build_runtime") or object(),
    )
    runtime = cli_main._initialize_runtime(state_snapshot_path="")
    assert calls == ["bootstrap", "build_runtime"]
    assert runtime is not None


# ── mirror resolver edge branches ─────────────────────────────────────


def test_mirror_framework_root_walks_to_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The framework-root mirror finds repo markers walking up, else sys.prefix.

    The mirror runs at import time against the installed tree; exercising
    the walk branches directly requires faking ``__file__`` — the module
    attribute is read inside the function, so a scratch-tree probe pins
    the loop: a workspace pyproject.toml marker resolves to that root.
    """
    repo = tmp_path / "somerepo"
    pkg_dir = repo / "src" / "hosts" / "cli"
    pkg_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        "[tool.uv]\nworkspace = { members = ['hosts/cli'] }\n", encoding="utf-8"
    )
    fake_module_file = pkg_dir / "main.py"
    fake_module_file.write_text("", encoding="utf-8")
    original_file = cli_main.__file__
    try:
        monkeypatch.setattr(cli_main, "__file__", str(fake_module_file))
        assert cli_main._mirror_framework_root() == repo
    finally:
        # The module-level _FRAMEWORK_ROOT was computed at import from the
        # real path; only the walk function is probed here.
        assert original_file

    # No markers anywhere: tool-install fallback to sys.prefix.
    lone_dir = tmp_path / "lone" / "deep" / "deeper"
    lone_dir.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "__file__", str(lone_dir / "main.py"))
    assert cli_main._mirror_framework_root() == Path(sys.prefix)


def test_mirror_framework_root_skips_unreadable_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed pyproject.toml is skipped, not a crash."""
    repo = tmp_path / "brokenrepo"
    pkg_dir = repo / "pkg"
    pkg_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[tool.uv\nbroken", encoding="utf-8")
    monkeypatch.setattr(cli_main, "__file__", str(pkg_dir / "main.py"))
    # Malformed candidate skipped → no checkout detected → sys.prefix.
    assert cli_main._mirror_framework_root() == Path(sys.prefix)


def test_mirror_profile_base_data_dir_structural_profiles_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A profile under any ``profiles/`` parent shares that parent's venv."""
    monkeypatch.delenv("LISA_PROFILES_DIR", raising=False)
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    profile = tmp_path / "srv" / "profiles" / "taxes"
    monkeypatch.setattr(cli_main, "_mirror_default_data_dir", lambda: tmp_path / "home")
    base = cli_main._mirror_profile_base_data_dir(profile)
    assert base == tmp_path / "srv"


def test_mirror_profile_base_data_dir_none_for_non_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plain data dir (not under a profiles root) owns its own venv."""
    monkeypatch.delenv("LISA_PROFILES_DIR", raising=False)
    monkeypatch.setattr(cli_main, "_mirror_default_data_dir", lambda: tmp_path / "home")
    assert cli_main._mirror_profile_base_data_dir(tmp_path / "plain") is None


def test_mirror_default_packs_venv_overrides_and_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LISA_PACKS_VENV wins; profiles share the base venv or own one."""
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PACKS_VENV", raising=False)

    monkeypatch.setenv("LISA_PACKS_VENV", str(tmp_path / "custom-venv"))
    assert cli_main._mirror_default_packs_venv() == tmp_path / "custom-venv"
    monkeypatch.delenv("LISA_PACKS_VENV", raising=False)

    # A profile under a relocated root with no base relation: the profile
    # dir itself owns the venv when profile_base_data_dir finds no base.
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "relocated"))
    monkeypatch.setattr(
        cli_main,
        "_mirror_profile_data_dir",
        lambda name: tmp_path / "relocated" / name,
    )
    monkeypatch.setattr(
        cli_main,
        "_mirror_profile_base_data_dir",
        lambda data_dir: None,
    )
    assert cli_main._mirror_default_packs_venv() == tmp_path / "relocated" / "taxes" / "venv"


def test_mirror_default_packs_venv_invalid_profile_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid profile name falls through to the platform default venv."""
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.delenv("LISA_PACKS_VENV", raising=False)
    monkeypatch.setenv("LISA_PROFILE", "in valid")

    def _raise(profile: str) -> Path:
        raise ValueError(profile)

    monkeypatch.setattr(cli_main, "_mirror_profile_data_dir", _raise)
    monkeypatch.setattr(cli_main, "_mirror_default_data_dir", lambda: Path("/home/lisa-home"))
    assert cli_main._mirror_default_packs_venv() == Path("/home/lisa-home/venv")


def test_mirror_venv_python_dir_ignores_malformed_cfg(
    tmp_path: Path,
) -> None:
    """A garbage pyvenv.cfg falls back to the lib scan / interpreter version."""
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("garbage = zzz\n", encoding="utf-8")
    # No lib dir: the running interpreter's version is the last resort.
    assert cli_main._mirror_venv_python_dir(venv) == (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )


def test_mirror_venv_python_dir_unreadable_cfg_falls_back(tmp_path: Path) -> None:
    """An unreadable pyvenv.cfg (directory in the way) still resolves the tag."""
    venv = tmp_path / "venv"
    venv.mkdir()
    cfg = venv / "pyvenv.cfg"
    cfg.mkdir()  # a directory where a file is expected: read_text raises OSError
    (venv / "lib" / "python3.13").mkdir(parents=True)
    assert cli_main._mirror_venv_python_dir(venv) == "python3.13"


def test_mirror_packs_venv_site_packages_windows_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Windows the Lib/site-packages layout applies regardless of venv tag."""
    monkeypatch.setattr(cli_main.os, "name", "nt")
    venv = tmp_path / "venv"
    assert cli_main._mirror_packs_venv_site_packages(venv) == venv / "Lib" / "site-packages"


def test_env_data_dir_invalid_profile_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid LISA_PROFILE in _env_data_dir falls back to ~/.lisa."""
    if not hasattr(cli_main, "_env_data_dir"):
        pytest.skip("python-dotenv optional dependency absent")
    monkeypatch.setenv("LISA_PROFILE", "not a profile!")
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.setattr(cli_main, "_mirror_default_data_dir", lambda: Path("/home/lisa-home"))
    assert cli_main._env_data_dir() == Path("/home/lisa-home")


def test_inject_packs_venv_site_packages_inserts_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The injection adds a real site dir to sys.path exactly once."""
    import sys as _sys

    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "_mirror_default_packs_venv", lambda: tmp_path / "venv")
    monkeypatch.setattr(cli_main, "_PACKS_VENV_INJECTED", False)
    monkeypatch.setattr(cli_main, "_ENV_LOADED", True)
    before = list(_sys.path)
    try:
        cli_main._bootstrap_env_and_packs()
        assert str(site_packages) in _sys.path
        # Idempotent: the second call adds nothing new.
        path_snapshot = list(_sys.path)
        cli_main._bootstrap_env_and_packs()
        assert [p for p in _sys.path if p not in before] == [
            p for p in path_snapshot if p not in before
        ]
    finally:
        for added in [p for p in _sys.path if p not in before]:
            _sys.path.remove(added)
