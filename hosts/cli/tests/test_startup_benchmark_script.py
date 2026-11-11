"""Tests for the CLI-half startup benchmark (``scripts/startup_benchmark.py``).

The benchmark is the Slice E gate evidence generator (spec §3): it must
probe the *installed* ``lisa`` executable, reuse the framework harness
(``runtime.startup_benchmark``) for aggregation, and never invent its own
measurement semantics. These tests pin the harness-reuse contract, the
telemetry phase parsing, and the help/version no-framework-sessions gate
without paying full benchmark runs per test.

The script is loaded by path rather than imported as a package: it lives
in ``scripts/``, which is not an importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "startup_benchmark.py"


def _load_module() -> ModuleType:
    """Import ``scripts/startup_benchmark.py`` as a module."""
    spec = importlib.util.spec_from_file_location("cli_startup_benchmark_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    """Load the benchmark module once (it imports the framework harness)."""
    return _load_module()


# ── harness reuse (spec §3: prefer the framework harness) ─────────────


def test_module_reuses_framework_harness_symbols(bench: ModuleType) -> None:
    """The benchmark imports the framework harness, not a reimplementation.

    ``runtime.startup_benchmark`` (complete at framework 1f19c5f2) owns the
    sample/run dataclasses, percentile math, and report formatting; the
    CLI script must consume them so measurement semantics cannot drift.
    """
    from runtime import startup_benchmark as framework_harness

    assert bench.BenchmarkSample is framework_harness.BenchmarkSample
    assert bench.BenchmarkRun is framework_harness.BenchmarkRun
    assert bench.format_report is framework_harness.format_report
    assert bench.summarize_variant is framework_harness.summarize_variant
    assert bench.parse_startup_lines is framework_harness.parse_startup_lines


def test_stub_manifest_points_model_at_local_endpoint(bench: ModuleType) -> None:
    """The deterministic stub manifest routes the model to the loopback server."""
    manifest = bench._stub_models_yaml("http://127.0.0.1:1/v1")
    parsed: Any = __import__("yaml").safe_load(manifest)
    model = parsed["models"][0]
    assert model["model_id"] == bench._MODEL_ID
    assert model["base_url"] == "http://127.0.0.1:1/v1"
    assert parsed["mode_models"]["default"] == bench._MODEL_ID


def test_stub_server_serves_openai_compatible_completion(
    bench: ModuleType,
) -> None:
    """The stub answers ``/chat/completions`` with the fixed deterministic reply."""
    import urllib.request

    server, base_url = bench._start_stub_server()
    try:
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
    assert payload["choices"][0]["message"]["content"] == bench._STUB_ANSWER
    assert payload["choices"][0]["finish_reason"] == "stop"


# ── telemetry phase parsing (spec §7: phases reported separately) ─────


def test_parse_turn_log_phases_reads_jsonl_host_log(bench: ModuleType, tmp_path: Path) -> None:
    """JSONL host-log records unwrap to the stage telemetry events."""
    stages = [
        {"boot_id": "boot-a", "stage": "setup_framework", "event": "begin", "pid": 1},
        {
            "boot_id": "boot-a",
            "stage": "setup_framework",
            "event": "end",
            "pid": 1,
            "seconds": 0.25,
        },
    ]
    lines = [
        json.dumps(
            {"ts": "t", "level": "info", "name": "lisa.startup", "msg": "startup " + json.dumps(s)}
        )
        for s in stages
    ]
    host_log = tmp_path / "host.log"
    host_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    phases = bench._parse_turn_log_phases(tmp_path, host_log)
    assert phases["setup_invocations"] == 1
    assert phases["stages"]["setup_framework"] == pytest.approx(0.25)


def test_parse_turn_log_phases_reports_only_last_boot(bench: ModuleType, tmp_path: Path) -> None:
    """Warm probes reuse the scratch dir; only the final boot's stages count."""

    def record(boot: str, event: str, stage: str, seconds: float) -> str:
        return json.dumps(
            {
                "ts": "t",
                "level": "info",
                "name": "lisa.startup",
                "msg": "startup "
                + json.dumps(
                    {
                        "boot_id": boot,
                        "stage": stage,
                        "event": event,
                        "pid": 1,
                        "seconds": seconds,
                    }
                ),
            }
        )

    host_log = tmp_path / "host.log"
    host_log.write_text(
        "\n".join(
            [
                record("boot-1", "begin", "setup_framework", 0.0),
                record("boot-1", "end", "setup_framework", 1.0),
                record("boot-2", "begin", "setup_framework", 0.0),
                record("boot-2", "begin", "setup_framework", 0.0),
                record("boot-2", "end", "setup_framework", 2.0),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    phases = bench._parse_turn_log_phases(tmp_path, host_log)
    # Last boot only: two begins counted (a real duplicate would fire the
    # framework's repeat_setup_detected warning), one summed duration.
    assert phases["setup_invocations"] == 2
    assert phases["stages"]["setup_framework"] == pytest.approx(2.0)


def test_parse_turn_log_phases_missing_log_is_empty(bench: ModuleType, tmp_path: Path) -> None:
    """An absent host log yields zeroed phases, never a crash."""
    phases = bench._parse_turn_log_phases(tmp_path, tmp_path / "missing.log")
    assert phases["stages"] == {}
    assert phases["setup_invocations"] == 0
    assert phases["duplicate_setups"] == 0
    assert phases["teardown_wall_seconds"] == 0.0


# ── help/version gate (spec §3: no framework sessions) ────────────────


def test_assert_no_framework_sessions_detects_telemetry(bench: ModuleType) -> None:
    """Boot-telemetry lines in probe output fail the help/version gate."""
    assert bench._assert_no_framework_sessions("", "") == []
    leaked = bench._assert_no_framework_sessions(
        "out line",
        'startup {"boot_id": "x", "stage": "setup_framework", "event": "begin"}',
    )
    assert leaked and leaked[0].startswith("startup ")


# ── module-root verification (spec §3) ────────────────────────────────


def test_module_paths_from_env_resolves_installed_roots(bench: ModuleType) -> None:
    """The probe reports the resolved package roots for the report."""
    paths = bench._module_paths_from_env()
    assert paths, "module roots must resolve"
    for name in ("kernel", "runtime"):
        assert name in paths, f"{name} root missing from verification"
        assert Path(paths[name]).is_absolute()


# ── probe failure semantics ───────────────────────────────────────────


def test_run_executable_probe_timeout_yields_failed_sample(bench: ModuleType) -> None:
    """A timing-out probe becomes a failed sample, not an exception."""
    import subprocess

    sample = bench._run_executable_probe(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        variant="warm",
        timeout_seconds=0.2,
        runner=subprocess.run,
    )
    assert sample.exit_code == -1
    assert sample.wall_seconds < 5.0


# ── probe helpers with fake runners ───────────────────────────────────


class _FakeCompleted:
    """A CompletedProcess stand-in the fake runner returns."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_help_version_variant_returns_runs_and_samples(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The help/version probe aggregates cold + warm and enforces the gate.

    The subprocess layer is stubbed: the assertion targets the probe's
    orchestration (sample counts, variant names, gate success), not the
    CLI's own runtime behavior (covered by the CLI suite).
    """
    probes: list[dict[str, Any]] = []

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        probes.append({"command": list(command), **kwargs})
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.5, exit_code=0)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=0, stdout="lisa v0\n", stderr="")

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)
    probed = bench._probe_help_version_variant(
        "/nonexistent/lisa",
        "version",
        ["--version"],
        samples=2,
        timeout_seconds=5.0,
        gate_runner=fake_gate_runner,
    )
    assert not isinstance(probed, int), "gate must pass with clean output"
    runs, samples = probed
    assert [r.variant for r in runs] == ["cold-version", "version"]
    assert len(samples) == 3  # cold + 2 warm
    assert all(s.exit_code == 0 for s in samples)
    assert all(p["command"] == ["/nonexistent/lisa", "--version"] for p in probes)


def test_probe_help_version_variant_gate_fails_on_telemetry(
    bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Startup-telemetry lines in probe output fail the gate with exit 1."""

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.5, exit_code=0)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(
            returncode=0, stderr='startup {"stage": "setup_framework", "event": "begin"}'
        )

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)
    probed = bench._probe_help_version_variant(
        "/nonexistent/lisa",
        "version",
        ["--version"],
        samples=1,
        timeout_seconds=5.0,
        gate_runner=fake_gate_runner,
    )
    assert probed == 1
    assert "gate FAILED" in capsys.readouterr().err


def test_probe_help_version_variant_gate_fails_on_nonzero_exit(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe whose command exits nonzero fails the gate."""

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.5, exit_code=0)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=3, stdout="", stderr="")

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)
    probed = bench._probe_help_version_variant(
        "/nonexistent/lisa",
        "help",
        ["--help"],
        samples=1,
        timeout_seconds=5.0,
        gate_runner=fake_gate_runner,
    )
    assert probed == 1


def test_probe_boot_teardown_uses_installed_interpreter(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boot probe resolves the interpreter beside the installed lisa.

    When ``<lisa-dir>/python`` exists it is probed (the installed CLI's
    own interpreter); otherwise the parent interpreter serves the probe.
    """
    captured: list[list[str]] = []

    def fake_run_child_probe(command, **_kwargs: Any):
        captured.append(list(command))
        return bench.BenchmarkSample(
            variant="boot", wall_seconds=0.1, exit_code=0, teardown_wall_seconds=0.001
        )

    monkeypatch.setattr(bench, "run_child_probe", fake_run_child_probe)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    samples = bench._probe_boot_teardown(str(bin_dir / "lisa"), 2, timeout_seconds=5.0)
    assert len(samples) == 2
    assert captured[0][0] == str(bin_dir / "python")
    assert captured[0][1:] == ["-c", bench.BOOT_PROBE_SOURCE]

    # No sibling python: the parent interpreter is used.
    captured.clear()
    lone = tmp_path / "lone"
    lone.mkdir()
    bench._probe_boot_teardown(str(lone / "lisa"), 1, timeout_seconds=5.0)
    assert captured[0][0] == sys.executable


def test_report_phase_detail_prints_prompt_stages(bench: ModuleType) -> None:
    """The spec §7 phase split prints the aggregated prompt stage seconds."""
    run = bench.BenchmarkRun(
        variant="prompt",
        sample_count=1,
        p50_wall_seconds=1.0,
        p95_wall_seconds=1.0,
        min_wall_seconds=1.0,
        max_wall_seconds=1.0,
        max_setup_invocations=1,
        total_duplicate_setups=0,
        max_shared_sessions=0,
        module_root="",
    )
    sample = bench.BenchmarkSample(
        variant="prompt",
        wall_seconds=1.0,
        exit_code=0,
        stage_durations={"setup_framework": 0.25},
    )
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        bench._report_phase_detail([run], [sample])
    out = buffer.getvalue()
    assert "Prompt phases" in out
    assert "'setup_framework': 0.25" in out

    # Without a prompt run: no output at all.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        bench._report_phase_detail([], [])
    assert buffer.getvalue() == ""


def test_assert_no_leaked_children_counts_direct_children(bench: ModuleType) -> None:
    """The leak scan counts living direct children of this process."""
    import subprocess

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        leaked = bench._assert_no_leaked_children()
        assert leaked >= 1
    finally:
        sleeper.kill()
        sleeper.wait()
    assert bench._assert_no_leaked_children() == 0


def test_seed_scratch_data_dir_writes_manifest_and_session(
    bench: ModuleType, tmp_path: Path
) -> None:
    """The scratch dir gets the stub manifest + the bundled default session."""
    bench._seed_scratch_data_dir(tmp_path, "http://127.0.0.1:1/v1")
    manifest = (tmp_path / "config" / "models.yaml").read_text(encoding="utf-8")
    assert "http://127.0.0.1:1/v1" in manifest
    session = tmp_path / "config" / "session_context.md"
    # The bundled session ships with lisa-pack-core; when present it must
    # be copied verbatim.
    from importlib import resources

    source = resources.files("sdk.packs.core") / "sessions" / "default.md"
    if source.is_file():
        assert session.is_file()
        assert session.read_bytes() == source.read_bytes()


def test_unwrap_host_log_messages_passthrough_plain_lines(bench: ModuleType) -> None:
    """Non-JSON lines pass through; JSON records yield their ``msg``."""
    messages = bench._unwrap_host_log_messages(
        "\n".join(
            [
                "plain startup-ish line",
                json.dumps({"msg": "startup payload"}),
                json.dumps({"other": "ignored"}),
            ]
        )
    )
    # The no-msg record yields an empty message; parse_startup_lines skips
    # non-telemetry lines, so an empty unwrap is inert downstream.
    assert messages == ["plain startup-ish line", "startup payload", ""]


# ── main() integration (help/version path, repo venv lisa) ───────────


def test_main_skip_prompt_smoke(bench: ModuleType, tmp_path: Path) -> None:
    """``main --skip-prompt`` runs the help/version probes end to end.

    Uses this repository venv's installed ``lisa`` console script (the
    spec's installed-executable requirement) with a single warm sample so
    the test stays fast; the gates assert clean output and a zero exit.
    """
    lisa = str(_REPO_ROOT / ".venv" / "bin" / "lisa")
    if not Path(lisa).is_file():
        pytest.skip("repo .venv lisa console script not present")
    json_path = tmp_path / "out.json"
    exit_code = bench.main(
        [
            "--samples",
            "1",
            "--lisa",
            lisa,
            "--skip-prompt",
            "--json",
            str(json_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    variants = {r["variant"] for r in payload["runs"]}
    assert {"cold-version", "version", "cold-help", "help"} <= variants
    assert payload["leaked_children"] == 0


@pytest.mark.integration
def test_main_full_run_with_prompt_and_boot(bench: ModuleType, tmp_path: Path) -> None:
    """``main`` end to end: prompt probes + boot teardown + lifecycle gate.

    The full Slice E protocol against the installed CLI: one cold + one
    warm prompt probe through the loopback stub (first model submission
    measured), the framework boot probe (bounded teardown), and the
    leaked-children gate. Marked ``integration`` (multi-process, ~10s).
    """
    lisa = str(_REPO_ROOT / ".venv" / "bin" / "lisa")
    if not Path(lisa).is_file():
        pytest.skip("repo .venv lisa console script not present")
    json_path = tmp_path / "out.json"
    exit_code = bench.main(["--samples", "1", "--lisa", lisa, "--json", str(json_path)])
    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    variants = {r["variant"] for r in payload["runs"]}
    assert {
        "cold-version",
        "version",
        "cold-help",
        "help",
        "cold-prompt",
        "prompt",
        "boot",
    } <= variants
    # Prompt probes: deterministic stub answered every turn (exit 0).
    prompt_samples = [s for s in payload["samples"] if s["variant"] in {"cold-prompt", "prompt"}]
    assert prompt_samples and all(s["exit_code"] == 0 for s in prompt_samples)
    assert all(s["setup_invocations"] == 1 for s in prompt_samples)
    # Boot probes: bounded teardown, one setup per boot, no duplicates.
    boot_samples = [s for s in payload["samples"] if s["variant"] == "boot"]
    assert boot_samples and all(s["exit_code"] == 0 for s in boot_samples)
    assert all(s["teardown_wall_seconds"] < 30.0 for s in boot_samples)
    assert all(s["duplicate_setups"] == 0 for s in boot_samples)
    # Lifecycle gate: zero leaked owned children.
    assert payload["leaked_children"] == 0


def test_main_without_lisa_returns_two(bench: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """No installed lisa anywhere: exit 2 (spec: never benchmark a checkout import)."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert bench.main([]) == 2


def test_probe_stdout_command_copies_argv(bench: ModuleType) -> None:
    """The stdout probe wraps the argv into a fresh list."""
    argv = ["lisa", "--version"]
    wrapped = bench._probe_stdout_command(argv)
    assert wrapped == argv
    assert wrapped is not argv


def test_parse_turn_log_phases_reads_session_inventory_and_teardown(
    bench: ModuleType, tmp_path: Path
) -> None:
    """shared_session_inventory and pool_readiness end feed the phase map."""

    def record(msg_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "ts": "t",
                "level": "info",
                "name": "lisa.startup",
                "msg": "startup " + json.dumps(msg_payload),
            }
        )

    host_log = tmp_path / "host.log"
    host_log.write_text(
        "\n".join(
            [
                record({"boot_id": "b1", "event": "shared_session_inventory", "sessions": 3}),
                record(
                    {"boot_id": "b1", "event": "end", "stage": "pool_readiness", "seconds": 0.4}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phases = bench._parse_turn_log_phases(tmp_path, host_log)
    assert phases["shared_sessions"] == 3
    assert phases["teardown_wall_seconds"] == pytest.approx(0.4)


def test_assert_no_leaked_children_handles_missing_proc(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without /proc (non-Linux) the leak scan degrades to zero."""

    def _raise_oserror(*_args: object, **_kwargs: object) -> object:
        raise OSError("no procfs")

    monkeypatch.setattr(bench.Path, "iterdir", _raise_oserror, raising=True)
    assert bench._assert_no_leaked_children() == 0


def test_assert_no_leaked_children_skips_malformed_proc_entries(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed /proc entry is skipped, not a crash."""

    class _FakeEntry:
        def __init__(self, name: str) -> None:
            self.name = name

        def __truediv__(self, other: str) -> _FakeEntry:
            return self

        def read_text(self) -> str:
            raise ValueError("malformed stat")

    class _FakeProc:
        def iterdir(self) -> list[_FakeEntry]:
            return [_FakeEntry("1"), _FakeEntry("notapid")]

    monkeypatch.setattr(bench, "Path", lambda _p: _FakeProc(), raising=True)
    assert bench._assert_no_leaked_children() == 0


def test_module_paths_from_env_imports_hosts_cli_root(bench: ModuleType) -> None:
    """The hosts.cli root resolves via import inside the verifier."""
    paths = bench._module_paths_from_env()
    assert "hosts.cli" in paths
    assert paths["hosts.cli"].endswith(("main.py", "__init__.py"))


def test_main_full_run_reports_failed_probe(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing probe (nonzero exit) fails the whole run with exit 1."""
    lisa = str(_REPO_ROOT / ".venv" / "bin" / "lisa")
    if not Path(lisa).is_file():
        pytest.skip("repo .venv lisa console script not present")

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.1, exit_code=7)

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: fake_gate_runner(None))
    exit_code = bench.main(
        ["--samples", "1", "--lisa", lisa, "--skip-prompt", "--json", str(tmp_path / "o.json")]
    )
    assert exit_code == 1


def test_main_propagates_help_version_gate_failure(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gate failure in the first variant stops main with the gate's exit code."""
    import shutil

    lisa = str(_REPO_ROOT / ".venv" / "bin" / "lisa")
    if not Path(lisa).is_file():
        pytest.skip("repo .venv lisa console script not present")

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.1, exit_code=0)

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=0, stderr='startup {"event": "begin"}')

    monkeypatch.setattr(
        bench,
        "_probe_help_version_variant",
        lambda lisa, variant, extra_argv, samples, timeout_seconds, gate_runner=None: (
            (fake_gate_runner(None).returncode and 1)
            or (print(f"gate FAILED for {variant}", file=sys.stderr) or 1)
        ),
    )
    monkeypatch.setattr(shutil, "which", lambda _name: lisa)
    assert bench.main(["--skip-prompt", "--json", str(tmp_path / "o.json")]) == 1


def test_main_fails_on_leaked_children(
    bench: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nonzero leak count fails the lifecycle gate after the report prints."""
    import shutil

    lisa = str(_REPO_ROOT / ".venv" / "bin" / "lisa")
    if not Path(lisa).is_file():
        pytest.skip("repo .venv lisa console script not present")

    def fake_executable_probe(command, **kwargs: Any) -> bench.BenchmarkSample:
        return bench.BenchmarkSample(variant=str(kwargs["variant"]), wall_seconds=0.1, exit_code=0)

    def fake_gate_runner(command, **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench, "_run_executable_probe", fake_executable_probe)
    monkeypatch.setattr(
        bench,
        "_probe_help_version_variant",
        lambda lisa, variant, extra_argv, samples, timeout_seconds, gate_runner=None: (
            [
                bench.summarize_variant(
                    f"cold-{variant}",
                    [bench.BenchmarkSample(variant=variant, wall_seconds=0.1, exit_code=0)],
                ),
                bench.summarize_variant(
                    variant, [bench.BenchmarkSample(variant=variant, wall_seconds=0.1, exit_code=0)]
                ),
            ],
            [bench.BenchmarkSample(variant=variant, wall_seconds=0.1, exit_code=0)],
        ),
    )
    monkeypatch.setattr(bench, "_assert_no_leaked_children", lambda: 2)
    monkeypatch.setattr(shutil, "which", lambda _name: lisa)
    assert bench.main(["--skip-prompt", "--json", str(tmp_path / "o.json")]) == 1


def test_script_entry_point_guard_runs_main() -> None:
    """``python scripts/startup_benchmark.py`` executes main (the __main__ guard).

    A ``--lisa`` pointing at a nonexistent executable fails the gate with
    exit 1 (a failed probe sample, never a crash) — this also pins the
    script's executable entry behavior end to end.
    """
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--lisa", "/nonexistent/lisa", "--skip-prompt"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1
    assert "gate FAILED" in completed.stderr
