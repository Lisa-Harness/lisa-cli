#!/usr/bin/env python3
"""Measure lisa CLI startup latency against the *installed* executable.

Runs the Slice E protocol from
``docs/SPEC/PROMPT_STARTUP_CLI.md`` (CLI half) — the CLI-side consumer of
the framework harness shipped complete at ``Lisa-Harness/lisa`` ``dev`` ≥
``1f19c5f2`` (``runtime.startup_benchmark``): fresh-process probes under
identical interpreter/environment conditions, cold effects separated from
warm samples, p50/p95 aggregation, ``lisa.startup`` telemetry capture
(distinct ``setup_framework`` invocations — the dedup-investigation
evidence; cProfile call counts are NOT evidence), per-sample teardown
timing, and module-root verification of whichever package tree served each
probe.

Variants:

- ``--version`` / ``--help`` — the spec §3 commands. Help/version must run
  with zero framework sessions or model calls (asserted), so these probes
  measure the CLI's own import/dispatch latency, not framework boot.
- ``prompt`` — one ``lisa prompt`` turn against a deterministic local
  OpenAI-compatible stub server seeded into a scratch data dir. The stub
  is a threaded ``HTTPServer`` in the parent process; the probe records
  first-model-submission time from the boot telemetry (the ``pool_readiness``
  stage end to the provider call) plus complete command wall time.

The benchmarked entry must be the installed ``lisa`` executable, never an
editable-checkout import (spec §3 release coordination): the default
``--lisa`` resolves ``shutil.which("lisa")`` and the report carries the
child probes' resolved ``__file__`` paths (module-root verification).

Usage::

    python scripts/startup_benchmark.py --samples 10 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from runtime.startup_benchmark import (
    BOOT_PROBE_SOURCE,
    BenchmarkRun,
    BenchmarkSample,
    child_python_command,
    format_report,
    parse_startup_lines,
    run_child_probe,
    stage_durations,
    summarize_variant,
)

_MODEL_ID = "bench-stub-model"
_STUB_ANSWER = "bench stub response"


def _stub_models_yaml(base_url: str) -> str:
    """Return the deterministic stub manifest for the scratch data dir.

    The model entry points ``base_url`` at the loopback stub server, so the
    first model submission is a local deterministic HTTP call — no network,
    no credentials (spec §7: a deterministic local model stub isolates
    framework startup from provider latency).
    """
    return (
        "\n".join(
            [
                "version: 1",
                f"mode_models: {{default: {_MODEL_ID}}}",
                "models:",
                f"  - model_id: {_MODEL_ID}",
                "    provider: ollama",
                f"    base_url: {base_url}",
            ]
        )
        + "\n"
    )


def _seed_scratch_data_dir(data_dir: Path, base_url: str) -> None:
    """Seed the scratch data dir with the stub manifest + default session.

    The prompt probe needs the same session scaffolding any real boot uses:
    the framework's bundled default session context (from the installed
    ``lisa-pack-core``) provides the routing vocabulary and provider
    selection the kernel wires at boot. Copying it verbatim keeps the probe
    deterministic — one session file, one stub model, no ambient operator
    configuration beyond the pack content shipped with the install.
    """
    from sdk.layout import CONFIG_DIR

    config = data_dir / CONFIG_DIR
    config.mkdir(parents=True, exist_ok=True)
    (config / "models.yaml").write_text(_stub_models_yaml(base_url), encoding="utf-8")

    import importlib.resources as _resources

    try:
        core_sessions = _resources.files("sdk.packs.core") / "sessions"
        source = core_sessions / "default.md"
        if source.is_file():
            (config / "session_context.md").write_bytes(source.read_bytes())
    except (OSError, FileNotFoundError):  # pragma: no cover — no bundled session
        # No bundled session in this install: the probe still runs; the
        # boot surfaces the framework's own startup error if one is
        # required.
        pass


class _StubHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/chat/completions`` returning one fixed answer."""

    def do_POST(self) -> None:  # http.server API
        length = int(self.headers.get("Content-Length", "0") or 0)
        self.rfile.read(length)
        payload = json.dumps(
            {
                "id": "bench-stub",
                "object": "chat.completion",
                "model": _MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _STUB_ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default request logging (benchmark output stays clean)."""
        return


def _start_stub_server() -> tuple[ThreadingHTTPServer, str]:
    """Start the loopback stub on an ephemeral port; return (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/v1"


def _probe_stdout_command(lisa_argv: list[str]) -> list[str]:
    """Wrap one installed-executable probe command (stdout captured by the runner)."""
    return list(lisa_argv)


def _assert_no_leaked_children() -> int:
    """Count leftover processes whose parent was this benchmark run.

    Lifecycle gate (spec §3): zero leaked owned children after teardown.
    A prompt probe's runtime aclose() terminates its MCP/proxy children
    before the process exits; anything still running with a benchmark
    parent means teardown failed. Scans ``/proc`` for children of the
    current process that are still alive.
    """
    leaked = 0
    try:
        proc_children = Path("/proc")
        for entry in proc_children.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == os.getpid() and entry.name != str(os.getpid()):
                leaked += 1
    except OSError:
        # /proc unavailable (non-Linux): the gate degrades to a no-op; the
        # bounded-teardown half of the lifecycle gate still runs.
        return 0
    return leaked


def _run_executable_probe(
    command: Sequence[str],
    *,
    variant: str,
    timeout_seconds: float = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    env: dict[str, str] | None = None,
) -> BenchmarkSample:
    """Run one installed-executable probe and fold it into a sample.

    Mirrors :func:`runtime.startup_benchmark.run_child_probe` (same
    failure-to-sample semantics) but for a console-script argv whose child
    prints no JSON payload — the executable's own wall time, exit code, and
    (for prompt probes) the turn log's ``lisa.startup`` telemetry are the
    measurement. Teardown time comes from the child's own
    ``pool_readiness``/``boot_complete`` boundaries where present, and from
    the parent-observed process exit otherwise.
    """
    started = time.perf_counter()
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkSample(
            variant=variant, wall_seconds=time.perf_counter() - started, exit_code=-1
        )
    except OSError:
        # The executable vanished or is unreadable (same failure semantics
        # as the framework's run_child_probe: a failed sample, not a crash,
        # so one bad probe cannot destroy the sweep).
        return BenchmarkSample(
            variant=variant, wall_seconds=time.perf_counter() - started, exit_code=-1
        )
    wall = time.perf_counter() - started
    return BenchmarkSample(variant=variant, wall_seconds=wall, exit_code=completed.returncode)


def _prompt_probe_sample(
    command: Sequence[str],
    *,
    data_dir: Path,
    host_log: Path,
    variant: str,
    timeout_seconds: float = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    env: dict[str, str] | None = None,
) -> tuple[BenchmarkSample, dict[str, object]]:
    """Run one ``lisa prompt`` probe; return its sample + phase map.

    The scratch data dir (``LISA_DATA_DIR`` via ``env``) holds the host
    log with every ``lisa.startup`` line the boot emitted; after the
    child exits, the probe parses them — the same telemetry payload the
    framework boot probe captures — and derives the spec §7 phase split:

    - ``cli_imports`` — the child's own import wall is not observable from
      a console script; the import share is reported through the
      ``--help`` variant instead.
    - ``framework_setup`` — summed ``runtime_construct`` +
      ``state_snapshot_load`` + ``discover_framework`` +
      ``setup_framework`` + ``configure_runtime`` + ``compile_mcp_configs``
      + ``compile_proxy_configs`` stage seconds.
    - ``required_server_readiness`` — ``pool_readiness`` stage seconds.
    - ``first_model_submission`` — the stub answers on loopback
      immediately; the reported prompt wall includes the kernel turn
      pipeline and the model round trip on top of the setup phases.
    - ``complete_wall`` — the parent-observed process wall.
    """
    sample = _run_executable_probe(
        command,
        variant=variant,
        timeout_seconds=timeout_seconds,
        runner=runner,
        env=env,
    )
    phases = _parse_turn_log_phases(data_dir, host_log)
    sample = BenchmarkSample(
        variant=sample.variant,
        wall_seconds=sample.wall_seconds,
        exit_code=sample.exit_code,
        import_wall_seconds=0.0,
        module_paths=_module_paths_from_env(),
        stage_durations=phases["stages"],  # type: ignore[arg-type]
        setup_invocations=phases["setup_invocations"],  # type: ignore[typeddict-item]
        duplicate_setups=phases["duplicate_setups"],  # type: ignore[typeddict-item]
        shared_sessions=phases["shared_sessions"],  # type: ignore[typeddict-item]
        teardown_wall_seconds=phases["teardown_wall_seconds"],  # type: ignore[typeddict-item]
    )
    return sample, phases


def _module_paths_from_env() -> dict[str, str]:
    """Resolve the installed package roots from this (parent) environment.

    Module-root verification (spec §3): the report must state which tree
    served the samples. The child probes run the installed executable,
    so the parent resolves the same interpreter's installed roots.
    """
    import importlib

    paths: dict[str, str] = {}
    for name in ("kernel", "runtime", "hosts.cli"):
        try:
            module = importlib.import_module(name)
            resolved = str(getattr(module, "__file__", "") or "")
            if resolved:
                paths[name] = resolved
        except ImportError:
            continue
    return paths


def _unwrap_host_log_messages(text: str) -> list[str]:
    """Extract log messages from a host-log JSONL file (``msg`` per record).

    The rotating JSONL handler writes one record per line with the message
    in ``msg``; non-JSON lines pass through verbatim so the same unwrap
    works for a plain-text host log.
    """
    messages: list[str] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            messages.append(line)
            continue
        msg = record.get("msg", "")
        if isinstance(msg, str):
            messages.append(msg)
    return messages


def _parse_turn_log_phases(data_dir: Path, host_log: Path) -> dict[str, object]:
    """Parse one data dir's startup telemetry into stage durations + counts.

    The prompt probes pass ``--host-log <data>/host.log`` so every
    ``lisa.startup`` line the framework emits lands in one file the parent
    controls. That file is JSONL (one record per line, the message in
    ``msg``), so each record is unwrapped first and then handed to
    ``runtime.startup_benchmark.parse_startup_lines`` — the same parser
    the framework boot probe uses, so the payload stream stays identical.

    Only the LAST boot's stages are reported: warm probes reuse the
    scratch dir, so the file accumulates one boot per sample.
    """
    events: list[dict[str, object]] = []
    if host_log.is_file():
        for message in _unwrap_host_log_messages(host_log.read_text(encoding="utf-8")):
            events.extend(parse_startup_lines(message))
    # Keep only the final boot's events (one boot_id per prompt probe).
    last_boot = ""
    for event in events:
        boot = str(event.get("boot_id", ""))
        if boot:
            last_boot = boot
    boot_events = [e for e in events if not last_boot or e.get("boot_id") == last_boot]
    stage_seconds = stage_durations(boot_events)  # type: ignore[arg-type]
    setup_count = sum(
        1 for e in boot_events if e.get("stage") == "setup_framework" and e.get("event") == "begin"
    )
    duplicate = sum(1 for e in boot_events if e.get("event") == "repeat_setup_detected")
    shared = 0
    for e in boot_events:
        if e.get("event") == "shared_session_inventory":
            shared = int(e.get("sessions", 0) or 0)
    teardown = 0.0
    for e in boot_events:
        if e.get("event") == "end" and e.get("stage") == "pool_readiness":
            teardown = max(teardown, float(e.get("seconds", 0.0) or 0.0))
    return {
        "stages": stage_seconds,
        "setup_invocations": setup_count,
        "duplicate_setups": duplicate,
        "shared_sessions": shared,
        "teardown_wall_seconds": teardown,
    }


def _assert_no_framework_sessions(stdout: str, stderr: str) -> list[str]:
    """Fail the help/version gate when framework boot machinery printed.

    Spec §3 acceptance: help/version must show no framework sessions or
    model calls. The probes assert the negative: no startup telemetry lines
    in the child output (a booted runtime always emits stage lines to the
    turn log/stderr at debug level and to the console activity path).
    """
    text = f"{stdout}\n{stderr}"
    return [line for line in text.splitlines() if line.startswith("startup ")]


class _GateResult:
    """A minimal gate-check result for timeout/missing-executable failures."""

    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _probe_help_version_variant(
    lisa: str,
    variant: str,
    extra_argv: list[str],
    samples: int,
    timeout_seconds: float,
    *,
    gate_runner: Callable[..., Any] | None = None,
) -> tuple[list[BenchmarkRun], list[BenchmarkSample]] | int:
    """Probe one help/version variant: cold + N warm + the no-sessions gate.

    Returns the new runs and samples, or an int exit code when the gate
    fails (spec §3: help/version must show no framework sessions).
    ``gate_runner`` injects the subprocess entry for the gate check
    (tests); production leaves it ``None`` for ``subprocess.run``.
    """
    command = [lisa, *extra_argv]
    cold = _run_executable_probe(
        command, variant=f"cold-{variant}", timeout_seconds=timeout_seconds
    )
    warm = [
        _run_executable_probe(command, variant=variant, timeout_seconds=timeout_seconds)
        for _ in range(max(samples, 1))
    ]
    # Help/version gate: one probe with output capture asserts no framework
    # telemetry leaked into the console path.
    try:
        gate: Any = (
            gate_runner(
                command, capture_output=True, text=True, check=False, timeout=timeout_seconds
            )
            if gate_runner is not None
            else subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=timeout_seconds
            )
        )
    except subprocess.TimeoutExpired:
        gate = _GateResult(returncode=-1, stdout="", stderr="")
    except OSError:
        gate = _GateResult(returncode=-1, stdout="", stderr="")
    leaked = _assert_no_framework_sessions(gate.stdout, gate.stderr)
    if leaked or gate.returncode != 0:
        print(
            f"gate FAILED for {variant}: leaked {leaked[:2]}, exit {gate.returncode}",
            file=sys.stderr,
        )
        return 1
    return [
        summarize_variant(f"cold-{variant}", [cold]),
        summarize_variant(variant, warm),
    ], [cold, *warm]


def _probe_prompt_variant(
    lisa: str,
    samples: int,
    timeout_seconds: float,
) -> tuple[list[BenchmarkRun], list[BenchmarkSample]]:
    """Probe the ``lisa prompt`` variant against the deterministic stub."""
    server, base_url = _start_stub_server()
    try:
        with tempfile.TemporaryDirectory(prefix="lisa-cli-bench-") as scratch:
            data_dir = Path(scratch)
            _seed_scratch_data_dir(data_dir, base_url)
            host_log = data_dir / "host.log"
            env = {**os.environ, "LISA_DATA_DIR": str(data_dir)}
            command = [
                lisa,
                "prompt",
                "--raw",
                "--streaming",
                "none",
                "--host-log",
                str(host_log),
                "bench stub turn",
            ]
            cold, _ = _prompt_probe_sample(
                command,
                data_dir=data_dir,
                host_log=host_log,
                variant="cold-prompt",
                timeout_seconds=timeout_seconds,
                env=env,
            )
            warm: list[BenchmarkSample] = []
            for _ in range(max(samples, 1)):
                # Each warm sample reuses the same data dir; the host log
                # appends, so parse only the LAST boot's stages.
                sample, _ = _prompt_probe_sample(
                    command,
                    data_dir=data_dir,
                    host_log=host_log,
                    variant="prompt",
                    timeout_seconds=timeout_seconds,
                    env=env,
                )
                warm.append(sample)
            return (
                [
                    summarize_variant("cold-prompt", [cold]),
                    summarize_variant("prompt", warm),
                ],
                [cold, *warm],
            )
    finally:
        server.shutdown()
        server.server_close()


def _probe_boot_teardown(
    lisa: str,
    samples: int,
    timeout_seconds: float,
) -> list[BenchmarkSample]:
    """Run the framework boot probe from the installed CLI's interpreter.

    Lifecycle gate (spec §3): the framework's boot probe (shipped at
    ``1f19c5f2``) builds a full runtime against a stub manifest and times
    ``HostRuntime.aclose()`` directly, producing the bounded-teardown
    metric. Running it with the same interpreter that served the installed
    ``lisa`` keeps the measurement against the release-coordinated tree.
    """
    boot_python = str(Path(lisa).resolve().parent / "python")
    boot_command = child_python_command(
        boot_python if Path(boot_python).exists() else sys.executable,
        BOOT_PROBE_SOURCE,
    )
    return [
        run_child_probe(boot_command, variant="boot", timeout_seconds=timeout_seconds)
        for _ in range(max(samples, 1))
    ]


def _report_phase_detail(runs: list[BenchmarkRun], samples: list[BenchmarkSample]) -> None:
    """Print the spec §7 phase split for the prompt variant (stage seconds)."""
    prompt_run = next((r for r in runs if r.variant == "prompt"), None)
    if prompt_run is None:
        return
    print("## Prompt phases (lisa.startup telemetry, seconds)")
    for sample in samples:
        if sample.variant == "prompt" and sample.stage_durations:
            stages = {k: round(v, 4) for k, v in sorted(sample.stage_durations.items())}
            print(f"- boot stages: {stages}")
            break


def main(argv: list[str] | None = None) -> int:
    """Run the CLI-half startup benchmark and publish the report."""
    args, lisa = _parse_args(argv)
    if lisa is None:
        return 2

    # The parent must import the harness from the same environment the
    # installed CLI runs against; its own module roots double as the
    # report's module-path verification.
    _module_paths_from_env()

    runs: list[BenchmarkRun] = []
    samples: list[BenchmarkSample] = []

    for variant, extra_argv in (("version", ["--version"]), ("help", ["--help"])):
        probed = _probe_help_version_variant(lisa, variant, extra_argv, args.samples, args.timeout)
        if isinstance(probed, int):
            return probed
        variant_runs, variant_samples = probed
        runs.extend(variant_runs)
        samples.extend(variant_samples)

    if not args.skip_prompt:
        prompt_runs, prompt_samples = _probe_prompt_variant(lisa, args.samples, args.timeout)
        runs.extend(prompt_runs)
        samples.extend(prompt_samples)

        # Lifecycle gate (spec §3): bounded teardown + zero leaked owned
        # children, measured by the framework boot probe against the same
        # interpreter that served the installed lisa.
        boot_samples = _probe_boot_teardown(lisa, args.samples, args.timeout)
        runs.append(summarize_variant("boot", boot_samples))
        samples.extend(boot_samples)

    print(format_report(runs))
    _report_phase_detail(runs, samples)

    # Lifecycle gate (spec §3): zero leaked owned children after the
    # prompt probes — the runtime's aclose must have terminated every
    # MCP/proxy child before each probe process exited.
    leaked = _assert_no_leaked_children()
    print("## Lifecycle gate")
    print(f"- leaked owned children after prompt probes: {leaked}")

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "lisa": lisa,
                    "leaked_children": leaked,
                    "runs": [run.to_dict() for run in runs],
                    "samples": [asdict(sample) for sample in samples],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    failed = [sample for sample in samples if sample.exit_code != 0]
    if failed:
        print(f"{len(failed)} probe(s) failed; treat the run as invalid", file=sys.stderr)
        return 1
    if leaked:
        print(f"lifecycle gate FAILED: {leaked} leaked owned children", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, str | None]:
    """Parse argv and resolve the installed lisa executable to probe."""
    import shutil

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--samples", type=int, default=10, help="warm fresh-process samples (spec: 10)"
    )
    parser.add_argument(
        "--lisa",
        default=None,
        help="lisa executable to probe (default: shutil.which('lisa') — the installed CLI)",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="skip the prompt/stub variant (e.g. on a machine without the framework)",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-probe timeout")
    parser.add_argument("--json", type=Path, default=None, help="write full JSON results")
    args = parser.parse_args(argv)
    lisa = args.lisa or shutil.which("lisa")
    if not lisa:
        print(
            "No installed `lisa` executable on PATH (spec §3: probe the installed CLI).",
            file=sys.stderr,
        )
    return args, lisa


if __name__ == "__main__":
    raise SystemExit(main())
