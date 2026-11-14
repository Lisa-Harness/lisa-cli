# CLI integration test layer — scripted-model scenarios + wheel-pinned compatibility

Status: approved direction (2026-11-13). Sequencing vs the `main.py`
decomposition is decided in [MAIN_PY_DECOMPOSITION.md](MAIN_PY_DECOMPOSITION.md).

## Problem

The CLI's 501-test suite is unit-shaped: every runtime-touching test
monkeypatches at least one seam (`_initialize_runtime_or_exit`,
`runtime.handle_message.handle_message`, `PromptTransport.__init__`). Those
mocks are exactly the seams a future refactor moves, so they cannot catch a
decomposition that rewires them. Nothing asserts the un-patched pipeline
(runtime construction → gateway wiring → turn → transport → teardown →
persistence), and nothing certifies the CLI against the framework *release*
its users install.

`lisa verify` in the `diagnose-smoke` job covers startup diagnostics only —
not a turn.

## Validated facts (measured 2026-11-13, lisa-cli v0.4.0 workspace)

- A full offline turn through the production pipeline costs ~1.7 s boot +
  ~0.2 s turn (session-scoped catalog subprocess amortizes across tests).
- The gateway override seam
  (`runtime.session.provider_gateways_override`, set before the first turn)
  replaces every provider without monkeypatching; entry-point gateway
  discovery is not consulted while it is set (asserted upstream by
  `runtime/tests/test_gateway_provider.py`).
- `runtime.testing.build_test_runtime(tmp_path, setup_framework=True)` wires
  the full framework (routing vocabulary, catalog, bundled-pack tool surface)
  offline; real tools (`web_search` stub, memory operations) appear in
  `request.tools`.
- `lisa-harness-e2e` publishes `e2etest.scripted`
  (`ScriptedGatewayProvider`, `Reply`, `ScriptTurn`) to PyPI since v0.5.4, so
  the scripted double the scenarios need is importable from the wheel-pinned
  dev environment.

## Design

### Layer definition

New suite: `hosts/cli/tests/integration/` (a subpackage, so pytest discovers
it under the existing `hosts.cli.tests` testpath — no pyproject change). Every
test is marked `pytest.mark.integration` (marker already registered in both
pyprojects). The suite runs in the standard `-m "not live"` scope
(`scripts/check_coverage.py --run` passes `-m "not live"`), so it is required
CI scope, not opt-in.

Fixtures in `hosts/cli/tests/integration/conftest.py`:

1. `offline_runtime` — builds a fresh runtime per test via
   `hosts.cli.main._initialize_runtime` with `data_dir=<tmp>`,
   `session_context=<fixture plan>`, `in_process_mcp=True`; the catalog
   subprocess is session-scoped (mirroring the framework suite's
   `lisa-catalog` contract).
2. `scripted_gateway` — factory over `e2etest.scripted.ScriptedGatewayProvider`;
   each test supplies `respond(turn)` / `auxiliary(turn)` as inline functions
   or small modules under `integration/model/`, and the fixture sets
   `runtime.session.provider_gateways_override` (clearing it on teardown —
   one override object per runtime).
3. `offline_overlay` — writes the web_search overlay
   (`tools/web_search/web_search.py` re-exporting the core stub +
   `web_search.tool.yml` pinning `tool_provider_id` / `default_available`,
   copied from `e2etest/base_data/config/overlays` at package time) into
   `<tmp>/config/overlays/`, plus the session-plan Markdown selecting the
   overlay provider.

### Scenario entry points (the seams the refactor must preserve)

The suite targets module-level entry points, not Typer internals, so the
decomposition slices keep them callable:

- Runtime construction: `hosts.cli.main._initialize_runtime(**kwargs)`.
- Gateway scripting: `runtime.session.provider_gateways_override`.
- The turn: `hosts.cli.main._run_prompt_with_teardown(runtime, message,
  transport=<StringIO PromptTransport>)` — the real CLI turn path including
  bounded teardown and the activity display.
- Decision round-trip: `runtime.handle_message.decide_pending_approval`
  against the durable approval record the first run left in `<data_dir>`.

### Scenarios (M1 scope, five tests)

| Scenario | Drives | Asserts |
| --- | --- | --- |
| scripted one-shot happy path | full pipeline, no mocks | response text; `on_response` / `model_call` stages reached the transport; thread persisted under `<data_dir>/threads`; zero leaked children after bounded teardown |
| tool dispatch → deterministic error | scripted model calls `web_search` (overlay stub), then synthesizes | error reached the model (script reads the stub's message via `turn.outcomes()`); turn completes `not is_error`; one dispatch stage pair |
| approval mint → resume | scripted model calls a write-effect operation (`synthesize_and_store_learning`); second runtime over the same tmp dir | first run surfaces the pending approval; record durable; `decide_pending_approval` approves it; post-approval continuation turn runs; ledger shows one settled record |
| restart / profile isolation | two runtimes, one profile dir, one pack-catalog subprocess | second run sees the first run's thread state; no duplicate framework setups (lifecycle gate from the startup benchmark) |
| decision-only run | `_run_prompt_with_teardown` with a decision + empty message | decision reply rendered, no turn dispatched, pool closed |

Out of scope for M1: live providers (the `live` marker stays opt-in), MCP
subprocess servers (in-process pool only), and the `/approve` string grammar
(the typed `decide_pending_approval` path is what the CLI owns).

### CI wiring

1. No new required job: the suite runs inside `test` (matrix, 3 OS ×
   Python 3.11/3.12) via `scripts/check_coverage.py --run`, which already
   passes `-m "not live"`.
2. `diagnose-smoke` gains one step after `lisa verify`:
   `uv run --no-sync pytest hosts/cli/tests/integration -q -m integration`.
   No `pytest-timeout` dependency is added; wall-clock safety stays with the
   CLI's bounded teardown (`_aclose_runtime_bounded`).

### Wheel pinning

Problem: local dev resolves `lisa-harness-*` from editable `../lisa` sources;
CI's `uv sync --no-sources` (quality/test jobs) resolves against the
registry, so a registry-side framework release can silently drift the
resolved wheel set between runs, and nothing checks newer-than-pinned
releases.

The pinning mechanism is the **committed `uv.lock`**: `--no-sources`
re-resolves against PyPI but keeps the locked version whenever one is still
available, so the resolved wheel set does not drift between runs. A newer
framework release is only picked up when the lock is deliberately bumped.
(`--frozen` is *not* used here: uv rejects `--no-sources` together with
`--frozen` — they are mutually exclusive — and CI has no sibling `../lisa`
checkout for the lock's editable paths to resolve against anyway.)

Two parts:

1. **Pinned matrix** — every CI sync that installs framework wheels keeps
   `--no-sources` (quality, test, changed-coverage, diagnose-smoke) and
   resolves the versions committed in `uv.lock`. The scripted double the
   scenarios need (`e2etest.scripted`) ships in the published
   `lisa-harness-e2e` distribution, which is a dev dependency of the root
   workspace so CI installs it as a wheel like every other framework package.
2. **Drift probe (advisory)** — a weekly scheduled workflow
   (`framework-compat.yml`, shaped like mutation.yml) installs the latest
   satisfying framework wheels (`uv sync --no-sources --extra dev
   --extra prompt` then `uv pip install --upgrade <five framework dists>`)
   and runs only
   `pytest hosts/cli/tests/integration -m integration -q`. Red ⇒ the CLI
   needs a lock bump (or a floor fix) before users hit it; red does **not**
   gate merges.

### Gate-parity ride-alongs

- `xfail_strict = true` in `[tool.pytest.ini_options]` (framework parity;
  no xfails exist today, so this only converts silent passes into failures
  later).
- Coverage denominators are unaffected: the scenarios live under `tests/`,
  which the coverage config already omits.

## Acceptance

- `pytest hosts/cli/tests/integration -m integration -q` green offline with
  no provider credentials on all three OSes in CI.
- `scripts/check_coverage.py --run` (required-CI gate) passes with the suite
  included; the 85 % package floor is unchanged.
- Killing the gateway seam (asserting against an unset
  `provider_gateways_override`) fails the suite — no mock sits between the
  CLI and the kernel turn runner.
