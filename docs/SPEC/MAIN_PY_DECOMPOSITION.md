# main.py decomposition — should we, and in what order relative to the integration layer?

Status: recommendation (2026-11-13). This memo decides sequencing; the
layer spec lives in [CLI_INTEGRATION_LAYER.md](CLI_INTEGRATION_LAYER.md).

## The short answer

**Yes, refactor — it is the highest-leverage maintainability fix in this
repo — but write the integration suite first, then decompose under it.**
The two workstreams are sequenced for a reason: the unit suite mocks the
very seams a decomposition rewires, so decomposing first means testing the
surgery with paper over the wound. The spec's five scenarios are cheap
(~10 s total offline) and directly assert the module-boundary contracts the
slices must preserve.

## Why refactor (evidence)

State of `hosts/cli/src/hosts/cli/main.py` (measured today):

| Fact | Value |
| --- | --- |
| Size | 3,463 lines (76 % of the package's 4,560 non-transport lines) |
| Top-level defs | 108 |
| Sections (the author's own `# ---` banners) | 16 |
| Commands on `app` | 18 static + dynamic pack commands |
| Tests importing `cli_main` | 19 of 32 suites, via `runner.invoke` and private callables (`cli_main._models_command(...)`) |
| Churn, last 6 months | 59 of 173 commits touch it (34 %) |

Symptoms that are architectural, not cosmetic:

1. **Every test works around the module.** Suites patch
   `_initialize_runtime_or_exit`, rebuild `Console`s, and call private
   functions rather than `app` because the app is one flat namespace with
   52 module-global `console.` uses and a module-level `Console()`
   singleton. The verify suite alone carries hand-rolled stub runtimes
   (`_StubRuntime`, `_StubConfig`) to avoid constructing the real one.
2. **The startup-deferral contract couples everything.**
   `test_startup_deferred_imports.py` (816 lines) audits *this module's
   body*; the lazy-import discipline it enforces is why ~19 functions
   carry local imports with rationale comments. A decomposition into
   command modules preserves the audit (module bodies stay light) while
   resting most commands on normal top-level imports again — the deferral
   was fighting the monolith, not the architecture.
3. **The natural seams already exist on the framework side.**
   `render.py` (presentation), `transport.py` (turn UI), `update.py`
   (pack-update policy), and `expand.py` (editor substitution) already
   live outside main.py, and heavy framework imports route through
   `runtime.*` lazily. The monolith is the CLI's *dispatch table plus IO
   helpers* layered on that structure.

### What main.py owns that is *not* incidental

- `_mirror_*` functions: import-free mirrors of the framework's
  data-dir/profile/venv resolvers, required by the PROMPT_STARTUP
  contract and locked to framework behavior by
  `test_startup_deferred_imports.py`. These stay together in one module —
  they are a designed seam, not accidents.
- `_LisaGroup` / `_cli_callback` / dynamic `_pack_command` factory:
  the argv grammar. One module.
- Shared `Annotated` option aliases (~140 lines). One module, imported
  by every command module.
- `_initialize_runtime[_or_exit]` + `_bootstrap_env_and_packs` +
  `_render_runtime_error`: the runtime-construction layer. One module.

## Slice plan (each slice: extract → retarget the touching tests → green)

Order chosen so dependencies flow one way and each slice is independently
shippable.

| Slice | Extract to | Contents (source lines) | Test moves |
| --- | --- | --- | --- |
| 0 (pre) | — | land the integration suite (5 scenarios, per the layer spec) — required before slice 1 | none |
| 1 | `hosts/cli/options.py` | the `Annotated` option aliases + exit-code constants (809–1050, ~240 lines) | import-path updates only |
| 2 | `hosts/cli/bootstrap.py` | `_mirror_*`, `_load_env_once`, `_inject_packs_venv_site_packages_once`, `_bootstrap_env_and_packs`, `_initialize_runtime[_or_exit]`, `_render_runtime_error` (≈ 660 lines) | `test_profile_*`, the deferred-import audit retargets |
| 3 | `hosts/cli/group.py` | `_LisaGroup`, `_cli_callback`, `_prompt_unknown_active`, `_pack_commands`, `_make_pack_command` (506–808) | `test_group_resolution`, the routing half of `test_pack_update` |
| 4 | `hosts/cli/commands/<verb>.py` | one module per existing banner: `verify.py`, `catalog.py` (modes/operations/tools/last/models), `runtime_cmds.py` (settings/tokens/debug), `mcp.py`, `packs.py`, `update.py` (1230–2631, ~1,390 lines) | `test_models_command`, `test_verify_command`, `test_render_catalogs`, `test_update_command`, the per-command suites |
| 5 | `hosts/cli/prompt.py` | `_PrearmedDecision`, `_compose_prompt_message`, `_prompt_command`, `_collect_interactive_or_error`, `_decide_approval_turn`, `_run_decision`, `_run_host_command[_turn]`, `_run_prompt_with_teardown`, `_collect_model_prices` (2632–3410, ~780 lines) | `test_prompt_*` (≈ 3,500 existing test lines), `test_runtime_teardown`, `test_restart` |
| 6 | `hosts/cli/app.py` | `Typer` construction, `intake`, `main()`; `main.py` remains the entry-point module re-exporting `app` / `main` so the `[project.scripts]` target and 15 test import sites stay untouched | none (imports unchanged) |

Invariants every slice must hold:

- No CLI module body has a heavy module-level import; the fresh-process
  probe audit widens from "this module imports no heavy dependency" to
  "the `hosts.cli` package imports none" (the probe imports the package).
- `[project.scripts] lisa = "hosts.cli.main:main"` is untouched.
- `lisa --help` and every `lisa <cmd> --help` output is byte-identical
  (panels, epilogs, option help included).
- The 85 % branch floor never dips: extraction is line-neutral; moving
  code out of `main.py` changes the denominator only if lines become
  dead, which the suite prevents.

## Effort and risk

- Slices 1–3 are mechanical (~330 lines out) with focused test
  retargets; a day including CI.
- Slice 4 is the bulk but the best-covered (the per-command suites
  already assert behavior through `runner.invoke` / private callables —
  the callables' import path changes, their behavior does not).
- Slice 5 is where behavior can silently change (the prompt section is
  the file's most stateful: hoist flags, transport lifecycle,
  decisions, bounded teardown). This is exactly why the integration
  approval / decision-only scenarios land first.
- Six small PRs, no big-bang. After slice 6, `main.py` is ~100 lines and
  each command module is 100–800 lines with its own focused suite.

## What we deliberately do NOT do

- No re-introduction of eager `runtime.*` imports: the deferral contract
  stays — it is a user-visible latency feature with a benchmark and a
  spec of its own.
- No console-injection framework: presentation-owning modules keep a
  module-scoped `Console()`; transports already parametrize consoles
  where it matters (`PromptTransport(console=...)`).
- No decomposition of `transport.py` (2,220 lines) in this effort: it
  has the densest test coverage in the repo (1,501 lines) and is not the
  churn hotspot main.py is.
- No Typer → Click conversion or callback rewrite; only movement.

## Sequencing decision

```
PR 1  integration scenarios + --frozen wheels + xfail_strict   (layer spec M1)
PR 2  slices 1–3 (options, bootstrap, group)
PR 3  slice 4 (command modules)
PR 4  slice 5 (prompt) — approval / decision-only scenarios green before + after
PR 5  slice 6 (app.py + re-exports) — main.py shrinks to ~100 lines
```

The integration suite is the refactor's safety net *and* its
boundary spec: each scenario names the entry points the slices must not
break, so PR 1 contains no `main.py` movement at all.
