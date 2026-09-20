# lisa-host-cli

Terminal CLI host for the [Lisa framework](https://github.com/Lisa-Framework/lisa).

## Install

Run the installer (works pre-PyPI: it resolves the packages straight from
git and wires up `~/.lisa/.env`):

```bash
curl -fsSL https://raw.githubusercontent.com/Lisa-Framework/lisa-host-cli/main/install.sh | bash
```

The installer accepts the same options as a direct `uv tool install`
(e.g. `--provider ollama,openai`, `--source local --lisa-repo ~/Code/lisa`).

Or with pip / uv directly:

```bash
uv tool install lisa-host-cli
pip install lisa-host-cli
```

## Usage

```bash
lisa            # show help (every command is one-shot)
lisa prompt "summarize the thread"   # run a single turn
lisa verify     # run the diagnostic smoke check
lisa --help     # show all commands
```

### One-shot pipelines (`prompt` with stdin and `--raw`)

`prompt` reads piped stdin automatically (when stdin is not a tty) and attaches
it to the turn ahead of your instruction, so config/data can flow through the
model into another command:

```bash
cat /etc/sysconfig/whatever.conf | lisa prompt "recommend improvements to this config"
cat whatever.conf | lisa prompt -r "fix it for me." > whatever.conf.new
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

Commands execute without an approval prompt only when the bash pack reports a
working, contained Bubblewrap sandbox. Missing, degraded, or unknown sandbox
state fails closed and shows an approval panel instead. A prompt run decides
with the host machinery (the request id survives the process boundary):

```bash
lisa prompt "approve <request-id>"                        # decide + re-run
lisa prompt "deny <request-id>"
lisa prompt "approve <request-id> --remember [prefix]"
```

### Policy (`lisa prompt policy`)

Inspect and change the approval policy without hand-editing YAML. The command is
`lisa prompt policy` (also reachable as the `policy` decision subcommand):

```bash
lisa prompt policy                     # show the active document
lisa prompt "policy mode trust"        # auto-allow write/execute (ask|trust|plan)
lisa prompt "policy bare off"          # stop prompting for unsandboxed execute
```

Changes take effect immediately — the policy engine re-reads
`${LISA_DATA_DIR:-~/.lisa}/policy.yaml` on every approval request. The default
is `mode: ask` with `sandbox_downgrade: true`. Rules are checked in the
order `deny`, `ask`, then `allow`; a remembered rule is scoped to the tool and
the command prefix, rather than granting broad shell access. `external_write`
actions always require an explicit policy decision because they leave the
sandbox boundary, and `mode: trust` still prompts for unsandboxed `execute`
(`bare_requires_approval: true`, toggle with `policy bare off`).

### Lazy `prompt` routing

Set `LISA_PROMPT_UNKNOWN=1` to let an unrecognized first positional become
the `prompt` message, so you can skip typing `prompt` for ad-hoc one-shots:

```bash
LISA_PROMPT_UNKNOWN=1 lisa continue
LISA_PROMPT_UNKNOWN=1 lisa "do something else"
```

Registered subcommands (`verify`, `pack`, `models`, …) are never rerouted,
so the env var is safe to leave on. Runtime options belong to each
subcommand (`lisa prompt --log-level debug …`). Prompt-only flags
(`-r/--raw`, `-q/--quiet`, `--stdin/--no-stdin`) appearing before the
positional are hoisted after the injected `prompt` token, so
`lisa -r "fix it"` parses as `lisa prompt -r "fix it"`.

### Profiles (`--profile`, `LISA_PROFILE`)

A **profile** is a named sub-workspace of the data dir:
`~/.lisa/profiles/<name>`. It isolates everything derived from the data
dir — threads, `memory.db`, `models.yaml`, `policy.yaml`, approvals,
session context, `files/` — while **sharing the base packs venv**, so
packs are installed once and discovered from every profile.

```bash
lisa --profile taxes            # one-shot: use the 'taxes' profile
LISA_PROFILE=taxes lisa         # env var form (persistent for the shell)
lisa --profile taxes verify     # works on every subcommand
```

Precedence: `--data-dir` (explicit) > `LISA_DATA_DIR` > `--profile` /
`LISA_PROFILE` > `~/.lisa`. A fresh profile starts empty (defaults, like a
fresh install); seed it from the base dir when you want a head start:

```bash
mkdir -p ~/.lisa/profiles/taxes
cp ~/.lisa/models.yaml ~/.lisa/profiles/taxes/
```

In a shell, a small helper makes switching sticky for the session:

```bash
lprof taxes     # export LISA_PROFILE=taxes (creates ~/.lisa/profiles/taxes)
lprof --list    # list profiles, mark the active one
lprof off       # return to the base data dir
```

(The profile name must be a single path-safe segment — letters, digits,
`-`, `_`, `.`.)

### Data directory layout

The data dir (base or profile) groups everything into lifecycle
categories; the names live in one place (`sdk.layout`):

```text
~/.lisa/
├── config/    models.yaml · policy.yaml · mcp_servers.yaml · overlays/ · session_context.md
├── state/     installed-packs.json · state.json · approvals.json ·
│              recovery-items.json · code-change-artifacts.json ·
│              repl_history.txt · models.proposed.yaml
├── data/      memory.db · threads/ · files/ · backups/
├── cache/     fetch/ · layers/    (rm-safe)
├── logs/      host.jsonl · mcp/<server_id>.jsonl · token-usage.jsonl
├── run/       per-session execution state, keyed by session hash:
│              jobs/ · scratch/ · profile/ · sandbox-facts.json · logs (rm-safe)
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
git clone https://github.com/Lisa-Framework/lisa-hosts-cli.git lisa-hosts
# (cli is the lisa-hosts repo; lisa is the framework repo)
git clone https://github.com/Lisa-Framework/lisa.git

cd lisa-hosts/cli
# edit pyproject.toml: uncomment the lisa-* path sources (sibling ../lisa)
uv sync --extra dev
uv run pytest -m "not live"
uv run lisa --help
```

## Architecture

`lisa-host-cli` wraps the transport-neutral `host` runtime (from `lisa-host-runtime`)
with a Typer + Rich terminal interface. Every command is a one-shot process; turns
run through `lisa prompt` and its `PromptTransport` renders the turn trajectory to
scrollback. The CLI implements `TransportCallbacks`; the runtime never imports
Rich, Typer, or any transport-specific code.

See `../lisa/docs/BOUNDARIES.md` for the full architecture.
