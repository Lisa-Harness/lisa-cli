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
lisa            # start the REPL
lisa verify     # run the diagnostic smoke check
lisa once "summarize the thread"   # run a single turn non-interactively
lisa --help     # show all commands
```

### Approvals

Commands execute without an approval prompt only when the bash pack reports a
working, contained Bubblewrap sandbox. Missing, degraded, or unknown sandbox
state fails closed and shows an approval panel instead. In the REPL, use `y` to
approve, `n` to deny, `a` to approve and remember a command prefix, `v` to show
the panel again, or `Esc` to dismiss it. Explicit commands remain available:

```bash
/approve <request-id>
/deny <request-id>
/approve <request-id> --remember [command-prefix]
```

The host reads `${LISA_DATA_DIR:-~/.lisa}/policy.yaml` for durable policy. The
default is `mode: ask` with `sandbox_downgrade: true`. Rules are checked in the
order `deny`, `ask`, then `allow`; a remembered rule is scoped to the tool and
the command prefix, rather than granting broad shell access. `external_write`
actions always require an explicit policy decision because they leave the
sandbox boundary.

### Lazy `once` routing

Set `LISA_ONCE_UNKNOWN=1` (or pass `--once-unknown`) to let an unrecognized
first positional become the `once` message, so you can skip typing
`once` for ad-hoc one-shots:

```bash
LISA_ONCE_UNKNOWN=1 lisa continue
LISA_ONCE_UNKNOWN=1 lisa "do something else"
lisa --once-unknown continue          # one-shot, no env var
```

Registered subcommands (`verify`, `pack`, `models`, …) are never rerouted,
so the flag is safe to leave on. Group-level options and their values
(e.g. `--log-level debug`) are skipped when locating the subcommand
position. The env var is the intended default; the flag is redundant and
exists only for discoverability.

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
with a Typer + Rich + prompt_toolkit terminal interface. The CLI implements
`TransportCallbacks` to render turns in the terminal; the runtime never imports
Rich, Typer, or any transport-specific code.

See `../lisa/docs/BOUNDARIES.md` for the full architecture.
