# lisa-host-cli

Terminal CLI host for the [Lisa framework](https://github.com/Lisa-Framework/lisa).

## Install

```bash
uv tool install lisa-host-cli
```

Or with pip:

```bash
pip install lisa-host-cli
```

## Usage

```bash
lisa            # start the REPL
lisa verify     # run the diagnostic smoke check
lisa once "summarize the thread"   # run a single turn non-interactively
lisa --help     # show all commands
```

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

This repo is a uv workspace whose only member is `hosts/cli/`. The default
install resolves all framework packages (`lisa-host-runtime`, `lisa-kernel`,
`lisa-sdk`, `lisa-mcp-servers`, the bundled `lisa-pack-*` packs) from PyPI, so
the CLI works as a standalone clone with no sibling framework checkout:

```bash
git clone https://github.com/Lisa-Framework/lisa-host-cli.git
cd lisa-host-cli
uv sync --extra dev
uv run lisa --help
```

### Developing against framework HEAD

When a sibling `../lisa` framework checkout is present, uncomment the
`[tool.uv.sources]` path overrides in `pyproject.toml` to resolve the
framework packages as editable path deps instead of from PyPI:

```text
c:\Code\
├── lisa/              # framework (kernel, sdk, mcp_server, host)
└── lisa-host-cli/     # this repo
```

```bash
cd lisa-host-cli
# edit pyproject.toml: uncomment the lisa-* path sources
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
