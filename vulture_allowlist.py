"""Vulture false-positive allowlist for lisa-cli.

These symbols are live at runtime but invisible to static analysis because
they are reached via dynamic dispatch (entry-point loading, getattr, importlib)
or are public API surface declared in ``__all__`` but never imported internally.

Every entry carries a ``# reason:`` comment explaining why vulture should
ignore it.  Run::

    uv run vulture hosts/cli vulture_allowlist.py
"""

# ────────────────────────────────────────────────────────────────────────
# Console-script entry points — called by pip-installed [project.scripts].
# ────────────────────────────────────────────────────────────────────────
# reason: console_script target — lisa CLI
hosts.cli.main.main  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Typer command functions — dispatched via Typer's command registration.
# ──────────────────────────────────────────────────────────────────────────
# reason: Typer command callback — lisa verify
hosts.cli.commands.verify.verify  # noqa: Vulture

# reason: Typer command callback — lisa models
hosts.cli.commands.models.models  # noqa: Vulture

# reason: Typer command callback — lisa packs
hosts.cli.commands.packs.packs  # noqa: Vulture

# reason: Typer command callback — lisa prompt
hosts.cli.commands.prompt.prompt  # noqa: Vulture

# reason: Typer command callback — lisa update
hosts.cli.commands.update.update  # noqa: Vulture

# reason: Typer command callback — lisa restart
hosts.cli.commands.restart.restart  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Protocol / ABC method parameters — vulture cannot see that these are
# interface contracts.  The parameters are part of the signature callers
# must satisfy, even if the Protocol/ABC body (``...`` / ``raise`` / pass)
# never reads them.
# ──────────────────────────────────────────────────────────────────────────
# reason: CLI command protocol parameter — interface contract
hosts.cli.commands.base.CommandProtocol.execute  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Pydantic models used for structured output / config — loaded via
# string dispatch or dynamic import.
# ──────────────────────────────────────────────────────────────────────────
# reason: Pydantic model for CLI config loaded dynamically
hosts.cli.config.Config  # noqa: Vulture

# reason: Pydantic model for prompt transport loaded dynamically
hosts.cli.transport.PromptTransport  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Rich renderables / console components — used via Rich's render protocol.
# ──────────────────────────────────────────────────────────────────────────
# reason: Rich renderable protocol — __rich_console__ / __rich__
hosts.cli.render.Renderable.__rich_console__  # noqa: Vulture

# reason: Rich renderable protocol — __rich_console__ / __rich__
hosts.cli.render.Renderable.__rich__  # noqa: Vulture

# reason: Rich renderable protocol — __rich_console__ requires ConsoleOptions parameter
hosts.cli.transport.Spinner.__rich_console__.options  # noqa: Vulture

# reason: Rich renderable protocol — __rich_console__ requires ConsoleOptions parameter
hosts.cli.transport.ActivityDisplay.__rich_console__.options  # noqa: Vulture