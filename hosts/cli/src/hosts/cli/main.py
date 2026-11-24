"""Compatibility entry point for the decomposed Lisa CLI."""

from __future__ import annotations

# Typer command registration follows the original CLI help-panel order.
# ruff: noqa: I001

import sys
from types import ModuleType
from typing import Any

from hosts.cli import application, bootstrap, cli_runtime
from hosts.cli.commands import verify
from hosts.cli.commands import catalog
from hosts.cli.commands import host
from hosts.cli.commands import mcp
from hosts.cli.commands import packs
from hosts.cli.commands import sessions
from hosts.cli.commands import update_command
from hosts.cli.commands import workspace_cli
from hosts.cli import prompt
from hosts.cli.commands import intake

app = application.app
console = application.console

_COMPAT_MODULES: tuple[ModuleType, ...] = (
    prompt,
    catalog,
    host,
    sessions,
    mcp,
    packs,
    update_command,
    workspace_cli,
    intake,
    verify,
    cli_runtime,
    bootstrap,
    application,
)


class _CompatibilityModule(ModuleType):
    """Forward legacy private-seam monkeypatches to the modules that own them."""

    def __setattr__(self, name: str, value: Any) -> None:
        modules = (
            (bootstrap,)
            if name == "__file__"
            else tuple(module for module in _COMPAT_MODULES if name in module.__dict__)
        )
        if modules:
            history = ModuleType.__getattribute__(self, "_forwarded_values")
            previous = history.get(name, [])
            if previous and all(
                value is old_value or value == old_value for _, old_value in previous[-1]
            ):
                records = previous.pop()
                for module, old_value in records:
                    setattr(module, name, old_value)
                if not previous:
                    history.pop(name, None)
            else:
                previous.append([(module, getattr(module, name)) for module in modules])
                history[name] = previous
                for module in modules:
                    setattr(module, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        history = ModuleType.__getattribute__(self, "_forwarded_values")
        previous = history.get(name, [])
        if previous:
            records = previous.pop()
            for module, old_value in records:
                setattr(module, name, old_value)
            if not previous:
                history.pop(name, None)
        super().__delattr__(name)


def __getattr__(name: str) -> Any:
    """Preserve internal import compatibility while command code lives in focused modules."""
    for module in _COMPAT_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module in _COMPAT_MODULES:
        names.update(dir(module))
    return sorted(names)


def main() -> None:
    """CLI entry point for the Lisa terminal host."""
    app(args=list(sys.argv[1:]))


if __name__ == "__main__":
    main()

_compat_module = sys.modules[__name__]
ModuleType.__setattr__(_compat_module, "__class__", _CompatibilityModule)
ModuleType.__setattr__(_compat_module, "_forwarded_values", {})
