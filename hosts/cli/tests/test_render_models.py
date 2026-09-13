"""Tests for ``render_models`` — the enriched model table.

Covers the new metadata columns (Context, Cost, Capabilities) that
``render_models`` surfaces now that ``ManifestModelEntry`` carries
``prompt_price``/``completion_price``/``pricing_currency`` and the
manifest schema admits them.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from hosts.cli.render import render_models
from host.testing import build_test_runtime


def test_render_models_shows_context_cost_capabilities(tmp_path: Path) -> None:
    """An openrouter model with real pricing renders the new columns."""
    models_yaml = "\n".join(
        [
            "mode_models: {default: 'openai/gpt-5.4'}",
            "models:",
            "  - model_id: openai/gpt-5.4",
            "    provider: openrouter",
            "    context_window: 256000",
            "    prompt_price: 0.0000025",
            "    completion_price: 0.000015",
            "    reasoning_model: true",
            "    capabilities: [text, tools, structured_output, reasoning]",
            "",
        ]
    )
    runtime = build_test_runtime(tmp_path, models_yaml=models_yaml)
    console = Console(record=True, width=200)
    render_models(runtime, console=console)
    text = console.export_text()
    # Context column: 256k formatted.
    assert "256k" in text
    # Cost column: $/M-token in/out (2.5 in, 15 out).
    assert "$2.5/$15" in text
    # Capabilities column.
    assert "tools" in text
    assert "structured_output" in text


def test_render_models_falls_back_to_cost_tier_when_no_price(tmp_path: Path) -> None:
    """A model without pricing shows its cost_tier label instead of a blank."""
    models_yaml = "\n".join(
        [
            "mode_models: {default: 'gemma3:latest'}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
            "    cost_tier: budget",
            "",
        ]
    )
    runtime = build_test_runtime(tmp_path, models_yaml=models_yaml)
    console = Console(record=True, width=200)
    render_models(runtime, console=console)
    text = console.export_text()
    assert "budget" in text


def test_render_models_dash_when_no_metadata(tmp_path: Path) -> None:
    """A model with no enrichable metadata renders dashes, not blanks."""
    models_yaml = "\n".join(
        [
            "mode_models: {default: 'mistral:latest'}",
            "models:",
            "  - model_id: mistral:latest",
            "    provider: ollama",
            "",
        ]
    )
    runtime = build_test_runtime(tmp_path, models_yaml=models_yaml)
    console = Console(record=True, width=200)
    render_models(runtime, console=console)
    text = console.export_text()
    row_lines = [line for line in text.splitlines() if "mistral:latest" in line]
    assert row_lines
    # Context, Cost, Capabilities columns all render as "—" for this model.
    assert row_lines[0].count("—") >= 3
