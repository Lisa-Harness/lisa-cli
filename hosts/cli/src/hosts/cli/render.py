"""Rich rendering helpers for the CLI host.

These functions build Rich renderables (Tables, Panels, Text) from the
transport-neutral data surfaced by ``host``. They are the CLI-host
analogue of the plain-text renderers in ``runtime.commands`` — the core
returns Markdown strings, and here we wrap them in Rich panels/tables for
the terminal.

Nothing in this module imports from ``hosts.cli.repl`` or
``hosts.cli.main``; it is a leaf rendering module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    # ``HostRuntime`` is used only as a string annotation (stringified by
    # ``from __future__ import annotations``); the runtime symbols are
    # passed in at call time. Deferring this import keeps ``render`` from
    # dragging in ``runtime.runtime`` -> ``kernel.ipsar_fsm`` -> ``openai``
    # (~2s) at module load.
    from runtime.runtime import HostRuntime as HostRuntime


def _format_tokens(tokens: int) -> str:
    """Format a token budget compactly (e.g. ``256000`` → ``256k``)."""

    tokens = int(tokens or 0)
    if tokens <= 0:
        return "—"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:g}k"
    return str(tokens)


def _format_cost(model: object) -> str:
    """Format a model's per-token price as $/M-tokens (in/out) when known.

    Reads ``prompt_price`` / ``completion_price`` (per-token USD, the units
    OpenRouter publishes). Falls back to the qualitative ``cost_tier`` label
    so providers without pricing still show something, and a dash when
    neither is present.
    """

    prompt_price = float(getattr(model, "prompt_price", 0.0) or 0.0)
    completion_price = float(getattr(model, "completion_price", 0.0) or 0.0)
    if prompt_price > 0.0 or completion_price > 0.0:
        return f"${prompt_price * 1e6:g}/${completion_price * 1e6:g}"
    cost_tier = str(getattr(model, "cost_tier", "") or "").strip()
    return cost_tier or "—"


# Heavy imports are deferred into the functions that use them so that
# ``--version`` / ``modes`` / ``models`` / ``verify`` don't pay the cost of
# ``rich.markdown`` (-> ``markdown_it`` + ``pygments`` ~= 1.4s) or
# ``kernel.ipsar_fsm`` / operation / tool registries (~= 1.5s) unless a
# render path that needs them actually runs.

# ── Style constants ───────────────────────────────────────────────────
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CATALOG = "slate_blue1"
STYLE_BORDER_APPROVAL = "gold3"
STYLE_CONTAINMENT_CONTAINED = "dark_sea_green4"
STYLE_CONTAINMENT_DEGRADED = "gold3"
STYLE_CONTAINMENT_NONE = "indian_red"
STYLE_TEXT_MUTED = "grey58"
STYLE_TEXT_PRIMARY = "bright_white"
STYLE_ROW_MODEL = "dark_sea_green4"
STYLE_TABLE_HEADER = "bold steel_blue3"
_SECONDS_PER_MINUTE = 60


def _styled_table(title: str) -> Table:
    """Build a catalog table with the shared Lisa presentation defaults.

    Minimal-head box (title rule + header rule only) and bold steel-blue
    headers. Deliberately NO zebra striping: terminal backgrounds are
    unknown — dark-grey rows vanish on black consoles, and a lighter pick
    inverts the problem on light ones. The box rules carry the structure.
    """
    return Table(
        title=title,
        box=SIMPLE_HEAD,
        header_style=STYLE_TABLE_HEADER,
    )


def render_modes(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of supported modes from the live workflow catalog.

    Uses ``kernel.ipsar_fsm.get_workflow_catalog`` (wired by
    ``runtime.mcp_pool.setup_framework``) and ``mode_support_gap`` to
    separate supported from unsupported modes.
    """
    from runtime.commands import mode_support_gap
    from kernel.ipsar_fsm import get_workflow_catalog

    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_workflow_catalog()
    table = _styled_table("Lisa Modes")
    table.add_column("Mode", style="bold")
    table.add_column("Kind")
    table.add_column("Command")
    table.add_column("Description")

    for mode_id, mode in sorted(catalog.items()):
        gap = mode_support_gap(mode)
        command = mode.command_pattern or "default route"
        description = mode.description or command
        if gap is not None:
            description = f"[yellow]requires {gap}[/yellow]"
        table.add_row(mode_id, mode.mode_kind, command, description)

    if console is not None:
        console.print(table)
    return table


def render_models(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of configured models from the model manifest.

    Uses ``runtime.model_manifest.load_model_manifest`` against the
    canonical runtime data-dir path.
    """
    from runtime.model_manifest import load_model_manifest

    path = runtime.paths.models_path
    manifest = load_model_manifest(path)
    table = _styled_table("Lisa Models")
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Aliases")
    table.add_column("Reasoning")
    table.add_column("Context", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Capabilities")

    for model in manifest.models:
        aliases = ", ".join(sorted(model.aliases)) or "—"
        reasoning = "yes" if model.reasoning_model else "no"
        context = _format_tokens(model.context_window)
        cost = _format_cost(model)
        capabilities = ", ".join(model.capabilities) or "—"
        table.add_row(
            model.model_id, model.provider, aliases, reasoning, context, cost, capabilities
        )

    if console is not None:
        console.print(table)
    return table


def render_pack_list_panel(runtime: HostRuntime, *, console: Console | None = None) -> Panel:
    """Render the pack activation list from ``runtime.commands.render_pack_list``.

    The core returns a Markdown string; we wrap it in a Rich Panel for the
    terminal host.
    """
    from runtime.commands import render_pack_list
    from rich.markdown import Markdown

    markdown_text = render_pack_list(runtime)
    panel = Panel(
        Markdown(markdown_text),
        title="Pack Activation",
        border_style=STYLE_BORDER_SESSION,
    )
    if console is not None:
        console.print(panel)
    return panel


def render_operations(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of internal operations with IDs for allowlist authoring.

    Lists every operation discovered from content packs via the
    ``lisa-catalog`` MCP server. The ID column is exactly the string to copy
    into a mode's ``allowed_operation_ids`` front-matter list.
    """
    from kernel.operations.registry import get_operation_catalog

    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_operation_catalog()
    table = _styled_table("Lisa Operations")
    table.add_column("Operation ID", style="bold")
    table.add_column("Kind")
    table.add_column("Description")
    table.add_column("Tools", overflow="fold")

    for operation_id, definition in sorted(catalog.items()):
        description = definition.description or "—"
        tools = ", ".join(definition.tool_ids) if definition.tool_ids else "—"
        table.add_row(operation_id, definition.execution_kind, description, tools)

    if console is not None:
        console.print(table)
    return table


def render_tools(
    runtime: HostRuntime,
    *,
    console: Console | None = None,
) -> Table:
    """Build a Rich Table of internal tools with IDs for allowlist authoring.

    Lists every internal tool discovered from content packs via the
    ``lisa-catalog`` MCP server. The ID column is exactly the string to copy
    into a mode's ``allowed_tool_ids`` front-matter list. MCP server tools
    are listed by the dedicated ``lisa mcp`` command (and REPL ``/mcp``).
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    table = _styled_table("Lisa Internal Tools")
    table.add_column("Tool ID", style="bold")
    table.add_column("Description")
    table.add_column("Max/Turn")
    table.add_column("Timeout")

    from kernel.tools.registry import get_tool_catalog

    catalog = get_tool_catalog()
    for tool_id, definition in sorted(catalog.items()):
        description = definition.description or "—"
        table.add_row(
            tool_id,
            description,
            str(definition.max_calls_per_turn),
            f"{definition.timeout_seconds}s",
        )

    if console is not None:
        console.print(table)
    return table


def render_mcp_servers(
    health: dict[str, McpServerHealth],
    *,
    console: Console | None = None,
) -> Table:
    """Render a Rich Table summarizing every configured MCP server.

    Columns: Server | Status | Transport | URL/Command | Tools | Last Error.
    Reads the per-server :class:`McpServerHealth` collected by warming the
    MCP pool (see :func:`hosts.cli.main._collect_mcp_tools`). The status,
    tool count, and last error reflect the current runtime state — a cold
    server that failed to start shows ``failed`` with the failure reason,
    not ``cold``.
    """
    table = _styled_table("MCP Servers")
    table.add_column("Server", style="bold")
    table.add_column("Status")
    table.add_column("Transport")
    table.add_column("URL/Command", overflow="fold")
    table.add_column("Tools", justify="right")
    table.add_column("Last Error", overflow="fold")

    for server_id in sorted(health):
        h = health[server_id]
        endpoint = h.url or h.command or "—"
        last_error = _truncate(h.last_error, 200) or "—"
        status_style = _status_style(h.status)
        # A cold server that hasn't been connected yet shows tool_count=0
        # because tools/list hasn't run — render an em-dash (like the empty
        # Last Error column) instead of a misleading 0.
        tool_display = "—" if (h.status == "cold" and h.tool_count == 0) else str(h.tool_count)
        table.add_row(
            server_id,
            f"[{status_style}]{_status_pill(h.status)}[/]",
            h.transport,
            endpoint,
            tool_display,
            last_error,
        )

    if console is not None:
        console.print(table)
    return table


def render_mcp_server_tools(
    specs: list[MCPToolSpec],
    health: McpServerHealth,
    *,
    console: Console | None = None,
) -> Table:
    """Render a Rich Table of one MCP server's tools with descriptions + required args.

    Used by ``lisa mcp <server_id>`` to drill into a single server. The
    header (status, transport, URL/command) comes from ``health``; the tool
    rows come from ``specs``. Tool names are the bare strings referenced by
    per-server ``tool_policy.allowed_tools`` / ``denied_tools`` in
    ``mcp_servers.yaml``.
    """
    table = _styled_table(f"MCP Server: {health.server_id}")
    table.add_column("Tool", style="bold", overflow="fold")
    table.add_column("Description", overflow="fold")
    table.add_column("Required args", overflow="fold")

    for spec in specs:
        description = _truncate(spec.description, 80) or "—"
        required = _required_args(spec)
        table.add_row(spec.name, description, required)

    if console is not None:
        console.print(table)
    return table


def _status_style(status: str) -> str:
    """Return a Rich style name for an MCP server status glyph."""
    if status == "warm":
        return "green"
    if status == "failed":
        return "red"
    if status == "cold":
        return "yellow"
    return "grey58"


def _status_pill(status: str) -> str:
    """Render an MCP server status as a glyph pill (``● warm``)."""
    glyph = {"warm": "●", "cold": "○", "failed": "✗"}.get(status, "·")
    return f"{glyph} {status}"


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _required_args(spec: MCPToolSpec) -> str:
    """Extract the comma-joined required arg names from a tool's input schema."""
    schema = spec.input_schema or {}
    required = schema.get("required")
    if isinstance(required, list) and required:
        return ", ".join(str(r) for r in required)
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        return ", ".join(str(k) for k in properties)
    return "—"


def render_markdown_panel(
    text: str,
    *,
    title: str = "Response",
    border_style: str = STYLE_BORDER_RESPONSE,
    console: Console | None = None,
) -> Panel:
    """Wrap a Markdown string in a Rich Panel.

    Convenience helper for rendering any plain-text/Markdown response from
    ``host`` in a terminal panel.
    """
    from rich.markdown import Markdown

    panel = Panel(Markdown(text), title=title, border_style=border_style)
    if console is not None:
        console.print(panel)
    return panel


#: Diff-detection thresholds: minimum hunk-header count and combined +/- lines
#: for the approval preview to render as a colored diff instead of flat text.
_DIFF_MIN_HUNKS = 1
_DIFF_MIN_CHANGED_LINES = 2
_DIFF_MIN_TOTAL_LINES = 3


def _looks_like_diff(text: str) -> bool:
    """Heuristic: does this approval preview carry a unified diff?"""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < _DIFF_MIN_TOTAL_LINES:
        return False
    hunk = sum(1 for ln in lines if ln.startswith("@@"))
    adds = sum(1 for ln in lines if ln.startswith("+"))
    dels = sum(1 for ln in lines if ln.startswith("-"))
    return hunk >= _DIFF_MIN_HUNKS and (adds + dels) >= _DIFF_MIN_CHANGED_LINES


def _diff_text(text: str) -> Text:
    """Render a unified diff with per-line colors (adds/dels/hunk headers)."""
    out = Text()
    for line in text.strip().splitlines():
        if line.startswith("+"):
            out.append(line + "\n", style="green3")
        elif line.startswith("-"):
            out.append(line + "\n", style="indian_red")
        elif line.startswith("@@"):
            out.append(line + "\n", style="steel_blue3")
        else:
            out.append(line + "\n", style=STYLE_TEXT_MUTED)
    return out


def _containment_badge(containment: str) -> tuple[str, str]:
    """Return (label, style) for a sandbox containment state."""
    state = (containment or "unknown").strip().lower()
    if state == "contained":
        return "● CONTAINED (sandbox)", STYLE_CONTAINMENT_CONTAINED
    if state == "degraded":
        return "▲ DEGRADED — sandbox cannot run", STYLE_CONTAINMENT_DEGRADED
    return "✗ NOT SANDBOXED", STYLE_CONTAINMENT_NONE


def _format_approval_expiry(seconds: object) -> str:
    """Format a host-supplied approval TTL for compact panel display."""
    try:
        remaining = max(0, int(seconds))
    except (TypeError, ValueError):
        return ""
    if remaining == 0:
        return "expires now"
    if remaining < _SECONDS_PER_MINUTE:
        return f"expires in {remaining}s"
    minutes, seconds = divmod(remaining, _SECONDS_PER_MINUTE)
    if minutes < _SECONDS_PER_MINUTE:
        return f"expires in {minutes}m {seconds}s"
    hours, minutes = divmod(minutes, _SECONDS_PER_MINUTE)
    return f"expires in {hours}h {minutes}m"


def render_approval_panel(
    payload: dict[str, object],
    *,
    console: Console | None = None,
) -> Panel:
    """Render the approval panel for a pending action.

    ``payload`` is the structured ``data["approval"]`` the host attaches to the
    turn response (``lisa.approval.v1``). It shows the tool/pack and effect
    tier, the proposed command, the sandbox containment badge, and the
    decision hints. The host decides approve/ask/deny and owns the policy
    write; this function only renders — it never decides. The hints are
    the ``lisa prompt "approve <id>"`` commands, so the operator can decide
    from the same shell that raised the request (the CLI has no interactive
    loop; keystroke hints went with the retired REPL).
    """

    target = f"{payload.get('pack_id') or ''}:{payload.get('target_id') or ''}".strip(":")
    effect = str(payload.get("effect") or "").strip() or "execute"
    command = str(payload.get("command") or "").strip()
    preview = str(payload.get("preview") or "").strip()
    request_id = str(payload.get("request_id") or "").strip()
    containment, containment_style = _containment_badge(str(payload.get("containment") or ""))
    reasons = str(payload.get("containment_reasons") or "").strip()
    policy_reason = str(payload.get("policy_reason") or "").strip()
    policy_tier = str(payload.get("policy_tier") or "").strip()
    expiry = _format_approval_expiry(payload.get("expires_in_seconds"))

    body = Text()
    if _looks_like_diff(preview or ""):
        # A file-edit approval ships a unified diff — color the diff lines
        # directly into the body (green adds, red dels, cyan hunk headers)
        # so edits read at a glance instead of as a flat preview block.
        body.append(_diff_text(preview))
        body.append("\n")
    else:
        body.append(f"{preview or 'Run an action'}\n", style=STYLE_TEXT_PRIMARY)
    if command:
        body.append("\n$ ", style=STYLE_TEXT_MUTED)
        body.append(command + "\n", style="bright_white")
    body.append("\ntool   ", style=STYLE_TEXT_MUTED)
    body.append(f"{target or '(unknown)'}\n")
    body.append("effect ", style=STYLE_TEXT_MUTED)
    body.append(f"{effect}\n")
    body.append("scope  ", style=STYLE_TEXT_MUTED)
    body.append(f"{containment}\n", style=containment_style)
    if reasons:
        body.append("       ", style=STYLE_TEXT_MUTED)
        body.append(f"{reasons}\n", style=STYLE_CONTAINMENT_DEGRADED)
    if policy_reason:
        body.append("policy ", style=STYLE_TEXT_MUTED)
        if policy_tier:
            body.append(f"{policy_tier}: ")
        body.append(policy_reason + "\n", style=STYLE_TEXT_MUTED)
    if expiry:
        body.append(expiry + "\n", style=STYLE_TEXT_MUTED)

    body.append("\nDecide from the same shell, without a session:\n", style=STYLE_TEXT_MUTED)
    body.append(
        f'  lisa prompt "approve {request_id}"          # run it\n',
        style=STYLE_TEXT_PRIMARY,
    )
    body.append(
        f'  lisa prompt "approve {request_id} --remember <prefix>"\n',
        style=STYLE_TEXT_MUTED,
    )
    body.append(
        f'  lisa prompt "deny {request_id}"   ·   lisa prompt policy\n',
        style=STYLE_TEXT_MUTED,
    )

    panel = Panel(
        body,
        title="Approval required",
        border_style=STYLE_BORDER_APPROVAL,
        padding=(0, 1),
    )
    if console is not None:
        console.print(panel)
    return panel


__all__ = [
    "render_approval_panel",
    "render_markdown_panel",
    "render_mcp_server_tools",
    "render_mcp_servers",
    "render_models",
    "render_modes",
    "render_operations",
    "render_pack_list_panel",
    "render_tools",
]
