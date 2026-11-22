"""Interactive prompt, approval, and host-command execution."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, cast

import typer
from rich.console import Console
from rich.markup import escape

from hosts.cli.application import (
    _ABORT_EXIT,
    _APPROVAL_REQUIRED_EXIT,
    _GROUP_HOIST,
    _PANEL_APPROVALS,
    _PENDING_APPROVAL_TARGET,
    _PROMPT_EPILOG,
    _YIELD_UNRESOLVED_EXIT,
    app,
    console,
    policy_app,
    policy_rule_app,
)
from hosts.cli.cli_runtime import _aclose_runtime_bounded, _initialize_runtime_or_exit
from hosts.cli.options import (
    DataDirOption,
    HostLogOption,
    LogLevelOption,
    LogOption,
    NoLogOption,
    NoThreadOption,
    OverlaysDirOption,
    ProfileOption,
    PromptApproveOption,
    PromptDenyOption,
    PromptRememberOption,
    PromptSwapKeysOption,
    QuietOption,
    RawOption,
    SessionContextOption,
    StateSnapshotOption,
    StdinOption,
    StreamingOption,
    ThreadDirOption,
    TimezoneOption,
    ToolDetailOption,
)

if TYPE_CHECKING:
    from runtime.runtime import HostRuntime

    from hosts.cli.transport import PromptTransport

logger = logging.getLogger(__name__)


# Prepended to the turn message in --raw mode: the response is consumed
# verbatim as a machine artifact, so the model must emit only the result.
# The yield prohibition is explicit: a prior turn ended in the `yield`
# tool (asking the operator for missing input), which wrote the question
# prose into redirected output as if it were the artifact.
_RAW_CONTRACT = (
    "OUTPUT CONTRACT (raw mode): Your entire response text is consumed verbatim "
    "as a machine artifact (redirected to a file or piped to another tool). "
    "Emit ONLY the requested result — no preamble, no closing commentary, no "
    "restating the task, no follow-up questions, and do not wrap the whole "
    "output in a markdown code fence. "
    "Do NOT call the yield/ask tool and do not end with a question. If required "
    "input is missing, stop without emitting an artifact and report what is "
    "missing on the error channel; never guess or invent a substitute result. "
    "A question or an empty response is a failed turn."
)


class _PrearmedDecision(NamedTuple):
    """One pre-armed approval decision parsed from prompt CLI flags."""

    request_id: str
    approve: bool
    remember: str


#: Interactive reader seam: assigned the real function lazily on the
#: no-args interactive path (keeps --help and piped runs free of the
#: entry module). Tests monkeypatch this to script terminal input.
_prompt_reader: Callable[..., str | None] | None = None


def _collect_interactive_or_error(
    full_instruction: str,
    payload: str,
    data_dir: str,
    profile: str,
    *,
    swap_keys: bool | None = None,
    on_submit: Callable[[], None] | None = None,
) -> str:
    """Resolve the no-input case: editor on a tty, usage error otherwise.

    Called when argv, piped stdin, and pre-armed decisions all carried
    nothing. On an interactive terminal (stdin AND stdout ttys) the
    message is collected outside the shell's argument grammar — quote
    characters arrive literally (see :mod:`hosts.cli.entry`), and
    ``$(cmd)`` is expanded over the editor text (:mod:`hosts.cli.expand`).

    ``swap_keys`` is tri-state and names the EXCEPTIONAL layout:
    ``True`` (``--swap-keys``) = chat-box, Enter sends; ``False``
    (``--no-swap-keys``) = pin the shipped swapped layout against a
    truthy ``LISA_PROMPT_KEYS``; ``None`` (flag absent) = the env
    decides in the entry module (truthy env = chat-box, else the
    shipped default). The reader receives the resolved layout as
    ``swapped_keys`` (True = Enter-newline shipped default; False =
    chat-box). Under redirection there is no editor: a pipeline wants
    an error, not a prompt. Returns the message or raises ``typer.Exit``.

    ``on_submit`` runs after a non-empty submission, before shell
    substitutions, so feedback need not wait for an embedded command.
    """
    del full_instruction  # empty by contract; parameter documents intent
    if sys.stdin.isatty() and sys.stdout.isatty():
        from hosts.cli import entry as entry_mod

        if _prompt_reader is not None:
            # Custom reader seam (tests): the plain-reader contract.
            terminal_message = _prompt_reader(console)
        else:
            # Flag state -> reader layout (True=shipped, False=chat-box):
            #   absent (None)      -> env decides (swapped_keys=None)
            #   --no-swap-keys     -> shipped (swapped_keys=True)
            #   --swap-keys        -> chat-box (swapped_keys=False)
            resolved: bool | None
            if swap_keys is None:
                resolved = None
            elif swap_keys is False:
                resolved = True
            else:
                resolved = False
            terminal_message = entry_mod.read_prompt_message_tui(
                console,
                history_path=entry_mod.prompt_history_path(data_dir, profile),
                swapped_keys=resolved,
                data_dir=data_dir,
                profile=profile,
            )
        if terminal_message is None:
            raise typer.Exit(code=_ABORT_EXIT) from None
        if terminal_message.strip():
            if on_submit is not None:
                on_submit()
            return _expand_editor_message(terminal_message)
    console.print(
        "[red]No message and no piped input. Pass a message and/or pipe data on stdin.[/red]"
    )
    raise typer.Exit(code=2)


def _expand_editor_message(text: str) -> str:
    """Run ``$()`` substitution over editor-collected text (see expand.py).

    A failed substitution is loud: the failing command is printed to
    stderr and the run exits 2 — never a silently-mutated message.
    """
    from hosts.cli.expand import expand_shell_substitutions

    expanded, failures = expand_shell_substitutions(text)
    if failures:
        shown = "; ".join(f"$({cmd})" for cmd in failures)
        console.print(f"[red]Shell substitution failed: {escape(shown)}[/red]")
        raise typer.Exit(code=2)
    return expanded


def _prompt_prearmed_decision(approve: str, deny: str, remember: str) -> _PrearmedDecision | None:
    """Parse the pre-armed decision flags for a prompt turn, or None.

    ``--approve``/``--deny`` accept either a request id or, bare, the
    sentinel "pending" (an empty ``request_id`` here, resolved to the
    single pending approval via the host's recorded-owner identity).
    ``--remember <prefix>`` only applies to approve; deny + remember is a
    usage error.
    """
    approve = approve.strip()
    deny = deny.strip()
    remember = remember.strip()
    if approve and deny:
        console.print("[red]--approve and --deny are mutually exclusive.[/red]")
        raise typer.Exit(code=2)
    if remember and not approve:
        console.print("[red]--remember requires --approve.[/red]")
        raise typer.Exit(code=2)
    if not approve and not deny:
        return None
    if approve:
        target = "" if approve.lower() == _PENDING_APPROVAL_TARGET else approve
        return _PrearmedDecision(request_id=target, approve=True, remember=remember)
    target = deny if deny.lower() != _PENDING_APPROVAL_TARGET else ""
    return _PrearmedDecision(request_id=target, approve=False, remember="")


async def _run_host_command_turn(runtime: HostRuntime, command: str) -> str:
    """Run one host-command message (no model turn), then close the pool.

    Mirrors the prompt-turn path without transport rendering: the
    command path returns text (or a structured result) that this helper
    prints through the shared console. The MCP pool starts in this event
    loop and closes in the same one (anyio same-loop contract).
    """
    from runtime import NullTransport
    from runtime.handle_message import handle_message

    try:
        await runtime.start()
        result = await handle_message(runtime, command, NullTransport())
    finally:
        await _aclose_runtime_bounded(runtime)
    return getattr(result, "text", "") or str(result)


async def _resolve_pending_request_id(runtime: HostRuntime) -> str:
    """Return the single pending approval's id, or '' when ambiguous/none.

    The approval store is process-durable and identity-scoped, so a fresh
    prompt process sees the request a prior prompt run left pending. With
    exactly one pending record the bare decision targets it; zero or
    multiple pendings return '' so the typed core replies with its own
    diagnostic (usage/expired/ambiguous).
    """
    service = getattr(runtime.services, "approval_service", None)
    if service is None or not hasattr(service, "list_pending"):
        return ""
    from runtime.approval import approval_identity

    try:
        pending = await service.list_pending(approval_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("prompt: pending-approval resolution failed: %s", exc)
        return ""
    if len(pending) == 1:
        # The service reaches the CLI as Any (duck-typed seam), so reassert
        # the declared kernel field type for the return contract.
        request_id = cast("str", pending[0].invocation.request_id)
        logger.info("prompt: resolved bare decision to pending %s", request_id[:8])
        return request_id
    return ""


async def _decide_approval_turn(
    runtime: HostRuntime,
    *,
    request_id: str,
    approve: bool,
    remember: str,
) -> tuple[str, bool]:
    """Decide one approval through the typed core, then close the pool.

    Mirrors ``_run_host_command_turn``: no transport rendering — the
    decision path returns text (or a structured continuation result) that
    this helper prints through the shared console. A bare decision (empty
    ``request_id``) resolves to the single pending approval first — the
    "decide the thing that just stopped me" case.
    """
    if not request_id:
        request_id = await _resolve_pending_request_id(runtime)
    from runtime import NullTransport
    from runtime.handle_message import decide_pending_approval

    try:
        await runtime.start()
        result = await decide_pending_approval(
            runtime,
            request_id,
            approve=approve,
            # The CLI's --remember is a value flag (explicit prefix or
            # absent): the default-prefix behavior belongs to the slash
            # grammar, not here. No --remember means no rule write.
            remember=remember or None,
            remember_default=False,
            transport=NullTransport(),
        )
    finally:
        await _aclose_runtime_bounded(runtime)
    return (getattr(result, "text", "") or str(result)), bool(getattr(result, "is_error", False))


def _run_decision(
    request_id: str,
    *,
    approve: bool,
    remember: str,
    **runtime_kwargs: Any,
) -> None:
    """Decide one pending approval through the typed core in a fresh runtime.

    The CLI's runtime options are forwarded verbatim; the decision is a
    value (id + direction + optional remember prefix), never a command
    string, so no quoting round trip exists between the CLI and the host.
    Prints the reply and exits 0.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    text, failed = asyncio.run(
        _decide_approval_turn(runtime, request_id=request_id, approve=approve, remember=remember)
    )
    if text:
        console.print(text)
    raise typer.Exit(code=1 if failed else 0)


def _run_host_command(command: str, **runtime_kwargs: Any) -> None:
    """Run a host-owned admin command through a fresh runtime.

    The CLI's runtime options are forwarded verbatim; the command string is
    normalized to its slash form and dispatched via ``handle_message``
    exactly as a persistent host would send it. This keeps pack commands and
    other host-admin replies host-owned while bypassing the model-turn path.
    Prints the reply and exits 0.
    """
    if not command.startswith("/"):
        command = "/" + command
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return
    text = asyncio.run(_run_host_command_turn(runtime, command))
    if text:
        console.print(text)
    raise typer.Exit(code=0)


def _run_policy_action(action: Callable[[HostRuntime], str], **runtime_kwargs: Any) -> None:
    """Run one typed policy action against a fresh runtime.

    ``action`` receives the runtime and returns the reply text — the typed
    cores (:func:`set_policy_mode`, :func:`add_policy_rule`, ...) take
    values, never command strings, so no quoting round trip exists between
    the CLI flags and the policy write. The host owns the write.
    """
    runtime = _initialize_runtime_or_exit(**runtime_kwargs)
    if runtime is None:
        return

    console.print(action(runtime))
    raise typer.Exit(code=0)


@policy_app.callback()
def _policy_group(  # Typer exposes shared runtime options as CLI flags.
    ctx: typer.Context,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    """Set up shared host options for policy commands; bare `policy` shows status."""
    ctx.obj = {
        "log_level": log_level,
        "log_path": log,
        "no_log": no_log,
        "data_dir": data_dir,
        "profile": profile,
        "overlays_dir": overlays_dir,
        "thread_dir": thread_dir,
        "no_thread": no_thread,
        "session_context": session_context,
        "timezone": timezone,
        "state_snapshot_path": state_snapshot,
        "streaming": streaming,
        "host_log_path": host_log,
    }
    if ctx.invoked_subcommand is None:

        def _show_document(runtime: HostRuntime) -> str:
            from runtime.commands import render_policy_document

            return render_policy_document(runtime)

        _run_policy_action(_show_document, **ctx.obj)


@policy_app.command("mode", help="Set the fallback approval mode: ask, trust, or plan.")
def _policy_mode(ctx: typer.Context, mode: Literal["ask", "trust", "plan"]) -> None:
    def _set_mode(runtime: HostRuntime) -> str:
        from runtime.commands import set_policy_mode
        from runtime.policy import PolicyMode

        return set_policy_mode(runtime, PolicyMode(mode))

    _run_policy_action(_set_mode, **ctx.obj)


@policy_app.command("bare", help="Toggle approval for unsandboxed execute actions.")
def _policy_bare(ctx: typer.Context, setting: Literal["on", "off"]) -> None:
    def _set_bare(runtime: HostRuntime) -> str:
        from runtime.commands import set_policy_bare

        return set_policy_bare(runtime, setting == "on")

    _run_policy_action(_set_bare, **ctx.obj)


@policy_rule_app.command("list", help="List allow, ask, and deny rules with remove addresses.")
def _policy_rule_list(ctx: typer.Context) -> None:
    def _list_rules(runtime: HostRuntime) -> str:
        from runtime.commands import render_policy_rules

        return render_policy_rules(runtime)

    _run_policy_action(_list_rules, **ctx.obj)


@policy_rule_app.command("remove", help="Remove a rule by its list address, e.g. allow:1.")
def _policy_rule_remove(ctx: typer.Context, address: str) -> None:
    def _remove_rule(runtime: HostRuntime) -> str:
        from runtime.commands import remove_policy_rule

        return remove_policy_rule(runtime, address)

    _run_policy_action(_remove_rule, **ctx.obj)


@policy_rule_app.command("add", help="Add an allow, ask, or deny rule.")
def _policy_rule_add(  # Typer requires each rule matcher as a CLI option.
    ctx: typer.Context,
    rule_list: Literal["allow", "ask", "deny"],
    tool: str = typer.Option("", "--tool", help="Exact capability tool ID."),
    effect: str = typer.Option("", "--effect", help="Exact effect kind."),
    pack: str = typer.Option("", "--pack", help="Exact owning pack ID."),
    prefix: str = typer.Option("", "--prefix", help="Command prefix matcher."),
    when: str = typer.Option("", "--when", help="CEL expression; quote the whole expression."),
) -> None:
    """Parse policy matchers with Typer and pass their values to the host policy writer."""

    def _add_rule(runtime: HostRuntime) -> str:
        from runtime.commands import add_policy_rule

        return add_policy_rule(
            runtime,
            rule_list,
            tool=tool,
            effect=effect,
            pack=pack,
            command_prefix=prefix,
            when=when,
        )

    _run_policy_action(_add_rule, **ctx.obj)


@app.command(
    "approve",
    rich_help_panel=_PANEL_APPROVALS,
    help="Approve one pending action without starting a model turn.",
)
def _approve_command(  # Match shared runtime options used by one-shot commands.
    request_id: str = typer.Argument(
        _PENDING_APPROVAL_TARGET,
        help="Approval request ID, or omit it to target the single pending request.",
    ),
    remember: str = typer.Option(
        "",
        "--remember",
        help="Persist an allow rule for the action's command prefix.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    target = "" if request_id.lower() == _PENDING_APPROVAL_TARGET else request_id
    _run_decision(
        target,
        approve=True,
        remember=remember,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


@app.command(
    "deny",
    rich_help_panel=_PANEL_APPROVALS,
    help="Deny one pending action without starting a model turn.",
)
def _deny_command(  # Match shared runtime options used by one-shot commands.
    request_id: str = typer.Argument(
        _PENDING_APPROVAL_TARGET,
        help="Approval request ID, or omit it to target the single pending request.",
    ),
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
) -> None:
    target = "" if request_id.lower() == _PENDING_APPROVAL_TARGET else request_id
    _run_decision(
        target,
        approve=False,
        remember="",
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        profile=profile,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )


def _compose_prompt_message(instruction: str, payload: str) -> str:
    """Compose the prompt-mode turn message from the CLI instruction and stdin.

    When both are present the payload is wrapped in a ``<piped-input>`` block
    ahead of the instruction so the model can tell data from directive;
    otherwise the non-empty one is the message.
    """
    instruction = instruction.strip()
    payload = payload.strip()
    if payload and instruction:
        return f"<piped-input>\n{payload}\n</piped-input>\n\n{instruction}"
    return payload or instruction


@app.command(
    "prompt",
    epilog=_PROMPT_EPILOG,
    help="Run a single turn and print the response.",
)
def _prompt_command(
    message: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Message to route through the host; multiple words are joined "
                "with spaces, so short prompts work unquoted: "
                "`lisa prompt fix the config parser`. Quote only to preserve "
                "inner spaces, quote characters, or shell-expansion semantics. "
                "When stdin is piped (not a tty), its contents are attached as "
                "the turn payload, e.g. "
                "`cat f.conf | lisa prompt -r 'fix this config'`. "
                "Approval decisions that skip the turn: `lisa approve <id> "
                "[--remember <prefix>]` and `lisa deny <id>`."
            ),
        ),
    ] = None,
    stdin: StdinOption = True,
    raw: RawOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    profile: ProfileOption = "",
    overlays_dir: OverlaysDirOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "",
    host_log: HostLogOption = "",
    quiet: QuietOption = False,
    tool_detail: ToolDetailOption = 0,
    approve: PromptApproveOption = "",
    deny: PromptDenyOption = "",
    remember: PromptRememberOption = "",
    swap_keys: PromptSwapKeysOption = None,
) -> None:
    # Group-level hoist: `lisa -r "fix it"` parses -r at group level (see
    # _cli_callback) and lazy-routing lands here; OR the hoisted values in.
    # Typer commands cannot reliably read click's thread-local context
    # (get_current_context returns None inside a Typer command body), so
    # the group callback records the hoisted flags on this module-level
    # slot — one CLI process dispatches exactly one group callback before
    # the command body runs.
    group_obj = _GROUP_HOIST
    raw = raw or bool(group_obj.get("raw"))
    quiet = quiet or bool(group_obj.get("quiet"))
    stdin = stdin and not bool(group_obj.get("no_stdin"))
    # swap_keys is tri-state (None = absent): let the env decide — unless
    # the hoist ORing pattern would clobber it, so read the hoist only
    # when this command's own flag is absent.
    if swap_keys is None:
        swap_keys = group_obj.get("swap_keys")
    # Pre-armed decisions are handled below; standalone approve/deny
    # commands are registered with Typer and never enter this function.
    words = message or []
    full_instruction = " ".join(words)
    payload = ""
    if stdin and not sys.stdin.isatty():
        try:
            payload = sys.stdin.read()
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[red]Failed to read piped stdin: {escape(str(exc))}[/red]")
            raise typer.Exit(code=2) from exc
    # A pre-armed decision (--approve/--deny) carries no message and no
    # piped payload; it is consumed at the end of the turn.
    decision = _prompt_prearmed_decision(approve, deny, remember)
    from hosts.cli.transport import PromptTransport

    transport = PromptTransport(
        console=Console(stderr=True) if raw else console,
        quiet=quiet,
        raw=raw,
        tool_detail=tool_detail,
    )
    try:
        if not full_instruction.strip() and not payload.strip() and decision is None:
            full_instruction = _collect_interactive_or_error(
                full_instruction,
                payload,
                data_dir,
                profile,
                swap_keys=swap_keys,
                on_submit=transport.start_activity,
            )
        transport.start_activity()
        full_message = _compose_prompt_message(full_instruction, payload)
        if raw and full_message.strip():
            full_message = f"{_RAW_CONTRACT}\n\n{full_message}"
        runtime = _initialize_runtime_or_exit(
            log_level=log_level,
            log_path=log,
            no_log=no_log,
            data_dir=data_dir,
            profile=profile,
            overlays_dir=overlays_dir,
            thread_dir=thread_dir,
            no_thread=no_thread,
            session_context=session_context,
            timezone=timezone,
            state_snapshot_path=state_snapshot,
            streaming=streaming,
            host_log_path=host_log,
            in_process_mcp=True,
        )
        if runtime is None:
            return
        transport.streaming_mode = runtime.config.streaming.mode
        transport = asyncio.run(
            _run_prompt_with_teardown(
                runtime,
                full_message,
                transport=transport,
                decision=decision,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        # A SIGINT mid-turn cancels the asyncio.run loop, surfacing as
        # CancelledError here (typer does not convert it). Print the
        # one-line abort notice — not the 40-line traceback the framework
        # excepthook logs — and exit with the conventional SIGINT code so
        # shells read "cancelled", not "crashed". Raw mode keeps stdout
        # clean for pipelines: the notice goes to stderr like all other
        # activity output.
        abort_console = Console(stderr=True) if raw else console
        abort_console.print("[red]⏹ Turn aborted by operator (Ctrl-C).[/red]")
        raise typer.Exit(code=_ABORT_EXIT) from None
    finally:
        transport.stop_activity()
    raise typer.Exit(code=_prompt_exit_code(transport, raw=raw))


def _prompt_exit_code(transport: PromptTransport, *, raw: bool) -> int:
    """Map structured terminal outcomes to the documented process statuses."""
    if transport.approval_unresolved:
        return _APPROVAL_REQUIRED_EXIT
    if transport.yield_unresolved:
        return _YIELD_UNRESOLVED_EXIT
    if transport.protocol_errors or transport.turn_failed:
        if transport.protocol_errors:
            Console(stderr=True).print(
                f"[red]Transport protocol failure: {escape(transport.protocol_errors[0])}[/red]"
            )
        return 1
    if raw and transport.raw_response_missing:
        Console(stderr=True).print("[red]The turn produced no raw response artifact.[/red]")
        return 1
    return 0


async def _run_prompt_with_teardown(
    runtime: HostRuntime,
    message: str,
    *,
    transport: PromptTransport,
    decision: _PrearmedDecision | None = None,
) -> PromptTransport:
    """Run one turn (and any pre-armed decision), then close the MCP pool.

    A pre-armed decision (``--approve``/``--deny``) is dispatched FIRST,
    through the typed approval core — one dispatch, one value, no command
    string — so the pending approval the prior run left behind is settled
    before the turn runs with it. ``--remember`` persists an allow rule for
    the decided prefix (default prefix when no explicit value is given).

    Returns the transport so the caller can inspect post-turn state
    (``approval_unresolved``, ``yield_unresolved``, and ``turn_failed``).
    A failed pre-armed decision never continues into a model turn. The
    caller starts its activity display before
    runtime construction and owns cleanup if construction fails.

    In raw mode the transport console moves to stderr so the prompt/activity
    screen, ⚡ footer, and errors never land in redirected stdout; the
    response itself is written verbatim to stdout by the transport.
    """
    from runtime.handle_message import (  # lazy: defers turn import cost
        decide_pending_approval,
        handle_message,
    )

    transport.model_prices, transport.model_cache_prices = _collect_model_pricing(runtime)
    transport.start_turn(message)
    try:
        await runtime.start()
        if decision is not None:
            request_id = decision.request_id
            if not request_id:
                request_id = await _resolve_pending_request_id(runtime)
            result = await decide_pending_approval(
                runtime,
                request_id,
                approve=decision.approve,
                # --remember is a value flag here: explicit prefix or none.
                remember=decision.remember or None,
                remember_default=False,
                transport=transport,
            )
            if not message:
                # Decision-only run: surface the decision's reply text and
                # stop — there is no turn to run.
                decision_text = getattr(result, "text", "") or str(result)
                transport.set_turn_outcome(is_error=bool(getattr(result, "is_error", False)))
                await transport.on_response(decision_text)
                return transport
            if getattr(result, "is_error", False):
                transport.set_turn_outcome(is_error=True)
                await transport.on_error(getattr(result, "text", "") or "Approval decision failed.")
                return transport
        if message:
            result = await handle_message(runtime, message, transport)
            transport.set_turn_outcome(is_error=bool(getattr(result, "is_error", False)))
    finally:
        transport.stop_activity()
        # Bounded teardown: the prompt command previously awaited bare
        # ``runtime.aclose()`` — when the MCP pool's sequential anyio
        # task-group teardown stalls (the 2026-09-21 headroom incident
        # recurred on 2026-10-26 ~16:11: the turn had already completed
        # and persisted, yet the CLI sat idle in epoll for the rest of
        # the prompt run because the unbounded close never returned).
        # The bounded helper falls back to terminating every tracked
        # MCP/proxy subprocess after the grace window.
        await _aclose_runtime_bounded(runtime)
    return transport


def _collect_model_pricing(
    runtime: HostRuntime,
) -> tuple[
    dict[str, tuple[float | None, float | None]],
    dict[str, tuple[float | None, float | None]],
]:
    """Load model prices once, preserving unknown versus explicit zero."""
    from runtime.model_manifest import load_model_manifest

    loaded = getattr(getattr(runtime, "session", None), "manifest", None)
    if loaded is None and not runtime.paths.models_path.exists():
        return {}, {}
    manifest = loaded if loaded is not None else load_model_manifest(runtime.paths.models_path)
    prices: dict[str, tuple[float | None, float | None]] = {}
    cache_prices: dict[str, tuple[float | None, float | None]] = {}
    for model in manifest.models:
        model_id = str(model.model_id)
        prices[model_id] = (model.prompt_price, model.completion_price)
        cache_prices[model_id] = (model.cache_read_price, model.cache_write_price)
    return prices, cache_prices
