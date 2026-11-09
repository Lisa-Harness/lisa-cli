"""Version selection for uv tool installs.

This module uses only the standard library so installers and Windows users can
run it outside the tool environment that uv is about to replace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, quote, urlencode, urlsplit

TOOL_NAME: Final = "lisa-cli"
INSTALL_PYTHON: Final = "3.12"
CLI_REPO_URL: Final = "https://github.com/Lisa-Harness/lisa-cli.git"
FRAMEWORK_REPO_URL: Final = "https://github.com/Lisa-Harness/lisa.git"
FRAMEWORK_MEMBERS: Final = (
    ("lisa-harness-runtime", "runtime"),
    ("lisa-harness-kernel", "kernel"),
    ("lisa-harness-sdk", "sdk"),
    ("lisa-harness-mcp-servers", "mcp_server"),
    ("lisa-pack-core", "sdk/packs/core"),
    ("lisa-pack-memory", "sdk/packs/memory"),
    ("lisa-pack-introspect", "sdk/packs/introspect"),
    ("lisa-pack-recovery", "sdk/packs/recovery"),
)
_FRAMEWORK_NAMES: Final = frozenset(name for name, _ in FRAMEWORK_MEMBERS)
_MODES: Final = ("keep", "latest", "version", "pin", "channel", "ref")
_SOURCE_KEYS: Final = frozenset(
    {"specifier", "git", "url", "path", "directory", "editable", "index", "rev", "tag", "branch"}
)
_VERSION_RE = re.compile(
    r"v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?[0-9]*)?"
    r"(?:-[0-9]+|[-_.]?(?:post|rev|r)[-_.]?[0-9]*)?"
    r"(?:[-_.]?dev[-_.]?[0-9]*)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?",
    re.IGNORECASE,
)
_CHANNEL_RE = re.compile(r"[0-9]{1,6}\.[0-9]{1,6}")


class UpdateError(Exception):
    """A version change could not be completed safely."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code if code > 0 else 1


@dataclass(frozen=True)
class Policy:
    mode: str = "keep"
    value: str = ""

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"Unknown version selection: {self.mode}")
        if self.mode in {"keep", "latest"}:
            if self.value:
                raise ValueError(f"{self.mode} does not take a value")
        elif self.mode in {"version", "pin"}:
            if not _VERSION_RE.fullmatch(self.value):
                raise ValueError(f"Invalid release version: {self.value!r}")
        elif self.mode == "channel":
            if not _CHANNEL_RE.fullmatch(self.value):
                raise ValueError("A channel must be a major.minor release, such as 0.3")
        elif (
            not self.value
            or self.value.startswith("-")
            or any(char.isspace() or not char.isprintable() for char in self.value)
        ):
            raise ValueError("A Git ref must be nonempty and contain no whitespace")


@dataclass(frozen=True)
class Selection:
    cli: Policy = Policy()
    framework: Policy = Policy()

    @property
    def changes_receipt(self) -> bool:
        return self.cli.mode != "keep" or self.framework.mode != "keep"

    @property
    def temporary(self) -> bool:
        return self.cli.mode == "version" or self.framework.mode == "version"


def choose_policy(
    *, version: str = "", pin: str = "", channel: str = "", ref: str = "", latest: bool = False
) -> Policy:
    choices = [
        (name, value)
        for name, value in (("version", version), ("pin", pin), ("channel", channel), ("ref", ref))
        if value
    ]
    if latest:
        choices.append(("latest", ""))
    if len(choices) > 1:
        raise ValueError(
            "Choose only one version, pin, channel, ref, or latest option per component"
        )
    return Policy(*choices[0]) if choices else Policy()


def specifier(policy: Policy) -> str:
    if policy.mode in {"version", "pin"}:
        return f"=={policy.value}"
    if policy.mode == "channel":
        major, minor = (int(part) for part in policy.value.split("."))
        return f">={major}.{minor},<{major}.{minor + 1}"
    return ""


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _receipt_data(text: str) -> dict[str, object]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise UpdateError(f"Cannot read the uv tool receipt: {exc}") from exc
    tool = data.get("tool")
    if not isinstance(tool, dict):
        raise UpdateError("The uv tool receipt has no [tool] table")
    return tool


def _entries(tool: dict[str, object], key: str) -> list[dict[str, object]]:
    entries = tool.get(key, [])
    if not isinstance(entries, list):
        raise UpdateError(f"The uv tool receipt has an invalid {key} array")
    result: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise UpdateError(f"The uv tool receipt has an invalid entry in {key}")
        result.append(dict(entry))
    return result


def _entry_name(entry: dict[str, object]) -> str:
    name = entry.get("name")
    if not isinstance(name, str):
        raise UpdateError("The uv tool receipt contains a requirement without a name")
    return _normalize_name(name)


def framework_names(text: str) -> frozenset[str]:
    return frozenset(
        _entry_name(entry)
        for entry in _entries(_receipt_data(text), "requirements")
        if _entry_name(entry) in _FRAMEWORK_NAMES
    )


def _git_source(repo: str, subdirectory: str, ref: str) -> str:
    return f"{repo}?{urlencode({'subdirectory': subdirectory, 'rev': ref})}"


def _selected_entry(
    entry: dict[str, object], policy: Policy, repo: str, subdirectory: str
) -> dict[str, object]:
    selected = {key: value for key, value in entry.items() if key not in _SOURCE_KEYS}
    if policy.mode == "ref":
        selected["git"] = _git_source(repo, subdirectory, policy.value)
    elif policy.mode != "latest":
        selected["specifier"] = specifier(policy)
    return selected


def _select_cli(entries: list[dict[str, object]], policy: Policy) -> None:
    matches = [index for index, entry in enumerate(entries) if _entry_name(entry) == TOOL_NAME]
    if len(matches) != 1:
        raise UpdateError("The uv tool receipt must contain exactly one lisa-cli requirement")
    index = matches[0]
    entries[index] = _selected_entry(entries[index], policy, CLI_REPO_URL, "hosts/cli")


def _select_framework(
    entries: list[dict[str, object]], policy: Policy, discard: frozenset[str]
) -> None:
    for name, subdirectory in FRAMEWORK_MEMBERS:
        matches = [index for index, entry in enumerate(entries) if _entry_name(entry) == name]
        if len(matches) > 1:
            raise UpdateError(f"The uv tool receipt contains multiple {name} requirements")
        if name in discard:
            if matches:
                entries.pop(matches[0])
        elif matches:
            index = matches[0]
            entries[index] = _selected_entry(
                entries[index], policy, FRAMEWORK_REPO_URL, subdirectory
            )
        elif policy.mode != "latest":
            entries.append(
                _selected_entry({"name": name}, policy, FRAMEWORK_REPO_URL, subdirectory)
            )


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise UpdateError("The uv tool receipt contains a non-string requirement key")
        return (
            "{ "
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)} = {_toml_value(item)}"
                for key, item in value.items()
            )
            + " }"
        )
    raise UpdateError(f"Unsupported value in the uv tool receipt: {type(value).__name__}")


def _array_end(text: str, start: int) -> int:
    depth = 0
    index = start
    quote_char = ""
    triple = False
    while index < len(text):
        char = text[index]
        if quote_char:
            delimiter = quote_char * (3 if triple else 1)
            if text.startswith(delimiter, index):
                index += len(delimiter)
                quote_char = ""
                continue
            if quote_char == '"' and char == "\\":
                index += 2
                continue
        elif char in "\"'":
            quote_char = char
            triple = text.startswith(char * 3, index)
            index += 3 if triple else 1
            continue
        elif char == "#":
            newline = text.find("\n", index)
            index = len(text) if newline == -1 else newline
            continue
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise UpdateError("Cannot locate the end of an array in the uv tool receipt")


def _replace_array(text: str, key: str, entries: list[dict[str, object]]) -> str:
    section = re.search(r"(?m)^[ \t]*\[tool\][ \t]*(?:\#[^\r\n]*)?$", text)
    if section is None:
        raise UpdateError("Cannot locate the [tool] table in the uv tool receipt")
    remainder = text[section.end() :]
    next_section = re.search(
        r"(?m)^[ \t]*\[{1,2}[^\]\r\n]+\]{1,2}[ \t]*(?:\#[^\r\n]*)?$", remainder
    )
    end = section.end() + next_section.start() if next_section else len(text)
    assignment = re.search(rf"(?m)^[ \t]*{key}[ \t]*=[ \t]*", text[section.end() : end])
    if assignment is None:
        raise UpdateError(f"Cannot locate {key} in the uv tool receipt")
    start = section.end() + assignment.end()
    if start >= len(text) or text[start] != "[":
        raise UpdateError(f"The uv tool receipt has an unsupported {key} format")
    stop = _array_end(text, start)
    if stop > end:
        raise UpdateError(f"The uv tool receipt has an unsupported {key} format")
    content = "[\n" + "".join(f"    {_toml_value(entry)},\n" for entry in entries) + "]"
    return text[:start] + content + text[stop:]


def rewrite_receipt(
    text: str, selection: Selection, *, discard_framework: frozenset[str] = frozenset()
) -> str:
    """Replace only selected requirements, retaining uv's other receipt settings."""
    tool = _receipt_data(text)
    data = tomllib.loads(text)
    data["tool"] = tool
    entries = _entries(tool, "requirements")
    targets: set[str] = set()
    if selection.cli.mode != "keep":
        _select_cli(entries, selection.cli)
        targets.add(TOOL_NAME)
    if selection.framework.mode != "keep":
        _select_framework(entries, selection.framework, discard_framework)
        targets.update(_FRAMEWORK_NAMES)
    if not targets:
        return text

    updates = {"requirements": entries}
    for key in ("constraints", "overrides"):
        if key in tool:
            updates[key] = [
                entry for entry in _entries(tool, key) if _entry_name(entry) not in targets
            ]
    for key, value in updates.items():
        text = _replace_array(text, key, value)
        tool[key] = value
    try:
        rewritten = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise UpdateError("The uv tool receipt could not be rewritten as valid TOML") from exc
    if rewritten != data:
        raise UpdateError(
            "The uv tool receipt could not be rewritten without changing other settings"
        )
    return text


def _write_receipt(receipt_path: Path, text: str) -> None:
    mode = stat.S_IMODE(receipt_path.stat().st_mode)
    descriptor, name = tempfile.mkstemp(prefix=".lisa-receipt-", dir=receipt_path.parent)
    temp_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, receipt_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _read_receipt(tool_dir: Path) -> str:
    try:
        return (tool_dir / "uv-receipt.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise UpdateError(f"Cannot read the uv tool receipt in {tool_dir}: {exc}") from exc


def _finish_temporary(
    tool_dir: Path, selection: Selection, original_framework: frozenset[str]
) -> None:
    if not selection.temporary:
        return
    after = Selection(
        Policy("latest") if selection.cli.mode == "version" else Policy(),
        Policy("latest") if selection.framework.mode == "version" else Policy(),
    )
    discard = (
        _FRAMEWORK_NAMES - original_framework
        if selection.framework.mode == "version"
        else frozenset()
    )
    try:
        receipt = tool_dir / "uv-receipt.toml"
        _write_receipt(
            receipt, rewrite_receipt(_read_receipt(tool_dir), after, discard_framework=discard)
        )
    except (OSError, UpdateError) as exc:
        raise UpdateError(
            "The requested version was installed, but latest tracking could not be restored. "
            "Run the installer with --latest (Bash) or -Latest (PowerShell) to recover: " + str(exc)
        ) from exc


def upgrade_tool(tool_dir: Path, selection: Selection, uv: str, *, force: bool = False) -> None:
    """Upgrade using uv's receipt, rolling its policy back if resolution fails."""
    receipt = tool_dir / "uv-receipt.toml"
    original = _read_receipt(tool_dir)
    original_framework = (
        framework_names(original) if selection.framework.mode == "version" else frozenset()
    )
    staged = rewrite_receipt(original, selection) if selection.changes_receipt else original
    if staged != original:
        try:
            _write_receipt(receipt, staged)
        except OSError as exc:
            raise UpdateError(f"Cannot update the uv tool receipt: {exc}") from exc
    argv = [uv, "tool", "upgrade", *(["--reinstall"] if force else []), TOOL_NAME]
    env = {**os.environ, "UV_TOOL_DIR": str(tool_dir.parent)}
    try:
        completed = subprocess.run(argv, check=False, env=env)
        if completed.returncode:
            raise UpdateError(
                f"uv tool upgrade failed (exit {completed.returncode})", completed.returncode
            )
    except (OSError, UpdateError, KeyboardInterrupt) as exc:
        if staged != original:
            try:
                current = _read_receipt(tool_dir)
            except UpdateError as read_exc:
                raise UpdateError(
                    f"Update failed and the previous uv receipt could not be restored: {read_exc}"
                ) from exc
            if current != staged:
                raise UpdateError(
                    "Update failed and the uv receipt changed during the operation; "
                    "it was not overwritten. Check the selection before retrying."
                ) from exc
            try:
                _write_receipt(receipt, original)
            except OSError as restore_exc:
                raise UpdateError(
                    "Update failed and the previous uv receipt could not be restored: "
                    f"{restore_exc}"
                ) from exc
        if isinstance(exc, OSError):
            raise UpdateError(f"Could not run uv: {exc}") from exc
        raise
    _finish_temporary(tool_dir, selection, original_framework)


def policy_status(text: str) -> list[str]:
    tool = _receipt_data(text)
    descriptions: dict[str, set[str]] = {"CLI": set(), "Framework": set()}
    for key in ("requirements", "constraints", "overrides"):
        for entry in _entries(tool, key):
            name = _entry_name(entry)
            group = "CLI" if name == TOOL_NAME else "Framework" if name in _FRAMEWORK_NAMES else ""
            if not group:
                continue
            version = entry.get("specifier")
            git = entry.get("git")
            if isinstance(version, str) and version:
                descriptions[group].add(version)
            if isinstance(git, str):
                ref = parse_qs(urlsplit(git).query).get("rev", [])
                if ref:
                    descriptions[group].add(f"Git ref {ref[0]}")
    return [
        f"{group} tracking: {', '.join(sorted(values))}"
        for group, values in descriptions.items()
        if values
    ]


def success_messages(selection: Selection) -> list[str]:
    messages = []
    for label, policy in (("CLI", selection.cli), ("Framework", selection.framework)):
        if policy.mode == "version":
            messages.append(
                f"{label} {policy.value} is temporary; "
                "the next normal update tracks the latest release."
            )
        elif policy.mode in {"pin", "channel", "ref"}:
            reset = "--framework-latest" if label == "Framework" else "--latest"
            messages.append(
                f"{label} is tracking {policy.mode} {policy.value}; "
                f"use lisa update {reset} to return to releases."
            )
    if messages:
        messages.append(
            "If this older CLI lacks the update command or --latest, "
            "rerun the installer with --latest (Bash) or -Latest (PowerShell)."
        )
    return messages


def helper_arguments(selection: Selection) -> list[str]:
    args = [
        "--cli-mode",
        selection.cli.mode,
        "--framework-mode",
        selection.framework.mode,
    ]
    if selection.cli.value:
        args.extend(("--cli-value", selection.cli.value))
    if selection.framework.value:
        args.extend(("--framework-value", selection.framework.value))
    return args


def _direct_git(repo: str, subdirectory: str, ref: str = "") -> str:
    revision = f"@{quote(ref, safe='/')}" if ref else ""
    return f"git+{repo}{revision}#subdirectory={subdirectory}"


def _install_requirement(name: str, policy: Policy, repo: str, subdirectory: str) -> str:
    if policy.mode == "ref":
        return f"{name} @ {_direct_git(repo, subdirectory, policy.value)}"
    return name + specifier(policy)


def _local_root(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not value or not path.is_dir():
        raise UpdateError(f"{label} checkout not found: {value or '(not specified)'}")
    return path.resolve()


def _install_cli(source: str, cli_repo: str, lisa_repo: str, policy: Policy) -> str:
    name = f"{TOOL_NAME}[prompt]"
    if policy.mode != "keep":
        return _install_requirement(name, policy, CLI_REPO_URL, "hosts/cli")
    if source == "git":
        return f"{name} @ {_direct_git(CLI_REPO_URL, 'hosts/cli')}"
    if source == "local":
        path = _local_root(
            cli_repo or str(_local_root(lisa_repo, "Framework").parent / "lisa-cli"), "CLI"
        )
        return f"{name} @ {path / 'hosts' / 'cli'}"
    return name


def _install_framework(source: str, lisa_repo: str, policy: Policy) -> list[str]:
    if policy.mode == "latest" or (policy.mode == "keep" and source == "pypi"):
        return []
    root = (
        _local_root(lisa_repo, "Framework") if source == "local" and policy.mode == "keep" else None
    )
    args: list[str] = []
    for name, subdirectory in FRAMEWORK_MEMBERS:
        if policy.mode != "keep":
            requirement = _install_requirement(name, policy, FRAMEWORK_REPO_URL, subdirectory)
        elif root is not None:
            requirement = f"{name} @ {root / subdirectory}"
        else:
            requirement = f"{name} @ {_direct_git(FRAMEWORK_REPO_URL, subdirectory)}"
        args.extend(("--with", requirement))
    return args


def install_tool(source: str, cli_repo: str, lisa_repo: str, selection: Selection, uv: str) -> None:
    if source not in {"pypi", "git", "local"}:
        raise UpdateError(f"Unknown install source: {source}")
    cli = _install_cli(source, cli_repo, lisa_repo, selection.cli)
    framework = _install_framework(source, lisa_repo, selection.framework)
    base = [uv, "tool", "install", "--python", INSTALL_PYTHON]
    first = subprocess.run([*base, "--no-sources", "--force", cli, *framework], check=False)
    if first.returncode:
        retry = subprocess.run(
            [*base, "--no-cache", "--no-sources", "--force", "--reinstall", cli, *framework],
            check=False,
        )
        if retry.returncode:
            raise UpdateError(f"uv tool install failed (exit {retry.returncode})", retry.returncode)
    if selection.temporary:
        result = subprocess.run([uv, "tool", "dir"], check=False, text=True, capture_output=True)
        if result.returncode or not result.stdout.strip():
            raise UpdateError(
                "The requested version was installed, but latest tracking could not be restored "
                "because uv tool dir failed. Rerun the installer with --latest (Bash) "
                "or -Latest (PowerShell)."
            )
        _finish_temporary(Path(result.stdout.strip()) / TOOL_NAME, selection, frozenset())


def standalone_command(tool_dir: Path, selection: Selection, uv: str, *, force: bool) -> list[str]:
    return [
        uv,
        "run",
        "--isolated",
        "--no-project",
        "--no-config",
        "--python",
        ">=3.11",
        "python",
        str(Path(__file__).resolve()),
        "update",
        "--tool-dir",
        str(tool_dir),
        *helper_arguments(selection),
        *(["--force"] if force else []),
    ]


def format_command(argv: list[str], *, windows: bool) -> str:
    if not windows:
        return shlex.join(argv)
    return "& " + " ".join("'" + part.replace("'", "''") + "'" for part in argv)


def _standalone_main() -> int:
    parser = argparse.ArgumentParser(description="Manage lisa-cli uv tool version policies")
    parser.add_argument("action", choices=("update", "install", "validate"))
    parser.add_argument("--tool-dir", type=Path)
    parser.add_argument("--source", choices=("pypi", "git", "local"), default="pypi")
    parser.add_argument("--cli-repo", default="")
    parser.add_argument("--lisa-repo", default="")
    parser.add_argument("--cli-mode", choices=_MODES, default="keep")
    parser.add_argument("--cli-value", default="")
    parser.add_argument("--framework-mode", choices=_MODES, default="keep")
    parser.add_argument("--framework-value", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        selection = Selection(
            Policy(args.cli_mode, args.cli_value), Policy(args.framework_mode, args.framework_value)
        )
        if args.action == "update":
            if args.tool_dir is None:
                parser.error("--tool-dir is required for update")
            upgrade_tool(args.tool_dir, selection, "uv", force=args.force)
        elif args.action == "install":
            install_tool(args.source, args.cli_repo, args.lisa_repo, selection, "uv")
        else:
            _install_cli(args.source, args.cli_repo, args.lisa_repo, selection.cli)
            _install_framework(args.source, args.lisa_repo, selection.framework)
        if args.action != "validate":
            for message in success_messages(selection):
                print(message)
    except (ValueError, UpdateError, OSError) as exc:
        print(f"Version selection failed: {exc}", file=sys.stderr)
        return exc.code if isinstance(exc, UpdateError) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_standalone_main())
