"""Import-free CLI bootstrap and operator path mirrors."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

# The framework functions remain the resolution authority for every command
# that builds a runtime. The local mirrors below re-implement only the
# import-free env contract ``default_data_dir``,
# ``profile_data_dir``, and ``framework_root`` document, and
# ``tests/test_startup_deferred_imports.py`` locks the mirror to the
# framework's own functions so drift fails CI.
_ENV_LOADED = False
"""Whether the .env candidate sweep has run in this process."""


def _mirror_default_data_dir() -> Path:
    """Return ``runtime.config.default_data_dir``'s platform default (~/.lisa).

    Import-free mirror of the framework function: the spec defers
    ``runtime.config`` off the ``--help`` / ``--version`` paths (PROMPT_STARTUP
    Slice C, CLI half), and the .env candidate order needs this value before
    any runtime command has run. ``runtime.config.default_data_dir`` is
    ``Path.home() / ".lisa"`` (``_DEFAULT_DATA_DIR_NAME``), resolved through
    ``Path.home()`` so Windows ``%USERPROFILE%`` and POSIX ``$HOME`` both
    work. A regression test asserts equality against the framework
    function.
    """
    return Path.home() / ".lisa"


_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _mirror_profile_data_dir(profile: str) -> Path:
    """Return ``runtime.config.profile_data_dir``'s value for ``profile``.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir`). Matches the framework contract:
    ``<profiles_dir>/<name>`` where ``profiles_dir`` honors
    ``LISA_PROFILES_DIR``, else ``<default_data_dir>/profiles``; the name
    must be a single path-safe segment or ``ValueError`` raises.
    """
    name = profile.strip()
    if not name:
        raise ValueError("LISA_PROFILE must not be empty")
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid profile name {name!r}: use letters, digits, '-', '_', '.' "
            "(single path segment, no separators)"
        )
    override = os.getenv("LISA_PROFILES_DIR", "").strip()
    if override:
        return (Path(override).expanduser() / name).resolve()
    return (_mirror_default_data_dir() / "profiles" / name).resolve()


def _resolve_data_dir(data_dir: str | Path = "", profile: str = "") -> Path:
    """Resolve one effective data dir without importing the framework.

    Precedence is explicit CLI data dir > explicit CLI profile >
    ``LISA_DATA_DIR`` > ``LISA_PROFILE`` > ``~/.lisa``. Use this for every
    CLI-owned artifact as well as runtime construction so history, hints,
    intake state, models, policy, and approvals cannot land in different
    profiles within one invocation.
    """
    override = str(data_dir).strip()
    if override:
        return Path(override).expanduser()
    name = profile.strip()
    if not name:
        override = os.getenv("LISA_DATA_DIR", "").strip()
        if override:
            return Path(override).expanduser()
        name = os.getenv("LISA_PROFILE", "").strip()
    if name:
        return _mirror_profile_data_dir(name)
    return _mirror_default_data_dir()


def _mirror_framework_root() -> Path:
    """Return ``runtime.paths.framework_root``'s value without importing runtime.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir``). Matches the framework contract:
    a repo checkout (a ``.git`` dir or a ``[tool.uv.workspace]``
    ``pyproject.toml`` walking up from the *framework's* module) resolves
    to the framework repo root; a tool install resolves to ``sys.prefix``.

    The framework root is resolved from where ``runtime`` actually lives,
    not from this CLI module's own location: the two are different trees
    in the sibling-checkout dev layout (the framework in ``../lisa``, the
    CLI in its own repo), so walking up from here would report the CLI
    repo while the framework reports its own. ``importlib.util.find_spec``
    resolves the installed framework's location *without executing* the
    ``runtime`` package — the import chain this deferral exists to avoid
    (``runtime/__init__`` → kernel → mcp) is never entered. A framework
    that cannot be located falls back to walking up from this module.
    """
    framework_root = _framework_root_from_installed_runtime()
    if framework_root is not None:
        return framework_root

    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if isinstance(payload.get("tool", {}).get("uv", {}).get("workspace"), dict):
                return candidate
    return Path(sys.prefix)


def _framework_root_from_installed_runtime() -> Path | None:
    """Return the framework repo root for the installed ``runtime``, or None.

    Mirrors ``runtime.paths.framework_root``'s repo-detection walk, but
    anchored at the installed ``runtime`` package's location rather than
    at this module. Returns ``None`` when the framework is not installed
    or is not in a repo checkout (the tool-install case, where the
    framework root is ``sys.prefix`` and the caller falls back to its own
    walk).
    """
    try:
        spec = importlib.util.find_spec("runtime")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    locations = list(spec.submodule_search_locations)
    if not locations:
        return None
    # ``runtime`` lives at ``<root>/runtime/src/runtime`` (src layout) or
    # ``<root>/runtime`` (flat layout); the repo root is above both.
    package_dir = Path(locations[0]).resolve()
    for candidate in (package_dir, *package_dir.parents):
        if (candidate / ".git").exists():
            return candidate
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if isinstance(payload.get("tool", {}).get("uv", {}).get("workspace"), dict):
                return candidate
    return None


try:
    from dotenv import dotenv_values

    def _env_data_dir() -> Path:
        """Resolve the .env data dir from the env: LISA_DATA_DIR > LISA_PROFILE.

        Honor LISA_DATA_DIR (and LISA_PROFILE) so tool-mode installs
        configured with a custom data dir find their .env before any
        HostRuntime exists to do the full config resolution. Falls back to
        the platform default (~/.lisa). Mirrors HostConfig.from_env's
        precedence: LISA_DATA_DIR > LISA_PROFILE > default_data_dir().
        """
        override = os.getenv("LISA_DATA_DIR", "").strip()
        if override:
            return Path(override).expanduser()
        profile = os.getenv("LISA_PROFILE", "").strip()
        if not profile:
            return _mirror_default_data_dir()
        try:
            return _mirror_profile_data_dir(profile)
        except ValueError:
            # Invalid profile name: HostConfig.from_env raises the real
            # startup error later; the .env loader falls back to the
            # platform default so the error panel (not an import crash)
            # is what the user sees.
            return _mirror_default_data_dir()

    def _env_candidates() -> tuple[Path, tuple[Path, ...]]:
        """Return (data-dir .env candidate, full low→high precedence order).

        Only explicitly configured locations participate. In particular, the
        current working directory is never an implicit configuration source:
        merely entering a downloaded checkout must not let it select a packs
        venv (an executable-code trust root) for the next CLI invocation.

        The base data-dir .env provides shared settings. A profile .env and
        then the installed framework's own .env are progressively more
        specific sources. The original process environment is applied last
        and remains authoritative over every file.
        """
        env_dir = _env_data_dir()
        base_dir = _mirror_default_data_dir()
        base_env = base_dir / ".env"
        candidates: list[Path] = []
        for candidate in (
            base_env,
            env_dir / ".env",
            _FRAMEWORK_ROOT / ".env",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
        return env_dir / ".env", tuple(candidates)

    # Static trust anchors, ordered lowest→highest precedence. The original
    # process environment is restored last and therefore wins over every file.
    _FRAMEWORK_ROOT = _mirror_framework_root()
    (_BASE_ENV_CANDIDATE, _ENV_SEARCH_CANDIDATES) = _env_candidates()

    def _load_env_candidates(candidates: tuple[Path, ...]) -> Path | None:
        """Merge trusted candidates in memory, then apply them once.

        Later files override earlier files only for keys absent from the
        original process environment. No intermediate file is exported, so a
        lower-precedence file can never influence resolution of a later one.
        """
        original = dict(os.environ)
        merged: dict[str, str] = {}
        loaded: Path | None = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            merged.update(
                {
                    key: value
                    for key, value in dotenv_values(candidate, interpolate=False).items()
                    if key not in original and value is not None
                }
            )
            loaded = candidate
        os.environ.update(merged)
        return loaded

    def _load_env_once() -> None:
        """Run the .env candidate sweep once per process (idempotent).

        PROMPT_STARTUP Slice C (CLI half): the sweep is invoked lazily by
        :func:`_bootstrap_env_and_packs` instead of at import time, so
        ``lisa --version`` (which reads no operator configuration) pays
        nothing. ``--help`` reaches it through pack-command discovery;
        runtime commands reach it before ``build_runtime`` so
        ``LISA_*`` env values from .env still configure the runtime.
        """
        global _ENV_LOADED  # noqa: PLW0603  # module-level idempotence seam
        if _ENV_LOADED:
            return
        _ENV_LOADED = True
        _load_env_candidates(_ENV_SEARCH_CANDIDATES)
except ImportError:  # pragma: no cover — python-dotenv is optional
    _FRAMEWORK_ROOT = _mirror_framework_root()

    def _load_env_once() -> None:  # pragma: no cover — optional-dependency stub
        """No-op when python-dotenv is absent (no candidates can load)."""
        global _ENV_LOADED  # noqa: PLW0603
        _ENV_LOADED = True


_PACKS_VENV_INJECTED = False
"""Whether the packs venv site-packages injection has run in this process."""

_PACKS_VENV_DIR_NAME = "venv"
_PROFILES_SUBDIR_NAME = "profiles"
_PYVER_MIN_PARTS = 2


def _mirror_base_data_dir() -> Path:
    """Return ``runtime.config.base_data_dir``'s value (venv-owning base dir).

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir`): ``LISA_DATA_DIR`` when set, else the
    platform default. The packs venv is shared with profiles through this
    base dir (``runtime.paths.profile_base_data_dir`` contract).
    """
    override = os.getenv("LISA_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _mirror_default_data_dir()


def _mirror_profile_base_data_dir(data_dir: Path) -> Path | None:
    """Return ``runtime.paths.profile_base_data_dir``'s value for ``data_dir``.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir``). Matches the framework contract: a
    profile ``<root>/<name>`` resolves to the base dir that owns the packs
    venv — the default root's parent when the profiles root sits under the
    data dir, the base data dir itself for a relocated root
    (``LISA_PROFILES_DIR``), the structural ``profiles/`` parent otherwise,
    and ``None`` when ``data_dir`` is not a profile.
    """
    data_dir = Path(data_dir).resolve()
    override = os.getenv("LISA_PROFILES_DIR", "").strip()
    root = Path(override).expanduser().resolve() if override else None
    default_root = (_mirror_default_data_dir() / _PROFILES_SUBDIR_NAME).resolve()
    if root is None:
        root = default_root
    if data_dir.parent == root:
        # Inside the profiles root. The default root sits under the data
        # dir, so its parent is the base; a relocated root has no such
        # relation, so the base is the data dir itself.
        if root == default_root:
            return root.parent
        return _mirror_base_data_dir()
    if data_dir.parent.name == _PROFILES_SUBDIR_NAME:
        return data_dir.parent.parent
    return None


def _mirror_default_packs_venv() -> Path:
    """Return ``runtime.paths.default_packs_venv``'s value.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir``). Matches the framework contract:
    ``LISA_PACKS_VENV`` > ``<LISA_DATA_DIR>/venv`` > the profile's BASE
    data dir's venv > ``<~/.lisa>/venv``. An invalid profile name falls
    through to the platform default (the startup error belongs to
    ``HostConfig.from_env``, not this resolver).
    """
    override = os.getenv("LISA_PACKS_VENV", "").strip()
    if override:
        return Path(override).expanduser()
    data_dir_env = os.getenv("LISA_DATA_DIR", "").strip()
    if data_dir_env:
        return Path(data_dir_env).expanduser().resolve() / _PACKS_VENV_DIR_NAME
    profile_env = os.getenv("LISA_PROFILE", "").strip()
    if profile_env:
        try:
            profile = _mirror_profile_data_dir(profile_env)
        except ValueError:
            pass
        else:
            base = _mirror_profile_base_data_dir(profile)
            if base is not None:
                return base / _PACKS_VENV_DIR_NAME
            return profile / _PACKS_VENV_DIR_NAME
    return _mirror_default_data_dir() / _PACKS_VENV_DIR_NAME


def _mirror_venv_python_dir(venv_path: Path) -> str:
    """Return ``runtime.paths._venv_python_dir``'s value for a venv.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir``). The packs venv may target a
    different Python than the host interpreter, so the ``pythonX.Y`` tag is
    read from the venv's own ``pyvenv.cfg`` (``version_info``/``version``)
    or scanned from ``<venv>/lib/``, falling back to the running
    interpreter's version.
    """
    cfg = venv_path / "pyvenv.cfg"
    if cfg.is_file():
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key in ("version_info", "version") and value:
                    parts = value.split(".")
                    if len(parts) >= _PYVER_MIN_PARTS and parts[0].isdigit() and parts[1].isdigit():
                        return f"python{parts[0]}.{parts[1]}"
        except OSError:  # pragma: no cover — unreadable cfg (root ignores modes)
            pass
    lib_dir = venv_path / "lib"
    if lib_dir.is_dir():
        for child in sorted(lib_dir.iterdir()):
            if (
                child.is_dir()
                and child.name.startswith("python")
                and child.name[6:].replace(".", "").isdigit()
            ):
                return child.name
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _mirror_packs_venv_site_packages(venv_path: Path) -> Path:
    """Return ``runtime.paths.packs_venv_site_packages``'s value.

    Import-free mirror (same deferral reason as
    :func:`_mirror_default_data_dir``): ``<venv>/Lib/site-packages`` on
    Windows, ``<venv>/lib/pythonX.Y/site-packages`` on POSIX with the tag
    from :func:`_mirror_venv_python_dir`.
    """
    if os.name == "nt":
        return venv_path / "Lib" / "site-packages"
    return venv_path / "lib" / _mirror_venv_python_dir(venv_path) / "site-packages"


def _inject_packs_venv_site_packages_once() -> None:
    """Insert the packs venv site-packages into ``sys.path`` once per process.

    The injection must precede the first ``importlib.metadata.entry_points``
    scan (pack-command discovery and pack discovery) so that packs
    installed via /pack install into the packs venv are discoverable.
    PROMPT_STARTUP Slice C (CLI half): the injection is an import-free
    mirror of ``runtime.paths.inject_packs_venv_site_packages`` —
    importing ``runtime.paths`` at this call site would drag the
    framework's eager ``runtime/__init__`` -> ``runtime.config`` chain
    (kernel.session_plan -> kernel.ipsar_fsm -> mcp) onto the ``--help``
    path, and pack-command discovery is what ``--help`` runs it for. The
    mirror is locked to the framework function by a parity regression
    test (``tests/test_startup_deferred_imports.py``).

    Like the framework original, this uses ``site.addsitedir()`` (not
    just ``sys.path.insert``) so ``.pth`` files in the packs venv
    site-packages are processed — critical for editable installs whose
    ``.pth`` ``import`` lines register the meta-path finders that make
    ``import lisa_packs.<name>`` work. Idempotent within the process.
    """
    global _PACKS_VENV_INJECTED  # noqa: PLW0603  # module-level idempotence seam
    if _PACKS_VENV_INJECTED:
        return
    _PACKS_VENV_INJECTED = True
    import site as _site

    venv = _mirror_default_packs_venv()
    site_packages = _mirror_packs_venv_site_packages(venv)
    if site_packages.is_dir():
        site_str = str(site_packages)
        # site.addsitedir processes .pth files (including the import lines
        # that register editable-install meta-path finders) and appends to
        # sys.path. Safe to call even if already on sys.path.
        _site.addsitedir(site_str)
        # Defensive (mirrors runtime.paths verbatim): addsitedir already
        # appends the directory, so the insert fires only when a .pth hook
        # removed it mid-processing — not reachable on a normal filesystem.
        if site_str not in sys.path:  # pragma: no cover — defensive, mirrors framework
            sys.path.insert(0, site_str)
            logger.debug("Injected packs venv site-packages: %s", site_str)


def _bootstrap_env_and_packs() -> None:
    """Run the CLI entry bootstrap (.env sweep + packs-venv injection) once.

    Ordering contract (unchanged from the import-time sequence): the .env
    candidates load FIRST so ``LISA_DATA_DIR`` / ``LISA_PACKS_VENV`` /
    ``LISA_PROFILE`` are honored when the packs venv is resolved, and the
    injection runs before any framework entry-point scan. Callers:
    ``_pack_commands`` (pack-command discovery for ``--help``, completion,
    and the dynamic pack subcommands) and ``_initialize_runtime`` (every
    runtime-building command).
    """
    _load_env_once()
    _inject_packs_venv_site_packages_once()
