#!/usr/bin/env bash
# Build the full lisa release wheel set into a single directory.
#
# Usage:
#   ./build-release.sh [OUT_DIR]
#
# Builds every distributable package from both uv workspaces
# (lisa + lisa-hosts/cli) into OUT_DIR (default: dist-release/), then
# verifies each wheel actually contains files (uv caches by name+version,
# so a silent EMPTY wheel is the classic failure mode — we check for it).
#
# Note: setuptools' sdist hook can leave workspace litter (e.g. kernel/kernel,
# <pkg>/build/, *.egg-info/). This script builds wheel-only (no sdists), which
# does not trigger it, and cleans litter before and after anyway. Re-run this
# script instead of hand-rolling `uv build` loops.

set -euo pipefail

REPO_LISA="${LISA_REPO:-$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)/lisa}"
REPO_CLI="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-dist-release}"

# Allow running from either checkout.
if [[ ! -f "$REPO_CLI/pyproject.toml" || ! -d "$REPO_CLI/hosts/cli/src" ]]; then
    REPO_CLI="$(pwd)"
fi

# Packages per workspace, in dependency order (deps first).
LISA_PKGS=(lisa-kernel lisa-sdk lisa-mcp-servers lisa-pack-memory lisa-pack-core lisa-pack-introspect lisa-pack-recovery lisa-host-runtime)
CLI_PKGS=(lisa-host-cli)

step() { printf '==> %s\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

clean_workspace_litter() {
    # build/ and *.egg-info/ dirs break/redirect setuptools discovery; nested
    # package copies (kernel/kernel etc.) appear after some sdist passes.
    # Remove ALL of them: wheel-only builds never need them.
    local repo=$1
    [[ -d "$repo" ]] || return 0
    find "$repo" \( -path '*/.venv' -prune \) -o \
        \( -type d \( -name build -o -name '*.egg-info' -o -name '__pycache__' \) \
           -path "$repo/kernel/*" -prune \) -print0 2>/dev/null \
        | xargs -0 -r rm -rf 2>/dev/null || true
    # Simpler and safe: every build dir + egg-info under the member roots
    local root
    for root in kernel sdk host mcp_server; do
        [[ -d "$repo/$root" ]] || continue
        find "$repo/$root" \( -path '*/__pycache__' -o -name build -o -name '*.egg-info' \) \
            -type d -prune -exec rm -rf {} + 2>/dev/null || true
    done
    rm -rf "$repo"/lisa_*.egg-info 2>/dev/null || true
    # CLI member: clean its src/ and tests/ roots too.
    for root in src hosts/cli/src hosts/cli/tests; do
        [[ -d "$repo/$root" ]] || continue
        find "$repo/$root" \( -path '*/__pycache__' -o -name build -o -name '*.egg-info' \) \
            -type d -prune -exec rm -rf {} + 2>/dev/null || true
    done
}

# Absolutize NOW, before any subshell `cd`s: `uv build --out-dir` resolves a
# relative path against ITS cwd, so a bare "dist-release" would scatter the
# wheels across each repo's own dist-release/ directory.
OUT_DIR="$(readlink -f "${1:-dist-release}")"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.whl 2>/dev/null || true

step_lisa_repo="${LISA_REPO:-}"
# Standard multi-repo layout: Code/lisa next to Code/lisa-hosts/cli — that is
# two levels up from this script (script lives at lisa-hosts/cli/).
if [[ -z "$step_lisa_repo" ]]; then
    for candidate in \
        "$REPO_CLI/../../lisa" \
        "$REPO_CLI/../lisa" \
        "$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)/lisa"; do
        if [[ -n "$candidate" && -d "$candidate" && -f "$candidate/pyproject.toml" ]]; then
            step_lisa_repo="$(readlink -f "$candidate")"
            break
        fi
    done
fi
if [[ -z "$step_lisa_repo" || ! -d "$step_lisa_repo" ]]; then
    # Fall back: this script also works when placed inside the lisa repo itself.
    if [[ -f "$REPO_CLI/kernel/pyproject.toml" ]]; then
        step_lisa_repo="$REPO_CLI"
        LISA_PKGS=(lisa-kernel lisa-sdk lisa-mcp-servers lisa-pack-memory lisa-pack-core lisa-pack-introspect lisa-pack-recovery lisa-host-runtime)
        # No CLI repo in this layout; build only the framework wheels.
        CLI_PKGS=()
    else
        fail "Could not locate the lisa framework checkout. Set LISA_REPO=/path/to/lisa."
    fi
fi

step "Cleaning workspace litter in $step_lisa_repo"
clean_workspace_litter "$step_lisa_repo"
clean_workspace_litter "$REPO_CLI"

step "Building ${#LISA_PKGS[@]} framework wheels from $step_lisa_repo"
for pkg in "${LISA_PKGS[@]}"; do
    (cd "$step_lisa_repo" && uv build --package "$pkg" --no-cache --wheel --out-dir "$OUT_DIR") \
        || fail "uv build $pkg failed"
done

step "Building lisa-host-cli from $REPO_CLI"
(cd "$REPO_CLI" && uv build --package lisa-host-cli --no-cache --wheel --out-dir "$OUT_DIR") \
    || fail "uv build lisa-host-cli failed"

step "Verifying wheels (non-zero file count, expected top-level module present)"
python3 - "$OUT_DIR" <<'PY'
import sys, zipfile, pathlib

EXPECT_TOP = {
    "lisa_kernel": "kernel/",
    "lisa_sdk": "sdk/",
    "lisa_mcp_servers": "mcp_server/",
    "lisa_pack_core": "sdk/packs/core/",
    "lisa_pack_memory": "sdk/packs/memory/",
    "lisa_pack_introspect": "sdk/packs/introspect/",
    "lisa_pack_recovery": "sdk/packs/recovery/",
    "lisa_host_runtime": "host/",
    "lisa_host_cli": "hosts/cli/",
}

bad = []
for whl in sorted(pathlib.Path(sys.argv[1]).glob("*.whl")):
    names = [n for n in zipfile.ZipFile(whl).namelist() if ".dist-info" not in n]
    dist = whl.name.split("-")[0]
    expect = EXPECT_TOP.get(dist, "")
    top_ok = any(n.startswith(expect) for n in names)
    print(f"  {whl.name:47s} {len(names):4d} files  top={expect} ok={top_ok}")
    if not names or not top_ok:
        bad.append(whl.name)

if bad:
    print("\nBROKEN WHEELS:", ", ".join(bad))
    sys.exit(1)
print("All wheels verified.")
PY

# Post-build litter sweep: wheel builds regenerate build/ dirs in member
# roots (harmless but noisy); nested package copies would be real trouble.
clean_workspace_litter "$step_lisa_repo"
clean_workspace_litter "$REPO_CLI"

step "Done: $(ls "$OUT_DIR"/*.whl | wc -l) wheels in $(realpath "$OUT_DIR")"
