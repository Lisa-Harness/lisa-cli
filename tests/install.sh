#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$repo_root/install.sh"

# Load only the helper functions; never run setup or contact the network.
source <(sed -n '/^install_lisa() {$/,/^}$/p' "$repo_root/install.sh")
error_exit() { printf '%s\n' "$1" >&2; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }
for helper in packs_venv_path remove_path clean_install_state reset_seeded_state; do
    source <(sed -n "/^${helper}() {\$/,/^}\$/p" "$repo_root/install.sh")
done

CLI_REPO_URL="https://github.com/Lisa-Harness/lisa-cli.git"
LISA_REPO_URL="https://github.com/Lisa-Harness/lisa.git"
LISA_REPO="$repo_root"
CLI_REPO="$repo_root"
export UV_PYTHON=3.15

uv() {
    calls=$((calls + 1))
    [[ "$1" == tool && "$2" == install ]] || error_exit "Expected uv tool install"
    [[ "$3" == --python && "$4" == 3.12 ]] || error_exit "Expected Python 3.12"
    shift 4
    [[ "$1" == --no-sources || "$1" == --no-cache ]] || error_exit "Expected --no-sources"
    if [[ "$1" == --no-cache ]]; then
        # Retry path: --no-cache --no-sources --force --reinstall.
        [[ "$calls" == 2 ]] || error_exit "Unexpected --no-cache on the first attempt"
        shift
        [[ "$1" == --no-sources && "$2" == --force && "$3" == --reinstall ]] ||
            error_exit "Expected --no-sources --force --reinstall on retry"
        shift 3
    else
        shift
    fi
    [[ "$1" == "$expected_requirement" ]] || error_exit "Incorrect CLI source or missing prompt extra"
    shift
    [[ "$#" == 16 ]] || error_exit "Expected all eight framework requirements"
    while [[ "$#" -gt 0 ]]; do
        [[ "$1" == --with && "$2" == *" @ "* ]] || error_exit "Invalid framework requirement"
        shift 2
    done
    if [[ "$retry" == true && "$calls" == 1 ]]; then
        return 1
    fi
    return 0
}

for SOURCE in git local; do
    if [[ "$SOURCE" == git ]]; then
        expected_requirement="lisa-cli[prompt] @ git+$CLI_REPO_URL#subdirectory=hosts/cli"
    else
        expected_requirement="lisa-cli[prompt] @ $CLI_REPO/hosts/cli"
    fi
    for retry in false true; do
        calls=0
        install_lisa
        if [[ "$retry" == true ]]; then
            [[ "$calls" == 2 ]]
        else
            [[ "$calls" == 1 ]]
        fi
    done
done

# --clean / --reset: run against a throwaway data dir with uv stubbed so no
# real install or uninstall is ever touched.
sandbox="$(mktemp -d)"
trap 'rm -rf "$sandbox"' EXIT
LISA_HOME="$sandbox/.lisa"
mkdir -p "$LISA_HOME/venv/lib" "$LISA_HOME/state" "$LISA_HOME/config"
for seeded in \
    "$LISA_HOME/venv/pyvenv.cfg" \
    "$LISA_HOME/.packs-venv.lock" \
    "$LISA_HOME/state/installed-packs.json" \
    "$LISA_HOME/state/installed-packs.json.lock" \
    "$LISA_HOME/state/models.proposed.yaml" \
    "$LISA_HOME/config/models.yaml" \
    "$LISA_HOME/config/session_context.md"; do
    : > "$seeded"
done

uv() {
    [[ "$1" == tool && "$2" == uninstall && "$3" == lisa-cli ]] || error_exit "Expected uv tool uninstall"
    return "${UV_UNINSTALL_EXIT:-0}"
}
unset LISA_PACKS_VENV

[[ "$(packs_venv_path)" == "$LISA_HOME/venv" ]] || error_exit "Expected the data dir's venv"

clean_install_state
[[ ! -e "$LISA_HOME/venv" ]] || error_exit "Packs venv survived --clean"
[[ ! -e "$LISA_HOME/.packs-venv.lock" ]] || error_exit "Packs venv lock survived --clean"
[[ ! -e "$LISA_HOME/state/installed-packs.json" ]] || error_exit "Pack lock survived --clean"
[[ ! -e "$LISA_HOME/state/installed-packs.json.lock" ]] || error_exit "Pack lock sidecar survived --clean"
[[ -e "$LISA_HOME/config/models.yaml" ]] || error_exit "--clean must not touch the model manifest"

reset_seeded_state
[[ ! -e "$LISA_HOME/config/models.yaml" ]] || error_exit "Model manifest survived --reset"
[[ ! -e "$LISA_HOME/state/models.proposed.yaml" ]] || error_exit "Model proposal survived --reset"
[[ ! -e "$LISA_HOME/config/session_context.md" ]] || error_exit "Session context survived --reset"

# A uv failure other than "not installed" (exit 2) must not abort the install.
mkdir -p "$LISA_HOME/venv" "$LISA_HOME/state"
: > "$LISA_HOME/state/installed-packs.json"
UV_UNINSTALL_EXIT=1
clean_install_state >/dev/null || error_exit "A failed tool uninstall must not abort --clean"
[[ ! -e "$LISA_HOME/state/installed-packs.json" ]] || error_exit "Pack lock survived a failed uninstall"
unset UV_UNINSTALL_EXIT

# LISA_PACKS_VENV is an explicit override and must win over the data dir.
LISA_PACKS_VENV="$sandbox/custom-venv"
[[ "$(packs_venv_path)" == "$sandbox/custom-venv" ]] || error_exit "LISA_PACKS_VENV override ignored"
unset LISA_PACKS_VENV

# A profile shares the base data dir's packs venv.
[[ "$(LISA_HOME="$LISA_HOME/profiles/work" packs_venv_path)" == "$LISA_HOME/venv" ]] ||
    error_exit "Profile must share the base packs venv"

printf 'Bash installer tests passed\n'
