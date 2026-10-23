#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$repo_root/install.sh"

# Load only the installation function; never run setup or contact the network.
source <(sed -n '/^install_lisa() {$/,/^}$/p' "$repo_root/install.sh")
error_exit() { printf '%s\n' "$1" >&2; exit 1; }

CLI_REPO_URL="https://github.com/Lisa-Harness/lisa-cli.git"
LISA_REPO_URL="https://github.com/Lisa-Harness/lisa.git"
LISA_REPO="$repo_root"
CLI_REPO="$repo_root"
export UV_PYTHON=3.15

uv() {
    calls=$((calls + 1))
    [[ "$1" == tool && "$2" == install ]] || error_exit "Expected uv tool install"
    [[ "$3" == --python && "$4" == 3.12 && "$5" == --no-sources ]] ||
        error_exit "Expected Python 3.12"
    shift 5
    if [[ "$calls" == 2 ]]; then
        [[ "$1" == --reinstall ]] || error_exit "Expected reinstall on retry"
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
printf 'Bash installer tests passed\n'
