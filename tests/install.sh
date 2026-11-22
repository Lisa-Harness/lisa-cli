#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$repo_root/install.sh"

# Load only the helper functions; never run setup or contact the network.
source <(sed -n '/^install_lisa() {$/,/^}$/p' "$repo_root/install.sh")
error_exit() { printf '%s\n' "$1" >&2; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }
for helper in run_version_helper packs_venv_path remove_path clean_install_state reset_seeded_state acquire_session_template cleanup_install_temps generate_session_context bootstrap_models; do
    source <(sed -n "/^${helper}() {\$/,/^}\$/p" "$repo_root/install.sh")
done
for helper in acquire_env_template cleanup_env_template env_template_value env_template_key_from_line existing_env_has_key create_env_template_file merge_env_template generate_env_file; do
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
    if [[ "$SOURCE" == pypi ]]; then
        [[ "$#" == 0 ]] || error_exit "Published installs should resolve framework dependencies"
    else
        [[ "$#" == 16 ]] || error_exit "Expected all eight framework requirements"
    fi
    while [[ "$#" -gt 0 ]]; do
        [[ "$1" == --with && "$2" == *" @ "* ]] || error_exit "Invalid framework requirement"
        shift 2
    done
    if [[ "$retry" == true && "$calls" == 1 ]]; then
        return 1
    fi
    return 0
}

for SOURCE in pypi git local; do
    if [[ "$SOURCE" == pypi ]]; then
        expected_requirement="lisa-cli[prompt]"
    elif [[ "$SOURCE" == git ]]; then
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

uv() {
    calls=$((calls + 1))
    [[ "$1" == run && "$2" == --isolated && "$3" == --no-project &&
       "$4" == --no-config && "$5" == --python && "$6" == 3.12 ]] ||
        error_exit "Expected isolated version helper"
    shift 6
    [[ "$1" == python && "$2" == "$VERSION_HELPER" && "$3" == install ]] ||
        error_exit "Expected version helper install"
    shift 3
    [[ "$1" == --source && "$2" == pypi && "$3" == --cli-mode && "$4" == version &&
       "$5" == --framework-mode && "$6" == pin && "$7" == --cli-value &&
       "$8" == 0.3.0 && "$9" == --framework-value && "${10}" == 0.5.2 ]] ||
        error_exit "Version selections were not forwarded"
}
SOURCE=pypi
CLI_MODE=version
CLI_VALUE=0.3.0
FRAMEWORK_MODE=pin
FRAMEWORK_VALUE=0.5.2
VERSION_HELPER="$repo_root/hosts/cli/src/hosts/cli/update.py"
calls=0
install_lisa
[[ "$calls" == 1 ]] || error_exit "Expected one version helper invocation"

if bash "$repo_root/install.sh" --version 0.3.0 --pin 0.3.1 >/dev/null 2>&1; then
    error_exit "Conflicting CLI selections must fail"
fi
if bash "$repo_root/install.sh" --framework-ref >/dev/null 2>&1; then
    error_exit "Missing framework ref must fail"
fi

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

# A profile under a relocated profiles root (LISA_PROFILES_DIR) still shares
# the base data dir's packs venv, not a venv beside the relocated root.
base_home="$LISA_HOME"
relocated_root="$sandbox/OneDrive/Lisa.Profiles"
[[ "$(LISA_HOME="$relocated_root/work" LISA_DATA_DIR="$base_home" LISA_PROFILES_DIR="$relocated_root" packs_venv_path)" == "$base_home/venv" ]] ||
    error_exit "Relocated profile must share the base packs venv"

# .env template copy, substitution, merge, and failure behavior.
env_sandbox="$sandbox/env"
mkdir -p "$env_sandbox/repo"
cat > "$env_sandbox/repo/.env.example" <<'ENV_TEMPLATE'
# Lisa template
# LISA_OPENAI_API_KEY=
# LISA_OPENAI_BASE_URL=https://api.openai.com/v1/
MAX_TOOL_ITERATIONS=300
# LISA_COMMENTED_KEY=
ENV_TEMPLATE
INSTALLER_DIR="$env_sandbox/repo"
LISA_HOME="$env_sandbox/.lisa"
unset LISA_OPENAI_API_KEY OPENAI_API_KEY
rm -rf -- "$LISA_HOME"
mkdir -p "$LISA_HOME"
LISA_OPENAI_API_KEY="test-secret"
generate_env_file
grep -q '^LISA_OPENAI_API_KEY=test-secret$' "$LISA_HOME/.env" ||
    error_exit "First-run template substitution was not applied"
grep -q '^MAX_TOOL_ITERATIONS=300$' "$LISA_HOME/.env" ||
    error_exit "First-run template defaults were not copied"
grep -q '^# LISA_COMMENTED_KEY=$' "$LISA_HOME/.env" ||
    error_exit "Commented template placeholder was not preserved"

cat > "$LISA_HOME/.env" <<'ENV_EXISTING'
CUSTOM_SETTING=keep
# LISA_OPENAI_API_KEY=
ENV_EXISTING
unset LISA_OPENAI_API_KEY OPENAI_API_KEY
merge_env_template "$LISA_HOME/.env"
grep -q '^CUSTOM_SETTING=keep$' "$LISA_HOME/.env" ||
    error_exit "Existing custom setting was overwritten"
grep -q '^MAX_TOOL_ITERATIONS=300$' "$LISA_HOME/.env" ||
    error_exit "Missing active template key was not merged"
[[ $(grep -c '^# LISA_OPENAI_API_KEY=' "$LISA_HOME/.env") == 1 ]] ||
    error_exit "Existing commented key was duplicated"

rm -rf -- "$env_sandbox/no-template"
mkdir -p "$env_sandbox/no-template"
cat > "$env_sandbox/no-template/.env" <<'ENV_EXISTING'
CUSTOM_SETTING=keep
ENV_EXISTING
LISA_HOME="$env_sandbox/no-template"
INSTALLER_DIR="$env_sandbox/missing"
ENV_TEMPLATE_URL="http://127.0.0.1:1/.env.example"
curl() { return 22; }
generate_env_file >/dev/null
grep -q '^CUSTOM_SETTING=keep$' "$LISA_HOME/.env" ||
    error_exit "Existing env file was changed after template download failure"
unset -f curl

# Canonical starter session is copied exactly and existing files survive.
INSTALLER_DIR="$repo_root"
LISA_HOME="$sandbox/session/.lisa"
generate_session_context
cmp "$repo_root/config/session_context.md" "$LISA_HOME/config/session_context.md" ||
    error_exit "Bash did not copy the canonical starter session"
grep -q '^execution:$' "$LISA_HOME/config/session_context.md" ||
    error_exit "Canonical starter session lost execution budgets"
printf 'custom user context\n' > "$LISA_HOME/config/session_context.md"
generate_session_context
grep -q '^custom user context$' "$LISA_HOME/config/session_context.md" ||
    error_exit "Existing session context was overwritten"

# Model bootstrap failures are transactional, never successful no-ops.
LISA_HOME="$sandbox/models/.lisa"
mkdir -p "$LISA_HOME/state"
lisa() { return 1; }
if (bootstrap_models >/dev/null 2>&1); then
    error_exit "Model discovery failure must fail installation"
fi
unset -f lisa

lisa() {
    [[ "$1 $2" == "models discover" ]] && return 0
    [[ "$1 $2" == "models apply" ]] && return 1
    return 2
}
: > "$LISA_HOME/state/models.proposed.yaml"
if (bootstrap_models >/dev/null 2>&1); then
    error_exit "Model apply failure must fail installation"
fi
unset -f lisa

lisa() {
    [[ "$1 $2" == "models discover" ]] && return 0
    [[ "$1 $2" == "models apply" ]] && return 0
    return 2
}
if (bootstrap_models >/dev/null 2>&1); then
    error_exit "Apply without an active manifest must fail installation"
fi
unset -f lisa

# Top-level orchestration: verify failures cannot print success, and skipping
# diagnostics does not suppress model bootstrap failures.
source <(sed -n '/^main() {$/,/^}$/p' "$repo_root/install.sh")
source <(sed -n '/^post_install_guidance() {$/,/^}$/p' "$repo_root/install.sh")
cleanup_install_temps() { :; }
install_uv() { :; }
install_lisa() { :; }
generate_env_file() { :; }
generate_session_context() { :; }
post_install_guidance() { printf 'UNEXPECTED SUCCESS\n'; }
uv() { printf 'uv 0.0.0\n'; }
lisa() {
    if [[ "$1" == --version ]]; then printf 'lisa v0.5.0\n'; return 0; fi
    if [[ "$1 $2" == "models discover" ]]; then
        printf 'model_id: test\n' > "$LISA_HOME/state/models.proposed.yaml"
        return 0
    fi
    if [[ "$1 $2" == "models apply" ]]; then
        mkdir -p "$LISA_HOME/config"
        printf 'model_id: test\n' > "$LISA_HOME/config/models.yaml"
        return 0
    fi
    if [[ "$1" == verify ]]; then return 1; fi
    return 2
}
SOURCE=pypi
CLI_MODE=keep
FRAMEWORK_MODE=keep
DO_CLEAN=false
DO_RESET=false
SKIP_DIAGNOSTICS=false
LISA_HOME="$sandbox/main/.lisa"
mkdir -p "$LISA_HOME/state"
if (main > "$sandbox/main-output" 2>&1); then
    error_exit "Final verification failure must fail installation"
fi
grep -q 'UNEXPECTED SUCCESS' "$sandbox/main-output" &&
    error_exit "Success banner printed after verification failure"
unset -f lisa

SKIP_DIAGNOSTICS=true
lisa() { return 1; }
if (main >/dev/null 2>&1); then
    error_exit "--no-verify must not suppress model discovery failure"
fi
unset -f lisa

printf 'Bash installer tests passed\n'
