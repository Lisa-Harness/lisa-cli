#!/usr/bin/env bash
# Lisa Framework installer for Linux/macOS
#
# Installs the `lisa` CLI as a uv tool, creates ~/.lisa/.env for the selected
# model providers, and bootstraps the model manifest via `lisa models
# discover` + `lisa models apply`. After this script, `lisa` works from any
# directory — no checkout, no `uv run`.
#
# Usage examples:
#   curl -fsSL https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --provider ollama,openai
#   curl -fsSL ... | bash -s -- --provider anthropic --no-verify
#
# Until the packages are published on PyPI, the installer resolves them
# straight from git (default) or from local checkouts of both repos:
#   curl -fsSL ... | bash -s --                          # git source (default)
#   ./install.sh --source local --lisa-repo /home/me/Code/lisa
#
# Requires: git (for git-mode installs) and network access to
# github.com/Lisa-Harness/{lisa-cli,lisa}.

set -euo pipefail

# --- Parameter parsing -------------------------------------------------------

# Install source: "git" (resolve packages from GitHub) or "local" (use local
# checkouts). Empty => git. Local mode needs --lisa-repo (or LISA_GIT_REPO)
# for the framework repo; the CLI repo defaults to ../cli next to it.
SOURCE="${LISA_SOURCE:-git}"
LISA_REPO="${LISA_GIT_REPO:-}"
CLI_REPO="${LISA_CLI_REPO:-}"

# Default provider: ollama. Accepts comma-separated list.
IFS=',' read -ra PROVIDERS <<< "${LISA_PROVIDER:-ollama}"

SKIP_DIAGNOSTICS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)         SOURCE="$2";         shift 2 ;;
        --lisa-repo)      LISA_REPO="$2";      shift 2 ;;
        --cli-repo)       CLI_REPO="$2";       shift 2 ;;
        --provider)       IFS=',' read -ra PROVIDERS <<< "$2"; shift 2 ;;
        --no-verify)      SKIP_DIAGNOSTICS=true; shift ;;
        --help|-h)
            cat <<HELP
Usage: install.sh [OPTIONS]

Install the Lisa CLI as a uv tool and configure model providers. After this
script, \`lisa\` works from anywhere — no repo checkout, no \`uv run\`.

Options:
  --source MODE           Install source: "git" (default) or "local".
                          Also settable via \$LISA_SOURCE.
  --lisa-repo DIR         Local checkout of the framework repo (lisa) for
                          --source local. Also settable via \$LISA_GIT_REPO.
  --cli-repo DIR          Local checkout of the CLI repo (defaults to
                          DIR/../cli relative to --lisa-repo).
  --provider LIST         Comma-separated provider(s): ollama,openai,azure_openai,
                          anthropic,google,openrouter,aws (default: \$LISA_PROVIDER or ollama)
  --no-verify             Do not run lisa verify after install
  --help, -h              Show this help

Examples:
  install.sh                                   # git source (default)
  install.sh --source git                      # same, explicit
  install.sh --source local --lisa-repo ~/Code/lisa
  install.sh --provider ollama,openai
  install.sh --provider anthropic --no-verify
HELP
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Constants ---------------------------------------------------------------

# Package source repos (git mode).
CLI_REPO_URL="https://github.com/Lisa-Harness/lisa-cli.git"
LISA_REPO_URL="https://github.com/Lisa-Harness/lisa.git"

# Data dir: tool-mode lisa loads .env from here (load order: CWD → ~/.lisa).
LISA_HOME="${LISA_DATA_DIR:-$HOME/.lisa}"

# Provider metadata: label, env var names, default values.
# The CLI host reads LISA_* with fallbacks to bare OPENAI_API_KEY /
# ANTHROPIC_API_KEY / AZURE_OPENAI_* / GOOGLE_API_KEY / OLLAMA_HOST / AWS_*.
declare -A _PROVIDER_LABEL
_PROVIDER_LABEL[ollama]="Ollama"
_PROVIDER_LABEL[openai]="OpenAI"
_PROVIDER_LABEL[azure_openai]="Azure OpenAI"
_PROVIDER_LABEL[anthropic]="Anthropic"
_PROVIDER_LABEL[google]="Google Gemini"
_PROVIDER_LABEL[openrouter]="OpenRouter"
_PROVIDER_LABEL[aws]="AWS Bedrock"

declare -A _PROVIDER_KEYS
_PROVIDER_KEYS[ollama]="LISA_OLLAMA_HOST"
_PROVIDER_KEYS[openai]="LISA_OPENAI_API_KEY"
_PROVIDER_KEYS[azure_openai]="LISA_AZURE_OPENAI_ENDPOINT,LISA_AZURE_OPENAI_API_KEY,LISA_AZURE_OPENAI_API_VERSION"
_PROVIDER_KEYS[anthropic]="LISA_ANTHROPIC_API_KEY,LISA_ANTHROPIC_BASE_URL,LISA_ANTHROPIC_VERSION"
_PROVIDER_KEYS[google]="LISA_GOOGLE_API_KEY,LISA_GOOGLE_API_BASE_URL"
_PROVIDER_KEYS[openrouter]="LISA_OPENROUTER_API_KEY,LISA_OPENROUTER_BASE_URL"
_PROVIDER_KEYS[aws]="LISA_AWS_ACCESS_KEY_ID,LISA_AWS_SECRET_ACCESS_KEY,LISA_AWS_REGION"

declare -A _PROVIDER_DEFAULTS
_PROVIDER_DEFAULTS[ollama]="127.0.0.1:11434"
_PROVIDER_DEFAULTS[openai]=""
_PROVIDER_DEFAULTS[azure_openai]=",,2024-10-21"
_PROVIDER_DEFAULTS[anthropic]=",https://api.anthropic.com,2023-06-01"
_PROVIDER_DEFAULTS[google]=",https://generativelanguage.googleapis.com/v1beta/openai/"
_PROVIDER_DEFAULTS[openrouter]=",https://openrouter.ai/api/v1"
_PROVIDER_DEFAULTS[aws]=",,us-east-1"

# --- Utility functions -------------------------------------------------------

step() {
    local s=$1 total=$2 msg=$3
    printf "  [%d/%d] %s\n" "$s" "$total" "$msg"
}

error_exit() {
    printf "  ERROR: %s\n" "$1" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

install_uv() {
    if have_cmd uv; then
        return 0
    fi
    printf "  Installing uv...\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

install_lisa() {
    # Install (or upgrade) the lisa tool. uv manages the Python interpreter,
    # so no system Python >= 3.11 check is needed here.
    #
    # The framework packages (lisa-kernel, lisa-sdk, ...) are not on PyPI yet,
    # so every install passes --no-sources (ignore the repos' path-based
    # tool.uv.sources) and supplies each framework package as an explicit
    # requirement. Git mode points them at GitHub; local mode at checkouts.
    # The CLI repo is a uv workspace whose umbrella root is not a package;
    # the installable member lives at hosts/cli (hence #subdirectory).
    #
    # Until lisa-sdk is published, the foreign PyPI package of the same name
    # is irrelevant: the explicit requirement always wins over the index.
    local -a git_withs=(
        --with "lisa-host-runtime @ git+${LISA_REPO_URL}#subdirectory=host"
        --with "lisa-kernel @ git+${LISA_REPO_URL}#subdirectory=kernel"
        --with "lisa-sdk @ git+${LISA_REPO_URL}#subdirectory=sdk"
        --with "lisa-mcp-servers @ git+${LISA_REPO_URL}#subdirectory=mcp_server"
        --with "lisa-pack-core @ git+${LISA_REPO_URL}#subdirectory=sdk/packs/core"
        --with "lisa-pack-memory @ git+${LISA_REPO_URL}#subdirectory=sdk/packs/memory"
        --with "lisa-pack-introspect @ git+${LISA_REPO_URL}#subdirectory=sdk/packs/introspect"
        --with "lisa-pack-recovery @ git+${LISA_REPO_URL}#subdirectory=sdk/packs/recovery"
    )
    local cli_requirement

    if [[ "$SOURCE" == "local" ]]; then
        [[ -n "$LISA_REPO" ]] || error_exit "--source local requires --lisa-repo (or \$LISA_GIT_REPO)"
        [[ -d "$LISA_REPO" ]] || error_exit "framework repo not found: $LISA_REPO"
        if [[ -z "$CLI_REPO" ]]; then
            CLI_REPO="$(dirname "$LISA_REPO")/lisa-hosts/cli"
        fi
        [[ -d "$CLI_REPO" ]] || error_exit "CLI repo not found: $CLI_REPO (pass --cli-repo)"

        local -a local_withs=(
            --with "lisa-host-runtime @ $LISA_REPO/host"
            --with "lisa-kernel @ $LISA_REPO/kernel"
            --with "lisa-sdk @ $LISA_REPO/sdk"
            --with "lisa-mcp-servers @ $LISA_REPO/mcp_server"
            --with "lisa-pack-core @ $LISA_REPO/sdk/packs/core"
            --with "lisa-pack-memory @ $LISA_REPO/sdk/packs/memory"
            --with "lisa-pack-introspect @ $LISA_REPO/sdk/packs/introspect"
            --with "lisa-pack-recovery @ $LISA_REPO/sdk/packs/recovery"
        )
        cli_requirement="lisa-cli @ $CLI_REPO/hosts/cli"
        WITHS=("${local_withs[@]}")
    else
        cli_requirement="lisa-cli @ git+${CLI_REPO_URL}#subdirectory=hosts/cli"
        WITHS=("${git_withs[@]}")
    fi

    if ! uv tool install --no-sources "${cli_requirement}" "${WITHS[@]}" 2>/dev/null; then
        uv tool install --no-sources --reinstall "${cli_requirement}" "${WITHS[@]}"
    fi

    # Make bare `lisa` resolvable for the rest of this script (uv installs
    # executables into ~/.local/bin, which may not be on PATH yet).
    export PATH="$HOME/.local/bin:$PATH"
}

generate_env_file() {
    local env_file="$LISA_HOME/.env"
    mkdir -p "$LISA_HOME"

    if [[ -f "$env_file" ]]; then
        printf "  %s already exists — preserved\n" "$env_file"
        return
    fi

    # ShellCheck: we intentionally want word splitting below for providers
    # shellcheck disable=SC2068
    {
        printf "# Lisa Framework environment configuration\n"
        printf "# Loaded by lisa at startup (~/.lisa/.env in tool mode).\n"
        printf "# Generated by: install.sh\n"
        printf "\n"

        for prov in ${PROVIDERS[@]}; do
            local label="${_PROVIDER_LABEL[$prov]}"
            printf "# %s\n" "$label"

            local keys="${_PROVIDER_KEYS[$prov]}"
            local defaults="${_PROVIDER_DEFAULTS[$prov]}"

            old_ifs="$IFS"
            IFS=','
            local key_array default_array
            read -ra key_array <<< "$keys"
            read -ra default_array <<< "$defaults"
            IFS="$old_ifs"

            for idx in "${!key_array[@]}"; do
                local k="${key_array[$idx]}"
                local d="${default_array[$idx]:-}"
                if [[ -n "$d" ]]; then
                    printf "%s=%s\n" "$k" "$d"
                else
                    printf "# %s=   <-- fill in your value\n" "$k"
                fi
            done
            printf "\n"
        done
    } > "$env_file"

    printf "  Created %s — %d provider(s) configured\n" "$env_file" "${#PROVIDERS[@]}"
}

bootstrap_models() {
    # First-run model bootstrap: discover what the configured providers can
    # see, write models.proposed.yaml, then promote it to the active
    # models.yaml. Non-fatal: with no reachable provider, discover writes no
    # proposal — skip and tell the user what to do instead.
    if [[ -f "$LISA_HOME/models.yaml" ]]; then
        printf "  Existing models.yaml found — skipping model bootstrap\n"
        return 0
    fi

    printf "  Discovering models from configured provider(s)...\n"
    lisa models discover || true

    if [[ ! -f "$LISA_HOME/models.proposed.yaml" ]]; then
        printf "  No models discovered (no provider reachable yet).\n"
        printf "  After installing Ollama models or filling in your API key, run:\n"
        printf "    lisa models discover && lisa models apply\n"
        return 0
    fi

    printf "  Promoting discovered models to active manifest...\n"
    if ! lisa models apply; then
        printf "  WARN: models apply did not complete; run manually:\n"
        printf "    lisa models apply\n"
        return 0
    fi
    printf "  OK: %s/models.yaml ready (%d model(s))\n" "$LISA_HOME" \
        "$(grep -c 'name:' "$LISA_HOME/models.yaml" 2>/dev/null || echo 0)"
}

post_install_guidance() {
    printf "\n"
    printf "  ------------------------\n"
    printf "  Lisa installed!\n"
    printf "\n"

    local has_secrets=false
    local prov
    for prov in ${PROVIDERS[@]}; do
        [[ "$prov" != "ollama" ]] && { has_secrets=true; break; }
    done

    if $has_secrets; then
        printf "  To finish configuring your providers:\n"
        printf "    1. Edit %s\n" "$LISA_HOME/.env"
        for prov in ${PROVIDERS[@]}; do
            [[ "$prov" == "ollama" ]] && continue
            local label="${_PROVIDER_LABEL[$prov]}"
            local keys="${_PROVIDER_KEYS[$prov]}"
            local defaults="${_PROVIDER_DEFAULTS[$prov]}"

            old_ifs="$IFS"
            IFS=','
            local key_array default_array
            read -ra key_array <<< "$keys"
            read -ra default_array <<< "$defaults"
            IFS="$old_ifs"

            local blanks=()
            for idx in "${!key_array[@]}"; do
                [[ -z "${default_array[$idx]:-}" ]] && blanks+=("${key_array[$idx]}")
            done
            if [[ ${#blanks[@]} -gt 0 ]]; then
                local joined
                printf -v joined ', ' "${blanks[@]}"
                printf "       %s: uncomment and set %s\n" "$label" "${joined#, }"
            fi
        done
        printf "    2. Run: lisa models discover && lisa models apply\n"
        printf "       (to refresh your model manifest)\n"
        printf "\n"
    fi

    printf "  Quick start (from any directory):\n"
    printf "    lisa prompt \"Hello, Lisa!\"\n"
    printf "\n"

    local is_ollama=false
    for prov in ${PROVIDERS[@]}; do
        [[ "$prov" == "ollama" ]] && { is_ollama=true; break; }
    done
    if $is_ollama; then
        printf "  Ollama selected — make sure it's installed and models pulled:\n"
        printf "    curl -fsSL https://ollama.com/install.sh | sh\n"
        printf "    ollama pull glm-5.3-flash:cloud\n"
        printf "    ollama pull glm-5.3:cloud\n"
        printf "\n"
        printf "  Then refresh the manifest:\n"
        printf "    lisa models discover && lisa models apply\n"
        printf "\n"
    fi

    printf "  Verify your setup:\n"
    printf "    lisa verify\n"
    printf "\n"
}

# --- Main --------------------------------------------------------------------

main() {
    printf "\n"
    printf "  Lisa Framework Installer\n"
    printf "  ------------------------\n"
    printf "\n"

    if ! have_cmd git; then
        error_exit "git is required for the ${SOURCE} install source but was not found on PATH"
    fi
    step 1 5 "Checking uv..."
    install_uv
    local uv_version
    uv_version=$(uv --version 2>/dev/null | cut -d' ' -f2 || echo "installed")
    printf "  OK: uv %s\n" "$uv_version"

    step 2 5 "Installing lisa CLI (uv tool, source: $SOURCE)..."
    install_lisa
    local lisa_version
    lisa_version=$(lisa --version 2>/dev/null | tail -1 || echo "installed")
    printf "  OK: %s\n" "$lisa_version"

    step 3 5 "Configuring providers in $LISA_HOME..."
    generate_env_file

    step 4 5 "Bootstrapping model manifest (discover + apply)..."
    bootstrap_models

    if ! $SKIP_DIAGNOSTICS; then
        step 5 5 "Running diagnostics..."
        printf "\n"
        lisa verify || true
        printf "\n"
    else
        step 5 5 "Done."
    fi

    post_install_guidance
}

main "$@"