#!/usr/bin/env bash
# Lisa Framework installer for Linux/macOS
#
# Installs the `lisa` CLI as a uv tool (no repo clone needed), creates
# ~/.lisa/.env for the selected model providers, and bootstraps the model
# manifest via `lisa models discover` + `lisa models apply`.
#
# Usage examples:
#   curl -fsSL https://raw.githubusercontent.com/Lisa-Framework/lisa-host-cli/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --provider ollama,openai
#   curl -fsSL ... | bash -s -- --provider anthropic --skip-diagnostics
#
# Until lisa-host-cli is published on PyPI, pass a wheel bundle (a .tar.gz of
# the release *.whl files, e.g. a GitHub release asset):
#   curl -fsSL ... | bash -s -- --wheels https://github.com/Lisa-Framework/lisa-host-cli/releases/latest/download/lisa-wheels.tgz

set -euo pipefail

# --- Parameter parsing -------------------------------------------------------

# Wheel bundle for pre-PyPI installs. Empty => install from PyPI.
WHEELS_URL="${LISA_WHEELS_URL:-}"

# Default provider: ollama. Accepts comma-separated list.
IFS=',' read -ra PROVIDERS <<< "${LISA_PROVIDER:-ollama}"

SKIP_DIAGNOSTICS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheels)         WHEELS_URL="$2";     shift 2 ;;
        --provider)       IFS=',' read -ra PROVIDERS <<< "$2"; shift 2 ;;
        --skip-diagnostics) SKIP_DIAGNOSTICS=true; shift ;;
        --help|-h)
            cat <<HELP
Usage: install.sh [OPTIONS]

Install the Lisa CLI as a uv tool and configure model providers. After this
script, \`lisa\` works from anywhere — no repo checkout, no \`uv run\`.

Options:
  --wheels URL            URL of a .tar.gz wheel bundle (pre-PyPI installs).
                          Can also be set via \$LISA_WHEELS_URL. Supports
                          file:// URLs for local bundles.
  --provider LIST         Comma-separated provider(s): ollama,openai,azure_openai,
                          anthropic,google,openrouter,aws (default: \$LISA_PROVIDER or ollama)
  --skip-diagnostics      Skip running lisa verify after install
  --help, -h              Show this help

Examples:
  install.sh --wheels https://github.com/.../lisa-wheels.tgz      # pre-PyPI
  install.sh                                                       # PyPI (default)
  install.sh --provider ollama,openai
  install.sh --provider anthropic --skip-diagnostics
HELP
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Constants ---------------------------------------------------------------

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
    # NOTE on lisa-sdk: an unrelated "LISA" project owns the lisa-sdk name on
    # PyPI (v0.2.x). Until that is resolved, pre-PyPI installs pin
    # lisa-sdk==0.1.0 (our wheels) so the resolver can't grab the foreign
    # package from the index while satisfying all other deps via --find-links.
    if [[ -n "$WHEELS_URL" ]]; then
        local wheels_dir
        wheels_dir=$(mktemp -d)
        printf "  Downloading wheel bundle...\n"
        curl -fsSL "$WHEELS_URL" | tar -xz -C "$wheels_dir"
        local whl_count
        whl_count=$(find "$wheels_dir" -maxdepth 2 -name '*.whl' | wc -l)
        if [[ "$whl_count" -eq 0 ]]; then
            error_exit "No .whl files found in $WHEELS_URL"
        fi
        uv tool install --find-links "$wheels_dir" --with 'lisa-sdk<=0.1.0' lisa-host-cli \
            || uv tool install --find-links "$wheels_dir" --with 'lisa-sdk<=0.1.0' --reinstall lisa-host-cli
    else
        uv tool install lisa-host-cli \
            || uv tool upgrade lisa-host-cli
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

        printf "# --- Web search (optional) ---\n"
        printf "# WEB_SEARCH_API_KEY=   <-- fill in to enable /search\n"
        printf "\n"
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
    printf "    lisa once \"Hello, Lisa!\"\n"
    printf "    lisa              # interactive REPL\n"
    printf "\n"

    local is_ollama=false
    for prov in ${PROVIDERS[@]}; do
        [[ "$prov" == "ollama" ]] && { is_ollama=true; break; }
    done
    if $is_ollama; then
        printf "  Ollama selected — make sure it's installed and models pulled:\n"
        printf "    curl -fsSL https://ollama.com/install.sh | sh\n"
        printf "    ollama pull llama3.2:latest\n"
        printf "    ollama pull qwen2.5:7b\n"
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

    step 1 5 "Checking uv..."
    install_uv
    local uv_version
    uv_version=$(uv --version 2>/dev/null | cut -d' ' -f2 || echo "installed")
    printf "  OK: uv %s\n" "$uv_version"

    step 2 5 "Installing lisa CLI (uv tool)..."
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
