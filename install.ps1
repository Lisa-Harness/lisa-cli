<#
.SYNOPSIS
    Install the Lisa CLI as a uv tool and configure model providers.

.DESCRIPTION
    Installs the `lisa` CLI as a uv tool (no repo clone needed), creates
    ~\.lisa\.env for the selected model providers, and bootstraps the model
    manifest via `lisa models discover` + `lisa models apply`. After this
    script, `lisa` works from any directory — no checkout, no `uv run`.

    Until the packages are published on PyPI, the installer resolves them
    straight from git (default) or from local checkouts of both repos:

        .\install.ps1 -Source local -LisaRepo C:\Code\lisa

.PARAMETER Source
    Install source: "git" (default) resolves the packages from GitHub;
    "local" uses local checkouts (requires -LisaRepo). Also settable via the
    LISA_SOURCE environment variable.

.PARAMETER LisaRepo
    Local checkout of the framework repo (lisa) for -Source local. Also
    settable via the LISA_GIT_REPO environment variable. The CLI repo
    defaults to ..\lisa-hosts\cli relative to it (override with
    LISA_CLI_REPO).

.PARAMETER Provider
    One or more model providers to enable at install time.
    Accepted values: ollama, openai, azure_openai, anthropic, google,
    openrouter, aws. Default: ollama (local, no API key). When multiple are
    given, .env is pre-populated with keys for all of them.

.PARAMETER SkipDiagnostics
    Skip running lisa verify after installation.

.EXAMPLE
    # Git source (default), Ollama only
    .\install.ps1

.EXAMPLE
    # Local checkouts of both repos
    .\install.ps1 -Source local -LisaRepo C:\Code\lisa

.EXAMPLE
    # PyPI install (default), Ollama only
    .\install.ps1

.EXAMPLE
    # Install with Azure OpenAI + Anthropic
    .\install.ps1 -Provider azure_openai, anthropic

.EXAMPLE
    # Skip the post-install verification pass
    .\install.ps1 -SkipDiagnostics

.LINK
    https://github.com/Lisa-Framework/lisa-host-cli
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet("git", "local")]
    [string]$Source = $(
        if ($env:LISA_SOURCE) { $env:LISA_SOURCE }
        else { "git" }
    ),

    [Parameter()]
    [string]$LisaRepo = $(
        if ($env:LISA_GIT_REPO) { $env:LISA_GIT_REPO }
        else { "" }
    ),

    [Parameter()]
    [string]$CliRepo = $(
        if ($env:LISA_CLI_REPO) { $env:LISA_CLI_REPO }
        else { "" }
    ),

    [Parameter()]
    [ValidateSet("ollama", "openai", "azure_openai", "anthropic", "google", "openrouter", "aws")]
    [string[]]$Provider = @(
        if ($env:LISA_PROVIDER) { $env:LISA_PROVIDER -split "," }
        else { "ollama" }
    ),

    [Parameter()]
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

function Resolve-LisaHome {
    # Data dir: tool-mode lisa loads .env from here (load order: CWD → data dir).
    if ($env:LISA_DATA_DIR) {
        $raw = $env:LISA_DATA_DIR
        if ($raw -eq "~" -or $raw.StartsWith("~\") -or $raw.StartsWith("~/")) {
            Join-Path $HOME $raw.TrimStart("~", "\", "/")
        }
        else { $raw }
    }
    else {
        Join-Path $HOME ".lisa"
    }
}
$LisaHome = Resolve-LisaHome

# -- Provider metadata -------------------------------------------------------

# NOTE: env var names use the LISA_* prefix. The CLI host reads LISA_* with
# fallbacks to bare OPENAI_API_KEY / ANTHROPIC_API_KEY / AZURE_OPENAI_* /
# GOOGLE_API_KEY / OLLAMA_HOST / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_REGION.
$ProviderMeta = @{
    ollama       = @{ Label = "Ollama";       EnvVars = @("LISA_OLLAMA_HOST");         Defaults = @("127.0.0.1:11434") }
    openai       = @{ Label = "OpenAI";       EnvVars = @("LISA_OPENAI_API_KEY");       Defaults = @("") }
    azure_openai = @{ Label = "Azure OpenAI"; EnvVars = @("LISA_AZURE_OPENAI_ENDPOINT", "LISA_AZURE_OPENAI_API_KEY", "LISA_AZURE_OPENAI_API_VERSION"); Defaults = @("", "", "2024-10-21") }
    anthropic    = @{ Label = "Anthropic";    EnvVars = @("LISA_ANTHROPIC_API_KEY", "LISA_ANTHROPIC_BASE_URL", "LISA_ANTHROPIC_VERSION"); Defaults = @("", "https://api.anthropic.com", "2023-06-01") }
    google       = @{ Label = "Google Gemini"; EnvVars = @("LISA_GOOGLE_API_KEY", "LISA_GOOGLE_API_BASE_URL"); Defaults = @("", "https://generativelanguage.googleapis.com/v1beta/openai/") }
    openrouter   = @{ Label = "OpenRouter";   EnvVars = @("LISA_OPENROUTER_API_KEY", "LISA_OPENROUTER_BASE_URL"); Defaults = @("", "https://openrouter.ai/api/v1") }
    aws          = @{ Label = "AWS Bedrock";  EnvVars = @("LISA_AWS_ACCESS_KEY_ID", "LISA_AWS_SECRET_ACCESS_KEY", "LISA_AWS_REGION"); Defaults = @("", "", "us-east-1") }
}

function Write-Step {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$Step,

        [Parameter(Mandatory)]
        [int]$Total,

        [Parameter(Mandatory)]
        [string]$Message
    )

    Write-Host "  [$Step/$Total] $Message" -ForegroundColor Cyan
}

function Install-UvIfMissing {
    [CmdletBinding()]
    param()

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "  Installing uv package manager..." -ForegroundColor Cyan
    $installScript = Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1
    Invoke-Expression $installScript

    $uvBinPaths = @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:USERPROFILE ".cargo\bin")
    )
    foreach ($binPath in $uvBinPaths) {
        if ((Test-Path $binPath) -and ($env:PATH -notlike "*$binPath*")) {
            $env:PATH = "$binPath;$env:PATH"
        }
    }
}

function Install-LisaTool {
    # Install (or upgrade) the lisa tool. uv manages the Python interpreter,
    # so no system Python >= 3.11 check is needed here.
    #
    # The framework packages (lisa-kernel, lisa-sdk, ...) are not on PyPI yet,
    # so every install passes --no-sources (ignore the repos' path-based
    # tool.uv.sources) and supplies each framework package as an explicit
    # requirement. Git mode points them at GitHub; local mode at checkouts.
    # The CLI repo is a uv workspace whose umbrella root is not a package;
    # the installable member lives at hosts\cli (hence #subdirectory).
    #
    # Until lisa-sdk is published, the foreign PyPI package of the same name
    # is irrelevant: the explicit requirement always wins over the index.
    [CmdletBinding()]
    param()

    $lisaRepoUrl = "https://github.com/Lisa-Framework/lisa.git"
    $cliRepoUrl = "https://github.com/Lisa-Framework/lisa-host-cli.git"
    $frameworkMembers = @{
        "lisa-host-runtime" = "host"
        "lisa-kernel"       = "kernel"
        "lisa-sdk"          = "sdk"
        "lisa-mcp-servers"  = "mcp_server"
        "lisa-pack-core"    = "sdk/packs/core"
        "lisa-pack-memory"  = "sdk/packs/memory"
        "lisa-pack-introspect" = "sdk/packs/introspect"
        "lisa-pack-recovery"   = "sdk/packs/recovery"
    }

    if ($Source -eq "local") {
        if (-not $LisaRepo) { throw "-Source local requires -LisaRepo (or `$env:LISA_GIT_REPO)" }
        if (-not (Test-Path $LisaRepo)) { throw "framework repo not found: $LisaRepo" }
        if (-not $CliRepo) {
            $CliRepo = Join-Path (Split-Path $LisaRepo -Parent) "lisa-hosts\cli"
        }
        if (-not (Test-Path $CliRepo)) { throw "CLI repo not found: $CliRepo (pass -CliRepo)" }

        $cliRequirement = "lisa-host-cli @ $(Join-Path $CliRepo "hosts\cli")"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
            "--with", "$_ @ $(Join-Path $LisaRepo $frameworkMembers[$_])"
        })
    }
    else {
        $cliRequirement = "lisa-host-cli @ git+$cliRepoUrl#subdirectory=hosts/cli"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
            "--with", "$_ @ git+$lisaRepoUrl#subdirectory=$($frameworkMembers[$_])"
        })
    }

    Write-Host "  Installing lisa (uv tool, source: $Source)..." -ForegroundColor Cyan
    & uv tool install --no-sources $cliRequirement @withArgs
    if ($LASTEXITCODE -ne 0) {
        & uv tool install --no-sources --reinstall $cliRequirement @withArgs
        if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit $LASTEXITCODE)" }
    }

    # Make bare `lisa` resolvable for the rest of this script (uv installs
    # executables into %USERPROFILE%\.local\bin, which may not be on PATH yet).
    $localBin = Join-Path $HOME ".local\bin"
    if ((Test-Path $localBin) -and ($env:PATH -notlike "*$localBin*")) {
        $env:PATH = "$localBin;$env:PATH"
    }
}

function New-GenerateEnvFile {
    # Write ~\.lisa\.env (the tool-mode .env location) from the provider
    # metadata — no repo checkout to copy .env.example from.
    [CmdletBinding()]
    param()

    if (-not (Test-Path $LisaHome)) {
        New-Item -ItemType Directory -Path $LisaHome -Force | Out-Null
    }

    $envFile = Join-Path $LisaHome ".env"
    if (Test-Path $envFile) {
        Write-Host "  $envFile already exists — preserved" -ForegroundColor Yellow
        return
    }

    $lines = @(
        "# Lisa Framework environment configuration",
        "# Loaded by lisa at startup (~\.lisa\.env in tool mode).",
        "# Generated by: install.ps1",
        ""
    )

    foreach ($provider in $Provider) {
        $meta = $ProviderMeta[$provider]
        $lines += "# $($meta.Label)"
        for ($i = 0; $i -lt $meta.EnvVars.Count; $i++) {
            $key = $meta.EnvVars[$i]
            $default = $meta.Defaults[$i]
            if ($default) {
                $lines += "$key=$default"
            } else {
                $lines += "# $key=   <-- fill in your value"
            }
        }
        $lines += ""
    }

    $lines += "# --- Web search (optional) ---"
    $lines += "# WEB_SEARCH_API_KEY=   <-- fill in to enable /search"
    $lines += ""

    Set-Content -Path $envFile -Value $lines -Encoding UTF8
    Write-Host "  Created $envFile — $($Provider.Count) provider(s) configured" -ForegroundColor Green
}

function New-BootstrapModels {
    # First-run model bootstrap: discover what providers can see, write
    # models.proposed.yaml, then promote it to the active models.yaml.
    # Non-fatal: with no reachable provider nothing is proposed; we skip
    # and print the manual steps instead.
    if (Test-Path (Join-Path $LisaHome "models.yaml")) {
        Write-Host "  Existing models.yaml found — skipping model bootstrap" -ForegroundColor Yellow
        return
    }

    Write-Host "  Discovering models from configured provider(s)..."
    try { & lisa models discover 2>&1 | Out-Host } catch { Write-Verbose "discover exited: $_" }

    if (-not (Test-Path (Join-Path $LisaHome "models.proposed.yaml"))) {
        Write-Host "  No models discovered (no provider reachable yet)." -ForegroundColor Yellow
        Write-Host "  After installing Ollama models or filling in your API key, run:"
        Write-Host "    lisa models discover; lisa models apply"
        return
    }

    Write-Host "  Promoting discovered models to active manifest..."
    try { & lisa models apply 2>&1 | Out-Host } catch { Write-Verbose "apply exited: $_" }
    if (Test-Path (Join-Path $LisaHome "models.yaml")) {
        Write-Host "  OK: $(Join-Path $LisaHome "models.yaml") ready" -ForegroundColor Green
    }
    else {
        Write-Warning "models apply did not complete; run manually: lisa models apply"
    }
}

function Write-PostInstallGuidance {
    $hasSecrets = @($Provider | Where-Object { $_ -ne "ollama" }).Count -gt 0

    Write-Host ""
    Write-Host "  ------------------------" -ForegroundColor Green
    Write-Host "  Lisa installed!" -ForegroundColor Green
    Write-Host ""

    if ($hasSecrets) {
        Write-Host "  To finish configuring your providers:" -ForegroundColor Yellow
        Write-Host "    1. Edit $(Join-Path $LisaHome ".env")" -ForegroundColor Yellow
        foreach ($provider in $Provider) {
            $meta = $ProviderMeta[$provider]
            $fillThese = @()
            for ($i = 0; $i -lt $meta.EnvVars.Count; $i++) {
                if (-not $meta.Defaults[$i]) { $fillThese += $meta.EnvVars[$i] }
            }
            if ($fillThese.Count -gt 0) {
                Write-Host "       $($meta.Label): uncomment and set $($fillThese -join ', ')" -ForegroundColor Yellow
            }
        }
        Write-Host "    2. Run: lisa models discover; lisa models apply" -ForegroundColor Yellow
        Write-Host "       (to refresh your model manifest)"
        Write-Host ""
    }

    Write-Host "  Quick start (from any directory):"
    Write-Host "    lisa once `"Hello, Lisa!`""
    Write-Host ""

    if ($Provider -contains "ollama") {
        Write-Host "  Ollama selected — make sure it's installed and models pulled:"
        Write-Host "    winget install Ollama.Ollama"
        Write-Host "    ollama pull llama3.2:latest"
        Write-Host "    ollama pull qwen2.5:7b"
        Write-Host ""
        Write-Host "  Then refresh the manifest:"
        Write-Host "    lisa models discover; lisa models apply"
        Write-Host ""
    }

    Write-Host "  Verify your setup:"
    Write-Host "    lisa verify"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Main installation
# ---------------------------------------------------------------------------

function Install-LisaFramework {
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "  Lisa Framework Installer" -ForegroundColor Green
    Write-Host "  ------------------------"
    Write-Host ""

    Write-Step -Step 1 -Total 5 -Message "Checking uv..."
    Install-UvIfMissing
    $uvVersion = (uv --version 2>$null) -replace '^uv\s+', ''
    Write-Host "  OK: uv $uvVersion" -ForegroundColor Green

    Write-Step -Step 2 -Total 5 -Message "Installing lisa CLI (uv tool)..."
    Install-LisaTool
    $lisaCmd = Get-Command lisa -ErrorAction SilentlyContinue
    if (-not $lisaCmd) {
        Write-Error "lisa was installed but is not on PATH. Add $(Join-Path $HOME ".local\bin") to your PATH, then reopen your shell."
        return
    }
    $lisaVersion = (& lisa --version 2>$null | Select-Object -Last 1)
    Write-Host "  OK: $lisaVersion" -ForegroundColor Green

    Write-Step -Step 3 -Total 5 -Message "Configuring providers in $LisaHome..."
    New-GenerateEnvFile

    Write-Step -Step 4 -Total 5 -Message "Bootstrapping model manifest (discover + apply)..."
    New-BootstrapModels

    if (-not $SkipDiagnostics) {
        Write-Step -Step 5 -Total 5 -Message "Running diagnostics..."
        Write-Host ""
        try { & lisa verify 2>&1 | Out-Host } catch { Write-Warning "Diagnostics encountered an issue (non-fatal): $_" }
        Write-Host ""
    }
    else {
        Write-Step -Step 5 -Total 5 -Message "Done."
    }

    Write-PostInstallGuidance
}

Install-LisaFramework
