<#
.SYNOPSIS
    Install the Lisa CLI as a uv tool and configure model providers.

.DESCRIPTION
    Installs the `lisa` CLI as a uv tool (no repo clone needed), creates
    ~\.lisa\.env for the selected model providers, seeds a starter session
    context (~\.lisa\config\session_context.md), and bootstraps the model
    manifest via `lisa models discover` + `lisa models apply`. After this
    script, `lisa` works from any directory - no checkout, no `uv run`.

    Parameters work when the downloaded file is run directly. For piped
    execution (`irm ... | iex`), use the env-var equivalents (LISA_SOURCE,
    LISA_PROVIDER, ...) or pass arguments through the iex string:

        irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1 | iex
        iex "& { $(irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1) } -Provider anthropic"

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
    https://github.com/Lisa-Harness/lisa-cli
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet("git", "local")]
    [string]$Source = $(
        if ([Environment]::GetEnvironmentVariable("LISA_SOURCE")) { [Environment]::GetEnvironmentVariable("LISA_SOURCE") }
        else { "git" }
    ),

    [Parameter()]
    [string]$LisaRepo = $(
        if ([Environment]::GetEnvironmentVariable("LISA_GIT_REPO")) { [Environment]::GetEnvironmentVariable("LISA_GIT_REPO") }
        else { "" }
    ),

    [Parameter()]
    [string]$CliRepo = $(
        if ([Environment]::GetEnvironmentVariable("LISA_CLI_REPO")) { [Environment]::GetEnvironmentVariable("LISA_CLI_REPO") }
        else { "" }
    ),

    [Parameter()]
    [ValidateSet("ollama", "openai", "azure_openai", "anthropic", "google", "openrouter", "aws")]
    [string[]]$Provider = @(
        if ([Environment]::GetEnvironmentVariable("LISA_PROVIDER")) { [Environment]::GetEnvironmentVariable("LISA_PROVIDER") -split "," }
        else { "ollama" }
    ),

    [Parameter()]
    [switch]$SkipDiagnostics
)

# PSScriptAnalyzer doesn't like how we use our params as globals; this calms
# it (same no-op-read trick as astral.sh/uv's install.ps1).
$null = $Source, $LisaRepo, $CliRepo, $Provider, $SkipDiagnostics

# -- Script state ------------------------------------------------------------

# A null MyCommand.Path means this code is being evaluated in the caller's
# session (irm | iex) - there is no script file on disk.
$script:ExecutedFromIex = ($null -eq $MyInvocation.MyCommand.Path)

function Write-Step {
    <#
    .SYNOPSIS
        Write one numbered banner line for an installation phase.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
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

function Test-InstallEnvironment {
    <#
    .SYNOPSIS
        Fail fast (actionable throw) when the host can't run this installer:
        PowerShell version, execution policy, language mode, TLS 1.2 support.
        Mirrors the preflight gates in astral.sh/uv's and Scoop's installers.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Gate checks a plural environment as a single predicate.')]
    param()
    Set-StrictMode -Version Latest

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        throw "This installer requires PowerShell >= 5."
    }

    if ($PSVersionTable.PSEdition -eq "Desktop") {
        # Windows PowerShell 5.1 only: a locally-run .ps1 must pass execution
        # policy. Under irm | iex no file executes, so skip the gate (Scoop's
        # contract). TLS availability is checked unconditionally below.
        if (-not $script:ExecutedFromIex) {
            $policy = Get-ExecutionPolicy
            if ($policy -notin @("Unrestricted", "RemoteSigned", "Bypass")) {
                throw "Execution policy '$policy' blocks local scripts. Run: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
            }
        }
    }

    if ($ExecutionContext.SessionState.LanguageMode -ne "FullLanguage") {
        throw "This installer requires FullLanguage mode (current: $($ExecutionContext.SessionState.LanguageMode))."
    }

    if ([System.Enum]::GetNames([System.Net.SecurityProtocolType]) -notcontains "Tls12") {
        throw "TLS 1.2 support is required (GitHub requires TLS 1.2); install .NET Framework >= 4.5."
    }
}

function Get-LisaHome {
    <#
    .SYNOPSIS
        Resolve the lisa data dir: LISA_DATA_DIR (with ~ expansion) or $HOME/.lisa.
    #>
    # Data dir: tool-mode lisa loads .env from here (load order: CWD -> data dir).
    $raw = [Environment]::GetEnvironmentVariable("LISA_DATA_DIR")
    if ([string]::IsNullOrEmpty($raw)) {
        return (Join-Path $HOME ".lisa")
    }
    if ($raw -eq "~" -or $raw.StartsWith("~\") -or $raw.StartsWith("~/")) {
        return (Join-Path $HOME $raw.TrimStart("~", "\", "/"))
    }
    return $raw
}

$script:LisaHome = Get-LisaHome

# -- Provider metadata -------------------------------------------------------

# NOTE: env var names use the LISA_* prefix. The CLI host reads LISA_* with
# fallbacks to bare OPENAI_API_KEY / ANTHROPIC_API_KEY / AZURE_OPENAI_* /
# GOOGLE_API_KEY / OLLAMA_HOST / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_REGION.
$script:ProviderMeta = @{
    ollama       = @{ Label = "Ollama"; EnvVars = @("LISA_OLLAMA_HOST"); Defaults = @("127.0.0.1:11434") }
    openai       = @{ Label = "OpenAI"; EnvVars = @("LISA_OPENAI_API_KEY"); Defaults = @("") }
    azure_openai = @{ Label = "Azure OpenAI"; EnvVars = @("LISA_AZURE_OPENAI_ENDPOINT", "LISA_AZURE_OPENAI_API_KEY", "LISA_AZURE_OPENAI_API_VERSION"); Defaults = @("", "", "2024-10-21") }
    anthropic    = @{ Label = "Anthropic"; EnvVars = @("LISA_ANTHROPIC_API_KEY", "LISA_ANTHROPIC_BASE_URL", "LISA_ANTHROPIC_VERSION"); Defaults = @("", "https://api.anthropic.com", "2023-06-01") }
    google       = @{ Label = "Google Gemini"; EnvVars = @("LISA_GOOGLE_API_KEY", "LISA_GOOGLE_API_BASE_URL"); Defaults = @("", "https://generativelanguage.googleapis.com/v1beta/openai/") }
    openrouter   = @{ Label = "OpenRouter"; EnvVars = @("LISA_OPENROUTER_API_KEY", "LISA_OPENROUTER_BASE_URL"); Defaults = @("", "https://openrouter.ai/api/v1") }
    aws          = @{ Label = "AWS Bedrock"; EnvVars = @("LISA_AWS_ACCESS_KEY_ID", "LISA_AWS_SECRET_ACCESS_KEY", "LISA_AWS_REGION"); Defaults = @("", "", "us-east-1") }
}

function Invoke-Native {
    <#
    .SYNOPSIS
        Run a native command and return its exit code.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(ValueFromRemainingArguments)]
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList 2>&1 | Out-Host
    return $LASTEXITCODE
}

function Install-UvIfMissing {
    <#
    .SYNOPSIS
        Install uv only when it is absent from PATH, then add its bin dirs to
        the current process PATH.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Installs uv conditionally; that behavior is inherent to the name.')]
    param()
    Set-StrictMode -Version Latest

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }

    Write-Verbose "uv not found - bootstrapping from astral.sh"
    $uvScript = Join-Path ([System.IO.Path]::GetTempPath()) ("lisa-uv-install-" + [guid]::NewGuid().ToString("N") + ".ps1")
    try {
        # Download-to-file, then invoke the file with the call operator. This
        # keeps an internet installer from executing a fetched string via iex.
        Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $uvScript -ErrorAction Stop
        & $uvScript
        if ($LASTEXITCODE -ne 0) { throw "uv installer failed (exit $LASTEXITCODE)" }
    }
    catch {
        throw "failed to download or run the uv installer from astral.sh: $($_.Exception.Message)"
    }
    finally {
        Remove-Item -LiteralPath $uvScript -Force -ErrorAction SilentlyContinue
    }

    $uvBinPaths = @(
        (Join-Path $HOME ".local\bin"),
        (Join-Path $HOME ".cargo\bin")
    )
    foreach ($binPath in $uvBinPaths) {
        if ((Test-Path $binPath) -and ($env:PATH -notlike "*$binPath*")) {
            $env:PATH = "$binPath;$env:PATH"
        }
    }
}

function Install-LisaTool {
    <#
    .SYNOPSIS
        Install (or upgrade) the lisa tool, resolving the framework packages
        from git or local checkouts.
    .DESCRIPTION
        uv manages the Python interpreter, so no system Python check is needed.

        The framework packages (lisa-kernel, lisa-sdk, ...) are not on PyPI
        yet, so every install passes --no-sources (ignore the repos'
        path-based tool.uv.sources) and supplies each framework package as an
        explicit requirement. Git mode points them at GitHub; local mode at
        checkouts. The CLI repo is a uv workspace whose umbrella root is not
        a package; the installable member lives at hosts\cli (hence
        #subdirectory).

        Until lisa-sdk is published, the foreign PyPI package of the same name
        is irrelevant: the explicit requirement always wins over the index.
    #>
    [CmdletBinding()]
    param()
    Set-StrictMode -Version Latest

    $lisaRepoUrl = "https://github.com/Lisa-Harness/lisa.git"
    $cliRepoUrl = "https://github.com/Lisa-Harness/lisa-cli.git"
    $frameworkMembers = [ordered]@{
        "lisa-harness-kernel"      = "kernel"
        "lisa-harness-runtime"     = "runtime"
        "lisa-harness-sdk"         = "sdk"
        "lisa-harness-mcp-servers" = "mcp_server"
        "lisa-pack-core"           = "sdk/packs/core"
        "lisa-pack-memory"         = "sdk/packs/memory"
        "lisa-pack-introspect"     = "sdk/packs/introspect"
        "lisa-pack-recovery"       = "sdk/packs/recovery"
    }

    if ($Source -eq "local") {
        if (-not $LisaRepo) { throw "-Source local requires -LisaRepo (or `$env:LISA_GIT_REPO)" }
        if (-not (Test-Path $LisaRepo)) { throw "framework repo not found: $LisaRepo" }
        if (-not $CliRepo) {
            $CliRepo = Join-Path (Split-Path $LisaRepo -Parent) "lisa-hosts\cli"
        }
        if (-not (Test-Path $CliRepo)) { throw "CLI repo not found: $CliRepo (pass -CliRepo)" }

        $cliRequirement = "lisa-cli @ $(Join-Path $CliRepo "hosts\cli")"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
                "--with", "$_ @ $(Join-Path $LisaRepo $frameworkMembers[$_])"
            })
    }
    else {
        $cliRequirement = "lisa-cli @ git+$cliRepoUrl#subdirectory=hosts/cli"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
                "--with", "$_ @ git+$lisaRepoUrl#subdirectory=$($frameworkMembers[$_])"
            })
    }

    Write-Verbose "Installing lisa (uv tool, source: $Source)"
    $exitCode = Invoke-Native -FilePath "uv" -ArgumentList (@("tool", "install", "--no-sources", $cliRequirement) + $withArgs)
    if ($exitCode -ne 0) {
        $exitCode = Invoke-Native -FilePath "uv" -ArgumentList (@("tool", "install", "--no-sources", "--reinstall", $cliRequirement) + $withArgs)
        if ($exitCode -ne 0) { throw "uv tool install failed (exit $exitCode)" }
    }

    # Make bare `lisa` resolvable for the rest of this script (uv installs
    # executables into ~/.local/bin, which may not be on PATH yet).
    $localBin = Join-Path $HOME ".local\bin"
    if ((Test-Path $localBin) -and ($env:PATH -notlike "*$localBin*")) {
        $env:PATH = "$localBin;$env:PATH"
    }
}

function New-LisaEnvFile {
    <#
    .SYNOPSIS
        Write ~\.lisa\.env (the tool-mode .env location) from the provider
        metadata - no repo checkout to copy .env.example from. Existing files
        are preserved.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Creates one .env file; plural is a legacy artifact.')]
    param()
    Set-StrictMode -Version Latest

    $lisaHome = $script:LisaHome
    $envFile = Join-Path $lisaHome ".env"
    if (Test-Path $envFile) {
        Write-Host "  $envFile already exists - preserved" -ForegroundColor Yellow
        return
    }

    if ((-not (Test-Path $lisaHome)) -and $PSCmdlet.ShouldProcess($lisaHome, "Create data directory")) {
        New-Item -ItemType Directory -Path $lisaHome -Force | Out-Null
    }

    $lines = @(
        "# Lisa Framework environment configuration",
        "# Loaded by lisa at startup (~\.lisa\.env in tool mode).",
        "# Generated by: install.ps1",
        ""
    )

    foreach ($providerName in $Provider) {
        $meta = $script:ProviderMeta[$providerName]
        $lines += "# $($meta.Label)"
        for ($i = 0; $i -lt $meta.EnvVars.Count; $i++) {
            $key = $meta.EnvVars[$i]
            $default = $meta.Defaults[$i]
            if ($default) {
                $lines += "$key=$default"
            }
            else {
                $lines += "# $key=   <-- fill in your value"
            }
        }
        $lines += ""
    }

    if ($PSCmdlet.ShouldProcess($envFile, "Create provider .env")) {
        Set-Content -Path $envFile -Value $lines -Encoding UTF8
        Write-Host "  Created $envFile - $($Provider.Count) provider(s) configured" -ForegroundColor Green
    }
}

function New-LisaSessionContext {
    <#
    .SYNOPSIS
        Seed a minimal-but-useful starter session context. Existing files are
        preserved - this is a first-run convenience, not a policy override.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Creates one session context file; plural is a legacy artifact.')]
    param()
    Set-StrictMode -Version Latest

    $configDir = Join-Path $script:LisaHome "config"
    $ctxFile = Join-Path $configDir "session_context.md"
    if (Test-Path $ctxFile) {
        Write-Host "  $ctxFile already exists - preserved" -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    # The required external packs auto-install from GitHub on first use.
    $content = @'
---
schema_version: 2
name: my-session
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration: {}
    required:
      - lisa-pack-markdown
      - lisa-pack-personas
      - lisa-pack-search
    # - lisa-pack-tavily
# Web search requires a web search provider pack, e.g., lisa-pack-tavily
# Add one above then designate it as the web_search tool below
providers:
  capabilities:
    memory: lisa-pack-memory
  # tools: { web_search: lisa-pack-tavily }
routing: { from: { pack: lisa-pack-core, session: default } }
---
# My Session

The user is running you via the CLI host at github.com/Lisa-Harness/lisa-cli.
However they have not yet installed any web search provider packs (e.g., lisa-pack-tavily).
'@
    if ($PSCmdlet.ShouldProcess($ctxFile, "Seed starter session context")) {
        # UTF8 without BOM on PS5.1 (Set-Content -Encoding UTF8 emits a BOM
        # there, and the kernel front-matter parser does not strip one).
        [System.IO.File]::WriteAllText($ctxFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host "  Created $ctxFile (starter: memory, web search, personas, markdown)" -ForegroundColor Green
    }
}

function Update-LisaModelManifest {
    <#
    .SYNOPSIS
        First-run model bootstrap: discover what providers can see (writes
        state\models.proposed.yaml), then promote it to the active
        config\models.yaml. Graceful at every step: reinstall (existing
        active manifest skips), no provider reachable (print recovery
        commands), stale proposal (--force re-discovers so a re-run after
        pulling models actually picks them up). Non-fatal throughout.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    param()
    Set-StrictMode -Version Latest

    $lisaHome = $script:LisaHome
    if (-not $PSCmdlet.ShouldProcess($lisaHome, "Discover and apply models")) {
        return
    }

    $activeFile = Join-Path $lisaHome "config\models.yaml"
    $proposalFile = Join-Path $lisaHome "state\models.proposed.yaml"

    if (Test-Path $activeFile) {
        $count = (Select-String -LiteralPath $activeFile -Pattern "model_id:" -AllMatches -ErrorAction SilentlyContinue).Count
        Write-Host "  Existing $activeFile ($count model(s)) - skipping model bootstrap" -ForegroundColor Yellow
        return
    }

    Write-Verbose "Discovering models from configured provider(s)"
    if (Get-Command lisa -ErrorAction SilentlyContinue) {
        # Best-effort: never fail the install on a discovery hiccup, and don't
        # leak its exit code to callers. --force overwrites a stale proposal
        # from an earlier run so re-runs actually re-discover.
        $null = Invoke-Native -FilePath "lisa" -ArgumentList @("models", "discover", "--force")
    }
    else {
        Write-Verbose "lisa not on PATH; skipping discover"
    }

    if (-not (Test-Path $proposalFile)) {
        Write-Host "  No models discovered yet (no provider reachable or configured)." -ForegroundColor Yellow
        Write-Host "  After installing Ollama models or filling in your API keys, run:"
        Write-Host "    lisa models discover; lisa models apply"
        Write-Host "  (lisa verify will flag the missing manifest until then - expected.)"
        return
    }

    Write-Verbose "Promoting discovered models to active manifest"
    $null = Invoke-Native -FilePath "lisa" -ArgumentList @("models", "apply")
    if (Test-Path $activeFile) {
        $count = (Select-String -LiteralPath $activeFile -Pattern "model_id:" -AllMatches -ErrorAction SilentlyContinue).Count
        Write-Host "  OK: $activeFile ready ($count model(s))" -ForegroundColor Green
    }
    else {
        Write-Warning "models apply did not complete; run manually: lisa models apply"
    }
}

function Write-PostInstallGuidance {
    <#
    .SYNOPSIS
        Print the next-steps panel (provider secrets, Ollama pulls, verify).
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    param()
    Set-StrictMode -Version Latest

    $lisaHome = $script:LisaHome
    $hasSecrets = @($Provider | Where-Object { $_ -ne "ollama" }).Count -gt 0

    Write-Host ""
    Write-Host "  ------------------------" -ForegroundColor Green
    Write-Host "  Lisa installed!" -ForegroundColor Green
    Write-Host ""

    if ($hasSecrets) {
        Write-Host "  To finish configuring your providers:" -ForegroundColor Yellow
        Write-Host "    1. Edit $(Join-Path $lisaHome ".env")" -ForegroundColor Yellow
        foreach ($providerName in $Provider) {
            $meta = $script:ProviderMeta[$providerName]
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
    Write-Host "    lisa prompt `"Let's get started, Lisa!`""
    Write-Host ""
    Write-Host "  The starter session auto-installs a few packs from GitHub"
    Write-Host "  (markdown, personas, search, serper) on first use."
    Write-Host ""

    if ($Provider -contains "ollama") {
        Write-Host "  Ollama selected - make sure it's installed and models pulled:"
        Write-Host "    winget install Ollama.Ollama"
        Write-Host "    ollama show"
        Write-Host "    ollama signin # sign in to use Ollama cloud models"
        Write-Host "    ollama pull glm-5.3-flash:cloud"
        Write-Host "    ollama pull glm-5.3:cloud"
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
    <#
    .SYNOPSIS
        Run the five installation phases end to end.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Installs a framework (plural product name) as one unit.')]
    param()
    Set-StrictMode -Version Latest

    Write-Host ""
    Write-Host "  Lisa Framework Installer" -ForegroundColor Green
    Write-Host "  ------------------------"
    Write-Host ""

    Test-InstallEnvironment

    Write-Step -Step 1 -Total 5 -Message "Checking uv..."
    Install-UvIfMissing
    $uvVersion = (uv --version 2>$null) -replace '^uv\s+', ''
    Write-Host "  OK: uv $uvVersion" -ForegroundColor Green

    Write-Step -Step 2 -Total 5 -Message "Installing lisa CLI (uv tool, source: $Source)..."
    Install-LisaTool
    $lisaCmd = Get-Command lisa -ErrorAction SilentlyContinue
    if (-not $lisaCmd) {
        throw "lisa was installed but is not on PATH. Add $(Join-Path $HOME ".local\bin") to your PATH, then reopen your shell."
    }
    $lisaVersion = (& lisa --version 2>$null | Select-Object -Last 1)
    Write-Host "  OK: $lisaVersion" -ForegroundColor Green

    Write-Step -Step 3 -Total 6 -Message "Configuring providers in $($script:LisaHome)..."
    New-LisaEnvFile

    Write-Step -Step 4 -Total 6 -Message "Seeding starter session context..."
    New-LisaSessionContext

    Write-Step -Step 5 -Total 6 -Message "Bootstrapping model manifest (discover + apply)..."
    Update-LisaModelManifest

    if (-not $SkipDiagnostics) {
        Write-Step -Step 6 -Total 6 -Message "Running diagnostics..."
        Write-Host ""
        $null = Invoke-Native -FilePath "lisa" -ArgumentList @("verify")
        Write-Host ""
    }
    else {
        Write-Step -Step 6 -Total 6 -Message "Done."
    }

    Write-PostInstallGuidance
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Dot-sourcing loads the functions without installing (the Pester contract,
# same as Scoop's Test-ShouldRunInstall). LISA_INSTALL_NOINSTALL forces the
# no-install path for CI shape checks.
$script:ShouldInstall = ($MyInvocation.InvocationName -ne '.')
if ([Environment]::GetEnvironmentVariable("LISA_INSTALL_NOINSTALL")) {
    $script:ShouldInstall = $false
}

if ($script:ShouldInstall) {
    try {
        $ErrorActionPreference = "Stop"
        Install-LisaFramework
    }
    catch {
        Write-Error "Install failed: $_" -ErrorAction Continue
        if (-not $env:CI -and $script:ExecutedFromIex) {
            # Never `exit` out from under an interactive iex session - that
            # would kill the caller's shell (Scoop's Exit-Install contract).
            $global:LASTEXITCODE = 1
        }
        else {
            exit 1
        }
    }
}
