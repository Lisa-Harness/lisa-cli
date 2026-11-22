<#
.SYNOPSIS
    Install the Lisa CLI as a uv tool and configure model providers.

.DESCRIPTION
    Installs the `lisa` CLI as a uv tool (no repo clone needed), creates
    ~\.lisa\.env for the selected model providers, seeds a starter session
    context (~\.lisa\config\session_context.md), and bootstraps the model
    manifest via `lisa models discover` + `lisa models apply`. After this
    script, `lisa` works from any directory - no checkout, no `uv run`.

    The installer auto-detects execution mode:
      - Piped (irm ... | iex): defaults to -Source pypi
      - File (.\install.ps1): defaults to -Source local with -LisaRepo ..\lisa -CliRepo .

    Parameters work when the downloaded file is run directly. For piped
    execution (`irm ... | iex`), use the env-var equivalents (LISA_SOURCE,
    LISA_PROVIDER, ...) or pass arguments through the iex string:

        irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1 | iex
        iex "& { $(irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1) } -Provider anthropic"

    Piped installs use published releases by default. Git and local checkouts
    are available for development:

        .\install.ps1                                    # local source (default for file)
        .\install.ps1 -Source git                        # GitHub checkouts
        .\install.ps1 -Source local -LisaRepo C:\Code\lisa

    -Clean removes what the install owns outside uv's receipts (the packs
    venv and the pack lock) and lets uv remove the CLI tool itself; -Reset
    drops the seeded model manifest and session context so they are written
    again. Both then continue with a normal install.

.PARAMETER Source
    Install source: "pypi" (default for piped) uses published releases, "git"
    uses GitHub, and "local" (default for file) uses local checkouts.
    Also settable via the LISA_SOURCE environment variable.

.PARAMETER LisaRepo
    Local checkout of the framework repo (lisa) for -Source local.
    Default (file mode): ..\lisa. Also settable via the LISA_GIT_REPO
    environment variable. The CLI repo defaults to . in file mode, or to a
    sibling lisa-cli directory when unspecified. Override with LISA_CLI_REPO.

.PARAMETER Provider
    One or more model providers to enable at install time.
    Accepted values: ollama, openai, azure_openai, anthropic, google,
    openrouter, aws. Default: ollama (local, no API key). When multiple are
    given, .env is pre-populated with keys for all of them.

.PARAMETER Version
    Temporarily install a CLI release; the next normal update tracks latest.
.PARAMETER Pin
    Stay on an exact CLI release until changed.
.PARAMETER Channel
    Track CLI patch releases in a major.minor series.
.PARAMETER Ref
    Track a CLI Git tag, branch, or commit.
.PARAMETER Latest
    Track the latest published CLI and framework releases.
.PARAMETER FrameworkVersion
    Temporarily install a framework release.
.PARAMETER FrameworkPin
    Pin all framework packages to one release.
.PARAMETER FrameworkChannel
    Track a framework major.minor series.
.PARAMETER FrameworkRef
    Track a framework Git tag, branch, or commit.
.PARAMETER FrameworkLatest
    Track the latest published framework packages.

.PARAMETER SkipDiagnostics
    Skip running lisa verify after installation.

.PARAMETER Clean
    Remove the install's own state before installing: the packs venv,
    state\installed-packs.json, and the lisa uv tool (via
    `uv tool uninstall`). Deletes packs you installed into the runtime.

.PARAMETER Reset
    Delete the seeded model manifest (config\models.yaml,
    state\models.proposed.yaml) and config\session_context.md so this run
    writes them again. Combine with -Clean for a full teardown.

.EXAMPLE
    # Purge pack state and reinstall
    .\install.ps1 -Clean

.EXAMPLE
    # Full teardown: purge pack state and re-seed models + session
    .\install.ps1 -Clean -Reset

.EXAMPLE
    # Piped (irm | iex): published releases, Ollama only
    irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1 | iex

.EXAMPLE
    # Temporarily roll back, or return to releases after a pin
    iex "& { $(irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1) } -Version 0.3.0"
    iex "& { $(irm https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/install.ps1) } -Latest"

.EXAMPLE
    # File (.\install.ps1): local source (default), Ollama only
    .\install.ps1

.EXAMPLE
    # Local checkouts of both repos (explicit)
    .\install.ps1 -Source local -LisaRepo C:\Code\lisa

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
    [ValidateSet("pypi", "git", "local")]
    [string]$Source = $(
        # The invocation-mode check must live inside each default: parameter
        # defaults bind before the script body runs, so a script-scoped
        # variable assigned later (ExecutedFromIex) does not exist yet and
        # strict mode makes reading it a terminating error. A null
        # MyCommand.Path means the code is evaluating in the caller's session
        # (irm | iex); a real path means the saved script file was executed.
        if ([Environment]::GetEnvironmentVariable("LISA_SOURCE")) { [Environment]::GetEnvironmentVariable("LISA_SOURCE") }
        elseif ($null -eq $MyInvocation.MyCommand.Path) { "pypi" }  # piped: published releases
        else { "local" }  # file: default to local
    ),

    [Parameter()]
    [string]$LisaRepo = $(
        if ([Environment]::GetEnvironmentVariable("LISA_GIT_REPO")) { [Environment]::GetEnvironmentVariable("LISA_GIT_REPO") }
        elseif (($null -ne $MyInvocation.MyCommand.Path) -and ($Source -eq "local")) { "..\lisa" }  # file + local: default to ..\lisa
        else { "" }
    ),

    [Parameter()]
    [string]$CliRepo = $(
        if ([Environment]::GetEnvironmentVariable("LISA_CLI_REPO")) { [Environment]::GetEnvironmentVariable("LISA_CLI_REPO") }
        elseif (($null -ne $MyInvocation.MyCommand.Path) -and ($Source -eq "local")) { "." }  # file + local: default to .
        else { "" }
    ),

    [Parameter()]
    [ValidateSet("ollama", "openai", "azure_openai", "anthropic", "google", "openrouter", "aws")]
    [string[]]$Provider = @(
        if ([Environment]::GetEnvironmentVariable("LISA_PROVIDER")) { [Environment]::GetEnvironmentVariable("LISA_PROVIDER") -split "," }
        else { "ollama" }
    ),

    [Parameter()]
    [string]$Version = "",

    [Parameter()]
    [string]$Pin = "",

    [Parameter()]
    [string]$Channel = "",

    [Parameter()]
    [string]$Ref = "",

    [Parameter()]
    [switch]$Latest,

    [Parameter()]
    [string]$FrameworkVersion = "",

    [Parameter()]
    [string]$FrameworkPin = "",

    [Parameter()]
    [string]$FrameworkChannel = "",

    [Parameter()]
    [string]$FrameworkRef = "",

    [Parameter()]
    [switch]$FrameworkLatest,

    [Parameter()]
    [switch]$SkipDiagnostics,

    [Parameter()]
    [switch]$Clean,

    [Parameter()]
    [switch]$Reset
)

# PSScriptAnalyzer doesn't like how we use our params as globals; this calms
# it (same no-op-read trick as astral.sh/uv's install.ps1).
$null = $Source, $LisaRepo, $CliRepo, $Provider, $SkipDiagnostics, $Clean, $Reset,
    $Version, $Pin, $Channel, $Ref, $Latest, $FrameworkVersion, $FrameworkPin,
    $FrameworkChannel, $FrameworkRef, $FrameworkLatest

# -- Script state ------------------------------------------------------------

# A null MyCommand.Path means this code is being evaluated in the caller's
# session (irm | iex) - there is no script file on disk.
$script:ExecutedFromIex = ($null -eq $MyInvocation.MyCommand.Path)
$script:InstallerDir = if ($script:ExecutedFromIex) { "" } else { $PSScriptRoot }
$script:CliMode = "keep"
$script:CliValue = ""
$script:FrameworkMode = "keep"
$script:FrameworkValue = ""
$script:VersionHelper = ""
$script:VersionHelperTemporary = ""
$script:EnvTemplateTemporary = ""
$script:EnvTemplateUrl = "https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/.env.example"
$script:SessionTemplateUrl = "https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/config/session_context.md"
$script:SessionTemplateTemporary = ""

function Set-LisaVersionSelections {
    <#
    .SYNOPSIS
        Validate mutually exclusive version options before any clean or install.
    #>
    [CmdletBinding()]
    param()
    $cliChoices = @()
    if ($Version) { $cliChoices += @{ Mode = "version"; Value = $Version } }
    if ($Pin) { $cliChoices += @{ Mode = "pin"; Value = $Pin } }
    if ($Channel) { $cliChoices += @{ Mode = "channel"; Value = $Channel } }
    if ($Ref) { $cliChoices += @{ Mode = "ref"; Value = $Ref } }
    if ($Latest) { $cliChoices += @{ Mode = "latest"; Value = "" } }
    if ($cliChoices.Count -gt 1) { throw "Choose only one CLI version option" }
    if ($cliChoices.Count -eq 1) {
        $script:CliMode = $cliChoices[0].Mode
        $script:CliValue = $cliChoices[0].Value
    }

    $frameworkChoices = @()
    if ($FrameworkVersion) { $frameworkChoices += @{ Mode = "version"; Value = $FrameworkVersion } }
    if ($FrameworkPin) { $frameworkChoices += @{ Mode = "pin"; Value = $FrameworkPin } }
    if ($FrameworkChannel) { $frameworkChoices += @{ Mode = "channel"; Value = $FrameworkChannel } }
    if ($FrameworkRef) { $frameworkChoices += @{ Mode = "ref"; Value = $FrameworkRef } }
    if ($FrameworkLatest) { $frameworkChoices += @{ Mode = "latest"; Value = "" } }
    if ($frameworkChoices.Count -gt 1) { throw "Choose only one framework version option" }
    if ($frameworkChoices.Count -eq 1) {
        $script:FrameworkMode = $frameworkChoices[0].Mode
        $script:FrameworkValue = $frameworkChoices[0].Value
    }
    elseif ($Latest) {
        $script:FrameworkMode = "latest"
    }
}

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
    ollama       = @{ Label = "Ollama"; EnvVars = @("LISA_OLLAMA_HOST") }
    openai       = @{ Label = "OpenAI"; EnvVars = @("LISA_OPENAI_API_KEY") }
    azure_openai = @{ Label = "Azure OpenAI"; EnvVars = @("LISA_AZURE_OPENAI_ENDPOINT", "LISA_AZURE_OPENAI_API_KEY", "LISA_AZURE_OPENAI_API_VERSION") }
    anthropic    = @{ Label = "Anthropic"; EnvVars = @("LISA_ANTHROPIC_API_KEY", "LISA_ANTHROPIC_BASE_URL", "LISA_ANTHROPIC_VERSION") }
    google       = @{ Label = "Google Gemini"; EnvVars = @("LISA_GOOGLE_API_KEY", "LISA_GOOGLE_API_BASE_URL") }
    openrouter   = @{ Label = "OpenRouter"; EnvVars = @("LISA_OPENROUTER_API_KEY", "LISA_OPENROUTER_BASE_URL") }
    aws          = @{ Label = "AWS Bedrock"; EnvVars = @("LISA_AWS_ACCESS_KEY_ID", "LISA_AWS_SECRET_ACCESS_KEY", "LISA_AWS_REGION") }
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

function Initialize-LisaVersionHelper {
    <#
    .SYNOPSIS
        Locate the shared version helper or download it for a piped install.
    #>
    [CmdletBinding()]
    param()
    if ($script:InstallerDir) {
        $candidate = Join-Path $script:InstallerDir "hosts\cli\src\hosts\cli\update.py"
        if (Test-Path -LiteralPath $candidate) {
            $script:VersionHelper = $candidate
            return
        }
    }
    $script:VersionHelperTemporary = Join-Path ([System.IO.Path]::GetTempPath()) ("lisa-version-" + [guid]::NewGuid().ToString("N") + ".py")
    $url = "https://raw.githubusercontent.com/Lisa-Harness/lisa-cli/main/hosts/cli/src/hosts/cli/update.py"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $script:VersionHelperTemporary
    $script:VersionHelper = $script:VersionHelperTemporary
}

function Invoke-LisaVersionHelper {
    <#
    .SYNOPSIS
        Validate or apply version selections in an isolated Python environment.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][ValidateSet("validate", "install")][string]$Action)
    $arguments = @(
        "run", "--isolated", "--no-project", "--no-config", "--python", "3.12",
        "python", $script:VersionHelper, $Action, "--source", $Source,
        "--cli-mode", $script:CliMode, "--framework-mode", $script:FrameworkMode
    )
    if ($script:CliValue) { $arguments += @("--cli-value", $script:CliValue) }
    if ($script:FrameworkValue) { $arguments += @("--framework-value", $script:FrameworkValue) }
    if ($CliRepo) { $arguments += @("--cli-repo", $CliRepo) }
    if ($LisaRepo) { $arguments += @("--lisa-repo", $LisaRepo) }
    $exitCode = Invoke-Native -FilePath "uv" -ArgumentList $arguments
    if ($exitCode -ne 0) { throw "Version selection $Action failed (exit $exitCode)" }
}

function Install-LisaTool {
    <#
    .SYNOPSIS
        Install (or upgrade) the lisa tool from published, git, or local sources.
    .DESCRIPTION
        Use the CI-tested Python 3.12: newer system interpreters may lack
        native dependency wheels (notably google-re2). uv downloads it if needed.

        Git and local modes pass each framework package as an explicit
        requirement. The installable CLI workspace member is hosts\cli.
    #>
    [CmdletBinding()]
    param()
    Set-StrictMode -Version Latest

    if ($script:CliMode -ne "keep" -or $script:FrameworkMode -ne "keep") {
        Invoke-LisaVersionHelper -Action "install"
        $localBin = Join-Path $HOME ".local\bin"
        if ((Test-Path $localBin) -and ($env:PATH -notlike "*$localBin*")) {
            $env:PATH = "$localBin;$env:PATH"
        }
        return
    }

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

    if ($Source -eq "pypi") {
        $cliRequirement = "lisa-cli[prompt]"
        $withArgs = @()
    }
    elseif ($Source -eq "local") {
        if (-not $LisaRepo) { throw "-Source local requires -LisaRepo (or `$env:LISA_GIT_REPO)" }
        if (-not (Test-Path $LisaRepo)) { throw "framework repo not found: $LisaRepo" }
        if (-not $CliRepo) {
            $CliRepo = Join-Path (Split-Path $LisaRepo -Parent) "lisa-cli"
        }
        if (-not (Test-Path $CliRepo)) { throw "CLI repo not found: $CliRepo (pass -CliRepo)" }

        $cliRequirement = "lisa-cli[prompt] @ $(Join-Path $CliRepo "hosts\cli")"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
                "--with", "$_ @ $(Join-Path $LisaRepo $frameworkMembers[$_])"
            })
    }
    else {
        $cliRequirement = "lisa-cli[prompt] @ git+$cliRepoUrl#subdirectory=hosts/cli"
        $withArgs = @($frameworkMembers.Keys | ForEach-Object {
                "--with", "$_ @ git+$lisaRepoUrl#subdirectory=$($frameworkMembers[$_])"
            })
    }

    Write-Verbose "Installing lisa (uv tool, source: $Source)"
    $exitCode = Invoke-Native -FilePath "uv" -ArgumentList (@("tool", "install", "--python", "3.12", "--no-sources", $cliRequirement) + $withArgs)
    if ($exitCode -ne 0) {
        $exitCode = Invoke-Native -FilePath "uv" -ArgumentList (@("tool", "install", "--python", "3.12", "--no-cache", "--no-sources", "--force", "--reinstall", $cliRequirement) + $withArgs)
        if ($exitCode -ne 0) { throw "uv tool install failed (exit $exitCode)" }
    }

    # Make bare `lisa` resolvable for the rest of this script (uv installs
    # executables into ~/.local/bin, which may not be on PATH yet).
    $localBin = Join-Path $HOME ".local\bin"
    if ((Test-Path $localBin) -and ($env:PATH -notlike "*$localBin*")) {
        $env:PATH = "$localBin;$env:PATH"
    }
}

function Get-PacksVenvPath {
    <#
    .SYNOPSIS
        Resolve the packs venv the way runtime.paths.packs_venv_for does:
        LISA_PACKS_VENV wins, then <data dir>\venv - except for a profile dir
        (<root>\<name>), which resolves to the base data dir's venv because
        packs are shared across profiles. A relocated profiles root
        (LISA_PROFILES_DIR) still shares the base data dir's venv.
    #>
    param()
    Set-StrictMode -Version Latest

    $override = [Environment]::GetEnvironmentVariable("LISA_PACKS_VENV")
    if (-not [string]::IsNullOrWhiteSpace($override)) {
        # Match os.path.expanduser: a leading ~ means $HOME.
        if ($override -eq "~" -or $override.StartsWith("~\") -or $override.StartsWith("~/")) {
            return (Join-Path $HOME $override.TrimStart("~", "\", "/"))
        }
        return $override
    }

    $base = $script:LisaHome.TrimEnd("\", "/")
    $parent = Split-Path $base -Parent
    $profilesRoot = [Environment]::GetEnvironmentVariable("LISA_PROFILES_DIR")
    if (-not [string]::IsNullOrWhiteSpace($profilesRoot)) {
        # Match os.path.expanduser: a leading ~ means $HOME.
        if ($profilesRoot -eq "~" -or $profilesRoot.StartsWith("~\") -or $profilesRoot.StartsWith("~/")) {
            $profilesRoot = Join-Path $HOME $profilesRoot.TrimStart("~", "\", "/")
        }
        $profilesRoot = $profilesRoot.TrimEnd("\", "/")
        if ($parent -eq $profilesRoot) {
            # The profile lives in a relocated root, so the shared venv
            # stays in the base data dir.
            $dataDir = [Environment]::GetEnvironmentVariable("LISA_DATA_DIR")
            if ([string]::IsNullOrWhiteSpace($dataDir)) {
                $dataDir = Join-Path $HOME ".lisa"
            }
            $base = $dataDir.TrimEnd("\", "/")
        }
    }
    elseif ((Split-Path $parent -Leaf) -eq "profiles") {
        $base = Split-Path $parent -Parent
    }
    return (Join-Path $base "venv")
}

function Remove-LisaPath {
    <#
    .SYNOPSIS
        Delete one path, reporting it. A missing path is a no-op that returns
        $false so callers can report "nothing here".
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )
    Set-StrictMode -Version Latest

    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $Path) {
        Write-Warning "Could not fully remove $Path (in use?). Remove it manually."
    }
    else {
        Write-Host "  Removed $Path" -ForegroundColor Green
    }
    return $true
}

function Clear-LisaInstallState {
    <#
    .SYNOPSIS
        Remove what the install owns outside uv's receipts: the packs venv and
        the pack provenance lock. The CLI tool venv belongs to uv, so it is
        removed with `uv tool uninstall` rather than by deleting files.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Clears one install state as a single unit.')]
    param()
    Set-StrictMode -Version Latest

    $packsVenv = Get-PacksVenvPath
    Write-Host "  Removing packs venv: $packsVenv"
    if (-not (Remove-LisaPath -Path $packsVenv)) {
        Write-Host "  No packs venv at $packsVenv"
    }
    # Advisory lock the runtime keeps beside the packs venv.
    $null = Remove-LisaPath -Path (Join-Path (Split-Path $packsVenv -Parent) ".packs-venv.lock")

    # The json lock plus the sidecar .lock the runtime writes beside it.
    $packLock = Join-Path $script:LisaHome "state\installed-packs.json"
    Write-Host "  Removing pack lock: $packLock"
    if (-not (Remove-LisaPath -Path $packLock)) {
        Write-Host "  No pack lock at $packLock"
    }
    $null = Remove-LisaPath -Path "$packLock.lock"

    Write-Host "  Uninstalling lisa tool (uv)..."
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        # Seed the automatic variable so StrictMode can read it even when `uv`
        # resolves to a function (tests) rather than a native executable.
        $global:LASTEXITCODE = 0
        $null = & uv tool uninstall lisa-cli 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Removed the lisa tool install" -ForegroundColor Green
        }
        else {
            Write-Host "  No lisa tool install to remove"
        }
    }
    else {
        Write-Host "  uv is unavailable; skipping tool uninstall"
    }
}

function Reset-LisaSeededState {
    <#
    .SYNOPSIS
        Drop what the installer seeds - the model manifest (active + proposal)
        and the starter session context - so this run writes them again.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '',
        Justification = 'Resets one seeded state as a single unit.')]
    param()
    Set-StrictMode -Version Latest

    $targets = @(
        (Join-Path $script:LisaHome "config\models.yaml"),
        (Join-Path $script:LisaHome "state\models.proposed.yaml"),
        (Join-Path $script:LisaHome "config\session_context.md")
    )
    $removed = $false
    foreach ($target in $targets) {
        if (Remove-LisaPath -Path $target) { $removed = $true }
    }
    if (-not $removed) {
        Write-Host "  Nothing to reset in $script:LisaHome"
    }
}

function Get-LisaEnvTemplate {
    <#
    .SYNOPSIS
        Locate or download the canonical .env.example template.
    #>
    [CmdletBinding()]
    param()
    Set-StrictMode -Version Latest

    if ($script:InstallerDir) {
        $candidate = Join-Path $script:InstallerDir ".env.example"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $script:EnvTemplateTemporary = Join-Path ([System.IO.Path]::GetTempPath()) ("lisa-env-" + [guid]::NewGuid().ToString("N") + ".example")
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $script:EnvTemplateUrl -OutFile $script:EnvTemplateTemporary -ErrorAction Stop
        if (-not (Test-Path -LiteralPath $script:EnvTemplateTemporary)) {
            throw "download produced no template file"
        }
        if ((Get-Item -LiteralPath $script:EnvTemplateTemporary).Length -eq 0) {
            throw "downloaded template is empty"
        }
        return $script:EnvTemplateTemporary
    }
    catch {
        Remove-Item -LiteralPath $script:EnvTemplateTemporary -Force -ErrorAction SilentlyContinue
        $script:EnvTemplateTemporary = ""
        throw "Could not download $($script:EnvTemplateUrl): $($_.Exception.Message)"
    }
}

function Get-LisaEnvTemplateValue {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Key)
    Set-StrictMode -Version Latest

    $value = [Environment]::GetEnvironmentVariable($Key)
    if ([string]::IsNullOrEmpty($value) -and $Key.StartsWith("LISA_")) {
        $bareKey = $Key.Substring(5)
        $value = [Environment]::GetEnvironmentVariable($bareKey)
    }
    if ($null -eq $value -or [string]::IsNullOrEmpty($value)) { return $null }
    if ($value.Contains("`n") -or $value.Contains("`r")) { return $null }
    return $value
}

function Get-LisaEnvTemplateKey {
    [CmdletBinding()]
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Line)
    Set-StrictMode -Version Latest

    if ($Line -match '^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
        return $Matches[1]
    }
    if ($Line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
        return $Matches[1]
    }
    return $null
}

function Test-LisaEnvHasActiveKey {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$EnvFile,
        [Parameter(Mandatory)][string]$Key
    )
    Set-StrictMode -Version Latest

    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        if ($line -match '^\s*#?\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
            if ($Matches[1] -eq $Key) { return $true }
        }
    }
    return $false
}

function New-LisaEnvFile {
    <#
    .SYNOPSIS
        Copy .env.example to ~\.lisa\.env, fill values available in the
        environment, or merge missing keys into an existing file.
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

    if ((-not (Test-Path $lisaHome)) -and $PSCmdlet.ShouldProcess($lisaHome, "Create data directory")) {
        New-Item -ItemType Directory -Path $lisaHome -Force | Out-Null
    }

    try {
        $template = Get-LisaEnvTemplate
    }
    catch {
        if (Test-Path -LiteralPath $envFile) {
            Write-Host "  WARN: $($_.Exception.Message); existing $envFile preserved" -ForegroundColor Yellow
            return
        }
        throw
    }

    if (Test-Path -LiteralPath $envFile) {
        $additions = [System.Collections.Generic.List[string]]::new()
        foreach ($line in Get-Content -LiteralPath $template) {
            $key = Get-LisaEnvTemplateKey -Line $line
            if (-not $key -or (Test-LisaEnvHasActiveKey -EnvFile $envFile -Key $key)) {
                continue
            }
            $value = Get-LisaEnvTemplateValue -Key $key
            if ($null -ne $value) {
                $additions.Add("$key=$value")
            }
            elseif ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
                $additions.Add($line)
            }
            elseif ($line -match '^\s*#\s*[A-Za-z_][A-Za-z0-9_]*\s*=') {
                $additions.Add($line)
            }
        }
        if ($additions.Count -eq 0) {
            Write-Host "  $envFile already contains all template keys - preserved" -ForegroundColor Yellow
            return
        }
        if ($PSCmdlet.ShouldProcess($envFile, "Merge $($additions.Count) missing template key(s)")) {
            $tmpFile = "$envFile.tmp"
            Copy-Item -LiteralPath $envFile -Destination $tmpFile -Force
            Add-Content -LiteralPath $tmpFile -Value "", "# Added by Lisa Harness installer"
            foreach ($addition in $additions) {
                Add-Content -LiteralPath $tmpFile -Value $addition
            }
            Move-Item -LiteralPath $tmpFile -Destination $envFile -Force
            Write-Host "  Merged $($additions.Count) missing template key(s) into $envFile - existing values preserved" -ForegroundColor Green
        }
        return
    }

    if ($PSCmdlet.ShouldProcess($envFile, "Create .env from template")) {
        $tmpFile = "$envFile.tmp"
        $lines = [System.Collections.Generic.List[string]]::new()
        foreach ($line in Get-Content -LiteralPath $template) {
            $key = Get-LisaEnvTemplateKey -Line $line
            $value = if ($key) { Get-LisaEnvTemplateValue -Key $key } else { $null }
            if ($key -and $null -ne $value) {
                $lines.Add("$key=$value")
            }
            else {
                $lines.Add($line)
            }
        }
        Set-Content -LiteralPath $tmpFile -Value $lines -Encoding UTF8
        Move-Item -LiteralPath $tmpFile -Destination $envFile -Force
        Write-Host "  Created $envFile from $template" -ForegroundColor Green
    }
}

function Get-LisaSessionTemplate {
    <#
    .SYNOPSIS
        Locate or download the canonical starter session template.
    #>
    [CmdletBinding()]
    param()
    Set-StrictMode -Version Latest

    if ($script:InstallerDir) {
        $candidate = Join-Path $script:InstallerDir "config\session_context.md"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $script:SessionTemplateTemporary = Join-Path ([System.IO.Path]::GetTempPath()) ("lisa-session-" + [guid]::NewGuid().ToString("N") + ".md")
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $script:SessionTemplateUrl -OutFile $script:SessionTemplateTemporary -ErrorAction Stop
        if (-not (Test-Path -LiteralPath $script:SessionTemplateTemporary)) {
            throw "download produced no session template"
        }
        if ((Get-Item -LiteralPath $script:SessionTemplateTemporary).Length -eq 0) {
            throw "downloaded session template is empty"
        }
        return $script:SessionTemplateTemporary
    }
    catch {
        Remove-Item -LiteralPath $script:SessionTemplateTemporary -Force -ErrorAction SilentlyContinue
        $script:SessionTemplateTemporary = ""
        throw "Could not download $($script:SessionTemplateUrl): $($_.Exception.Message)"
    }
}

function New-LisaSessionContext {
    <#
    .SYNOPSIS
        Copy the canonical starter session. Existing user files are preserved.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'Installer UX is deliberately host-colored output.')]
    param()
    Set-StrictMode -Version Latest

    $configDir = Join-Path $script:LisaHome "config"
    $ctxFile = Join-Path $configDir "session_context.md"
    if (Test-Path -LiteralPath $ctxFile) {
        Write-Host "  $ctxFile already exists - preserved" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path -LiteralPath $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    $template = Get-LisaSessionTemplate
    if ($PSCmdlet.ShouldProcess($ctxFile, "Seed starter session context")) {
        # Preserve exact bytes and avoid a BOM on Windows PowerShell 5.1.
        $content = [System.IO.File]::ReadAllText($template)
        [System.IO.File]::WriteAllText($ctxFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host "  Created $ctxFile (starter: memory, web search, personas, markdown)" -ForegroundColor Green
    }
}


function Update-LisaModelManifest {
    <#
    .SYNOPSIS
        First-run model bootstrap: discover providers, promote the proposal,
        and require a nonempty active manifest. Existing active manifests are
        preserved and checked by final verification. Discovery/apply failures
        are fatal so a partial installation never reports success.
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
        $count = @([System.IO.File]::ReadAllLines($activeFile) | Where-Object { $_ -match "model_id:" } | Measure-Object).Count
        Write-Host "  Existing $activeFile ($count model(s)) - skipping model bootstrap" -ForegroundColor Yellow
        return
    }

    Write-Verbose "Discovering models from configured provider(s)"
    if (-not (Get-Command lisa -ErrorAction SilentlyContinue)) {
        throw "lisa is not on PATH; cannot discover models"
    }
    $exitCode = Invoke-Native -FilePath "lisa" -ArgumentList @("models", "discover", "--force")
    if ($exitCode -ne 0) {
        throw "Model discovery failed (exit $exitCode). Configure provider credentials/endpoints, then rerun: lisa models discover --force; lisa models apply"
    }
    if (-not (Test-Path -LiteralPath $proposalFile)) {
        throw "No models were discovered. Configure provider credentials/endpoints, then run: lisa models discover --force; lisa models apply"
    }

    Write-Verbose "Promoting discovered models to active manifest"
    $exitCode = Invoke-Native -FilePath "lisa" -ArgumentList @("models", "apply")
    if ($exitCode -ne 0) {
        throw "Model apply failed (exit $exitCode); fix the proposal and run: lisa models apply"
    }
    if (-not (Test-Path -LiteralPath $activeFile)) {
        throw "Model apply completed without creating $activeFile"
    }
    $count = @([System.IO.File]::ReadAllLines($activeFile) | Where-Object { $_ -match "model_id:" } | Measure-Object).Count
    if ($count -lt 1) {
        throw "Active model manifest $activeFile contains no models"
    }
    Write-Host "  OK: $activeFile ready ($count model(s))" -ForegroundColor Green
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
            $fillThese = $meta.EnvVars -join ', '
            Write-Host "       $($meta.Label): uncomment and set $fillThese as needed" -ForegroundColor Yellow
        }
        Write-Host "    2. Run: lisa models discover --force; lisa models apply" -ForegroundColor Yellow
        Write-Host "       (to refresh your model manifest)"
        Write-Host ""
    }

    Write-Host "  Quick start (from any directory):"
    Write-Host "    lisa prompt `"Let's get started, Lisa!`""
    Write-Host ""
    Write-Host "  The starter session auto-installs a few packs from GitHub"
    Write-Host "  (markdown, personas) on first use."
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
        Write-Host "    lisa models discover --force; lisa models apply"
        Write-Host ""
    }

    Write-Host "  Verify your setup:"
    Write-Host "    lisa verify"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Main installation
# ---------------------------------------------------------------------------

function Install-LisaHarness {
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
    Write-Host "  Lisa Harness Installer" -ForegroundColor Green
    Write-Host "  ------------------------"
    Write-Host ""

    Set-LisaVersionSelections
    Test-InstallEnvironment

    $total = 6
    if ($Clean) { $total++ }
    if ($Reset) { $total++ }
    $stepNo = 0

    $stepNo++
    Write-Step -Step $stepNo -Total $total -Message "Checking uv..."
    Install-UvIfMissing
    $uvVersion = (uv --version 2>$null) -replace '^uv\s+', ''
    Write-Host "  OK: uv $uvVersion" -ForegroundColor Green

    if ($script:CliMode -ne "keep" -or $script:FrameworkMode -ne "keep") {
        Initialize-LisaVersionHelper
        Invoke-LisaVersionHelper -Action "validate"
    }

    # Runs after uv exists: the tool uninstall step shells out to it.
    if ($Clean) {
        $stepNo++
        Write-Step -Step $stepNo -Total $total -Message "Cleaning prior install state..."
        Clear-LisaInstallState
    }
    if ($Reset) {
        $stepNo++
        Write-Step -Step $stepNo -Total $total -Message "Resetting seeded state..."
        Reset-LisaSeededState
    }

    $stepNo++
    Write-Step -Step $stepNo -Total $total -Message "Installing lisa CLI (uv tool, source: $Source)..."
    Install-LisaTool
    $lisaCmd = Get-Command lisa -ErrorAction SilentlyContinue
    if (-not $lisaCmd) {
        throw "lisa was installed but is not on PATH. Add $(Join-Path $HOME ".local\bin") to your PATH, then reopen your shell."
    }
    $lisaVersion = (& lisa --version 2>&1 | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) {
        throw "The installed lisa executable failed to launch (exit $LASTEXITCODE): $lisaVersion"
    }
    Write-Host "  OK: $lisaVersion" -ForegroundColor Green

    $stepNo++
    Write-Step -Step $stepNo -Total $total -Message "Configuring providers in $($script:LisaHome)..."
    New-LisaEnvFile

    $stepNo++
    Write-Step -Step $stepNo -Total $total -Message "Seeding starter session context..."
    New-LisaSessionContext

    $stepNo++
    Write-Step -Step $stepNo -Total $total -Message "Bootstrapping model manifest (discover + apply)..."
    Update-LisaModelManifest

    $stepNo++
    if (-not $SkipDiagnostics) {
        Write-Step -Step $stepNo -Total $total -Message "Running diagnostics..."
        Write-Host ""
        $verifyExit = Invoke-Native -FilePath "lisa" -ArgumentList @("verify")
        if ($verifyExit -ne 0) {
            throw "lisa verify failed (exit $verifyExit); resolve the reported configuration errors and rerun"
        }
        Write-Host ""
    }
    else {
        Write-Step -Step $stepNo -Total $total -Message "Done."
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
        Install-LisaHarness
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
    finally {
        if ($script:VersionHelperTemporary) {
            Remove-Item -LiteralPath $script:VersionHelperTemporary -Force -ErrorAction Continue
        }
        if ($script:EnvTemplateTemporary) {
            Remove-Item -LiteralPath $script:EnvTemplateTemporary -Force -ErrorAction Continue
        }
        if ($script:SessionTemplateTemporary) {
            Remove-Item -LiteralPath $script:SessionTemplateTemporary -Force -ErrorAction Continue
        }
    }
}
