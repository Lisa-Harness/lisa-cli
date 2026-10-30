$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $repoRoot "install.ps1")

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Invoke-Native {
    param([string]$FilePath, [string[]]$ArgumentList)
    $script:calls++
    Assert-True ($FilePath -eq "uv") "Expected uv"
    Assert-True (($ArgumentList[0..3] -join " ") -eq "tool install --python 3.12") "Expected Python 3.12"
    $remaining = @($ArgumentList | Select-Object -Skip 4)
    if ($script:calls -eq 2) {
        # Retry path: --no-cache --no-sources --force --reinstall.
        Assert-True (($remaining[0..3] -join " ") -eq "--no-cache --no-sources --force --reinstall") "Expected --no-cache --no-sources --force --reinstall on retry"
        $remaining = @($remaining | Select-Object -Skip 4)
    }
    else {
        Assert-True ($remaining[0] -eq "--no-sources") "Expected --no-sources"
        $remaining = @($remaining | Select-Object -Skip 1)
    }
    Assert-True ($remaining[0] -eq $script:expectedRequirement) "Incorrect CLI source or missing prompt extra"
    Assert-True ($remaining.Count -eq 17) "Expected all eight framework requirements"
    for ($i = 1; $i -lt $remaining.Count; $i += 2) {
        Assert-True ($remaining[$i] -eq "--with" -and $remaining[$i + 1] -like "* @ *") "Invalid framework requirement"
    }
    if ($script:retry -and $script:calls -eq 1) { return 1 }
    return 0
}

$LisaRepo = $repoRoot
$CliRepo = $repoRoot
$env:UV_PYTHON = "3.15"
foreach ($Source in @("git", "local")) {
    $script:expectedRequirement = if ($Source -eq "git") {
        "lisa-cli[prompt] @ git+https://github.com/Lisa-Harness/lisa-cli.git#subdirectory=hosts/cli"
    }
    else {
        "lisa-cli[prompt] @ $(Join-Path $CliRepo 'hosts\cli')"
    }
    foreach ($script:retry in @($false, $true)) {
        $script:calls = 0
        Install-LisaTool
        $expectedCalls = if ($script:retry) { 2 } else { 1 }
        Assert-True ($script:calls -eq $expectedCalls) "Unexpected install call count"
    }
}
Write-Output "PowerShell installer tests passed"

# -Clean / -Reset: run against a throwaway data dir with uv stubbed so no real
# install or uninstall is ever touched.
$sandbox = Join-Path ([System.IO.Path]::GetTempPath()) ("lisa-installer-test-" + [guid]::NewGuid().ToString("N"))
$savedDataDir = [Environment]::GetEnvironmentVariable("LISA_DATA_DIR")
$savedPacksVenv = [Environment]::GetEnvironmentVariable("LISA_PACKS_VENV")
try {
    $dataDir = Join-Path $sandbox ".lisa"
    [Environment]::SetEnvironmentVariable("LISA_DATA_DIR", $dataDir)
    [Environment]::SetEnvironmentVariable("LISA_PACKS_VENV", $null)
    $script:LisaHome = Get-LisaHome

    foreach ($dir in @("venv", "state", "config")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $dataDir $dir) | Out-Null
    }
    foreach ($seeded in @(
            "venv\pyvenv.cfg",
            ".packs-venv.lock",
            "state\installed-packs.json",
            "state\installed-packs.json.lock",
            "state\models.proposed.yaml",
            "config\models.yaml",
            "config\session_context.md"
        )) {
        Set-Content -LiteralPath (Join-Path $dataDir $seeded) -Value ""
    }

    # uv is stubbed by replacing the function the installer resolves at call time.
    function global:uv { $script:uvCall = ($args -join " "); return 0 }

    Assert-True ((Get-PacksVenvPath) -eq (Join-Path $dataDir "venv")) "Expected the data dir's venv"

    Clear-LisaInstallState
    Assert-True ($script:uvCall -eq "tool uninstall lisa-cli") "Expected uv tool uninstall"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "venv"))) "Packs venv survived -Clean"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir ".packs-venv.lock"))) "Packs venv lock survived -Clean"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "state\installed-packs.json"))) "Pack lock survived -Clean"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "state\installed-packs.json.lock"))) "Pack lock sidecar survived -Clean"
    Assert-True (Test-Path -LiteralPath (Join-Path $dataDir "config\models.yaml")) "-Clean must not touch the model manifest"

    Reset-LisaSeededState
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "config\models.yaml"))) "Model manifest survived -Reset"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "state\models.proposed.yaml"))) "Model proposal survived -Reset"
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "config\session_context.md"))) "Session context survived -Reset"

    # A uv failure other than "not installed" (exit 2) must not abort -Clean.
    New-Item -ItemType Directory -Force -Path (Join-Path $dataDir "venv"), (Join-Path $dataDir "state") | Out-Null
    Set-Content -LiteralPath (Join-Path $dataDir "state\installed-packs.json") -Value "{}"
    function global:uv { $script:uvCall = ($args -join " "); $global:LASTEXITCODE = 1; return 1 }
    Clear-LisaInstallState
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dataDir "state\installed-packs.json"))) "Pack lock survived a failed uninstall"

    function global:uv { $script:uvCall = ($args -join " "); $global:LASTEXITCODE = 0; return 0 }

    # LISA_PACKS_VENV is an explicit override and must win over the data dir.
    [Environment]::SetEnvironmentVariable("LISA_PACKS_VENV", (Join-Path $sandbox "custom-venv"))
    Assert-True ((Get-PacksVenvPath) -eq (Join-Path $sandbox "custom-venv")) "LISA_PACKS_VENV override ignored"
    [Environment]::SetEnvironmentVariable("LISA_PACKS_VENV", $null)

    # A profile shares the base data dir's packs venv.
    $script:LisaHome = Join-Path (Join-Path $dataDir "profiles") "work"
    Assert-True (
        (Get-PacksVenvPath) -eq (Join-Path $dataDir "venv")
    ) "Profile must share the base packs venv"

    Write-Output "PowerShell clean/reset tests passed"
}
finally {
    [Environment]::SetEnvironmentVariable("LISA_DATA_DIR", $savedDataDir)
    [Environment]::SetEnvironmentVariable("LISA_PACKS_VENV", $savedPacksVenv)
    Remove-Item -LiteralPath $sandbox -Recurse -Force -ErrorAction SilentlyContinue
}
