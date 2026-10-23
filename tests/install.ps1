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
    Assert-True (($ArgumentList[0..4] -join " ") -eq "tool install --python 3.12 --no-sources") "Expected Python 3.12"
    $remaining = @($ArgumentList | Select-Object -Skip 5)
    if ($script:calls -eq 2) {
        Assert-True ($remaining[0] -eq "--reinstall") "Expected reinstall on retry"
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
