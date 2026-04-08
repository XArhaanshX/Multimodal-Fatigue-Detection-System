param(
    [string]$PythonExe = "",
    [int]$Port = 8766,
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$certPath = Join-Path $repoRoot "certs\\mobile-alert-local-cert.pem"
$keyPath = Join-Path $repoRoot "certs\\mobile-alert-local-key.pem"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidates = @(
        "$env:LOCALAPPDATA\\Programs\\Python\\Python313\\python.exe",
        "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe",
        "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $PythonExe = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    throw "Python executable not found. Pass -PythonExe 'C:\\path\\to\\python.exe'."
}

if (-not (Test-Path $certPath) -or -not (Test-Path $keyPath)) {
    & (Join-Path $PSScriptRoot "generate_local_https_cert.ps1") -PythonExe $PythonExe
}

$args = @("backend\\alert\\mobile_alert_server.py", "--https", "--port", $Port.ToString())
if ($Reload.IsPresent) {
    $args += "--reload"
} else {
    $args += "--no-reload"
}

Push-Location $repoRoot
try {
    & $PythonExe @args
} finally {
    Pop-Location
}
