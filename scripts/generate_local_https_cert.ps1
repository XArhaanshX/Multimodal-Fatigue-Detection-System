param(
    [string]$PythonExe = "",
    [string]$CertsDir = (Join-Path $PSScriptRoot "..\\certs")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

& $PythonExe (Join-Path $PSScriptRoot "generate_local_https_cert.py") --certs-dir $CertsDir
