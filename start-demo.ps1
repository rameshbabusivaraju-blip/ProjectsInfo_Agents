<#
.SYNOPSIS
  Starts the whole ProjectPulse demo on this computer with one command.

.DESCRIPTION
  Stops any old copy of the API (port 8000) and the web app (port 5080), starts both in their own
  windows, waits until the web app answers, and opens it in the browser. The API key is read from
  PROJECTPULSE_API_KEY in .env, so the web app and the API always use the same key.

  Run from the repo root:      .\start-demo.ps1
  Stop everything again:       .\start-demo.ps1 -Stop
#>
param([switch]$Stop)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$apiPort = 8000
$webPort = 5080

# Stops whatever is listening on the two demo ports (old API or web app windows' processes).
function Stop-Demo {
    foreach ($port in $apiPort, $webPort) {
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    }
}

Stop-Demo
if ($Stop) {
    Write-Host "Demo stopped."
    return
}

# Read the API key from .env so both apps use the same value.
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) { throw ".env was not found in $root" }
$keyLine = Select-String -Path $envFile -Pattern '^\s*PROJECTPULSE_API_KEY\s*=\s*(.+)$' | Select-Object -First 1
if (-not $keyLine) { throw "PROJECTPULSE_API_KEY is not set in .env" }
$apiKey = $keyLine.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")

# Window 1: the API. Uses the pp_env virtual environment when it exists.
$activate = Join-Path $root "pp_env\Scripts\Activate.ps1"
$activateStep = ""
if (Test-Path $activate) { $activateStep = "& '$activate'; " }
$apiCommand = "Set-Location '$root'; $activateStep" + "uvicorn app.main:app --port $apiPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCommand

# Window 2: the web app, with the API key passed as an environment variable.
$webFolder = Join-Path $root "web\ProjectPulse.Web"
$webCommand = "`$env:ProjectPulseApi__ApiKey = '$apiKey'; Set-Location '$webFolder'; dotnet run --urls http://localhost:$webPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $webCommand

# Wait until the web app answers (the first build can take a little while), then open it.
Write-Host "Starting... the first run can take about a minute."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-WebRequest "http://localhost:$webPort" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    }
    catch { Start-Sleep -Seconds 2 }
}
if ($ready) {
    Start-Process "http://localhost:$webPort"
    Write-Host "Ready: http://localhost:$webPort   (stop with: .\start-demo.ps1 -Stop)"
}
else {
    Write-Host "The web app did not answer in time. Check the two new windows for errors."
}
