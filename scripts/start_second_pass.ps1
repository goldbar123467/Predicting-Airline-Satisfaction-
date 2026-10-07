$ErrorActionPreference = 'Stop'
$campaignRoot = Split-Path -Parent $PSScriptRoot
$campaignPython = Join-Path $campaignRoot '.venv\Scripts\python.exe'
$campaignConfig = Join-Path $campaignRoot 'configs\second_pass.json'
if (Test-Path -LiteralPath (Join-Path $campaignRoot 'state\second_pass\STOP')) {
    throw 'Second-pass STOP exists. Inspect state and resolve the stop before restarting.'
}
& $campaignPython (Join-Path $PSScriptRoot 'supervisor.py') --config $campaignConfig --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Second-pass configuration validation failed.' }
$campaignLogs = Join-Path $campaignRoot 'logs\second_pass\supervisor'
New-Item -ItemType Directory -Path $campaignLogs -Force | Out-Null
$campaignLabel = Get-Date -Format 'yyyyMMddTHHmmss'
$campaignProcess = Start-Process -FilePath $campaignPython `
    -ArgumentList @('-u', 'scripts/supervisor.py', '--config', 'configs/second_pass.json') `
    -WorkingDirectory $campaignRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $campaignLogs "launcher-$campaignLabel.stdout.log") `
    -RedirectStandardError (Join-Path $campaignLogs "launcher-$campaignLabel.stderr.log")
[pscustomobject]@{
    Pid = $campaignProcess.Id
    CreatedUtc = $campaignProcess.StartTime.ToUniversalTime().ToString('o')
    State = (Join-Path $campaignRoot 'state\second_pass\run_state.json')
    Deadline = '2026-10-02T17:00:00Z'
} | ConvertTo-Json
