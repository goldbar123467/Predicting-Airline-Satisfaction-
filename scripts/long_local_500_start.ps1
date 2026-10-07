param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$longRoot = Split-Path -Parent $PSScriptRoot
$longPython = Join-Path $longRoot '.venv\Scripts\python.exe'
$longWaiter = Join-Path $PSScriptRoot 'long_local_500_wait.py'
& $longPython $longWaiter --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Long-local registration validation failed.' }
if ($CheckOnly) { return }
foreach ($longStop in @('state\long_local_500\STOP', 'state\third_pass_batch07\STOP')) {
    if (Test-Path -LiteralPath (Join-Path $longRoot $longStop)) { throw 'Long-local STOP exists; inspect it before starting.' }
}
$longLogs = Join-Path $longRoot 'logs\long_local_500'
New-Item -ItemType Directory -Path $longLogs -Force | Out-Null
$longTag = Get-Date -Format 'yyyyMMddTHHmmssfff'
$longProcess = Start-Process -FilePath $longPython `
    -ArgumentList @('-u', 'scripts/long_local_500_wait.py') `
    -WorkingDirectory $longRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $longLogs "waiter-$longTag.stdout.log") `
    -RedirectStandardError (Join-Path $longLogs "waiter-$longTag.stderr.log")
[pscustomobject]@{ Pid = $longProcess.Id; CreatedUtc = $longProcess.StartTime.ToUniversalTime().ToString('o');
    State = (Join-Path $longRoot 'state\long_local_500\waiter_state.json'); TrainingNotBeforeUtc = '2026-10-02T17:05:00Z'
} | ConvertTo-Json
