param(
    [string]$Config = "configs\overnight.json",
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$supervisorPath = Join-Path $workspacePath 'scripts\supervisor.py'
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    [System.IO.Path]::GetFullPath($Config)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspacePath $Config))
}

foreach ($requiredPath in @($pythonPath, $supervisorPath, $configPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}

& $pythonPath $supervisorPath --config $configPath --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Supervisor configuration validation failed.' }
if ($DryRun) { return }

$stopPath = Join-Path $workspacePath 'state\STOP'
if (Test-Path -LiteralPath $stopPath) {
    throw "Stop request exists at $stopPath. Review it and remove it deliberately before restarting."
}

$logDirectory = Join-Path $workspacePath 'logs\supervisor'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$launchStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $logDirectory "launcher_$launchStamp.stdout.log"
$stderrPath = Join-Path $logDirectory "launcher_$launchStamp.stderr.log"

# Windows file paths cannot contain a double quote. Explicit quoting preserves
# spaces when Start-Process joins its ArgumentList into a command line.
$arguments = @('-u', ('"' + $supervisorPath + '"'), '--config', ('"' + $configPath + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
    -WorkingDirectory $workspacePath -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Start-Sleep -Seconds 2
$process.Refresh()
if ($process.HasExited) {
    Get-Content -LiteralPath $stderrPath -Tail 30
    throw "Supervisor exited during startup with code $($process.ExitCode)."
}

[pscustomobject]@{
    SupervisorPid = $process.Id
    StartedUtc = $process.StartTime.ToUniversalTime().ToString('o')
    State = (Join-Path $workspacePath 'state\run_state.json')
    Stdout = $stdoutPath
    Stderr = $stderrPath
} | ConvertTo-Json
