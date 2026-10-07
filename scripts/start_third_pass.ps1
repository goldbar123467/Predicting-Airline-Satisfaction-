param(
    [ValidatePattern('^third_pass(?:_batch[0-9]{2})?$')]
    [string]$CampaignId = 'third_pass'
)
$ErrorActionPreference = 'Stop'
$campaignRoot = Split-Path -Parent $PSScriptRoot
$campaignPython = Join-Path $campaignRoot '.venv\Scripts\python.exe'
$campaignConfig = Join-Path $campaignRoot "configs\$CampaignId.json"
if (Test-Path -LiteralPath (Join-Path $campaignRoot "state\$CampaignId\STOP")) {
    throw 'Third-pass STOP exists. Inspect state and resolve the stop before restarting.'
}
$campaignSettings = Get-Content -LiteralPath $campaignConfig -Raw | ConvertFrom-Json
if ($campaignSettings.campaign -ne 'third_pass') {
    throw 'The third-pass launcher requires campaign=third_pass.'
}
$configuredCampaignId = $campaignSettings.campaign_id
if ($null -eq $configuredCampaignId) { $configuredCampaignId = 'third_pass' }
if ($configuredCampaignId -ne $CampaignId) {
    throw 'The config campaign_id must match the requested launcher CampaignId.'
}
& $campaignPython (Join-Path $PSScriptRoot 'supervisor.py') --config $campaignConfig --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Third-pass configuration validation failed.' }
$campaignLogs = Join-Path $campaignRoot "logs\$CampaignId\supervisor"
New-Item -ItemType Directory -Path $campaignLogs -Force | Out-Null
$campaignLabel = Get-Date -Format 'yyyyMMddTHHmmssfff'
$campaignProcess = Start-Process -FilePath $campaignPython `
    -ArgumentList @('-u', 'scripts/supervisor.py', '--config', "configs/$CampaignId.json") `
    -WorkingDirectory $campaignRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $campaignLogs "launcher-$campaignLabel.stdout.log") `
    -RedirectStandardError (Join-Path $campaignLogs "launcher-$campaignLabel.stderr.log")
[pscustomobject]@{
    Pid = $campaignProcess.Id
    CreatedUtc = $campaignProcess.StartTime.ToUniversalTime().ToString('o')
    CampaignId = $CampaignId
    State = (Join-Path $campaignRoot "state\$CampaignId\run_state.json")
    Deadline = $campaignSettings.deadline_utc
} | ConvertTo-Json
