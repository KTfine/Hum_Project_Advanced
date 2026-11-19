param(
    [string]$Python = '',
    [int]$CameraDevice
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $Python) {
    $localPython = Join-Path $projectRoot '.venv-client\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { 'python' }
}
$clientLog = Join-Path $projectRoot ('experiments\runtime\live-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.jsonl')
$env:PYTHONIOENCODING = 'utf-8'
$clientArguments = @('--log-file', $clientLog)
if ($PSBoundParameters.ContainsKey('CameraDevice')) { $clientArguments += @('--camera-device', $CameraDevice) }
& $Python (Join-Path $projectRoot 'realtime_av_test.py') @clientArguments
exit $LASTEXITCODE
