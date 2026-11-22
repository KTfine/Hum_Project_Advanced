param(
    [string]$Python = '',
    [int]$CameraDevice,
    [string]$PersonaFile,
    [switch]$Microphone,
    [switch]$NoTts,
    [switch]$NoMicrophone,
    [string]$MicrophoneDevice
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
if ($NoTts) { $clientArguments += '--no-tts' }
if ($PSBoundParameters.ContainsKey('CameraDevice')) { $clientArguments += @('--camera-device', $CameraDevice) }
if ($PSBoundParameters.ContainsKey('PersonaFile')) { $clientArguments += @('--character-persona-file', $PersonaFile) }
if ($NoMicrophone) {
    if ($Microphone) { throw 'Choose only one of -Microphone and -NoMicrophone.' }
    $clientArguments += '--no-microphone'
} elseif ($PSBoundParameters.ContainsKey('Microphone')) {
    if ($Microphone) { $clientArguments += '--microphone' } else { $clientArguments += '--no-microphone' }
}
if ($PSBoundParameters.ContainsKey('MicrophoneDevice')) {
    $clientArguments += @('--microphone-device', $MicrophoneDevice)
}
& $Python (Join-Path $projectRoot 'realtime_av_test.py') @clientArguments
exit $LASTEXITCODE
