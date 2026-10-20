# AIR216D - ML-Agents Training Launcher
param (
    [string]$RunId = "RollerBall_Run1",
    [switch]$Resume,
    [switch]$Force
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   AIR216D: Unity ML-Agents Training Runner (PPO)        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$condaPython = "C:\Users\musac\miniconda3\envs\mlagents\python.exe"
$mlagentsLearn = "C:\Users\musac\miniconda3\envs\mlagents\Scripts\mlagents-learn.exe"
$configFile = "$PSScriptRoot\rollerball_config.yaml"

if (-not (Test-Path $mlagentsLearn)) {
    Write-Host "`n[!] ERROR: mlagents-learn not found at: $mlagentsLearn" -ForegroundColor Red
    Write-Host "Please ensure the mlagents conda environment is installed properly."
    Write-Host "`nPress Enter to exit..."
    Read-Host
    exit 1
}

if (-not (Test-Path $configFile)) {
    Write-Host "`n[!] ERROR: Config file not found: $configFile" -ForegroundColor Red
    Write-Host "`nPress Enter to exit..."
    Read-Host
    exit 1
}

Write-Host "`n[+] Starting ML-Agents training session: $RunId" -ForegroundColor Green
Write-Host "[+] Config: $configFile" -ForegroundColor Gray
Write-Host "[+] When prompted, press Play in the Unity Editor to begin training!`n" -ForegroundColor Yellow

$arguments = @("$configFile", "--run-id=$RunId", "--force")

if ($Resume) {
    # If resuming, remove force and add resume
    $arguments = @("$configFile", "--run-id=$RunId", "--resume")
}

try {
    # Call executable with individual arguments
    & $mlagentsLearn @arguments
}
catch {
    Write-Host "`n[!] Error during training: $_" -ForegroundColor Red
}
finally {
    Write-Host "`nTraining session stopped or finished." -ForegroundColor Cyan
    Write-Host "Press Enter to exit..."
    Read-Host
}
