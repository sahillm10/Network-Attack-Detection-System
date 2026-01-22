# Network Attack System - One-Click Startup Script

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Network Attack System Startup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Get the script's directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot "network-attack-backend"
$frontendDir = Join-Path $projectRoot "network-attack-frontend"

# ===================== BACKEND SETUP =====================
Write-Host "[1/4] Setting up Backend..." -ForegroundColor Yellow
$venvPath = Join-Path $backendDir "venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    Set-Location $backendDir
    python -m venv venv
}

Write-Host "Activating virtual environment..."
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing backend dependencies..."
pip install -q -r requirements.txt

# Check for .env file
$envFile = Join-Path $backendDir ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    @"
GEMINI_API_KEY=your_api_key_here
"@ | Out-File $envFile
    Write-Host ".env file created. Please add your GEMINI_API_KEY to: $envFile" -ForegroundColor Magenta
}

# ===================== FRONTEND SETUP =====================
Write-Host ""
Write-Host "[2/4] Setting up Frontend..." -ForegroundColor Yellow
Set-Location $frontendDir

if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    npm install --legacy-peer-deps -q
}

# ===================== START SERVICES =====================
Write-Host ""
Write-Host "[3/4] Starting Backend..." -ForegroundColor Green
Set-Location $backendDir
$backendCmd = "& `"$venvPath\Scripts\Activate.ps1`"; cd `"$backendDir`"; python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Write-Host "[4/4] Starting Frontend..." -ForegroundColor Green
Set-Location $frontendDir
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start"

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Both services are starting!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in either window to stop services" -ForegroundColor Yellow
