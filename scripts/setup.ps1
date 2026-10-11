param([string]$Python = 'python', [switch]$Models)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    & $Python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,14), 'Please use Python 3.10-3.13 (3.12 recommended)'"
    if ($LASTEXITCODE -ne 0) { throw 'Python version unsupported. Pass -Python with the absolute path to Python 3.12.' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    $taskRequirements = if ($Models) { 'requirements-models.txt' } else { 'requirements.txt' }
    & $taskPython -m pip install -r $taskRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    Write-Host 'Dependencies ready. Standard mode additionally requires prepare_models.py and a running Ollama.'
} finally { Pop-Location }
