$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPidFile = Join-Path $taskRoot 'data\runtime\server-processes.json'
if (-not (Test-Path -LiteralPath $taskPidFile)) { Write-Host 'No recorded project processes.'; exit }
$taskRecord = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
foreach ($taskProcessId in @($taskRecord.apiPid,$taskRecord.uiPid)) {
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId"
    if ($taskProcess -and $taskProcess.ExecutablePath -eq $taskRecord.python -and $taskProcess.CommandLine -match '(uvicorn src.api.main:app|streamlit run src/frontend/app.py)') {
        # Windows venv launchers may own a base-interpreter child. Restrict
        # shutdown to directly descended Python processes running this module.
        $taskChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId=$taskProcessId"
        foreach ($taskChild in $taskChildren) {
            if ($taskChild.Name -eq 'python.exe' -and $taskChild.CommandLine -match '(uvicorn src.api.main:app|streamlit run src/frontend/app.py)') {
                Stop-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $taskProcessId
    } elseif ($taskProcess) { Write-Warning "PID $taskProcessId does not match this project; skipped." }
}
Remove-Item -LiteralPath $taskPidFile
