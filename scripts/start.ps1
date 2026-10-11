param([ValidateSet('standard','demo')][string]$Profile = 'standard', [int]$ApiPort = 8000, [int]$UiPort = 8501,
      [ValidateRange(1024,65535)][int]$ModelPort = 11435)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/setup.ps1 first.' }
foreach ($taskPort in @($ApiPort,$UiPort)) {
    if (Get-NetTCPConnection -LocalPort $taskPort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $taskPort is already in use. Stop this project's previous processes or choose -ApiPort / -UiPort."
    }
}
$taskLogDir = Join-Path $taskRoot 'data\runtime'
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$env:RAG_PROFILE = $Profile
$env:RAG_API_URL = "http://127.0.0.1:$ApiPort"
$env:PYTHONUTF8 = '1'
$taskWorkspaceFile = Join-Path $taskLogDir "workspace-$Profile.json"
# Distinguish a caller-provided token from the token this launcher assigned in
# the same PowerShell session. Switching profiles must restore that profile's
# saved workspace instead of accidentally inheriting the previous profile.
$taskManagedWorkspace = $env:RAG_WORKSPACE_TOKEN -and ($env:RAG_WORKSPACE_TOKEN -eq $env:RAG_MANAGED_WORKSPACE_TOKEN)
if (-not $env:RAG_WORKSPACE_TOKEN -or $taskManagedWorkspace) {
    if (Test-Path -LiteralPath $taskWorkspaceFile) {
        $taskWorkspace = Get-Content -LiteralPath $taskWorkspaceFile -Raw -Encoding utf8 | ConvertFrom-Json
        $env:RAG_WORKSPACE_TOKEN = $taskWorkspace.workspace_token
    } else {
        $taskCourseWorkspace = Join-Path $taskLogDir 'course-workspace.json'
        $taskToken = if ($Profile -eq 'standard' -and (Test-Path -LiteralPath $taskCourseWorkspace)) {
            (Get-Content -LiteralPath $taskCourseWorkspace -Raw -Encoding utf8 | ConvertFrom-Json).workspace_token
        } else { [guid]::NewGuid().ToString() }
        @{workspace_token=$taskToken; profile=$Profile} | ConvertTo-Json | Set-Content -LiteralPath $taskWorkspaceFile -Encoding utf8
        $env:RAG_WORKSPACE_TOKEN = $taskToken
    }
    $env:RAG_MANAGED_WORKSPACE_TOKEN = $env:RAG_WORKSPACE_TOKEN
}
if (Test-Path -LiteralPath (Join-Path $taskRoot 'data/models/hub')) {
    $env:HF_HUB_CACHE = Join-Path $taskRoot 'data/models/hub'
}
if ($Profile -eq 'standard') {
    if (-not $env:RAG_DATA_DIR -or $env:RAG_DATA_DIR -in @((Join-Path $taskLogDir 'demo'), (Join-Path $taskLogDir 'standard'))) {
        $env:RAG_DATA_DIR = Join-Path $taskLogDir 'standard'
    }
    if (-not $env:RAG_CONFIG) { $env:RAG_CONFIG = Join-Path $taskRoot 'config.standard.yaml' }
    $taskConfiguredBackend = & $taskPython -c "import sys,yaml; print((yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {}).get('llm_backend','ollama'))" $env:RAG_CONFIG
    $taskBackendChoice = if ($env:RAG_LLM_BACKEND) { $env:RAG_LLM_BACKEND } else { $taskConfiguredBackend }
    $taskLlmManifest = Join-Path $taskRoot 'data/models/local-llm.json'
    if ($taskBackendChoice -eq 'ollama' -and (Test-Path -LiteralPath $taskLlmManifest)) {
        $taskLlm = Get-Content -LiteralPath $taskLlmManifest -Raw -Encoding utf8 | ConvertFrom-Json
        if ($taskLlm.status -in @('imported','verified','ready') -and -not $env:RAG_LLM_MODEL) { $env:RAG_LLM_MODEL = $taskLlm.ollama_model }
    }
    $taskPortableManifest = Join-Path $taskRoot 'data/tools/ollama-manifest.json'
    $taskManagedModel = -not $env:RAG_OLLAMA_URL -or $env:RAG_OLLAMA_URL -eq $env:RAG_MANAGED_OLLAMA_URL
    if ($taskBackendChoice -eq 'ollama' -and $taskManagedModel -and (Test-Path -LiteralPath $taskPortableManifest)) {
        # Keep the project cache/server separate from a user's system Ollama.
        $env:RAG_OLLAMA_URL = "http://127.0.0.1:$ModelPort"
        $env:RAG_MANAGED_OLLAMA_URL = $env:RAG_OLLAMA_URL
        $taskPortable = Get-Content -LiteralPath $taskPortableManifest -Raw -Encoding utf8 | ConvertFrom-Json
        $taskModelExecutable = (Resolve-Path -LiteralPath $taskPortable.executable).Path
        $taskAllowedTools = [IO.Path]::GetFullPath((Join-Path $taskRoot 'data/tools')) + [IO.Path]::DirectorySeparatorChar
        if (-not $taskModelExecutable.StartsWith($taskAllowedTools, [StringComparison]::OrdinalIgnoreCase)) { throw 'Portable model executable is outside the project tools directory.' }
        $taskModelListener = Get-NetTCPConnection -LocalPort $ModelPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($taskModelListener) {
            $taskExistingModel = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskModelListener.OwningProcess)"
            if ($taskExistingModel.ExecutablePath -ne $taskModelExecutable) { throw "Model port $ModelPort belongs to another application. Choose -ModelPort or explicitly set RAG_OLLAMA_URL." }
        } else {
        $env:OLLAMA_HOST = "127.0.0.1:$ModelPort"
        $env:OLLAMA_MODELS = Join-Path $taskRoot 'data/models/ollama'
        $env:OLLAMA_NUM_PARALLEL = '1'
        $env:OLLAMA_MAX_LOADED_MODELS = '1'
        $env:OLLAMA_NO_CLOUD = '1'
        $taskModelServer = Start-Process -FilePath $taskModelExecutable -ArgumentList @('serve') -WindowStyle Hidden -PassThru -WorkingDirectory (Split-Path -Parent $taskModelExecutable) -RedirectStandardOutput (Join-Path $taskLogDir "ollama-$ModelPort.stdout.log") -RedirectStandardError (Join-Path $taskLogDir "ollama-$ModelPort.stderr.log")
        $taskModelServer.Id | Set-Content -LiteralPath (Join-Path $taskLogDir "ollama-$ModelPort.pid")
        for ($taskModelAttempt=0; $taskModelAttempt -lt 20; $taskModelAttempt++) {
            try { $null=Invoke-RestMethod -Uri "$env:RAG_OLLAMA_URL/api/version" -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 250 }
        }
        }
    }
} elseif (-not $env:RAG_DATA_DIR -or $env:RAG_DATA_DIR -in @((Join-Path $taskLogDir 'demo'), (Join-Path $taskLogDir 'standard'))) {
    $env:RAG_DATA_DIR = Join-Path $taskLogDir 'demo'
}
$taskBackend = Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','src.api.main:app','--host','127.0.0.1','--port',"$ApiPort") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogDir 'api.stdout.log') -RedirectStandardError (Join-Path $taskLogDir 'api.stderr.log')
try {
    $taskApiReady = $false
    for ($taskAttempt=0; $taskAttempt -lt 30; $taskAttempt++) {
        if ($taskBackend.HasExited) { throw 'Backend startup failed. See data/runtime/api.stderr.log.' }
        try {
            $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
            if ($taskHealth.alive) { $taskApiReady=$true; break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $taskApiReady) { throw 'Backend did not become available. See data/runtime/api.stderr.log.' }
} catch {
    if (-not $taskBackend.HasExited) { Stop-Process -Id $taskBackend.Id }
    throw
}
$taskFrontend = Start-Process -FilePath $taskPython -ArgumentList @('-m','streamlit','run','src/frontend/app.py','--server.address','127.0.0.1','--server.port',"$UiPort",'--server.headless','true') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogDir 'ui.stdout.log') -RedirectStandardError (Join-Path $taskLogDir 'ui.stderr.log')
@{ apiPid=$taskBackend.Id; uiPid=$taskFrontend.Id; python=$taskPython; root=$taskRoot; apiPort=$ApiPort; uiPort=$UiPort;
   apiStarted=$taskBackend.StartTime.ToUniversalTime().ToString('o'); uiStarted=$taskFrontend.StartTime.ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLogDir 'server-processes.json') -Encoding utf8
Write-Host "UI: http://127.0.0.1:$UiPort   API docs: http://127.0.0.1:$ApiPort/docs"
Write-Host "Profile: $Profile. Logs: $taskLogDir. Stop with scripts/stop.ps1."
Write-Host 'The local workspace is saved privately and restored across browser refreshes. Different profiles keep separate data.'
if (-not $taskHealth.ready) { Write-Host 'The app is available; model dependencies are not yet ready. Check the sidebar service status.' }
