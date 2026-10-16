param(
    [ValidateSet('standard','demo')][string]$Profile='standard',
    [ValidateRange(1,65535)][int]$ApiPort=8015,
    [ValidateRange(1,65535)][int]$FrontendPort=8515,
    [string]$EvidenceDir=''
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if (-not $EvidenceDir) { $EvidenceDir=Join-Path $taskRoot ('reports/deployment/docker/run-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
$taskEvidence=[IO.Path]::GetFullPath($EvidenceDir)
New-Item -ItemType Directory -Force -Path $taskEvidence | Out-Null
$taskRecord=[ordered]@{started_at=(Get-Date).ToString('o');status='preflight';profile=$Profile;api_port=$ApiPort;frontend_port=$FrontendPort;project='nau-rag-acceptance';reboot_performed=$false}
function Save-TaskDeployment { $taskRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskEvidence 'deployment.json') -Encoding utf8 }
Save-TaskDeployment
$taskDocker=Get-Command docker -ErrorAction SilentlyContinue
if (-not $taskDocker) {
    $taskDockerPath='C:/Program Files/Docker/Docker/resources/bin'
    if (Test-Path -LiteralPath (Join-Path $taskDockerPath 'docker.exe')) { $env:PATH=$taskDockerPath+';'+$env:PATH; $taskDocker=Get-Command docker }
}
if (-not $taskDocker) { throw 'Install Docker Desktop first; see scripts/prepare_docker_host.ps1.' }
if (-not (Get-CimInstance Win32_ComputerSystem).HypervisorPresent) {
    $taskRecord.status='restart_required_no_containers_started'
    $taskRecord.reason='Windows virtual machine components have not become active in the current boot.'
    Save-TaskDeployment
    Write-Host 'Save work and restart Windows. Then rerun this same command; no containers have been started.'
    exit 3010
}
$taskPython=Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython='python' }
$taskPrevious=@{}
Start-Transcript -Path (Join-Path $taskEvidence 'deployment-transcript.txt') -Force | Out-Null
try {
    & $taskDocker.Source info --format '{{.ServerVersion}}' 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Start-Process -FilePath 'C:/Program Files/Docker/Docker/Docker Desktop.exe' -WindowStyle Hidden
        $taskDeadline=(Get-Date).AddMinutes(3)
        do {
            Start-Sleep -Seconds 5
            & $taskDocker.Source info --format '{{.ServerVersion}}' 2>$null | Out-Null
            $taskEngineReady=$LASTEXITCODE -eq 0
        } while (-not $taskEngineReady -and (Get-Date) -lt $taskDeadline)
        if (-not $taskEngineReady) { throw 'Docker Engine did not become ready. Open Docker Desktop to inspect its startup state.' }
    }
    $taskHub=Join-Path $taskRoot 'data/models/hub'
    $taskOllama=Join-Path $taskRoot 'data/models/ollama'
    $taskHasHub=(Test-Path -LiteralPath (Join-Path $taskHub 'models--BAAI--bge-m3')) -and (Test-Path -LiteralPath (Join-Path $taskHub 'models--BAAI--bge-reranker-base'))
    $taskHasLlm=Test-Path -LiteralPath (Join-Path $taskOllama 'manifests-v2/ollama.com/library/research-qwen/7b')
    $taskLlm=if ($taskHasLlm) { 'research-qwen:7b' } else { 'qwen2.5:7b-instruct-q4_K_M' }
    $taskEnv=@{RAG_PROFILE=$Profile;RAG_IMAGE_TAG=$Profile;RAG_INSTALL_MODELS=if ($Profile -eq 'standard') {'true'} else {'false'};RAG_API_PORT=[string]$ApiPort;RAG_FRONTEND_PORT=[string]$FrontendPort;RAG_LLM_MODEL=$taskLlm}
    if ($taskHasHub) { $taskEnv.RAG_HF_HUB_SOURCE=$taskHub.Replace('\','/') }
    if ($taskHasLlm) { $taskEnv.RAG_OLLAMA_MODELS_SOURCE=$taskOllama.Replace('\','/') }
    foreach ($taskName in $taskEnv.Keys) {
        $taskPrevious[$taskName]=[Environment]::GetEnvironmentVariable($taskName,'Process')
        [Environment]::SetEnvironmentVariable($taskName,$taskEnv[$taskName],'Process')
    }
    $taskDeploy=@{Profile=$Profile;ApiPort=$ApiPort;FrontendPort=$FrontendPort;ProjectName='nau-rag-acceptance';LlmModel=$taskLlm}
    if ($taskHasHub) { $taskDeploy.HfHubPath=$taskHub }
    if ($taskHasLlm) { $taskDeploy.OllamaModelsPath=$taskOllama }
    if ($taskHasHub -and $taskHasLlm) { $taskDeploy.SkipModelPreparation=$true }
    $taskRecord.status='building_and_starting'
    $taskRecord.reuse_existing_hf_cache=$taskHasHub
    $taskRecord.reuse_existing_ollama_models=$taskHasLlm
    $taskRecord.llm_model=$taskLlm
    Save-TaskDeployment
    & (Join-Path $PSScriptRoot 'deploy_docker.ps1') @taskDeploy
    $taskRecord.status='running_acceptance'
    Save-TaskDeployment
    $taskVerify=@('-X','utf8',(Join-Path $PSScriptRoot 'verify_docker_deployment.py'),'--docker',$taskDocker.Source,'--project','nau-rag-acceptance','--compose',(Join-Path $taskRoot 'docker/docker-compose.yml'),'--profile',$Profile,'--api',"http://127.0.0.1:$ApiPort",'--ui',"http://127.0.0.1:$FrontendPort",'--evidence-dir',(Join-Path $taskEvidence 'acceptance'),'--restart')
    $taskPaper=Join-Path $taskRoot 'data/corpus/cvt_2103.15808v1.pdf'
    if (Test-Path -LiteralPath $taskPaper) { $taskVerify+=@('--document',$taskPaper) }
    & $taskPython @taskVerify
    if ($LASTEXITCODE -ne 0) { throw 'Actual container acceptance failed; inspect acceptance/acceptance.json and logs.' }
    $taskRecord.status=if ($Profile -eq 'standard') {'standard_container_acceptance_passed'} else {'demo_control_flow_passed'}
    $taskRecord.finished_at=(Get-Date).ToString('o')
    Save-TaskDeployment
    Write-Host "Recorded evidence: $taskEvidence"
} catch {
    $taskRecord.status='failed'
    $taskRecord.error=$_.Exception.Message
    $taskRecord.finished_at=(Get-Date).ToString('o')
    Save-TaskDeployment
    throw
} finally {
    foreach ($taskName in $taskPrevious.Keys) { [Environment]::SetEnvironmentVariable($taskName,$taskPrevious[$taskName],'Process') }
    Stop-Transcript | Out-Null
}
