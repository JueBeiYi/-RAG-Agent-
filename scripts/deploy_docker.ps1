param(
    [ValidateSet('standard','demo')][string]$Profile='standard',
    [switch]$Gpu,
    [ValidateRange(1,65535)][int]$ApiPort=8000,
    [ValidateRange(1,65535)][int]$FrontendPort=8501,
    [ValidatePattern('^[a-z0-9][a-z0-9_-]*$')][string]$ProjectName='rag-course',
    [string]$HfHubPath,
    [string]$OllamaModelsPath,
    [string]$LlmModel='qwen2.5:7b-instruct-q4_K_M',
    [switch]$SkipModelPreparation
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if ($ApiPort -eq $FrontendPort) { throw 'API and frontend require different host ports.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is unavailable. Install and start Docker Desktop, then rerun this deployment command.' }
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is not running.' }
$taskComposeArgs=@('compose','--project-directory',$taskRoot,'--project-name',$ProjectName,'-f',(Join-Path $taskRoot 'docker/docker-compose.yml'))
if ($Gpu) { $taskComposeArgs+=@('-f',(Join-Path $taskRoot 'docker/docker-compose.gpu.yml')) }
if ($Profile -eq 'standard') { $taskComposeArgs+=@('--profile','models') }
function Invoke-TaskCompose([string[]]$TaskArguments) {
    & docker @taskComposeArgs @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw 'Docker deployment step failed; the previous output contains the actual error.' }
}
function Resolve-TaskModelDirectory([string]$TaskPath) {
    $taskResolved=(Resolve-Path -LiteralPath $TaskPath).Path
    if (-not (Test-Path -LiteralPath $taskResolved -PathType Container)) { throw "Model directory is unavailable: $TaskPath" }
    return $taskResolved.Replace('\','/')
}
$taskEnvironment=@{
    RAG_PROFILE=$Profile
    RAG_IMAGE_TAG=$Profile
    RAG_INSTALL_MODELS=if ($Profile -eq 'demo') { 'false' } else { 'true' }
    RAG_API_PORT=[string]$ApiPort
    RAG_FRONTEND_PORT=[string]$FrontendPort
    RAG_LLM_MODEL=$LlmModel
}
if ($HfHubPath) { $taskEnvironment.RAG_HF_HUB_SOURCE=Resolve-TaskModelDirectory $HfHubPath }
if ($OllamaModelsPath) { $taskEnvironment.RAG_OLLAMA_MODELS_SOURCE=Resolve-TaskModelDirectory $OllamaModelsPath }
$taskPreviousEnvironment=@{}
try {
    foreach ($taskName in $taskEnvironment.Keys) {
        $taskPreviousEnvironment[$taskName]=[Environment]::GetEnvironmentVariable($taskName,'Process')
        [Environment]::SetEnvironmentVariable($taskName,$taskEnvironment[$taskName],'Process')
    }
    Invoke-TaskCompose -TaskArguments @('config','--quiet')
    # Build exactly one application image; frontend and prepare reuse that tag.
    Invoke-TaskCompose -TaskArguments @('build','api')
    if ($Profile -eq 'standard') {
        Invoke-TaskCompose -TaskArguments @('up','-d','ollama')
        if (-not $SkipModelPreparation) {
            Invoke-TaskCompose -TaskArguments @('--profile','setup','run','--rm','prepare')
            Invoke-TaskCompose -TaskArguments @('exec','ollama','ollama','pull',$LlmModel)
        }
    }
    Invoke-TaskCompose -TaskArguments @('up','-d','--wait','--wait-timeout','240','api','frontend')
    Invoke-TaskCompose -TaskArguments @('ps')
    Write-Host "Application: http://127.0.0.1:$FrontendPort  API: http://127.0.0.1:$ApiPort  Profile: $Profile"
    Write-Host 'Uploaded documents persist in the project-specific Docker volume. Model preparation is explicit; demo does not start Ollama.'
} finally {
    foreach ($taskName in $taskPreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskName,$taskPreviousEnvironment[$taskName],'Process')
    }
}
