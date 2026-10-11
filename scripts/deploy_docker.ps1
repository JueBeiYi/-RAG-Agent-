param([ValidateSet('standard','demo')][string]$Profile='standard', [switch]$Gpu)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is unavailable. Install and start Docker Desktop, then rerun this deployment command.' }
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is not running.' }
$taskComposeArgs=@('compose','-f',(Join-Path $taskRoot 'docker/docker-compose.yml'))
if ($Gpu) { $taskComposeArgs+=@('-f',(Join-Path $taskRoot 'docker/docker-compose.gpu.yml')) }
$env:RAG_PROFILE=$Profile
function Invoke-TaskCompose([string[]]$TaskArguments) {
    & docker @taskComposeArgs @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw 'Docker deployment step failed; the previous output contains the actual error.' }
}
Invoke-TaskCompose -TaskArguments @('config','--quiet')
if ($Profile -eq 'standard') {
    Invoke-TaskCompose -TaskArguments @('up','-d','--build','ollama')
    # Dependencies and fixed identifiers are recorded by prepare_models.
    Invoke-TaskCompose -TaskArguments @('--profile','setup','run','--rm','prepare')
    Invoke-TaskCompose -TaskArguments @('exec','ollama','ollama','pull','qwen2.5:7b-instruct-q4_K_M')
}
Invoke-TaskCompose -TaskArguments @('up','-d','--build','api','frontend')
Write-Host "Application: http://127.0.0.1:8501  Profile: $Profile"
Write-Host 'First standard deployment downloads model weights. Uploaded documents persist in Docker volumes.'
