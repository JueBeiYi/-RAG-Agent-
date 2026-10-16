param([string]$EvidenceDir='reports/deployment/local-20261015')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskEvidence=[IO.Path]::GetFullPath((Join-Path $taskRoot $EvidenceDir))
New-Item -ItemType Directory -Force -Path $taskEvidence | Out-Null
$taskPidRecord=Get-Content -LiteralPath (Join-Path $taskRoot 'data/runtime/server-processes.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskPorts=@([int]$taskPidRecord.apiPort,[int]$taskPidRecord.uiPort,11435)
$taskListeners=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -In $taskPorts | Select-Object LocalAddress,LocalPort,OwningProcess)
$taskProcessRows=@($taskListeners | ForEach-Object {
    $taskProc=Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
    [ordered]@{pid=$taskProc.ProcessId;parent_pid=$taskProc.ParentProcessId;executable=$taskProc.ExecutablePath;command_line=$taskProc.CommandLine;created_at=$taskProc.CreationDate.ToString('o');port=$_.LocalPort;address=$_.LocalAddress}
})
$taskHealth=Invoke-RestMethod -Uri "http://127.0.0.1:$($taskPidRecord.apiPort)/health" -TimeoutSec 15
$taskModelPs=Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/ps' -TimeoutSec 15
$taskHealth | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $taskEvidence 'health.json') -Encoding utf8
$taskModelPs | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $taskEvidence 'ollama-model-ps.json') -Encoding utf8
$taskProcessRows | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $taskEvidence 'processes-and-listeners.json') -Encoding utf8
foreach ($taskLogName in @('api.stdout.log','api.stderr.log','ui.stdout.log','ui.stderr.log','ollama-11435.stdout.log','ollama-11435.stderr.log')) {
    $taskLog=Join-Path $taskRoot "data/runtime/$taskLogName"
    if (Test-Path -LiteralPath $taskLog) { Copy-Item -LiteralPath $taskLog -Destination (Join-Path $taskEvidence $taskLogName) -Force }
}
$taskAcceptance=Get-Content -LiteralPath (Join-Path $taskEvidence 'acceptance/acceptance.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskUniqueChecks=@($taskAcceptance.checks | Select-Object -ExpandProperty name -Unique)
$taskRecord=[ordered]@{
    captured_at=(Get-Date).ToString('o');timezone='Asia/Shanghai';status=$taskAcceptance.status;profile='standard';deployment='local-loopback';
    start_command='.\scripts\start.ps1 -Profile standard -ApiPort 8000 -UiPort 8501 -ModelPort 11435';
    ui_url="http://127.0.0.1:$($taskPidRecord.uiPort)";api_url="http://127.0.0.1:$($taskPidRecord.apiPort)";model_url='http://127.0.0.1:11435';
    all_owned_listeners_are_loopback=(@($taskListeners | Where-Object LocalAddress -NotIn @('127.0.0.1','::1')).Count -eq 0);
    api_alive=$taskHealth.alive;api_ready=$taskHealth.ready;demo=$taskHealth.demo;
    actual_model=$taskAcceptance.answer.metrics.model;embedding_backend=$taskHealth.retrieval.embedding.backend;
    embedding_model=$taskHealth.retrieval.embedding.model;reranker_model=$taskHealth.retrieval.reranker.model;vector_backend=$taskHealth.retrieval.vectors.backend;
    unique_check_count=$taskUniqueChecks.Count;poll_inclusive_passed_count=$taskAcceptance.passed_check_count;failed_check_count=$taskAcceptance.failed_check_count;
    document=$taskAcceptance.document;generation_metrics=$taskAcceptance.answer.metrics;
    quality_review_warning_count=@($taskAcceptance.answer.audit.needs_review).Count;
    boundary='Actual local startup and core workflow acceptance; answer semantic correctness and full benchmark scores are separate. No container or restart acceptance is claimed.';
    user_requested_scope='Docker validation cancelled by user; actual 127.0.0.1 startup and workflow records requested instead.'
}
if (-not $taskRecord.api_ready -or $taskRecord.demo -or -not $taskRecord.all_owned_listeners_are_loopback -or $taskRecord.status -ne 'passed') { throw 'The actual local runtime did not meet the required acceptance state.' }
$taskRecord | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $taskEvidence 'runtime-record.json') -Encoding utf8
$taskRecord | ConvertTo-Json -Depth 3
