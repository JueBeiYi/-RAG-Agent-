param([string]$EvidenceDir='', [switch]$SkipInstall)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if (-not $EvidenceDir) { $EvidenceDir=Join-Path $taskRoot ('reports/deployment/docker/host-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
$taskEvidence=[IO.Path]::GetFullPath($EvidenceDir)
$taskDownloads=Join-Path $taskRoot 'data/runtime/docker-host-installers'
New-Item -ItemType Directory -Force -Path $taskEvidence,$taskDownloads | Out-Null
$taskRecord=[ordered]@{ started_at=(Get-Date).ToString('o'); status='preparing'; reboot_performed=$false; commands=@(); sources=@('https://docs.docker.com/desktop/setup/install/windows-install/','https://learn.microsoft.com/en-us/windows/wsl/install'); installations=@() }
function Save-TaskRecord { $taskRecord | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $taskEvidence 'host-setup.json') -Encoding utf8 }
function Get-TaskFeatures {
    @('VirtualMachinePlatform','Microsoft-Windows-Subsystem-Linux') | ForEach-Object {
        $taskFeature=Get-WindowsOptionalFeature -Online -FeatureName $_
        [ordered]@{ feature=$taskFeature.FeatureName; state=$taskFeature.State.ToString(); restart_required=$taskFeature.RestartRequired.ToString() }
    }
}
function Install-TaskProcess([string]$Name,[string]$File,[string[]]$Arguments) {
    $taskRecord.commands+=@([ordered]@{at=(Get-Date).ToString('o');executable=$File;arguments=$Arguments})
    Save-TaskRecord
    $taskProcess=Start-Process -FilePath $File -ArgumentList $Arguments -PassThru -Wait -WindowStyle Hidden
    $taskRecord.installations+=@([ordered]@{name=$Name;exit_code=$taskProcess.ExitCode;finished_at=(Get-Date).ToString('o')})
    Save-TaskRecord
    if ($taskProcess.ExitCode -notin @(0,3010)) { throw "$Name installer returned $($taskProcess.ExitCode); see the recorded installer logs." }
}
Start-Transcript -Path (Join-Path $taskEvidence 'host-setup-transcript.txt') -Force | Out-Null
try {
    $taskIdentity=[Security.Principal.WindowsIdentity]::GetCurrent()
    $taskPrincipal=[Security.Principal.WindowsPrincipal]::new($taskIdentity)
    $taskOS=Get-CimInstance Win32_OperatingSystem
    $taskComputer=Get-CimInstance Win32_ComputerSystem
    $taskCPU=Get-CimInstance Win32_Processor | Select-Object -First 1
    $taskRecord.before=[ordered]@{os=$taskOS.Caption;version=$taskOS.Version;build=$taskOS.BuildNumber;administrator=$taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);hypervisor_present=$taskComputer.HypervisorPresent;ram_gb=[math]::Round($taskComputer.TotalPhysicalMemory/1GB,2);virtualization_firmware_enabled=$taskCPU.VirtualizationFirmwareEnabled;features=@(Get-TaskFeatures);docker_cli_found=[bool](Get-Command docker -ErrorAction SilentlyContinue)}
    Save-TaskRecord
    if (-not $taskRecord.before.administrator) { throw 'Run host preparation from an administrator PowerShell.' }
    if (-not $SkipInstall) {
        foreach ($taskName in @('VirtualMachinePlatform','Microsoft-Windows-Subsystem-Linux')) {
            $taskRecord.commands+=@([ordered]@{at=(Get-Date).ToString('o');executable='Enable-WindowsOptionalFeature';arguments=@('-Online','-FeatureName',$taskName,'-All','-NoRestart')})
            Save-TaskRecord
            $taskFeatureResult=Enable-WindowsOptionalFeature -Online -FeatureName $taskName -All -NoRestart
            Write-Host "$taskName result: $($taskFeatureResult.RestartNeeded)"
            $taskRecord.installations+=@([ordered]@{name=$taskName;restart_needed=$taskFeatureResult.RestartNeeded;at=(Get-Date).ToString('o')})
            Save-TaskRecord
        }
        $taskWsl=Invoke-RestMethod -Uri 'https://api.github.com/repos/microsoft/WSL/releases/latest' -Headers @{'User-Agent'='NAU-RAG-Agent-Deployment'} -TimeoutSec 60
        $taskAsset=$taskWsl.assets | Where-Object name -Match 'x64.msi$' | Select-Object -First 1
        if (-not $taskAsset -or $taskAsset.browser_download_url -notlike 'https://github.com/microsoft/WSL/releases/*') { throw 'Official WSL x64 MSI asset was not found.' }
        $taskWslFile=Join-Path $taskDownloads $taskAsset.name
        Write-Host "Downloading official WSL $($taskWsl.tag_name)"
        if (-not (Test-Path -LiteralPath $taskWslFile)) { Invoke-WebRequest -Uri $taskAsset.browser_download_url -OutFile $taskWslFile -TimeoutSec 900 }
        $taskWslHash=(Get-FileHash -LiteralPath $taskWslFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($taskAsset.digest -and $taskAsset.digest -ne "sha256:$taskWslHash") { throw 'WSL release SHA256 mismatch; installer was not executed.' }
        $taskWslSig=Get-AuthenticodeSignature -LiteralPath $taskWslFile
        if ($taskWslSig.Status -ne 'Valid' -or $taskWslSig.SignerCertificate.Subject -notmatch 'Microsoft') { throw 'WSL Microsoft signature verification failed.' }
        $taskRecord.wsl_installer=[ordered]@{url=$taskAsset.browser_download_url;version=$taskWsl.tag_name;sha256=$taskWslHash;signature=$taskWslSig.Status.ToString();publisher=$taskWslSig.SignerCertificate.Subject}
        Save-TaskRecord
        Install-TaskProcess 'WSL MSI' 'msiexec.exe' @('/i',('"'+$taskWslFile+'"'),'/qn','/norestart','/l*v',('"'+(Join-Path $taskEvidence 'wsl-msi.log')+'"'))
        $taskDockerUrl='https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe'
        $taskDockerFile=Join-Path $taskDownloads 'Docker Desktop Installer.exe'
        Write-Host 'Downloading official Docker Desktop installer'
        if (-not (Test-Path -LiteralPath $taskDockerFile)) { Invoke-WebRequest -Uri $taskDockerUrl -OutFile $taskDockerFile -TimeoutSec 1200 }
        $taskDockerSig=Get-AuthenticodeSignature -LiteralPath $taskDockerFile
        if ($taskDockerSig.Status -ne 'Valid' -or $taskDockerSig.SignerCertificate.Subject -notmatch 'Docker') { throw 'Docker publisher signature verification failed.' }
        $taskRecord.docker_installer=[ordered]@{url=$taskDockerUrl;sha256=(Get-FileHash -LiteralPath $taskDockerFile -Algorithm SHA256).Hash.ToLowerInvariant();signature=$taskDockerSig.Status.ToString();publisher=$taskDockerSig.SignerCertificate.Subject;version=(Get-Item -LiteralPath $taskDockerFile).VersionInfo.ProductVersion}
        Save-TaskRecord
        $taskDataRoot=Join-Path $taskRoot 'data/runtime/docker-desktop-wsl'
        Install-TaskProcess 'Docker Desktop' $taskDockerFile @('install','--quiet','--backend=wsl-2','--no-windows-containers','--accept-license',('--wsl-default-data-root="'+$taskDataRoot+'"'))
    }
    $taskRecord.after=[ordered]@{features=@(Get-TaskFeatures);hypervisor_present=(Get-CimInstance Win32_ComputerSystem).HypervisorPresent;docker_executable_exists=(Test-Path 'C:/Program Files/Docker/Docker/resources/bin/docker.exe')}
    $taskRecord.status=if (-not $taskRecord.after.hypervisor_present) { 'host_prepared_restart_required_container_acceptance_pending' } else { 'host_prepared_engine_validation_pending' }
    $taskRecord.finished_at=(Get-Date).ToString('o')
    Save-TaskRecord
    Write-Host "Evidence: $taskEvidence"
    Write-Host "Status: $($taskRecord.status)"
} catch {
    $taskRecord.status='host_preparation_failed'
    $taskRecord.error=$_.Exception.Message
    $taskRecord.finished_at=(Get-Date).ToString('o')
    Save-TaskRecord
    throw
} finally { Stop-Transcript | Out-Null }
