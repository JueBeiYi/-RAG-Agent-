[CmdletBinding()]
param(
    [string]$CompilerPath
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

function Get-CompilerKind {
    param([string]$Path)
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    if ($name -ieq 'xelatex') { return 'xelatex' }
    if ($name -ieq 'tectonic') { return 'tectonic' }
    throw "不支持的编译器文件名：$name。请指定 xelatex.exe 或 tectonic.exe。"
}

$selectedCompiler = $null
$compilerKind = $null

if ($CompilerPath) {
    if (-not (Test-Path -LiteralPath $CompilerPath -PathType Leaf)) {
        throw "指定的编译器不存在：$CompilerPath"
    }
    $selectedCompiler = (Resolve-Path -LiteralPath $CompilerPath).ProviderPath
    $compilerKind = Get-CompilerKind -Path $selectedCompiler
}
else {
    $xeCommand = Get-Command xelatex -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $tectonicCommand = Get-Command tectonic -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($xeCommand) {
        $selectedCompiler = $xeCommand.Source
        $compilerKind = 'xelatex'
    }
    elseif ($tectonicCommand) {
        $selectedCompiler = $tectonicCommand.Source
        $compilerKind = 'tectonic'
    }
    elseif ($env:CODEX_TECTONIC_PATH -and (Test-Path -LiteralPath $env:CODEX_TECTONIC_PATH -PathType Leaf)) {
        $selectedCompiler = (Resolve-Path -LiteralPath $env:CODEX_TECTONIC_PATH).ProviderPath
        $compilerKind = 'tectonic'
    }
    else {
        throw '未找到编译器。请安装包含中文宏包的 TeX Live/MiKTeX，或通过 -CompilerPath 指定便携 tectonic.exe；也可设置 CODEX_TECTONIC_PATH。'
    }
}

$sourcePath = Join-Path $PSScriptRoot 'main.tex'
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "未找到报告源文件：$sourcePath"
}

Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "使用编译器：$selectedCompiler"
    if ($compilerKind -eq 'xelatex') {
        foreach ($pass in 1..2) {
            Write-Host "XeLaTeX 第 $pass 遍编译"
            & $selectedCompiler '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' 'main.tex'
            if ($LASTEXITCODE -ne 0) {
                throw "XeLaTeX 第 $pass 遍编译失败，退出码：$LASTEXITCODE。请查看 main.log。"
            }
        }
    }
    else {
        & $selectedCompiler '--keep-logs' '--keep-intermediates' 'main.tex'
        if ($LASTEXITCODE -ne 0) {
            throw "Tectonic 编译失败，退出码：$LASTEXITCODE。请查看终端输出及 main.log。"
        }
    }

    $pdfPath = Join-Path $PSScriptRoot 'main.pdf'
    if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
        throw '编译器已退出，但未找到 main.pdf，不能确认编译成功。'
    }
    if ((Get-Item -LiteralPath $pdfPath).Length -le 0) {
        throw '生成的 main.pdf 为空，不能确认编译成功。'
    }
    Write-Host "编译成功：$pdfPath"
}
finally {
    Pop-Location
}
