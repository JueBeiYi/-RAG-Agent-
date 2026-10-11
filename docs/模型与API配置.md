# 模型与API配置

项目支持本地Ollama和兼容Chat Completions格式的云端接口。课程原文3.2明确要求可本地离线运行，并允许无GPU时使用CPU或云端API；因此无需API Key也能使用真实本地大模型。

## 当前本地模型

本项目缓存Qwen2.5 7B Instruct Q4权重，本地标签为`research-qwen:7b`。项目便携Ollama由`start.ps1`启动，默认独立使用`http://127.0.0.1:11435`，与系统安装的Ollama分开。检索使用BGE-M3，重排使用BGE-reranker-base。权重和运行数据位于被Git忽略的`data/models`与`data/runtime`。

```powershell
.\scripts\start.ps1 -Profile standard
# 自定义本项目模型服务端口：
.\scripts\start.ps1 -Profile standard -ModelPort 11436
```

侧栏“API已连接”表示科研助理后端8000端口可访问；模型状态单独检查Ollama与模型标签。`/health`会区分应用在线、模型已准备、模型尚未加载和失败状态。

## 使用已有Ollama

显式设置服务URL和实际模型标签，然后启动。显式URL优先于项目便携服务器，启动脚本不会终止用户的其他Ollama进程。

```powershell
$env:RAG_OLLAMA_URL='http://127.0.0.1:11434'
$env:RAG_LLM_MODEL='实际已安装的模型标签'
$env:RAG_LLM_BACKEND='ollama'
$env:RAG_OFFLINE='true'
.\scripts\start.ps1 -Profile standard
```

## 配置云端大模型API

配置在后端读取，当前页面的“服务地址”用于连接科研助理后端。云端参数由环境变量覆盖YAML，`.env.example`是字段示例，不会自动加载。

| 参数 | 内容 |
| --- | --- |
| `RAG_LLM_BACKEND` | `cloud` |
| `RAG_LLM_MODEL` | 服务商实际支持的模型ID |
| `RAG_CLOUD_URL` | 基础URL，例如`https://your-provider.example/v1`，不要附加`/chat/completions` |
| `RAG_CLOUD_API_KEY` | 服务商签发的API Key，通过本机终端输入 |
| `RAG_OFFLINE` | `false`，显式允许将问题与检索上下文发送至配置服务 |

在同一个PowerShell终端执行以下命令，将示例URL和模型ID换成自己的服务参数。

```powershell
.\scripts\stop.ps1
$env:RAG_LLM_BACKEND='cloud'
$env:RAG_LLM_MODEL='服务商模型ID'
$env:RAG_CLOUD_URL='https://your-provider.example/v1'
$env:RAG_OFFLINE='false'
$taskApiSecret=Read-Host '输入API Key' -AsSecureString
$env:RAG_CLOUD_API_KEY=([System.Net.NetworkCredential]::new('', $taskApiSecret)).Password
.\scripts\start.ps1 -Profile standard
```

这些环境变量作用于本终端启动的进程。Key不会出现在前端配置导出、公开健康状态或评测配置中。云端健康状态只说明字段配置完整，真实调用成功需要由所配置的服务验证。

## 恢复本地模式

```powershell
.\scripts\stop.ps1
Remove-Item Env:RAG_LLM_BACKEND,Env:RAG_LLM_MODEL,Env:RAG_CLOUD_URL,Env:RAG_CLOUD_API_KEY,Env:RAG_OFFLINE -ErrorAction SilentlyContinue
# 若曾显式连接其他Ollama，再清除此覆盖，让启动器使用项目服务器。
Remove-Item Env:RAG_OLLAMA_URL,Env:RAG_MANAGED_OLLAMA_URL -ErrorAction SilentlyContinue
.\scripts\start.ps1 -Profile standard
```

仅更换生成模型不需要重建论文索引；更换Embedding模型、PDF解析方式或分块参数时，应在论文库重建相关文档索引。
