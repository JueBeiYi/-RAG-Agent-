# 智能科研助理

通用论文知识库问答系统：RAG、手写 ReAct、八个本地工具、流式引用、会话记忆、证据审校、运行回放与可复现评测。已实现课程五个模块并完成真实模型实验；当前包含 10 篇真实论文、80 道来源核对题及 60 题实际生成答案。题目仍待学生确认，答案人工质量评分按用户要求保留待填，不预置虚构成绩。

## 课程交付

最终材料位于 [artifacts/delivery-final](artifacts/delivery-final)：课程报告、系统评测报告、技术设计文档、用户手册、实验与 Bad Case 附件、分块策略对比实验报告、Prompt 模板设计与实验记录、Bad Case 分析报告，以及 17 页演示 PPT。逐条完成状态见[课程要求逐条验收](docs/课程要求逐条验收.md)，实际答案与可填写评分字段见[60 题人工评分表](reports/experiments/generation/baseline/human_scoring_60.csv)。人工评分、题集人工确认和实际提交日期仍待填写；Docker 部署文件已提供，本机未启动容器验证。

最新生产索引含 **342 个片段**；三档检索测试和短答案生成实验使用冻结的历史 **359 个片段**索引。60 题生成实验只评测真实检索输入后的生成阶段，输出上限为 256 token，不代表最新生产索引的全流程质量结论。原始记录分别保存在 [retrieval](reports/experiments/retrieval)、[generation](reports/experiments/generation) 和[当前索引记录](reports/requirements/current-standard-index.json)。

课程交付包包含源码、环境说明、真实实验记录、最终材料与真实 Git 历史，排除 API 密钥、模型权重、虚拟环境和运行缓存。论文原 PDF 通过 `data/corpus/manifest.json` 记录版本、来源与 SHA256；换电脑后执行 `.venv\Scripts\python.exe scripts/prepare_corpus.py download` 恢复固定版本，再在论文库导入，模型按下方说明准备。

## 快速运行

推荐 Python 3.12，支持 3.10–3.13。本机已有 `.venv`、模型与真实论文索引，可直接启动标准模式；已有服务运行时直接打开界面，无需重复启动：

```powershell
.\scripts\start.ps1 -Profile standard
# 需要停止本项目进程时执行
.\scripts\stop.ps1
```

界面：http://127.0.0.1:8501，接口文档：http://127.0.0.1:8000/docs。需要单独检查 demo 时可用 `scripts/start.ps1 -Profile demo -ApiPort 8003 -UiPort 8503`；demo 使用哈希向量、关键词重排与原文抽取，不执行大模型推理，不能作为正式模型质量评测。

其他电脑首次准备基础环境：

```powershell
.\scripts\setup.ps1 -Python 'C:\Path\To\Python312\python.exe'
.\scripts\start.ps1 -Profile demo
.venv\Scripts\python.exe scripts/seed_demo.py
```

打开侧栏“工作区设置与恢复”，填写演示脚本打印的工作区标识并点击“连接此工作区”；也可保存或恢复工作区配置文件。`DEMO_` 资料均为明确标注的虚构文本，不是课程论文语料。配置文件用于恢复对应资料与历史，请自行保管。

## 标准本地模式

本机已缓存本地模型，并已在标准工作区导入 10 篇真实论文、建立 342 个最新检索片段。已有缓存可直接使用 `scripts/start.ps1 -Profile standard`；首次在其他电脑运行时，再准备 Ollama、模型依赖与权重。首次准备会联网、需要数 GB 空间，应用离线模式不会自动下载。模型与语料准备不等于完成正式答案质量评测，实际界面验证见[界面功能验收](docs/界面功能验收.md)。项目便携Ollama默认独立使用11435端口，避免与用户系统Ollama冲突；显式`RAG_OLLAMA_URL`仍优先。云端API Key、模型ID与服务URL的设置见[模型与API配置](docs/模型与API配置.md)。

```powershell
.\scripts\setup.ps1 -Python 'C:\Path\To\Python312\python.exe' -Models
.venv\Scripts\python.exe scripts/prepare_models.py --models BAAI/bge-m3 BAAI/bge-large-zh-v1.5 moka-ai/m3e-base BAAI/bge-reranker-base --cache-dir data/models/hub --ollama qwen2.5:7b-instruct-q4_K_M
.\scripts\start.ps1 -Profile standard
```

启动脚本会检查端口占用；已有服务运行时可使用 `-ApiPort` 和 `-UiPort` 选择不同端口。`config.standard.yaml` 使用 Chroma、bge-m3、bge-reranker-base 和 Qwen2.5 7B Q4，处理英文论文与中文提问；`config.yaml` 保留 bge-large-zh-v1.5 课程基线，m3e-base 用于课程比较。Embedding 与重排默认 CPU，Ollama 按本机资源使用 GPU。启动器优先复用项目中的模型清单、本地缓存与标准工作区；本机已验证的本地 LLM 名称为 `research-qwen:7b`，启动器从模型清单自动选用。模型清单保存实际 snapshot revision，可用本地 snapshot 路径固定版本。

更换模型权重、分块或解析配置后，旧文档会提示重新索引。未准备模型时健康接口明确显示未就绪。配置由 config.yaml 和 RAG_字段名大写环境变量控制；`.env.example` 只作示例，不会自动启用云端。

Linux/macOS 手动运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-models.txt
python scripts/prepare_models.py --models BAAI/bge-m3 BAAI/bge-large-zh-v1.5 moka-ai/m3e-base BAAI/bge-reranker-base
ollama pull qwen2.5:7b-instruct-q4_K_M
RAG_CONFIG=config.standard.yaml python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# 另一个终端
python -m streamlit run src/frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

## 应用功能

四个页面：论文库、科研问答、论文对比、实验与运行记录。支持 PDF/DOCX/TXT/Markdown、导入进度与重试、论文元信息编辑、原文与片段分页、索引重建、会话搜索/改名/导出、三档检索、流式回答、来源抽屉、运行回放与真实指标。论文选择跨页面保留，切换工作区会清理当前界面状态；同名记录可以明确区分。

八工具：知识库问答、论文元信息、论文对比、关键词、结构化摘要、当前时间、安全计算器、本地论文目录。当前注册这八个本地工具；离线模式使用本地文档与模型。

PDF 使用物理页码；跨页证据可以选择各页查看。其他格式使用段落，不编造页码。证据审校检查来源和数字支撑，不能证明全部语义正确。扫描 PDF 需先 OCR；公式、复杂表格要结合原页核查。比较论文时实验条件不同不能直接排名。

标准模式的论文对比按用户选择的维度生成中文证据表，每篇分别检索并核对来源；作者署名、同等贡献说明与参考文献不作为实验事实。模型未完成、返回结构不合格或引用不可靠时，结果会明确标为“原文摘录降级 · 对比不完整”，保留可回查的短证据和实际用量，不把降级片段当作完整解释。任务、数据集版本、划分、指标与训练条件未完整核对时，不自动计算跨论文差值或排名。

评测集可上传文件或粘贴 JSONL。“依次运行三档对比”显示实际任务状态，完成后可逐题排查并导出 JSON/CSV/XLSX。语料、题集或数据划分不同的实验会拒绝绘制方法优劣图。对比答案和运行记录可以恢复、导出或按确认操作删除；“请求停止”会在可停止的事件边界结束执行，当前模型或工具可能仍需释放资源。

## 数据与评测

在具体方向收集 10–20 篇论文后导入，生成 80 题标注模板（20 dev、60 test）：

```powershell
.venv\Scripts\python.exe scripts/build_dataset.py --scope YOUR_WORKSPACE_TOKEN --output reports/evaluation_cases.jsonl
.venv\Scripts\python.exe scripts/evaluate.py validate reports/evaluation_cases.jsonl
.venv\Scripts\python.exe scripts/evaluate.py retrieval --cases reports/evaluation_cases.jsonl --scope YOUR_WORKSPACE_TOKEN --split test --mode hybrid_rerank --output reports/runs/final
.venv\Scripts\python.exe scripts/evaluate.py matrix --cases reports/evaluation_cases.jsonl --scope YOUR_WORKSPACE_TOKEN --split dev --preset chunking --output reports/runs/chunking
.venv\Scripts\python.exe scripts/evaluate.py generation --cases reports/evaluation_cases.jsonl --scope YOUR_WORKSPACE_TOKEN --split test --output reports/runs/answers
```

模板 reviewed=false，须填写参考答案、真实文档ID、页/段、原文摘录、预期工具，并人工核对。未经核对的题不能正式评测。dev 调参后冻结配置，再比较两版在 test 上的真实结果；测试集用于修复后必须标明已经暴露。

实验预设覆盖三解析器、三分块与256/512/1024、两Embedding、三档检索、生成参数、路由、并行、缓存和上下文。分块大小是**轻量 token**（中文字符/英文词/标点），区别于模型 tokenizer；超长编码完整分窗聚合，拒绝静默截尾。独立 CLI 请在无索引变更时运行，参数变体使用隔离索引。

记录 Hit@5、MRR、Recall@5、工具准确率、轮次、延迟、Token、配置/语料hash、索引版本、模型revision。答案正确性、完整性、引用准确性需人工据原文评分。完整操作见[评测说明](docs/评测说明.md)。

## 测试

```powershell
New-Item -ItemType Directory data/runtime -Force | Out-Null
.venv\Scripts\python.exe -m pytest --basetemp data/runtime/pytest-check
.venv\Scripts\python.exe scripts/check_ui_workflows.py --report reports/ui/workflow-tests.json
# 旧入口继续兼容相同完整流程
.venv\Scripts\python.exe scripts/smoke_ui.py --report reports/ui/workflow-tests.json
# 如需连接已有的专用demo API，可显式加 --api http://127.0.0.1:8003
```

UI 集成检查默认临时启动独立随机端口的 demo API，只在自行创建的随机工作区操作与清理测试数据，不停止已有服务。当前覆盖 16 个完整工作流，报告保存真实执行时间与逐项结果。显式 `--api` 只允许本机 demo 服务；用于控制流时序的两项取消测试仅在自动创建的测试服务中执行。Chroma/FAISS 测试在对应包安装时运行。当前界面结果与浏览器实际文件传输证据见[界面功能验收](docs/界面功能验收.md)；早期实现记录保留在[实现与验证记录](docs/实现与验证记录.md)。

requirements-tested.txt 记录已有环境的依赖版本；标准模型环境按 requirements-models.txt 准备。依赖清单与模型权重清单用途不同，模型是否可用以本地模型准备记录和健康检查为准。

## Docker

当前 Windows 环境未执行 Docker；已核对Compose配置与一键部署脚本语法，容器运行状态不冒充验证通过。有Docker Engine的环境可直接执行`.\scripts\deploy_docker.ps1 -Profile standard`；无模型演示用`-Profile demo`，已配置NVIDIA容器运行时可加`-Gpu`。首次标准部署会下载权重。等价分步命令如下。

```bash
docker compose -f docker/docker-compose.yml up -d --build ollama
docker compose -f docker/docker-compose.yml --profile setup run --rm prepare
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose -f docker/docker-compose.yml up -d api frontend
```

模型与数据均使用持久化卷。首次联网准备后，本地问答链路可以断网；功能检查可以在创建服务前设置 RAG_PROFILE=demo。

## 代码

`src/core` 数据与配置，`data_loader/chunking/retrieval` 文档检索，`generation` RAG与引用，`agent` 手写决策和工具，`services/api` 任务与SSE，`frontend` 四页界面，`evaluation` 评测。脚本、测试和Docker分别位于 scripts、tests、docker。

详见[技术设计](docs/技术设计文档.md)、[用户手册](docs/用户使用手册.md)和[课程规划](docs/课程实践项目规划.md)。
