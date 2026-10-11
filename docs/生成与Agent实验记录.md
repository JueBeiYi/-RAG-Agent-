# 生成与Agent实验记录

计划与输入核对日期：2026 年 10 月 11 日。当前脚本已实现并完成轻量验证，真实模型生成阶段尚未执行；本记录不把输入准备或单元检查写成实验成绩。

## 真实冻结输入

生成脚本：`scripts/run_course_generation.py`。输入来自检索研究实际计算的 `reports/experiments/retrieval/frozen/hybrid.jsonl`，包含20道开发题、60道保留测试题、真实Top-20检索结果与1024维真实BGE查询向量。来源、模型版本、语料/题集hash与索引版本保留在输入和 `reports/experiments/generation/prepared_inputs.json` 中。

当前输入仍使用已持久化的359个真实论文片段；检索研究会另外比较重新按轻量Token构建的分块变体。生成研究不能把这组冻结结果称为重新在线检索，也不能将其延迟代替浏览器端到端时间。

脚本将冻结Top-20的前5个候选经当前应用相同的上下文预算与真实来源裁剪生成固定context，保存每题context hash。参数及新旧Prompt使用同一context与来源列表，避免改变检索输入形成混杂。公开输出的Chunk.scope清空，内部会话/缓存放在独立评测scope，不复制实际应用工作区凭据。

## 最少实验设计

| 研究 | 题目与条件 | 边界 |
|---|---|---|
| 保留测试集基线 | 60题各生成1次，实际本地Qwen、真实流式用量和引用规则审校 | generation-only；正确性和完整性等待人工评分 |
| 单因素参数 | 四题型各1道dev；基线0.1/0.9/40、temperature0.0、temperature0.5、top_p0.7、top_k20，共20条 | 小样本探索，不做整体质量优劣或显著性结论 |
| Prompt对比 | 同4道dev；简朴对照Prompt额外4次，当前Prompt复用参数研究4条基线 | 简朴Prompt是显式设计的对照，不声称对应历史代码版本；只比较引用/数字规则与执行现象 |
| Agent路由对照 | 概念、计算、时间、问题关键词、单论文事实，共5个样本；对照使用固定真实context RAG | 小样本；固定对照没有在线重检索；一般概念可来自普通模型知识 |
| 工具并行 | 真实安全计算器与输入关键词工具，workers1/2，各10次配对执行 | 只测独立本地工具，不含LLM或检索；并行不保证更快 |
| 冷热缓存 | 真实冻结query向量和应用缓存准入策略下的exact repeat | 若没有真实向量或答案未通过策略则明确不执行，不造hash向量 |

预定真实生成服务为独立项目Ollama `http://127.0.0.1:11435`、模型 `research-qwen:7b`。系统11434不用于本研究。检索重模型仍在串行执行，生成脚本须等其退出并释放内存；脚本还检查可用内存，不并行加载Embedding或Reranker。

每个题目与配置只执行一遍，没有固定随机种子或重复采样。研究保留实际运行顺序、每组指标、完整Prompt与源码hash；单次采样差异不能作为参数优劣的稳健结论。

为控制完整60题实际生成时间，研究统一使用256个输出Token上限，并在相同用户模板要求尽量100–150字保留必要事实与引用。这是显式的短答案研究配置，与应用标准配置的768个输出Token区分；长度要求是软目标，截断、遗漏和引用质量会如实记录并待人工复核。

## 人工评分填写表

基线实际执行时逐题写入 `reports/experiments/generation/baseline/human_scoring_60.csv`，包含实际答案、参考答案、原文位置与摘录、引用和错误。评分人、正确性/完整性/引用质量/流畅性1–5分、Bad Case类型与备注全部留空，已有真实人工填入内容不会被覆盖。

来源核对标为 `assistant_source_checked`，保留 `human_review_pending=true`；答案人工评分另标待完成。数字规则可能标记据公式正确推算但未逐字出现的结果，不能把这种规则计数当正确率。

## 运行与复现

```powershell
.venv\Scripts\python.exe scripts/run_course_generation.py prepare
# 重模型退出后再逐阶段执行，使用相同冻结输入与参数：
.venv\Scripts\python.exe scripts/run_course_generation.py baseline
.venv\Scripts\python.exe scripts/run_course_generation.py parameters
.venv\Scripts\python.exe scripts/run_course_generation.py prompt
.venv\Scripts\python.exe scripts/run_course_generation.py agent
.venv\Scripts\python.exe scripts/run_course_generation.py parallel
.venv\Scripts\python.exe scripts/run_course_generation.py cache
.venv\Scripts\python.exe scripts/run_course_generation.py summary
```

每阶段记录及时落盘，可按相同来源hash续跑；不同源码、配置或来源不能静默混入旧checkpoint。`results.json`兼容交付材料输入，包含metrics、records、config、split、hash、study、kind、demo=false；没有填写的人工作答质量保持null。实际执行后，本记录会由脚本更新逐阶段已测结果与出处。
