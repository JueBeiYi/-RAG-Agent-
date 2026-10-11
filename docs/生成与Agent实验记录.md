# 生成与Agent实验记录

记录日期：2026-10-11（UTC）。

所有论文生成研究使用真实冻结检索结果与本地Qwen，不加载Embedding或Reranker。记录的延迟属于generation-only，不是在线端到端性能。

60test基线答案可填写评分表：`reports/experiments/generation/baseline/human_scoring_60.csv`。评分人、正确性、完整性、引用质量和流畅性均留空，未伪造人工评分。

开发集取四题型各1题，5个单因素配置共20问；旧Prompt额外4问，新Prompt复用相同context的4条开发集基线。小样本只用于规则现象与参数探索，不据此概括整体质量。

| 研究 | 状态 | 实际记录数 | 成功执行率 | 平均耗时ms |
|---|---|---:|---:|---:|
| baseline | completed | 60 | 1 | 3055.9623 |
| parameters | completed | 20 | 1 | 2152.74825 |
| prompt | completed | 8 | 1 | 3403.202375 |
| small_sample_agent_routing | completed | 10 | 1 | 1867.345660009475 |
| actual_local_tool_parallelism | completed | 20 | 1 | 22.642689995700493 |
| exact_semantic_cache_cold_hot | completed | 2 | 1 | 1479.794800048694 |

出处与限制：

- 原始结果、来源编号、实际Token和自动规则审校均保存在各研究results.json；自动规则不等于人工正确性评分。
- Agent路由只有5个题型样本；固定RAG对照明确使用同一真实固定context，未重新在线检索。
- 并行研究只测两个真实独立本地工具，不含LLM；线程调度可能比串行更慢，结果照实保留。
- 缓存只测真实冻结query向量下的exactrepeat；没有向量或答案不满足缓存策略时明确不执行。
- 案例来源已自动据原文核对，人工来源复核与答案评分仍待完成。公开JSON不包含真实工作区凭据。
