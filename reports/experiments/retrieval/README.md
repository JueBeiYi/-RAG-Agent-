# 真实检索实验结果

10篇真实论文；20dev和60test。模型串行加载，已编码向量复用。

## m3_test

| 配置 | 查询数 | Hit@5 | MRR | Recall@5 |
| --- | --- | --- | --- | --- |
| vector | 60 | 0.417 | 0.299 | 0.325 |
| hybrid | 60 | 0.433 | 0.354 | 0.358 |
| hybrid_rerank | 60 | 0.617 | 0.431 | 0.475 |

## chunking_dev

| 配置 | 查询数 | Hit@5 | MRR | Recall@5 |
| --- | --- | --- | --- | --- |
| moka-ai/m3e-base/fixed-256 | 20 | 0.350 | 0.292 | 0.325 |
| moka-ai/m3e-base/fixed-512 | 20 | 0.400 | 0.270 | 0.375 |
| moka-ai/m3e-base/fixed-1024 | 20 | 0.500 | 0.360 | 0.400 |
| moka-ai/m3e-base/recursive-512 | 20 | 0.400 | 0.278 | 0.350 |
| moka-ai/m3e-base/semantic-512 | 20 | 0.450 | 0.308 | 0.400 |

## embedding_dev

| 配置 | 查询数 | Hit@5 | MRR | Recall@5 |
| --- | --- | --- | --- | --- |
| moka-ai/m3e-base/recursive-512 | 20 | 0.400 | 0.278 | 0.350 |
| BAAI/bge-large-zh-v1.5/recursive-512 | 20 | 0.350 | 0.296 | 0.300 |

耗时列仅记录检索/排名/精排计算，批量编码单独列出，不能作为完整在线问答延迟。
短页内证据锚点已由助理核对，人工审核仍待完成；不能以此宣称课程成绩或答案质量。
