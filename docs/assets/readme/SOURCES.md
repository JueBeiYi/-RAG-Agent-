# README 视觉资源来源

`hero.svg` 为项目原生手写矢量横幅，1280 × 360，未使用外部字体、网络图片或脚本。节点图属于抽象品牌图形，不表示实际系统界面或测量结果。

以下图片均为现有项目文件的逐字节复制，未裁切、重绘或修改界面内容。文件哈希见 `manifest.json`。

| README 资源 | 原始文件 | 用途 |
| --- | --- | --- |
| `library.jpg` | `docs/screenshots/新版论文库.jpg` | 文献检索、论文元信息与文档管理 |
| `chat.jpg` | `docs/screenshots/新版科研问答.jpg` | 论文问答与来源引用 |
| `comparison.jpg` | `docs/screenshots/新版论文对比.jpg` | 基于原文证据的论文对比 |
| `evidence.jpg` | `docs/screenshots/新版原文证据.jpg` | 引用片段对应的 PDF 物理页 |
| `retrieval-results.png` | `artifacts/course-report/figures/fig07_retrieval.png` | 历史冻结索引上的检索评测结果 |

四张界面图为开发阶段的真实截图。`library.jpg` 中的 359 个检索片段属于截图当时的索引快照，不代表当前默认索引规模。已检查这些图片，未发现 API Key 或工作区令牌；短文档编号是界面内的文献记录标识。

检索结果图的实验条件、原始数据、误差来源及解释边界见课程报告和 `reports/presentation/source-data/experiments.json`。图中的检索时间不包含共享的查询向量编码时间，不宜解释为完整问答耗时。
