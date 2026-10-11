# 视觉Transformer论文评测集

此处已准备80道具体问题、参考答案与物理页定位，覆盖10篇真实论文。它是经过助理逐页来源核对的评测初稿，尚待学生或教师人工复核；没有伪造人工审核人、评分、正式模型效果或课程成绩。

## 已完成的材料

| 文件 | 用途 |
| --- | --- |
| `vision_transformer_80.jsonl` | 80题主集，`document_id`暂为稳定`paper_key` |
| `vision_transformer_dev.jsonl` | 20道开发题，用于调参与Bad Case迭代 |
| `vision_transformer_test.jsonl` | 60道测试题，冻结配置后使用 |
| `dataset_manifest.json` | 数量、分类、拆分、数据集哈希与来源核对状态 |
| `source_validation.json` | 原件哈希、原页锚点、来源坐标候选及9种分块配置的实际验证 |
| `human_review_sheet.csv` | 待人工复核表，审核人、日期与意见均留空 |
| `source_checked_cases.py` | 将逐题编写的内容序列化，重建JSONL与复核表 |
| `previews/` | 使用Poppler渲染并核查的关键表格原页；不包含虚构图表 |
| `*_pages.json` | 本地原文抽取缓存，页号为固定PDF的物理页 |

分类数量为事实30、对比20、归纳15、推理15。开发集为8/4/4/4，测试集为22/16/11/11。测试答案在备题阶段被来源核对助理读过；开发与参数选择只应使用开发集，不能反复利用测试题调参后仍声称未见测试。

## 来源与标注口径

原件位于`data/corpus/`，`manifest.json`记录完整官方作者、题目、固定arXiv版本、原文链接、许可链接、SHA256、文件字节数和页数。首次预印本时间与会议出版年份分别记录，避免将ViT/DeiT的2020年提交误写成会议出版年份。10篇原件合计169个物理页。

每条证据保留短原文实体或事实锚点、物理页码，以及原PDF和整页文本的哈希。所有论文累计证据quote均不超过25个英文词；参考答案使用自行组织的中文表述。短锚点是定位条件，不是完整证明：同页可能存在多个包含该锚点的段落，候选段落和坐标记录在`source_validation.json`，人工复核仍应确认实际支持答案的上下文。实验报告须写明这一相关性标注限制。

已经区分准确率与错误率、准确率百分点与相对提升、FLOPs与MACs、预训练与微调标签、不同数据/分辨率的不可直接比较性。推理题另有`inference_basis`，标出作者解释、基于证据推断或依据报告数值计算。`reviewed=true`表示来源已核对，`review_method=assistant_source_checked`与`human_review_pending=true`表示尚非人工共识标签。

## 重现与导入

在项目根目录使用已安装依赖的Python执行：

```powershell
python scripts/prepare_corpus.py download
python reports/datasets/source_checked_cases.py
python scripts/prepare_corpus.py validate
```

`download`是明确联网操作；已有manifest时固定原版本并检查SHA256，不会悄悄更新论文版本。`validate`完全离线，核查原件、全部119条原页锚点、加载器保留情况，以及固定/递归/句段边界三种策略在256/512/1024轻量token、重叠64的九种配置。它不下载或运行模型，不产生模型质量分数。

启动真实模型API后，使用与界面相同的workspace token，通过环境变量传入；脚本不打印或保存token：

```powershell
python scripts/prepare_corpus.py import --base-url http://127.0.0.1:8002
```

该命令上传现有PDF，等待真实导入任务完成，更新官方元信息与常用别名，并保存`index-map.json`及映射到实际文档ID的`vision_transformer_80_mapped.jsonl`和对应dev/test文件。需在本地环境预先设置`RAG_WORKSPACE_TOKEN`，与界面选择的知识库范围一致。未完成导入时，主JSONL中的paper_key不能直接当服务器document_id。

已有文档ID映射时可重建映射文件：

```powershell
python scripts/prepare_corpus.py map --mapping reports/datasets/index-map.json
python scripts/prepare_corpus.py validate-index --data-dir data/runtime
```

`validate-index`核对真实持久化知识库中已映射ID、文档哈希、就绪状态和当前分块证据，生成`index_validation.json`。若API采用其他data_dir，应使用该目录。需要相同的`RAG_WORKSPACE_TOKEN`；脚本只写范围的哈希指纹。

## 人工复核与正式评测

先逐题检查原页和参考答案，将真实复核记录写入复核表；确认后才把相应`review_method`改为`human`、`human_review_pending`改为`false`。不要用空白审核人或助理核对冒充学生人工评分。生成质量评分需要针对系统实际答案另行填写正确性、完整性和引用准确性；本表只审核参考题与标注。

固定数据集与模型配置后执行60题测试集。若看到测试Bad Case后继续改进，应注明测试集已暴露，并准备新的未见题目用于重新估计泛化效果。原始PDF按manifest中的原文许可使用，本项目未为原论文额外赋予再许可；公开交付可优先保留来源清单与下载脚本。
