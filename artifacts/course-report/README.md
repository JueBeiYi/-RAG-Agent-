# 智能科研助理课程报告

本目录为《智能科研助理——基于 RAG + Agent 的论文知识库问答系统》课程报告的独立 LaTeX 交付目录。报告依据课程实践要求及项目交付模板编排，包含项目概述、团队分工、五个模块的交付清单、技术实现、技术设计、总结与附录，并附实验图表、界面截图及参考文献。

当前 `main.tex` 与46页 `main.pdf` 为用户修订稿，本次同步直接保留这些文件，未重新生成或编译。下述图表升级与 `build-records/` 中的哈希、日志属于先前构建记录，不代表当前修订稿的文件哈希。

2026-10-17 图表升级版已实际编译为47页PDF，包含26幅编号图（21幅学术图及5幅界面图）和29张编号表。17幅既有学术图统一重绘，新增4幅基于真实逐题记录的分析图。编译信息、源文件/PDF哈希及日志保存于 `build-records/`。

## 本次图表升级

图表采用统一的学术配色、字体层级、面板编号和注释规范，并提供可用于放大排版的矢量 PDF / SVG 及高清 PNG。新增分析分别回答以下问题：

| 新增图件 | 数据与方法 | 阅读重点 |
| --- | --- | --- |
| 配对检索效应森林图 | 同一60题，逐题配对，10,000次查询Bootstrap | 展示方法差值及95%区间，区分不同指标的变化 |
| 60题证据排名矩阵 | 三种检索方法，按四类题型分组 | 识别持续失败、命中救回与排名变化；未找到相关证据单独标记 |
| 回答质量关联图 | 60题四维AI辅助评分，离散计数气泡及Spearman秩相关 | 展示正确性、完整性、引用质量与流畅性的关系 |
| Prompt逐题配对图 | 4道开发题、相同冻结上下文，新旧Prompt对照 | 呈现题目间不一致的变化；当前Prompt复用参数基线答案 |

本次升级保持实验源数据快照不变，新增模型调用为0。Bootstrap区间反映固定实验条件下的查询抽样不确定性，不表示多次模型运行的方差；评分来源仍为AI辅助评审，未实施独立人工评分。

## 文件说明

| 文件或目录 | 用途 |
| --- | --- |
| `main.tex` | 完整报告源文件；正文、公式、表格和参考文献均在此文件内 |
| `main.pdf` | 由编译生成的阅读版，以实际编译结果为准 |
| `figures/` | 正文引用的图件；科学图表提供 PDF、PNG 和 SVG，截图保留原始图像及功能区标注版 |
| `figures/figure_manifest.json` | 图件清单、数据来源、文件哈希与区间估计方法记录 |
| `figures/original-ui/` | 图表生成器使用的原始界面截图 |
| `data/report-data.json` | 本报告使用的实验与本地验收数据快照 |
| `data/advanced-statistics.json` | 由冻结快照派生的配对Bootstrap、排名矩阵、评分关联及Prompt逐题统计 |
| `build_report.py` | 根据数据快照及内嵌正文重新生成 `main.tex` |
| `build_figures.py` | 根据本目录中的数据和截图重新生成全部图件，并调用高级分析模块 |
| `build_advanced_figures.py` | 计算并绘制4幅新增逐题分析图；单独执行时仅更新派生统计 |
| `DESIGN.md` | 视觉规范、新增统计方法、单位及设计参考说明 |
| `figure_style.py` | 学术图表的统一配色、字体、坐标轴及版式规范 |
| `compile.ps1` | Windows 编译入口；编译本脚本所在目录的报告 |
| `requirements-report.txt` | 图表重建所需的 Python 第三方依赖 |

移动或发送报告时请保留整个目录结构。图件使用相对路径，编译无需访问原项目工作目录，也无需模型 API。

## 编译

推荐使用包含 `ctex`、Fandol 字体及常用宏包的 TeX Live 或 MiKTeX，以 XeLaTeX 连续编译两遍，更新目录和交叉引用：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

也可使用 Tectonic；它会自动处理多轮编译，首次运行通常需要联网下载宏包和字体：

```powershell
tectonic --keep-logs --keep-intermediates main.tex
```

上述命令应在本目录运行。Windows 也可直接调用脚本：

```powershell
.\compile.ps1
.\compile.ps1 -CompilerPath "C:\Tools\tectonic.exe"
```

脚本优先采用显式指定的编译器；未指定时依次查找 PATH 中的 XeLaTeX、Tectonic，再检查环境变量 `CODEX_TECTONIC_PATH`。脚本不下载或安装软件，编译出错即停止；只有编译器成功退出且生成非空 PDF 后才报告成功。若系统执行策略限制脚本，可在允许的 PowerShell 会话中运行上述编译命令。

## 修改正文与重建图表

可以直接编辑 `main.tex`。**重新执行 `build_report.py` 会覆盖 `main.tex`**；需要持续使用生成器时，应同步修改生成器中的正文或数据。正文生成器只使用 Python 标准库。

图表重建需要 Python 3.10 或更高版本，以及 Matplotlib、NumPy、Pillow。建议使用独立虚拟环境：

```powershell
python -m pip install -r requirements-report.txt
python build_figures.py
```

图表脚本读取本地数据快照和原始截图，不调用模型、不重新执行基准实验。`build_figures.py` 会一并调用 `build_advanced_figures.py`，更新 `data/advanced-statistics.json` 并绘制新增图件。中文图件建议安装微软雅黑或 Noto Sans CJK SC 字体。重建会覆盖对应图件及其清单；修改图表后应重新编译报告。若仅需修改报告文字或阅读已有图表，无需安装这些 Python 依赖。

## 结果边界与提交前补充

- 报告已纳入 `127.0.0.1` 本地运行的实际验收记录；相关数据与历史参数实验分别注明实验时间、样本和计时范围。
- 回答质量评分据实标为“AI 辅助评分”，不表述为独立人工评分或专家盲评。Docker 实际部署未验收，本地运行记录不等同于 Docker 验收。
- 成员姓名、班级、学号及 40% / 30% / 30% 的贡献比例依据已提供信息填写；具体分工安排仍需成员确认。
- 教师成绩、教师签名及真实提交日期保留为空。报告编制日期不替代真实提交日期；模板示例日期不作为本次提交事实。

正式提交前，请补齐需要成员或教师确认的信息，并以最终 PDF 检查目录、页码和图表引用。
