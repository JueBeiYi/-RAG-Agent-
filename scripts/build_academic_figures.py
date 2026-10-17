"""Build publication and presentation graphics exclusively from recorded experiments.

Run with the project Python (matplotlib, numpy and Pillow are already installed).
AI reviews stay explicitly separate from human reviews and runtime verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/academic-figures"
EXP = ROOT / "reports/experiments"
NAVY, TEAL, BLUE, ORANGE, PURPLE = "#17324D", "#15877F", "#497AA8", "#D78B37", "#806A9A"
INK, MUTED, GRID, LIGHT = "#23394D", "#687C8D", "#E5EBF0", "#F3F7F9"
COLORS = [TEAL, BLUE, ORANGE, PURPLE, "#A8B7C7"]
CAT = {"factual": "事实型", "comparison": "对比型", "synthesis": "综合型", "reasoning": "推理型"}
MODE = {"vector": "向量检索", "hybrid": "+ BM25 / RRF", "hybrid_rerank": "+ Cross-Encoder"}
QUALITY = ["correctness", "completeness", "citation_quality", "fluency"]
QLABELS = ["正确性", "完整性", "引用支撑", "流畅性"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def result(name):
    return load(EXP / name / "results.json")


def shortpath(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def wrap(value, width=72):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=True, replace_whitespace=False))


def setup():
    for filename in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"):
        if Path(filename).exists():
            font_manager.fontManager.addfont(filename)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.titleweight": "bold", "axes.titlepad": 13,
        "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white", "xtick.major.size": 0, "ytick.major.size": 0,
    })
    for directory in ("figures", "tables", "data"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)


class Atlas:
    def __init__(self):
        self.items = []
        self.sources = {}
        self.pdf = PdfPages(OUT / "科研助理实验图表集.pdf", metadata={
            "Title": "智能科研助理 | 实验结果与学术图表", "Author": "课程项目组 / Codex AI辅助整理",
            "Subject": "Recorded experiments; AI-assisted answer review; historical and current scopes separated"})

    def page(self, title, subtitle, category="实验结果", kind="figure"):
        fig = plt.figure(figsize=(40/3, 7.5))
        number = len(self.items) + 1
        fig.text(.065, .95, f"RESEARCH ASSISTANT  /  {category.upper()}", fontsize=10, color=TEAL, weight="bold")
        fig.text(.065, .89, title, fontsize=23, weight="bold", color=NAVY)
        fig.text(.065, .842, subtitle, fontsize=10.8, color=MUTED)
        fig.text(.94, .945, f"{number:02}", ha="right", fontsize=13, color=NAVY, weight="bold")
        fig.add_artist(plt.Line2D([.065, .935], [.813, .813], color=GRID, lw=1))
        return fig

    def save(self, fig, slug, title, caption, sources, scope, category, kind="figure"):
        folder = "tables" if kind == "table" else "figures"
        for source in sources:
            path = ROOT / source
            if path.exists():
                self.sources[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        fig.text(.065, .106, wrap(caption, 90), fontsize=9.4, color=INK, va="top", linespacing=1.5)
        fig.text(.065, .048, scope, fontsize=8.3, color=MUTED, va="top")
        fig.text(.935, .048, "NAU  ·  2026.10", fontsize=8.3, color=MUTED, ha="right", va="top")
        stem = OUT / folder / slug
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".svg"))
        self.pdf.savefig(fig)
        self.items.append({"id": slug, "title": title, "category": category, "kind": kind,
                           "caption": caption, "scope": scope, "sources": sources,
                           "png": f"{folder}/{slug}.png", "svg": f"{folder}/{slug}.svg",
                           "pdf_page": len(self.items) + 1})
        plt.close(fig)

    def table(self, slug, title, subtitle, headers, rows, sources, scope, caption="", widths=None):
        # Keep a legible presentation table; detailed long-form data stay in CSV/XLSX.
        fig = self.page(title, subtitle, "结果表", "table")
        ax = fig.add_axes([.065, .21, .87, .55]); ax.axis("off")
        cell_text = [[str(v) if v is not None else "—" for v in row] for row in rows]
        table = ax.table(cellText=cell_text, colLabels=headers, cellLoc="center", colLoc="center",
                         bbox=[0, 0, 1, 1], colWidths=widths)
        table.auto_set_font_size(False); table.set_fontsize(11 if len(headers) <= 7 else 10)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white"); cell.set_linewidth(1.3)
            cell.PAD = .1
            if r == 0:
                cell.set_facecolor(NAVY); cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(LIGHT if r % 2 else "white")
                cell.set_text_props(color=INK)
            if c == 0:
                cell.set_text_props(ha="left")
        self.save(fig, slug, title, caption, sources, scope, "结果表", "table")


def axis(fig, rect=(.10, .23, .82, .52)):
    ax = fig.add_axes(rect); ax.grid(axis="y"); return ax


def bars(ax, labels, vals, color=TEAL, percent=False):
    x = np.arange(len(labels))
    bs = ax.bar(x, vals, width=.56, color=color, zorder=3)
    ax.set_xticks(x, labels)
    for bar, value in zip(bs, vals):
        if value is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{value:.1%}" if percent else f"{value:,.2f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_ylim(0, max(vals)*1.24 if max(vals) else 1)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    return bs


def grouped(ax, labels, groups, values, ylabel="", percent=False, ylim=None):
    x = np.arange(len(labels)); width = .72 / len(groups)
    for i, (group, vals) in enumerate(zip(groups, values)):
        pos = x + (i - (len(groups)-1)/2)*width
        bs = ax.bar(pos, vals, width*.92, color=COLORS[i], label=group)
        for b, v in zip(bs, vals):
            ax.text(b.get_x()+b.get_width()/2, v + (.012 if percent else .055),
                    f"{v:.1%}" if percent else f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    elif percent: ax.set_ylim(0, min(1.2, max(max(v) for v in values)*1.25))
    if percent: ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, ncol=len(groups), loc="upper left", bbox_to_anchor=(0, 1.16))


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261015)
    means = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def pipeline(atlas):
    fig = atlas.page("从论文证据到可追溯回答", "系统结构与实验覆盖  |  IMPLEMENTATION & EVIDENCE", "系统与数据")
    ax = fig.add_axes([.065, .22, .87, .54]); ax.axis("off")
    nodes = [(.02,.65,"多格式论文导入","PDF / DOCX / TXT / MD"), (.27,.65,"结构解析与分块","3个解析器 · 3类分块"),
             (.52,.65,"混合检索与重排","Dense + BM25 + RRF"), (.77,.65,"证据约束生成","流式输出 · 物理页引用"),
             (.02,.10,"ReAct 智能路由","8个工具 · 记忆 · 追踪"), (.27,.10,"真实实验比较","解析 / 分块 / 模型 / 参数"),
             (.52,.10,"质量与性能评审","AI质量评审 · Token · 延迟"), (.77,.10,"科研汇报输出","高清PNG · SVG · PDF · XLSX")]
    for i,(x,y,t,s) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((x,y),.21,.26,boxstyle="round,pad=0.014,rounding_size=0.025",fc=LIGHT if i>=4 else NAVY,ec="none"))
        ax.text(x+.105,y+.17,t,ha="center",fontsize=12,weight="bold",color=INK if i>=4 else "white")
        ax.text(x+.105,y+.075,s,ha="center",fontsize=9,color=MUTED if i>=4 else "#CEE5EA")
    for y in (.78,.23):
        for x in (.235,.485,.735):
            ax.annotate("",xy=(x+.026,y),xytext=(x-.006,y),arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.8})
    atlas.save(fig,"01_system_evidence","从论文证据到可追溯回答","系统结构图说明模块关系；节点不代表所有模块已完成最终课程验收。",
               ["docs/课程要求逐条验收.md"],"系统设计示意；评分来源为AI辅助评审，人工评分状态另列。","系统与数据")


def corpus_dataset(atlas):
    pars = result("retrieval/parsers")
    recs = [r for r in pars["records"] if r["parser"] == "pymupdf"]
    fig = atlas.page("10篇固定版本论文构成评测语料", "CORPUS PROFILE  |  物理页统计与可追溯版本", "系统与数据")
    ax = axis(fig, (.12,.22,.78,.55))
    recs.sort(key=lambda r:r["physical_pages"])
    labels = [r["name"].split("_")[0].upper() for r in recs]
    vals = [r["physical_pages"] for r in recs]
    ax.barh(labels,vals,color=TEAL,height=.6)
    ax.grid(False); ax.grid(axis="x"); ax.set_xlabel("原始PDF物理页数")
    ax.set_xlim(0,max(vals)*1.2)
    for i,v in enumerate(vals): ax.text(v+.4,i,str(v),va="center",fontsize=11)
    atlas.save(fig,"02_corpus","10篇固定版本论文构成评测语料",f"共{sum(vals)}个物理页；页数来自真实解析记录，不是正文页码或提取质量评分。",
               ["reports/experiments/retrieval/parsers/results.json","data/corpus/manifest.json"],"原始论文版本固定；当前生产索引342片段，正式历史检索实验359片段。","系统与数据")
    test = result("retrieval/m3/test/vector")["records"]
    dev = result("retrieval/m3/dev/vector")["records"]
    cats = list(CAT)
    fig = atlas.page("评测题集覆盖四类科研问题", "80 QUESTIONS  |  开发集与测试集独立展示", "系统与数据")
    ax = axis(fig)
    counts = [[sum(r.get("category")==c for r in rr) for c in cats] for rr in (dev,test)]
    bottom = np.zeros(4)
    for j,(name,values) in enumerate(zip(("开发集 dev · n=20","测试集 test · n=60"),counts)):
        ax.bar(range(4),values,bottom=bottom,color=COLORS[j],width=.55,label=name)
        for i,v in enumerate(values): ax.text(i,bottom[i]+v/2,str(v),ha="center",va="center",color="white",weight="bold",fontsize=14)
        bottom += np.array(values)
    ax.set_xticks(range(4),[CAT[c] for c in cats]); ax.set_ylabel("问题数"); ax.set_ylim(0,36)
    ax.legend(frameon=False,ncol=2,loc="upper right")
    atlas.save(fig,"03_dataset","评测题集覆盖四类科研问题","题型与集合划分来自真实题目记录；参数探索使用dev，三档主要检索结果使用独立test。",
               ["reports/datasets/dataset_manifest.json","reports/experiments/retrieval/m3/test/vector/results.json","reports/experiments/retrieval/m3/dev/vector/results.json"],"80题 = dev 20 + test 60；参考答案与证据由助手核对，真人答案评分尚未完成。","系统与数据")


def retrieval(atlas):
    modes=list(MODE); tests=[result(f"retrieval/m3/test/{m}") for m in modes]
    source=[f"reports/experiments/retrieval/m3/test/{m}/results.json" for m in modes]
    fig=atlas.page("重排提升历史测试集的检索覆盖", "RETRIEVAL ABLATION  |  同一60题、同一359片段索引", "检索实验")
    ax=axis(fig)
    grouped(ax,[MODE[m] for m in modes],["Hit@5","MRR","Recall@5"],[[r["metrics"][k] for r in tests] for k in ("hit_at_5","mrr","recall_at_5")],percent=True)
    diff=tests[-1]["metrics"]["hit_at_5"]-tests[0]["metrics"]["hit_at_5"]
    atlas.save(fig,"04_retrieval_ablation","重排提升历史测试集的检索覆盖",f"向量→混合+重排，Hit@5增加{diff*100:.1f}个百分点；该结论对应历史索引，不直接代表当前342片段版本。",source,"test n=60；历史359片段；MRR为源评测定义的倒数排名均值。","检索实验")
    fig=atlas.page("逐题重采样显示指标的不确定范围", "QUERY-LEVEL BOOTSTRAP  |  95% percentile interval · 5,000 resamples", "检索实验")
    axes=[fig.add_axes([.095+i*.29,.25,.235,.49]) for i in range(3)]
    for ax,key,title in zip(axes,("hit_at_5","mrr","recall_at_5"),("Hit@5","MRR","Recall@5")):
        for i,r in enumerate(tests):
            vals=[a[key] for a in r["records"]]
            mean=np.mean(vals); low,high=bootstrap(vals)
            ax.errorbar(i,mean,yerr=[[mean-low],[high-mean]],fmt="o",color=COLORS[i],capsize=6,lw=2,ms=9)
        ax.set_xticks(range(3),["Dense","Hybrid","Rerank"],fontsize=10)
        ax.set_ylim(0,1); ax.grid(axis="y"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title,loc="left")
    atlas.save(fig,"05_retrieval_uncertainty","逐题重采样显示指标的不确定范围","区间按每种模式的60道实际题目重采样估计；这是题目抽样不确定性，不是多次运行方差，也不是显著性检验。",source,"5,000次percentile bootstrap；固定seed 20261015；test n=60；359片段。","检索实验")
    fig=atlas.page("不同题型的检索难度存在差异", "CATEGORY ANALYSIS  |  Hit@5按问题类型分解", "检索实验")
    ax=axis(fig,(.14,.24,.72,.49)); cats=list(CAT)
    mat=np.array([[r["metrics"]["by_category"][c]["hit_at_5"] for c in cats] for r in tests])
    im=ax.imshow(mat,cmap="YlGnBu",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(4),[CAT[c] for c in cats]); ax.set_yticks(range(3),[MODE[m] for m in modes]); ax.grid(False)
    for i in range(3):
        for j in range(4): ax.text(j,i,f"{mat[i,j]:.1%}",ha="center",va="center",color="white" if mat[i,j]>.55 else INK,weight="bold",fontsize=14)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.035); cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
    atlas.save(fig,"06_category_heatmap","不同题型的检索难度存在差异","题型样本量不同，颜色展示Hit@5，不将不同题型直接作为可互换的难度标尺。",source,"test：事实22 / 对比16 / 综合11 / 推理11；历史359片段。","检索实验")
    fig=atlas.page("Top-1原始得分分布反映不同评分机制", "SCORE DISTRIBUTION  |  仅观察形态，不跨模型比较数值大小", "检索实验")
    axes=[fig.add_axes([.08+i*.30,.25,.25,.49]) for i in range(3)]
    for ax,r,m,col in zip(axes,tests,modes,COLORS):
        field={"vector":"vector_score","hybrid":"rrf_score","hybrid_rerank":"rerank_score"}[m]
        vals=[]
        for record in r["records"]:
            hits=record.get("retrieved",[])
            if hits and isinstance(hits[0].get(field),(float,int)): vals.append(hits[0][field])
        if len(vals)!=len(r["records"]): raise ValueError(f"Incomplete Top-1 field: {m}/{field}")
        ax.hist(vals,bins=12,color=col,edgecolor="white")
        ax.set_title(MODE[m],loc="left"); ax.set_xlabel(field,fontsize=10); ax.grid(axis="y")
    axes[0].set_ylabel("问题数")
    atlas.save(fig,"07_score_distribution","Top-1原始得分分布反映不同评分机制","向量相似度、RRF融合分与重排分不是同一量纲，不以原始得分的高低推断检索模式优劣。",source,"test n=60 / 模式；Top-1原始score从逐题检索记录提取。","检索实验")
    atlas.table("T01_retrieval","三档检索性能汇总","同一60题测试集；质量指标与计时范围分开说明",
        ["检索模式","n","Hit@5","MRR","Recall@5","均时 / ms","P95 / ms"],
        [[MODE[m],r["metrics"]["case_count"],*[f"{r['metrics'][k]:.4f}" for k in ("hit_at_5","mrr","recall_at_5")],f"{r['metrics']['latency_mean_ms']:.2f}",f"{r['metrics']['latency_p95_ms']:.2f}"] for m,r in zip(modes,tests)],
        source,"历史359片段；本表计时遵循原始离线检索实验口径，不等于在线问答总耗时。","所有指标来自同一冻结检索实验；不能与异构在线请求均时直接比较。")


def parsing_chunking(atlas):
    r=result("retrieval/parsers"); rows=r["summary"]; src=["reports/experiments/retrieval/parsers/results.json"]
    fig=atlas.page("解析速度与证据锚点保留率需要同时考虑", "PARSER COMPARISON  |  3个解析器 × 10篇真实论文", "文档与索引")
    ax=axis(fig,(.10,.25,.37,.49)); bars(ax,[v["parser"] for v in rows],[v["parse_ms"]/1000 for v in rows]); ax.set_ylabel("10篇累计解析耗时 / s")
    ax2=axis(fig,(.59,.25,.32,.49)); bars(ax2,[v["parser"] for v in rows],[v["anchor_retention"] for v in rows],BLUE,True); ax2.set_ylabel("119个标注锚点保留率"); ax2.set_ylim(0,1.2)
    atlas.save(fig,"08_parser_tradeoff","解析速度与证据锚点保留率需要同时考虑","锚点保留是证据片段的代理指标；表格/公式块数是检测数量，不能单独解释为语义准确率或版面还原率。",src,"n=10篇 / 解析器；169页；119个标注的来源锚点（非本轮真人答案评分）。","文档与索引")
    atlas.table("T02_parsers","解析器的结构与证据保留统计","统计口径为真实解析输出；检测数量不是正确率",
        ["解析器","耗时 / s","锚点保留","表格块","公式块","含文字页","带坐标块"],
        [[v["parser"],f"{v['parse_ms']/1000:.2f}",f"{v['matched_anchors']}/{v['total_anchors']}",v["table_blocks"],v["formula_blocks"],v["pages_with_text"],v["bbox_blocks"]] for v in rows],src,"3×10篇真实解析；来源锚点由助手核对。","PyPDF2未提供块坐标；不将0块简单视为整页正文丢失。")
    chunk_names=["fixed-256","fixed-512","fixed-1024","recursive-512","semantic-512"]
    chunks=[result(f"retrieval/chunking/m3e/{s}") for s in chunk_names]
    labels=["固定 256","固定 512","固定 1024","递归 512","语义 512"]
    sources=[f"reports/experiments/retrieval/chunking/m3e/{s}/results.json" for s in chunk_names]
    fig=atlas.page("分块策略改变证据命中与排序表现", "CHUNKING ABLATION  |  同一m3e模型、同一20题开发集", "文档与索引")
    ax=axis(fig)
    grouped(ax,labels,["Hit@5","MRR","Recall@5"],[[v["metrics"][k] for v in chunks] for k in ("hit_at_5","mrr","recall_at_5")],percent=True)
    atlas.save(fig,"09_chunking","分块策略改变证据命中与排序表现","固定、递归与语义切分使用各自真实索引；这里是dev探索结果，不能代替最终test性能。",sources,"dev n=20；m3e-base；长度参数为应用token预算，详见配置和分块报告。","文档与索引")
    atlas.table("T03_chunking","五种分块配置的开发集结果","混合检索（Dense + BM25 + RRF）；实际分块配置与同一m3e模型",
        ["策略","n","Hit@5","MRR","Recall@5","检索均时 / ms"],
        [[label,v["metrics"]["case_count"],*[f"{v['metrics'][k]:.4f}" for k in ("hit_at_5","mrr","recall_at_5")],f"{v['metrics']['latency_mean_ms']:.2f}"] for label,v in zip(labels,chunks)],sources,"dev n=20 / 配置；不同分块索引；未复用虚构编码耗时。","分块索引规模、编码统计与完整配置可在配套数据表追溯。")
    bge=result("retrieval/chunking/bge-zh/recursive-512"); models=[chunks[3],bge]
    msources=[sources[3],"reports/experiments/retrieval/chunking/bge-zh/recursive-512/results.json"]
    fig=atlas.page("Embedding比较采用一致的递归分块", "MODEL COMPARISON  |  m3e-base vs BGE-large-zh", "文档与索引")
    ax=axis(fig)
    grouped(ax,["m3e-base","BGE-large-zh"],["Hit@5","MRR","Recall@5"],[[v["metrics"][k] for v in models] for k in ("hit_at_5","mrr","recall_at_5")],percent=True)
    atlas.save(fig,"10_embedding","Embedding比较采用一致的递归分块","模型比较使用同一342片段与20题dev；向量库/主测试中的359片段实验另有范围，不混合排名。",msources,"dev n=20；342片段；recursive-512；模型速度另列编码/查询数据表。","文档与索引")
    atlas.table("T04_embeddings","Embedding检索质量与排序耗时","排序计时使用已预计算向量，排除查询编码和模型加载",
        ["模型","Hit@5","MRR","Recall@5","排序均时 / ms","排序P95 / ms"],
        [[name,*[f"{v['metrics'][k]:.4f}" for k in ("hit_at_5","mrr","recall_at_5")],f"{v['metrics']['latency_mean_ms']:.2f}",f"{v['metrics']['latency_p95_ms']:.2f}"] for name,v in zip(["m3e-base","BGE-large-zh"],models)],msources,"dev n=20；ranking_only_precomputed_vectors；实际查询编码另见编码速度图。","编码总时间、模型加载时间与摊销查询时间是不同计时阶段，不应相加后冒充在线均时。")
    stores=[result(f"retrieval/vectorstores/{s}") for s in ("chroma","faiss")]
    vsources=[f"reports/experiments/retrieval/vectorstores/{s}/results.json" for s in ("chroma","faiss")]
    fig=atlas.page("向量库比较复用同一份真实向量", "VECTOR STORE COMPARISON  |  控制语料、向量与问题集", "文档与索引")
    ax=axis(fig,(.10,.25,.38,.49)); grouped(ax,["Chroma","FAISS"],["Hit@5","MRR","Recall@5"],[[v["metrics"][k] for v in stores] for k in ("hit_at_5","mrr","recall_at_5")],percent=True)
    ax2=axis(fig,(.61,.25,.29,.49)); bars(ax2,["Chroma","FAISS"],[v["metrics"]["latency_mean_ms"] for v in stores],BLUE); ax2.set_ylabel("检索均时 / ms")
    atlas.save(fig,"11_vectorstores","向量库比较复用同一份真实向量","同一359片段与真实Embedding；结果反映当前数据规模和运行条件，不能外推大规模数据库吞吐。",vsources,"dev n=20 / 后端；359片段；模型推理复用。","文档与索引")
    atlas.table("T05_vectorstores","向量库质量与时延汇总","Chroma与FAISS使用相同的真实向量与查询",
        ["后端","n","Hit@5","MRR","Recall@5","均时 / ms","P95 / ms"],
        [[name,v["metrics"]["case_count"],*[f"{v['metrics'][k]:.4f}" for k in ("hit_at_5","mrr","recall_at_5")],f"{v['metrics']['latency_mean_ms']:.2f}",f"{v['metrics']['latency_p95_ms']:.2f}"] for name,v in zip(["Chroma","FAISS"],stores)],vsources,"dev n=20；历史359片段。","本表仅描述已有小规模后端实验。")


def rrf(atlas):
    r=result("retrieval/dev-parameters"); rows=r["summary"]; source=["reports/experiments/retrieval/dev-parameters/results.json"]
    fig=atlas.page("检索参数探索保留完整的比较矩阵", "RRF SENSITIVITY  |  融合常数 × 返回条数", "检索实验")
    for i,key in enumerate(("hit","recall")):
        ax=fig.add_axes([.12+i*.44,.26,.31,.47])
        mat=np.array([[next(v[key] for v in rows if v["rrf_k"]==k and v["metric_k"]==n) for n in (3,5,10)] for k in (20,60,100)])
        ax.imshow(mat,cmap="YlGnBu",vmin=0,vmax=1,aspect="auto"); ax.grid(False)
        ax.set_xticks(range(3),["Top-3","Top-5","Top-10"]); ax.set_yticks(range(3),["RRF 20","RRF 60","RRF 100"])
        ax.set_title("Hit@K" if key=="hit" else "Recall@K",loc="left")
        for y in range(3):
            for x in range(3): ax.text(x,y,f"{mat[y,x]:.1%}",ha="center",va="center",color="white" if mat[y,x]>.55 else INK,fontsize=15,weight="bold")
    atlas.save(fig,"12_rrf_grid","检索参数探索保留完整的比较矩阵","在同一20题dev上，RRF 60→20使MRR@20从0.3148变为0.3247；Hit@5与Recall@5不变，不能夸大为全面提升。",source,"20道dev题；复用真实排名；额外模型调用0次；Top-K改变评价截断位置。","检索实验")
    atlas.table("T06_rrf","RRF与Top-K参数网格完整结果","MRR@20使用同一候选深度，不随图中的K重新定义",
        ["RRF k","评价K","n","Hit@K","Recall@K","MRR@20"],
        [[v["rrf_k"],v["metric_k"],v["query_count"],f"{v['hit']:.4f}",f"{v['recall']:.4f}",f"{v['mrr_at_20']:.4f}"] for v in rows],source,"dev n=20；3×3网格；无额外Embedding调用。","一次受控优化已记录；这里的微小变化不构成统计显著性结论。")


def baseline_quality(atlas):
    path=EXP/"generation/baseline/ai_review_summary.json"; r=load(path)
    sources=[shortpath(path),"reports/experiments/generation/baseline/human_scoring_60.csv"]
    keys=["correctness_1_to_5","completeness_1_to_5","citation_quality_1_to_5","fluency_1_to_5"]
    fig=atlas.page("答案质量评审揭示完整性与引用短板", "AI-ASSISTED QUALITY REVIEW  |  60题实际短答案 · 1–5分", "答案质量")
    ax=axis(fig,(.15,.25,.69,.48)); vals=[r["means"][k] for k in keys]
    ax.hlines(range(4),1,5,color=GRID,lw=3)
    ax.scatter(vals,range(4),s=220,c=COLORS[:4],zorder=3)
    for i,v in enumerate(vals): ax.text(v+.13,i,f"{v:.2f}",va="center",fontsize=15,weight="bold")
    ax.set_yticks(range(4),QLABELS); ax.invert_yaxis(); ax.set_xlim(.95,5.45); ax.set_xticks(range(1,6)); ax.set_xlabel("AI辅助评审均分 / 5"); ax.grid(False)
    atlas.save(fig,"13_baseline_quality","答案质量评审揭示完整性与引用短板","评分由Codex AI辅助完成，保留逐题证据与评语；可用于质量分析，不能改称真人评分。流畅性较高不表示事实与引用可靠。",sources,"AI n=60；真人评分n=0；历史359片段冻结检索；256输出token上限。","答案质量")
    fig=atlas.page("评分分布比单一平均分更完整", "ORDINAL SCORE DISTRIBUTION  |  逐维呈现1–5分占比", "答案质量")
    ax=axis(fig,(.14,.25,.75,.48)); bottoms=np.zeros(4); cmap=["#CD7668","#DCA36D","#E1CF9A","#7FB3AC",TEAL]
    for score,color in zip(range(1,6),cmap):
        v=np.array([int(r["score_distribution"][k].get(str(score),0)) for k in keys])/r["case_count"]
        ax.barh(range(4),v,left=bottoms,color=color,height=.57,label=f"{score}分",edgecolor="white")
        for i,x in enumerate(v):
            if x>.065: ax.text(bottoms[i]+x/2,i,f"{x:.0%}",ha="center",va="center",color="white" if score in (1,5) else INK,fontsize=11)
        bottoms+=v
    ax.set_yticks(range(4),QLABELS); ax.invert_yaxis(); ax.set_xlim(0,1); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.grid(False)
    ax.legend(frameon=False,ncol=5,loc="upper left",bbox_to_anchor=(0,1.18))
    atlas.save(fig,"14_quality_distribution","评分分布比单一平均分更完整","1–5分为有序评分；分布展示回答质量差异，不假设评分间隔对应等量的真实能力变化。",sources,"每维n=60；AI辅助评审；分布使用逐题评分原值。","答案质量")
    with (EXP/"generation/baseline/human_scoring_60.csv").open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    # Column names intentionally follow the real exported rubric, not assumed categories.
    keymap={"correctness":"correctness_1_to_5","completeness":"completeness_1_to_5","citation_quality":"citation_quality_1_to_5","fluency":"fluency_1_to_5"}
    fig=atlas.page("质量评审按问题类型展开", "QUESTION-TYPE REVIEW  |  四类问题的AI评分均值", "答案质量")
    ax=axis(fig,(.14,.25,.72,.48)); cats=list(CAT); mat=[]
    for cat in cats:
        subset=[x for x in rows if x.get("category")==cat]
        mat.append([np.mean([float(x[keymap[k]]) for x in subset]) if subset else np.nan for k in QUALITY])
    mat=np.asarray(mat); im=ax.imshow(mat,cmap="YlGnBu",vmin=1,vmax=5,aspect="auto"); ax.grid(False)
    ax.set_xticks(range(4),QLABELS); ax.set_yticks(range(4),[f"{CAT[c]}  n={sum(x.get('category')==c for x in rows)}" for c in cats])
    for y in range(4):
        for x in range(4): ax.text(x,y,f"{mat[y,x]:.2f}",ha="center",va="center",color="white" if mat[y,x]>3.5 else INK,fontsize=14,weight="bold")
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    atlas.save(fig,"15_quality_by_category","质量评审按问题类型展开","不同题型的样本量与内容不同，均值用于定位薄弱环节，不推断系统在某类任务上已稳定达到该分数。",sources,"AI辅助评审；事实22 / 对比16 / 综合11 / 推理11；每维1–5分。","答案质量")
    atlas.table("T07_baseline_quality","60题答案质量评审汇总","评分来源：Codex AI辅助评审；不计入真人评分人数",
        ["评价维度","n","均分 / 5","1分","2分","3分","4分","5分"],
        [[label,r["case_count"],f"{r['means'][k]:.2f}",*[r["score_distribution"][k].get(str(v),0) for v in range(1,6)]] for label,k in zip(QLABELS,keys)],sources,"评分日期2026-10-14；历史短答案实验；真人评分待完成。","完整逐题回答、证据、评分和Bad Case记录保存在配套工作簿及源CSV。")


def generation_quality(atlas):
    path=EXP/"generation/quality-comparison/ai_quality_summary.json"
    if not path.exists():
        raise FileNotFoundError("Complete AI parameter/Prompt review before building the final atlas.")
    r=load(path); conditions=r["conditions"]
    sources=[shortpath(path),"reports/experiments/generation/quality-comparison/ai_quality_rows.json"]
    names={"baseline_current":"基线 T=.1 / p=.9 / k=40","temperature_0.0":"Temperature = 0.0","temperature_0.5":"Temperature = 0.5","top_p_0.7":"Top-p = 0.7","top_k_20":"Top-k = 20","naive_old_prompt":"简化提示词","current_prompt_reused":"证据约束提示词","current_prompt":"证据约束提示词","new_prompt":"证据约束提示词"}
    for study,slug,title in (("parameters","16_generation_parameters","生成参数的质量差异需结合实际回答解释"),("prompt","17_prompt_quality","提示词比较同时观察事实与引用表现")):
        cs=[c for c in conditions if c["study"]==study]
        if not cs: continue
        def q(c,k):
            v=c.get(k+"_mean")
            if v is None: v=c.get("means",{}).get(k)
            return float(v)
        labels=[names.get(c["condition"],c["condition"]) for c in cs]
        fig=atlas.page(title,"AI QUALITY COMPARISON  |  同一4道dev题 · 每条件一次实际生成", "生成与Prompt")
        ax=axis(fig,(.11,.28,.79,.43)); grouped(ax,labels,QLABELS,[[q(c,k) for c in cs] for k in QUALITY],ylabel="AI辅助评分 / 5",ylim=(0,5.8))
        ax.tick_params(axis="x",labelsize=9 if study=="parameters" else 12)
        atlas.save(fig,slug,title,"已有回答按统一量表评审，重复回答复用评分；样本仅4题且无固定采样seed，结果为探索性比较，不宣称稳定最优或统计显著。",sources,"dev n=4 / 条件；AI评审；历史359片段冻结检索；256输出token上限。","生成与Prompt")
        atlas.table("T08_parameters" if study=="parameters" else "T09_prompt", "生成参数质量与成本汇总" if study=="parameters" else "Prompt质量与成本汇总",
            "质量来自AI辅助评审；时间与Token来自原始真实生成记录",
            ["配置 / 提示词","n","正确性","完整性","引用","流畅性","均时 / ms","输出token/条"],
            [[names.get(c["condition"],c["condition"]),c["n"],*[f"{q(c,k):.2f}" for k in QUALITY],f"{c['latency_mean_ms']:.1f}" if c.get("latency_mean_ms") is not None else None,f"{c['completion_tokens_mean']:.1f}" if c.get("completion_tokens_mean") is not None else None] for c in cs],sources,"dev n=4 / 条件；单次采样；不据此声称普遍参数最优。","完整Temperature、Top-p、Top-k参数与逐题评语见配套数据表。",widths=[.30,.06,.09,.09,.09,.09,.14,.14])


def runtime(atlas):
    a=result("generation/agent"); src=["reports/experiments/generation/agent/results.json"]
    rows=[a["condition_metrics"][c] for c in ("fixed_rag","agent")]
    fig=atlas.page("Agent路由带来不同的Token与时延结构", "AGENT VS FIXED RAG  |  5个路由探针 · 每种方式各运行一次", "Agent与性能")
    ax=axis(fig,(.10,.25,.35,.48)); bars(ax,["固定RAG","Agent"],[r["latency_mean_ms"]/1000 for r in rows]); ax.set_ylabel("均时 / s")
    ax2=axis(fig,(.59,.25,.32,.48)); grouped(ax2,["固定RAG","Agent"],["输入token","输出token"],[[r[k]/r["run_count"] for r in rows] for k in ("prompt_tokens_total","completion_tokens_total")]); ax2.set_ylabel("平均token / 次")
    atlas.save(fig,"18_agent","Agent路由带来不同的Token与时延结构","5个探针上的Agent工具路由精确匹配率为1.00；这是小样本路由结果，不等于所有问题的答案正确率。",src,"n=5 / 方式；Agent包含路由与本地工具，固定RAG为对照；不外推大规模收益。","Agent与性能")
    atlas.table("T10_agent","Agent与固定RAG的探针结果","相同5个探针；每种方式5次执行",
        ["方式","次数","均时 / ms","P95 / ms","输入token合计","输出token合计"],
        [[label,r["run_count"],f"{r['latency_mean_ms']:.2f}",f"{r['latency_p95_ms']:.2f}",r["prompt_tokens_total"],r["completion_tokens_total"]] for label,r in zip(["固定RAG","Agent"],rows)],src,"Agent路由exact-match=1.00；小样本功能探针。","路由匹配与生成质量是不同指标；不得把路由匹配率写成答案正确率。")
    p=result("generation/parallel"); src=["reports/experiments/generation/parallel/results.json"]
    fig=atlas.page("独立本地工具并行的实测耗时", "LOCAL TOOL CONCURRENCY  |  10轮配对请求 × 2种执行方式", "Agent与性能")
    ax=axis(fig,(.13,.25,.72,.48)); rs={c:[v["elapsed_ms"] for v in p["records"] if v["condition"]==c] for c in ("workers_1","workers_2")}
    for i,(v1,v2) in enumerate(zip(rs["workers_1"],rs["workers_2"])):
        ax.plot([0,1],[v1,v2],color="#CBD6DF",alpha=.7,zorder=1)
    for i,(c,label) in enumerate(zip(rs,["串行 · 1 worker","并行 · 2 workers"])):
        vs=rs[c]; ax.scatter(np.full(len(vs),i),vs,color=COLORS[i],s=42,zorder=3)
        ax.scatter(i,np.mean(vs),s=180,marker="D",c=NAVY,edgecolors="white",zorder=4)
    ax.set_xticks([0,1],["串行 · 1 worker","并行 · 2 workers"]); ax.set_xlim(-.4,1.4); ax.set_ylabel("一组本地工具耗时 / ms")
    atlas.save(fig,"19_parallel","独立本地工具并行的实测耗时","每条连线对应同一轮工具请求；菱形为均值。仅测计算器与关键词等本地工具，不包含LLM并发推理。",src,"10轮 / 方式；共20次；无LLM调用；工具任务非常短，线程调度开销不可忽略。","Agent与性能")
    c=result("generation/cache"); src=["reports/experiments/generation/cache/results.json"]
    fig=atlas.page("精确重复缓存避免重复生成", "CACHE FUNCTIONAL PROBE  |  一道问题的冷请求与精确重复", "Agent与性能")
    ax=axis(fig,(.12,.25,.72,.48)); keys=["cold_exact_query","hot_exact_repeat"]
    vals=[c["condition_metrics"][k]["latency_mean_ms"] for k in keys]
    ax.barh(["首次冷请求","精确重复命中"],vals,color=[BLUE,TEAL],height=.48); ax.set_xscale("log"); ax.set_xlabel("耗时 / ms（对数坐标）"); ax.grid(False); ax.grid(axis="x")
    ax.set_xlim(min(vals)/2,max(vals)*3)
    for i,v in enumerate(vals): ax.text(v*1.15,i,f"{v:,.2f} ms",va="center",fontsize=13)
    ax.invert_yaxis()
    atlas.save(fig,"20_cache","精确重复缓存避免重复生成","本图仅有一次冷请求和一次精确重复，是缓存功能证据；不能据此宣称语义近邻缓存命中率或稳定加速倍数。",src,"n=1问题 / 2次请求；首次生成含LLM调用，重复命中新增token=0。","Agent与性能")
    online=result("online-functionality"); src=["reports/experiments/online-functionality/results.json"]
    records=online.get("records",[])
    fig=atlas.page("在线请求耗时分布保留任务差异", "ONLINE FUNCTIONALITY  |  10次真实请求的端到端记录", "Agent与性能")
    ax=axis(fig,(.12,.26,.78,.48))
    vs=[r.get("elapsed_ms",r.get("latency_ms",0))/1000 for r in records]
    bs=ax.bar(range(len(vs)),vs,color=[COLORS[i%3] for i in range(len(vs))],width=.64)
    ax.set_xticks(range(len(vs)),[str(i+1).zfill(2) for i in range(len(vs))]); ax.set_xlabel("请求序号（不同任务）"); ax.set_ylabel("端到端耗时 / s")
    for b,v in zip(bs,vs): ax.text(b.get_x()+b.get_width()/2,v+.6,f"{v:.1f}",ha="center",fontsize=10)
    if vs:
        mean=np.mean(vs); ax.axhline(mean,color=NAVY,lw=1.2,linestyle="--",label=f"记录均值 {mean:.2f} s"); ax.legend(frameon=False)
        ax.set_ylim(0,max(vs)*1.2)
    atlas.save(fig,"21_online_latency","在线请求耗时分布保留任务差异","10次请求的任务、工具与上下文不同；平均延迟是该批功能验证的描述统计，不是受控负载测试或正式性能上界。",src,"真实在线n=10；当前实验记录时间与索引范围详见源文件。","Agent与性能")
    # Long-form metrics appear in the workbook; this summary keeps scope explicit.
    rows=[]
    for title,study in (("本地工具串行",p["condition_metrics"]["workers_1"]),("本地工具并行",p["condition_metrics"]["workers_2"]),("缓存冷请求",c["condition_metrics"]["cold_exact_query"]),("精确重复命中",c["condition_metrics"]["hot_exact_repeat"])):
        rows.append([title,study["run_count"],f"{study['latency_mean_ms']:.2f}",study["prompt_tokens_total"],study["completion_tokens_total"]])
    atlas.table("T11_runtime","本地工具与缓存功能探针汇总","不同实验分组展示；不横向合并为统一性能排名",
        ["实验条件","n","均时 / ms","输入token合计","输出token合计"],rows,
        ["reports/experiments/generation/parallel/results.json","reports/experiments/generation/cache/results.json"],"并行：10轮 / 条件；缓存：1次 / 条件；全部保留各自计时范围。","缓存精确重复只说明这一样例命中；并行实验不调用LLM。")


def additional_results(atlas):
    data=load(ROOT/"reports/presentation/source-data/experiments.json")["tables"]
    es=data["embeddings"]["rows"]
    source=["reports/experiments/retrieval/encoding/m3e/results.json","reports/experiments/retrieval/encoding/bge-zh/results.json"]
    fig=atlas.page("Embedding速度比较区分推理阶段", "ENCODING PROFILE  |  同20题查询编码与真实文档编码记录", "文档与索引")
    ax=axis(fig,(.10,.25,.35,.48)); bars(ax,["m3e-base","BGE-large-zh"],[v["query_encoding_amortized_ms"] for v in es]); ax.set_ylabel("查询编码摊销耗时 / ms")
    ax2=axis(fig,(.60,.25,.30,.48)); bars(ax2,["m3e-base","BGE-large-zh"],[v["document_texts_per_second"] for v in es],BLUE); ax2.set_ylabel("文档编码吞吐 / 文本每秒")
    atlas.save(fig,"22_encoding_speed","Embedding速度比较区分推理阶段","左图为相同20个查询的真实编码总耗时除以20；右图编码文本集合不同（m3e为五种分块去重1836条，BGE为342条），仅作过程描述。",source,"CPU 8线程；batch=1；512-token窗口；摊销查询时间不是逐请求在线延迟。","文档与索引")
    atlas.table("T14_encoding","Embedding实际编码与加载统计","相同20题查询可比较；文档集合不同，不进行总耗时直接排名",
        ["模型","维度","加载 / s","查询均摊 / ms","文档条数","编码 / s","文本 / s"],
        [[name,v["dimension"],f"{v['model_load_ms']/1000:.2f}",f"{v['query_encoding_amortized_ms']:.2f}",v["unique_document_text_count"],f"{v['document_encoding_total_ms']/1000:.2f}",f"{v['document_texts_per_second']:.3f}"] for name,v in zip(["m3e-base","BGE-large-zh"],es)],source,"8 CPU线程；batch=1；模型加载、文档编码、查询编码为不同阶段。","m3e文档为五分块方案的去重文本；BGE文档为递归方案，长度分布不同。")
    q=load(EXP/"generation/quality-comparison/ai_quality_summary.json"); cs=[c for c in q["conditions"] if c["study"]=="parameters"]
    source=["reports/experiments/generation/quality-comparison/ai_quality_summary.json","reports/experiments/generation/parameters/results.json"]
    fig=atlas.page("质量变化与采样成本需要联合解读", "GENERATION TRADE-OFF  |  每条件4题 · 单次采样", "生成与Prompt")
    ax=axis(fig,(.10,.28,.78,.43)); labels=["基线","T=0.0","T=0.5","p=0.7","k=20"]
    for i,c in enumerate(cs):
        score=np.mean([c[k+"_mean"] for k in QUALITY[:3]])
        ax.scatter(c["latency_mean_ms"]/1000,score,s=c["completion_tokens_mean"]*3,color=COLORS[i],alpha=.85,edgecolor="white",lw=1.5)
        ax.annotate(labels[i],(c["latency_mean_ms"]/1000,score),xytext=(10,(i%2)*17-6),textcoords="offset points",fontsize=11,color=INK)
    ax.set_xlabel("生成均时 / s"); ax.set_ylabel("正确性 / 完整性 / 引用三维均分"); ax.set_ylim(2.9,4.35); ax.margins(x=.23); ax.grid(True)
    atlas.save(fig,"23_generation_tradeoff","质量变化与采样成本需要联合解读","气泡面积对应平均输出token；纵轴仅为三个同量表维度的描述性均值，不是额外人工标注指标。样本量不足以支持参数稳定最优结论。",source,"AI评审；dev n=4 / 条件；延迟受输出长度与运行顺序影响。","生成与Prompt")
    online=result("online-functionality"); records=online["records"]; source=["reports/experiments/online-functionality/results.json"]
    fig=atlas.page("Token与推理轮次保留逐请求记录", "ONLINE RESOURCE PROFILE  |  10个异构请求", "Agent与性能")
    ax=axis(fig,(.10,.25,.78,.48)); x=np.arange(len(records)); inp=[r["input_tokens"] for r in records]; out=[r["output_tokens"] for r in records]
    ax.bar(x,inp,color=BLUE,label="输入token",width=.65); ax.bar(x,out,bottom=inp,color=TEAL,label="输出token",width=.65)
    ax.set_xticks(x,[str(i+1).zfill(2) for i in x]); ax.set_ylabel("Token数 / 请求"); ax.set_xlabel("请求序号（与时延图一致）"); ax.legend(frameon=False,ncol=2)
    for i,r in enumerate(records): ax.text(i,inp[i]+out[i]+100,f"{r['agent_steps']}轮",ha="center",fontsize=9)
    ax.set_ylim(0,max(np.array(inp)+np.array(out))*1.2)
    atlas.save(fig,"24_online_resources","Token与推理轮次保留逐请求记录","总输入43,908 token、输出4,634 token，平均Agent轮次1.2；任务与上下文不同，不解释为同一任务的受控开销对比。",source,"10次真实在线请求；Token来源与估算标记保存在原始日志。","Agent与性能")
    metrics=online["metrics"]
    atlas.table("T15_online","真实在线功能样本的资源统计","这些描述统计只代表本批10次异构请求",
        ["样本数","均时 / s","输入token合计","输出token合计","平均Agent轮次"],
        [[metrics["sample_count"],f"{metrics['latency_mean_ms']/1000:.2f}",metrics["input_tokens_total"],metrics["output_tokens_total"],f"{metrics['average_agent_steps']:.2f}"]],source,"不等同于受控性能基准、并发负载测试或端到端准确率。","每次请求的具体问题、时延、工具执行和警告见配套数据工作簿。")
    baseline=EXP/"generation/baseline/human_scoring_60.csv"
    with baseline.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    counts=Counter(label.strip() for row in rows for label in re.split(r"[，,、;；]",row["bad_case_type"] or "未标记") if label.strip())
    fig=atlas.page("质量诊断同时记录失败类型", "BAD CASE PROFILE  |  60题AI评审的多标签统计", "答案质量")
    ax=axis(fig,(.22,.25,.66,.48)); labs=list(counts); vals=[counts[k] for k in labs]
    ax.barh(labs,vals,color=[COLORS[i%5] for i in range(len(labs))],height=.57); ax.invert_yaxis(); ax.grid(False); ax.grid(axis="x"); ax.set_xlabel("带该标签的回答数（一题可有多个标签）"); ax.set_xlim(0,max(vals)*1.18)
    for i,v in enumerate(vals): ax.text(v+.25,i,str(v),va="center",fontsize=12)
    atlas.save(fig,"25_bad_case_profile","质量诊断同时记录失败类型","类别直接从评分表拆分多标签；同一回答可同时计入检索、生成与引用问题，各类合计可以超过60。未标记不等于全面无误。",[shortpath(baseline)],"AI辅助评审n=60；历史短答案；同一回答可贡献多个标签。","答案质量")
    atlas.table("T16_bad_case_actions","Bad Case根因与改进记录","四类课程Bad Case与token窗口工程问题",
        ["问题类型","观察到的根因","已实施的改进与边界"],
        [["检索失败","实体跨语言与相近架构干扰排名","分块/模型/RRF比较；仍有漏召回"],["生成错误","自由文本未约束结构与数值归属","JSON schema与有界修复；语义仍需评审"],["引用错误","模型自写页码、合并引用未展开","服务器物理页追溯与来源逐项核对"],["Agent决策错误","文档范围再次交由规划模型猜测","服务器固定明确范围；仅5个路由探针"],["长文本编码失败","decode后再encode改变窗口长度","原token-ID完整分窗；真实模型回归"]],
        ["reports/experiments/bad-cases/actual-bad-cases.json"],"每类的实际请求、原始记录与改进证据详见源表；不声称已消除全部同类问题。","修复过程与剩余回答质量问题同时保留，方便说明工程迭代与研究局限。",widths=[.18,.35,.47])


def caveats_and_configs(atlas):
    source=["reports/experiments/bad-cases/actual-bad-cases.json","reports/requirements/course-audit.json"]
    atlas.table("T12_experimental_scope","展示与汇报应保留的实验边界","不同研究回答不同问题，使用各自的样本与版本",
        ["研究 / 结果","范围","可支持的结论"],
        [["三档检索测试","60题 / 历史359片段","同一冻结索引的检索模式比较"],["生成参数 / Prompt","4题 / 条件 / 单次生成","探索性质量与成本比较"],["60题质量评分","AI辅助 / 256输出token上限","历史短答案质量诊断"],["Agent路由探针","5个探针","给定探针的工具路由匹配"],["本地工具并行","10轮 / 条件 / 无LLM","短本地任务的并行耗时"],["缓存探针","1道题 / 冷+精确重复","该精确重复的缓存功能"],["在线功能记录","10个异构真实请求","这批请求的流程、时延与Token"],["当前生产版本","342片段","不能直接套用359片段正式成绩"]],source,"课程明确要求真人评分；本轮输出保持AI来源。Docker运行状态仍为未验证。","Bad Case根因与修复记录另见配套表；当前图册不改写历史实验和课程验收事实。",widths=[.25,.29,.46])
    config=result("retrieval/m3/test/hybrid_rerank")["config"]
    wanted=[("混合检索","candidate_k","候选数量"),("混合检索","result_k","最终来源数"),("混合检索","rrf_k","历史基线RRF常数"),("分块","chunk_size","递归块目标"),("分块","chunk_overlap","重叠预算"),("生成默认","temperature","Temperature"),("生成默认","top_p","Top-p"),("生成默认","top_k_sampling","采样Top-k"),("生成默认","context_tokens","上下文token预算"),("生成默认","answer_tokens","生产配置输出上限"),("Agent","max_agent_steps","最大推理轮次")]
    atlas.table("T13_key_parameters","关键参数与实验配置","源配置完整保存在工作簿和JSON；生成研究使用独立256-token上限",
        ["模块","配置键","含义","源记录值"],[[group,key,meaning,config.get(key)] for group,key,meaning in wanted],
        ["reports/experiments/retrieval/m3/test/hybrid_rerank/results.json","reports/experiments/generation/parameters/results.json"],"历史正式检索运行配置；当前生产配置及其他研究配置分别保留，不能默认为同一运行。","Top-k采样与检索返回Top-K含义不同；生产answer_tokens=768不等于历史生成研究的256。",widths=[.16,.28,.36,.20])


def gallery(atlas):
    cards=[]
    for i,item in enumerate(atlas.items):
        e=lambda s:html.escape(str(s),quote=True)
        cards.append(f'''<article class="card" data-category="{e(item['category'])}" data-search="{e(item['title']+' '+item['caption'])}">
<button class="image" data-index="{i}" aria-label="放大 {e(item['title'])}"><img loading="lazy" src="{item['png']}" alt="{e(item['title'])}"></button>
<div class="body"><div class="meta">{e(item['category'])} · PDF {item['pdf_page']:02d}</div><h2>{e(item['title'])}</h2><p>{e(item['caption'])}</p><small>{e(item['scope'])}</small><div class="links"><a href="{item['png']}" download>PNG · 300 dpi</a><a href="{item['svg']}" download>SVG · 矢量</a><button class="present" data-index="{i}">全屏展示 ↗</button></div><details><summary>数据来源</summary><ul>{''.join('<li>'+e(s)+'</li>' for s in item['sources'])}</ul></details></div></article>''')
    cats=list(dict.fromkeys(item["category"] for item in atlas.items))
    payload=json.dumps(atlas.items,ensure_ascii=False).replace("</","<\\/")
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>科研助理 · 学术实验图册</title>
<style>:root{--navy:#17324d;--teal:#15877f;--ink:#23394d;--muted:#687c8d;--line:#e1e9ef}*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.65 'Microsoft YaHei','Segoe UI',sans-serif}header{background:var(--navy);color:white;padding:64px max(5vw,24px) 48px}.eyebrow{font-size:12px;letter-spacing:3px;color:#8ed0c5}h1{font-size:clamp(28px,4vw,48px);margin:15px 0}header p{max-width:850px;color:#d0dce5}.stats{display:flex;gap:32px;margin-top:26px}.stats strong{display:block;font-size:28px;color:#a8dbd2}.toolbar{position:sticky;top:0;z-index:3;background:#ffffffed;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:16px 5vw;display:flex;gap:12px;align-items:center;flex-wrap:wrap}input,select{border:1px solid #cbd8e1;border-radius:8px;padding:10px 14px;font:inherit;background:white}input{flex:1;min-width:200px}.toolbar a{color:var(--teal);font-weight:bold;text-decoration:none}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:26px;padding:30px 5vw 60px}.card{background:white;border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 5px 18px #14314b06}.image{border:0;background:white;cursor:zoom-in;display:block;width:100%;padding:0}.image img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain}.body{padding:22px}.meta{color:var(--teal);font-size:12px;font-weight:bold;letter-spacing:.5px}h2{font-size:19px;margin:8px 0}.body p{color:#50677a;margin:10px 0;font-size:14px}.body small{display:block;font-size:12px;color:var(--muted)}.links{display:flex;gap:15px;margin-top:18px;align-items:center;flex-wrap:wrap}.links a,.present{color:var(--teal);font:600 13px 'Microsoft YaHei',sans-serif;text-decoration:none}.present{border:0;background:#edf6f4;padding:8px 11px;border-radius:6px;cursor:pointer}details{font-size:11px;color:var(--muted);margin-top:15px}li{overflow-wrap:anywhere}dialog{max-width:100vw;width:100vw;max-height:100vh;height:100vh;border:0;padding:0;background:#101f2e;color:white}dialog::backdrop{background:#101f2e}dialog img{width:100%;height:calc(100vh - 68px);object-fit:contain}nav{height:68px;display:flex;align-items:center;gap:20px;padding:12px 24px}nav button{background:#ffffff15;color:white;border:1px solid #ffffff30;border-radius:6px;padding:8px 18px;font:inherit;cursor:pointer}#slideTitle{flex:1;font-size:14px}footer{padding:0 5vw 35px;color:var(--muted);font-size:12px}#empty{padding:50px;display:none;text-align:center}</style>
<header><div class="eyebrow">NAU / RESEARCH ASSISTANT / EXPERIMENT ATLAS</div><h1>以证据呈现系统能力</h1><p>智能科研助理：基于RAG与Agent的论文知识库问答系统。统一的学术图表与可编辑数据，覆盖真实实验、质量评审与关键参数。保留样本量、指标口径和版本边界。</p><div class="stats"><div><strong>__COUNT__</strong>幅图与表</div><div><strong>300 dpi</strong>高清导出</div><div><strong>SVG / PDF</strong>矢量输出</div></div></header>
<div class="toolbar"><input id="search" placeholder="搜索指标、实验、参数或结论…" aria-label="搜索图表"><select id="category" aria-label="筛选类别"><option value="all">全部主题</option>__OPTIONS__</select><a href="科研助理实验图表集.pdf" download>下载PDF图册</a><a href="学术图表数据与结果.xlsx" download>下载Excel数据</a><span id="count"></span></div><main class="grid">__CARDS__</main><p id="empty">没有匹配的图表，请调整搜索条件。</p><footer>AI辅助评分保留真实来源，真人评分状态另列。主检索结果基于历史359片段；当前生产342片段。参数/Prompt每条件4题，单次采样。可在不联网的环境打开本图册。</footer>
<dialog id="viewer"><nav><button id="prev" aria-label="上一张">←</button><span id="slideTitle"></span><button id="next" aria-label="下一张">→</button><button id="close">关闭 Esc</button></nav><img id="slide" alt=""></dialog>
<script>const items=__DATA__;let current=0;const viewer=document.querySelector('#viewer'),slide=document.querySelector('#slide');function show(i){current=(i+items.length)%items.length;slide.src=items[current].svg;slide.alt=items[current].title;document.querySelector('#slideTitle').textContent=`${current+1} / ${items.length}  ${items[current].title}`;if(!viewer.open)viewer.showModal()}document.querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>show(+b.dataset.index));document.querySelector('#prev').onclick=()=>show(current-1);document.querySelector('#next').onclick=()=>show(current+1);document.querySelector('#close').onclick=()=>viewer.close();document.addEventListener('keydown',e=>{if(viewer.open&&e.key==='ArrowRight')show(current+1);if(viewer.open&&e.key==='ArrowLeft')show(current-1)});function filter(){let n=0;const q=document.querySelector('#search').value.toLowerCase(),cat=document.querySelector('#category').value;document.querySelectorAll('.card').forEach(c=>{const ok=(cat==='all'||cat===c.dataset.category)&&c.dataset.search.toLowerCase().includes(q);c.hidden=!ok;if(ok)n++});document.querySelector('#count').textContent=`${n} / ${items.length}`;document.querySelector('#empty').style.display=n?'none':'block'}document.querySelector('#search').oninput=filter;document.querySelector('#category').onchange=filter;filter();</script></html>'''
    doc=doc.replace("__COUNT__",str(len(atlas.items))).replace("__OPTIONS__","".join(f'<option>{html.escape(c)}</option>' for c in cats)).replace("__CARDS__","\n".join(cards)).replace("__DATA__",payload)
    (OUT/"学术图表展示.html").write_text(doc,encoding="utf-8")


def main():
    setup(); atlas=Atlas()
    try:
        pipeline(atlas); corpus_dataset(atlas); retrieval(atlas); parsing_chunking(atlas)
        rrf(atlas); baseline_quality(atlas); generation_quality(atlas); runtime(atlas); additional_results(atlas); caveats_and_configs(atlas)
    finally:
        atlas.pdf.close()
    source_dir=ROOT/"reports/presentation/source-data"
    if source_dir.exists():
        for p in source_dir.glob("*"):
            if p.is_file() and p.name != "raw-results.json" and p.suffix in (".csv",".json"):
                shutil.copy2(p,OUT/"data"/p.name)
    for p in (EXP/"generation/quality-comparison").glob("*"):
        if p.suffix in (".csv",".json"): shutil.copy2(p,OUT/"data"/p.name)
    manifest={"generated_at":datetime.now().isoformat(),"figure_count":sum(i["kind"]=="figure" for i in atlas.items),
              "table_count":sum(i["kind"]=="table" for i in atlas.items),"pdf_pages":len(atlas.items),"png_dpi":300,
              "png_pixels":[4000,2250],"ai_review_is_human_review":False,"items":atlas.items,
              "source_hashes":atlas.sources,"bootstrap":{"seed":20261015,"resamples":5000,"interval":"95% percentile; query sampling, not run variance"}}
    (OUT/"图表清单.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    gallery(atlas)
    intro=["# 科研助理学术图表包","",f"生成时间：{manifest['generated_at']}。", "",
        f"共{manifest['figure_count']}幅科研图、{manifest['table_count']}张结果表；PDF共{manifest['pdf_pages']}页。",
        "","- `学术图表展示.html`：离线图册，支持主题筛选、搜索、单图放大、左右键展示与下载。",
        "- `figures/`、`tables/`：300 dpi PNG（4000×2250）与SVG矢量图，可直接插入PPT或论文。",
        "- `科研助理实验图表集.pdf`：矢量图册，每页16:9，适合汇报与归档。",
        "- `学术图表数据与结果.xlsx`：可编辑结果、参数、逐题质量评审与来源。",
        "- `data/`：CSV、标准化JSON与完整实验配置；`图表清单.json`记录来源SHA与每图页码。",
        "","## 口径","","AI辅助评审始终标注AI来源，不改称真人评分。三档正式检索使用历史359片段，当前生产索引342片段。生成参数/Prompt每条件4题、单次采样、256输出token上限，只作探索性比较。图中误差区间为逐题bootstrap，不表示多次运行方差。",
        "","## 推荐展示顺序","","系统与题集 → 三档检索与不确定性 → 解析/分块/模型 → 参数与Prompt质量 → Agent与时延 → 局限与改进。",
        "","## 图表目录",""]
    intro += [f"{i+1}. {item['title']}（PDF第{item['pdf_page']}页，`{item['id']}`）" for i,item in enumerate(atlas.items)]
    (OUT/"使用说明.md").write_text("\n".join(intro),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"figures":manifest["figure_count"],"tables":manifest["table_count"],"pages":manifest["pdf_pages"]},ensure_ascii=False))


if __name__=="__main__":
    main()
