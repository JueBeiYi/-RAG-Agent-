"""Rebuild compact academic figures using only the bundled data and original UI assets.

Requires Python 3.10+, matplotlib, numpy and Pillow.  Run from any directory.
All plotted measurements are read from data/report-data.json; no model is called.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
SOURCE = BASE / "data/report-data.json"
D = json.loads(SOURCE.read_text(encoding="utf-8"))
T = D["tables"]
NAVY, TEAL, ORANGE, PURPLE = "#193A56", "#168980", "#D38C3C", "#7F6DAD"
BLUE, INK, MUTED, GRID = "#5688AD", "#263D4E", "#6E7E8A", "#E1E8EE"
PALETTE = [NAVY, TEAL, ORANGE, PURPLE, BLUE]
METRICS = ["hit_at_5", "mrr", "recall_at_5"]
MLABELS = ["Hit@5", "MRR", "Recall@5"]
MODES = ["vector", "hybrid", "hybrid_rerank"]
MODE_LABELS = ["纯向量", "混合检索", "混合 + 重排"]
CATS = ["factual", "comparison", "synthesis", "reasoning"]
CAT_LABELS = ["事实型", "对比型", "归纳型", "推理型"]
QLABELS = ["正确性", "完整性", "引用质量", "流畅性"]
QFIELDS = ["correctness", "completeness", "citation_quality", "fluency"]
MANIFEST = []
BOOTSTRAP = []


def rows(key):
    return T[key]["rows"]


def setup():
    OUT.mkdir(exist_ok=True)
    for path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "grid.linewidth": .65, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def canvas(n=1, height=3.1, widths=None):
    fig, axs = plt.subplots(1, n, figsize=(8, height), squeeze=False,
                            gridspec_kw={"width_ratios": widths} if widths else None)
    fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.92, wspace=.48)
    for ax in axs[0]:
        ax.grid(axis="y")
    return fig, list(axs[0])


def panel(ax, label):
    ax.figure.text(ax.get_position().x0, .975, label, weight="bold", fontsize=11, va="top")


def save(fig, name, sources, note, parameters=None):
    stem = OUT / name
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(stem.with_suffix("."+ext), dpi=400)
    MANIFEST.append({"id": name, "files": [f"figures/{name}.{e}" for e in ["pdf", "png", "svg"]],
                     "source": "data/report-data.json", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                     "source_tables_or_keys": sources, "parameters": parameters or {}, "note": note,
                     "native_width_inches": float(fig.get_size_inches()[0]),
                     "native_height_inches": float(fig.get_size_inches()[1]), "raster_dpi": 400})
    plt.close(fig)


def bars(ax, labels, values, colors=None, fmt=".2f", ylim=None):
    xx = np.arange(len(labels))
    bb = ax.bar(xx, values, width=.58, color=colors or TEAL, zorder=3)
    ax.set_xticks(xx, labels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(values)*1.22 if values else 1)
    for b, v in zip(bb, values):
        ax.annotate(format(v, fmt), (b.get_x()+b.get_width()/2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
    return bb


def grouped(ax, labels, groups, values, colors=None, top=1, fmt=".2f", show_values=True):
    xx = np.arange(len(labels)); width=.76/len(groups)
    for j, (g, vv) in enumerate(zip(groups, values)):
        xpos=xx+(j-(len(groups)-1)/2)*width
        bb=ax.bar(xpos,vv,width=width*.91,label=g,color=(colors or PALETTE)[j],zorder=3)
        if show_values:
            for b,v in zip(bb,vv):
                ax.annotate(format(v,fmt),(b.get_x()+b.get_width()/2,v),xytext=(0,3),
                            textcoords="offset points",ha="center",va="bottom",fontsize=9)
    ax.set_xticks(xx, labels); ax.set_ylim(0,top)
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(-.02,1.14),ncol=len(groups))


def heatmap(ax, values, xlabels, ylabels, vmin=0, vmax=1, fmt=".2f", cmap=None):
    cmap=cmap or LinearSegmentedColormap.from_list("navy_teal",["#F2F7FA","#B9DCD8",TEAL,NAVY])
    im=ax.imshow(values,vmin=vmin,vmax=vmax,cmap=cmap,aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)),xlabels)
    ax.set_yticks(np.arange(len(ylabels)),ylabels)
    ax.grid(False)
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            v=values[i][j]
            ax.text(j,i,format(v,fmt),ha="center",va="center",fontsize=10,
                    color="white" if (v-vmin)/(vmax-vmin)>.61 else INK)
    for spine in ax.spines.values():spine.set_visible(False)
    return im


def architecture():
    fig, aa=canvas(height=4.65); ax=aa[0];ax.axis("off");ax.set_xlim(0,10);ax.set_ylim(0,6)
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.12",facecolor=color+"12",edgecolor=color,lw=1.15))
        ax.text(x+w/2,y+h*.69,title,ha="center",va="center",fontsize=11,weight="bold",color=color)
        ax.text(x+w/2,y+h*.29,body,ha="center",va="center",fontsize=10,linespacing=1.55)
    def arrow(a,b,color=MUTED,reverse=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="<->" if reverse else "-|>",mutation_scale=11,color=color,lw=1.1,connectionstyle="arc3,rad=0"))
    box(.2,4.8,2.6,.94,"展示层 Streamlit","论文库 / 问答 / 对比 / 实验",NAVY)
    box(3.7,4.8,2.6,.94,"接口层 FastAPI","REST / SSE / 工作区与会话",NAVY)
    box(7.2,4.8,2.6,.94,"日志与监控","Token / 延迟 / 轨迹 / 健康",PURPLE)
    arrow((2.9,5.27),(3.6,5.27),reverse=True);arrow((6.4,5.27),(7.1,5.27))
    box(.2,3.15,2.6,1.03,"调度层 ReAct","路由 / 工具调用 / 有界恢复",TEAL)
    box(3.7,3.15,2.6,1.03,"工具与记忆","8 个工具 / 窗口 / 摘要",TEAL)
    box(7.2,3.15,2.6,1.03,"本地大模型","Ollama · research-qwen:7b",ORANGE)
    arrow((4.1,4.72),(1.5,4.27));arrow((2.9,3.67),(3.6,3.67),reverse=True)
    arrow((6.4,3.67),(7.1,3.67),reverse=True)
    box(.2,1.5,2.6,1.03,"生成层 RAG","上下文 / Prompt / 引用 / 缓存",TEAL)
    box(3.7,1.5,2.6,1.03,"检索层","向量 + BM25 → RRF → 重排",TEAL)
    box(7.2,1.5,2.6,1.03,"向量与重排模型","Embedding / Cross-Encoder",ORANGE)
    arrow((4.65,3.08),(1.5,2.61));arrow((2.9,2.02),(3.6,2.02),reverse=True)
    arrow((6.4,2.02),(7.1,2.02),reverse=True);arrow((8.5,3.08),(2.0,2.61))
    box(.2,.12,9.6,.8,"数据层","PDF / DOCX / TXT / MD → 解析与分块 → Chroma / FAISS · 元数据 · 原页",NAVY)
    arrow((5,1.43),(5,.99),reverse=True)
    save(fig,"fig01_architecture",["production_index","local_runtime","acceptance"],"依据真实模块职责与数据流绘制；箭头表示主要数据或调用关系，不表示并发时序。")


def dataset():
    rr=rows("dataset"); cc=rows("corpus")
    fig,aa=canvas(2,height=3.35,widths=[1,1.15]);a,b=aa
    dev=[r["dev"] for r in rr];test=[r["test"] for r in rr];xx=np.arange(4)
    a.bar(xx,dev,color=ORANGE,label="开发集",width=.58)
    a.bar(xx,test,bottom=dev,color=TEAL,label="测试集",width=.58)
    a.set_xticks(xx,CAT_LABELS);a.set_ylabel("问答对数量");a.set_ylim(0,36)
    for j,(dv,tv) in enumerate(zip(dev,test)):
        a.text(j,dv/2,str(dv),ha="center",va="center",color="white",fontsize=10)
        a.text(j,dv+tv/2,str(tv),ha="center",va="center",color="white",fontsize=10)
    a.legend(frameon=False,ncol=2,loc="upper left",bbox_to_anchor=(-.07,1.13));panel(a,"(a) 题集划分")
    bars(b,[r["paper_key"].upper() for r in cc],[r["physical_pages"] for r in cc],colors=BLUE,fmt=".0f",ylim=(0,36))
    b.set_ylabel("PDF 物理页数");b.tick_params(axis="x",labelrotation=50,labelsize=9);panel(b,"(b) 10 篇固定版本论文")
    fig.subplots_adjust(wspace=.4,bottom=.24,top=.82)
    save(fig,"fig02_dataset",["dataset","corpus"],"80题分为20题开发集和60题测试集；物理页统计对应固定arXiv版本。")


def parsers():
    rr=rows("parsers");labels=[{"pymupdf":"PyMuPDF","pdfplumber":"pdfplumber","pypdf2":"PyPDF2"}.get(r["parser"].lower(),r["parser"]) for r in rr]
    fig,aa=canvas(2,height=3.1)
    bars(aa[0],labels,[r["parse_ms"]/1000 for r in rr],colors=[NAVY,TEAL,ORANGE],fmt=".2f")
    aa[0].set_ylabel("10 篇累计解析时间 / s");panel(aa[0],"(a) 实际解析耗时")
    bars(aa[1],labels,[r["anchor_retention"] for r in rr],colors=[NAVY,TEAL,ORANGE],fmt=".1%",ylim=(0,1.2))
    aa[1].set_ylabel("锚点保留率");aa[1].yaxis.set_major_formatter(PercentFormatter(1));panel(aa[1],"(b) 119 个原文锚点")
    save(fig,"fig03_parsers",["parsers"],"30次解析；锚点保留是自动代理指标，不等于版面、表格或公式语义完全正确。")


def chunking():
    rr=[r for r in rows("chunking") if r["model"]=="moka-ai/m3e-base"]
    labels=[{"fixed-256":"固定\n256","fixed-512":"固定\n512","fixed-1024":"固定\n1024","recursive-512":"递归\n512","semantic-512":"句段\n512"}[r["variant"]] for r in rr]
    fig,aa=canvas(2,height=3.3,widths=[1.55,1]);a,b=aa
    grouped(a,labels,MLABELS,[[r[m] for r in rr] for m in METRICS],top=.68,show_values=False)
    a.set_ylabel("检索指标");panel(a,"(a) 相同 m3e / 20 题开发集")
    bars(b,labels,[r["chunk_count"] for r in rr],colors=[NAVY,BLUE,TEAL,ORANGE,PURPLE],fmt=".0f")
    b.set_ylabel("分块数量");panel(b,"(b) 同一语料的分块规模")
    fig.subplots_adjust(wspace=.43,bottom=.22,top=.83)
    save(fig,"fig04_chunking",["chunking"],"只对比同m3e模型的5种分块变体，不混入不同模型。")


def embeddings():
    rr=rows("embeddings");labels=["m3e-base","bge-large-zh"]
    fig,aa=canvas(2,height=3.25)
    grouped(aa[0],MLABELS,labels,[[r[m] for m in METRICS] for r in rr],top=.58,colors=[TEAL,NAVY])
    aa[0].set_ylabel("检索指标");panel(aa[0],"(a) 同 342 块 / 20 题开发集")
    bars(aa[1],labels,[r["query_encoding_amortized_ms"] for r in rr],colors=[TEAL,NAVY],fmt=".2f")
    aa[1].set_ylabel("每题平均编码时间 / ms");panel(aa[1],"(b) 同 20 条查询的真实编码")
    fig.subplots_adjust(top=.81)
    save(fig,"fig05_embeddings",["embeddings"],"查询编码在相同20条查询上比较；没有比较1836条与342条文档的总编码时间。")


def vectorstores():
    rr=rows("vectorstores");labels=[r["backend"].upper() if r["backend"]=="faiss" else "Chroma" for r in rr]
    fig,aa=canvas(2,height=3.05)
    bars(aa[0],labels,[r["build_and_persist_ms"] for r in rr],colors=[NAVY,TEAL],fmt=".1f")
    aa[0].set_ylabel("建库与持久化 / ms");panel(aa[0],"(a) 相同 359 块 / 1024 维")
    grouped(aa[1],["查询均值","查询 P95","重启首查"],labels,[[r["latency_mean_ms"],r["latency_p95_ms"],r["restart_first_query_ms"]] for r in rr],colors=[NAVY,TEAL],top=11)
    aa[1].set_ylabel("实际耗时 / ms");panel(aa[1],"(b) 库操作的计时范围")
    fig.subplots_adjust(top=.81)
    save(fig,"fig06_vectorstores",["vectorstores"],"不含共享向量编码；两种库在此20题上的检索质量相同。")


def retrieval():
    rr=rows("retrieval_test")
    fig,aa=canvas(2,height=3.3,widths=[1.35,1])
    grouped(aa[0],MLABELS,MODE_LABELS,[[r[m] for m in METRICS] for r in rr],top=.82)
    aa[0].set_ylabel("测试集检索指标");panel(aa[0],"(a) 60 题 / 历史 359 块")
    bars(aa[1],MODE_LABELS,[r["latency_mean_ms"] for r in rr],colors=PALETTE[:3],fmt=".1f")
    aa[1].set_yscale("log");aa[1].set_ylim(1,90000);aa[1].set_ylabel("排序阶段平均耗时 / ms（对数）")
    aa[1].tick_params(axis="x",labelsize=9);panel(aa[1],"(b) 质量与计算开销")
    fig.subplots_adjust(top=.80,wspace=.45)
    save(fig,"fig07_retrieval",["retrieval_test"],"实际60题结果；耗时排除共享查询编码，非完整在线端到端延迟。")


def retrieval_ci():
    rr=rows("retrieval_records");rng=np.random.default_rng(20261015)
    fig,aa=canvas(3,height=3.25);fig.subplots_adjust(wspace=.54,left=.075,right=.99,bottom=.22,top=.88)
    for ax,m,label in zip(aa,METRICS,MLABELS):
        for j,mode in enumerate(MODES):
            subset=sorted([r for r in rr if r["split"]=="test" and r["mode"]==mode],key=lambda r:r["case_id"])
            assert len(subset)==60
            vals=np.array([float(r[m]) for r in subset]);idx=rng.integers(0,len(vals),size=(5000,len(vals)))
            boot=vals[idx].mean(axis=1);lo,hi=np.percentile(boot,[2.5,97.5]);mean=vals.mean()
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=7,color=PALETTE[j],capsize=5,lw=1.7)
            ax.annotate(f"{mean:.3f}",(j,hi),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9.5)
            BOOTSTRAP.append({"mode":mode,"metric":m,"n":60,"mean":float(mean),"ci95_low":float(lo),"ci95_high":float(hi)})
        ax.set_xticks(range(3),["向量","混合","重排"]);ax.set_ylim(0,.9);ax.set_xlim(-.55,2.55);ax.set_ylabel(label)
    save(fig,"fig08_retrieval_ci",["retrieval_records"],"逐题非参数bootstrap百分位区间；仅描述此冻结题集抽样不确定性，不构成显著性检验。",{"bootstrap_samples":5000,"seed":20261015,"interval":"95% percentile","resampling_unit":"question","metric_order":METRICS,"mode_order":MODES})


def retrieval_categories():
    rr=[r for r in rows("retrieval_category") if r["split"]=="test"]
    vals=[[next(r["hit_at_5"] for r in rr if r["mode"]==mode and r["category"]==cat) for cat in CATS] for mode in MODES]
    fig,aa=canvas(height=2.65);fig.subplots_adjust(left=.16,right=.90,bottom=.20,top=.93)
    heatmap(aa[0],vals,[f"{c}\n(n={n})" for c,n in zip(CAT_LABELS,[22,16,11,11])],MODE_LABELS,vmin=0,vmax=1,fmt=".1%")
    save(fig,"fig09_retrieval_categories",["retrieval_category"],"各问题类别的Hit@5；类别样本量不同，不宜直接比较方差或显著性。")


def rrf():
    rr=rows("rrf_grid");fig,aa=canvas(2,height=3.05,widths=[1.15,1]);fig.subplots_adjust(left=.10,wspace=.50,bottom=.22)
    vals=[[next(r["hit"] for r in rr if r["rrf_k"]==k and r["metric_k"]==top) for top in [3,5,10]] for k in [20,60,100]]
    heatmap(aa[0],vals,["Top-3","Top-5","Top-10"],["RRF 20","RRF 60","RRF 100"],vmin=0,vmax=.8,fmt=".2f")
    panel(aa[0],"(a) 开发集 Hit@K")
    bars(aa[1],["RRF 60","RRF 20"],[next(r["mrr_at_20"] for r in rr if r["rrf_k"]==k) for k in [60,20]],colors=[NAVY,TEAL],fmt=".4f",ylim=(0,.42))
    aa[1].set_ylabel("MRR@20");panel(aa[1],"(b) 同题排序迭代")
    save(fig,"fig10_rrf",["rrf_grid"],"复用真实向量/BM25排名，无新增模型调用；RRF60到20仅小幅改善MRR，Hit@5与Recall@5不变。")


def ai_quality():
    rr=rows("baseline_ai_quality");dd=rows("baseline_ai_distribution")
    fig,aa=canvas(2,height=3.3,widths=[1,1.3]);fig.subplots_adjust(wspace=.46,top=.84)
    bars(aa[0],QLABELS,[r["mean"] for r in rr],colors=PALETTE[:4],fmt=".2f",ylim=(0,5.6));aa[0].set_ylabel("AI 评审均分（1–5）");panel(aa[0],"(a) 60 题短答案的四维质量")
    for i,(q,label) in enumerate(zip(QFIELDS,QLABELS)):
        counts=[next(r["proportion"] for r in dd if r["dimension"]==q+"_1_to_5" and r["score"]==s) for s in range(1,6)]
        left=0
        for j,prop in enumerate(counts):
            aa[1].barh(i,prop,left=left,color=["#D6DEE7","#AFBECF",BLUE,TEAL,NAVY][j],height=.6,label=str(j+1) if i==0 else None)
            if prop>=.095:aa[1].text(left+prop/2,i,f"{prop:.0%}",ha="center",va="center",fontsize=9,color="white" if j>=2 else INK)
            left+=prop
    aa[1].set_yticks(range(4),QLABELS);aa[1].invert_yaxis();aa[1].set_xlim(0,1);aa[1].xaxis.set_major_formatter(PercentFormatter(1));aa[1].set_xlabel("各分值所占比例");aa[1].grid(axis="x");aa[1].legend(frameon=False,ncol=5,title="评分",loc="upper left",bbox_to_anchor=(-.06,1.27),columnspacing=.75,handlelength=1)
    save(fig,"fig11_ai_quality",["baseline_ai_quality","baseline_ai_distribution"],"AI辅助评审明确保留来源；流畅性不等同事实正确性。")


def gen_parameters():
    rr=[r for r in rows("quality_conditions") if r["study"]=="parameters"]
    labels=[f"基线  T={rr[0]['temperature']}, p={rr[0]['top_p']}, k={rr[0]['top_k_sampling']}","Temperature = 0.0","Temperature = 0.5","Top-p = 0.7","Top-k = 20"]
    vals=[[r[q+"_mean"] for q in QFIELDS] for r in rr]
    fig,aa=canvas(height=3.15);fig.subplots_adjust(left=.33,right=.92,bottom=.16,top=.95)
    heatmap(aa[0],vals,QLABELS,labels,vmin=1,vmax=5)
    save(fig,"fig12_gen_parameters",["quality_conditions"],"5个单因素配置，每配置同4题，各题每配置仅一次历史生成；AI辅助质量评分。",{"score_range":[1,5],"case_count_per_condition":4})


def prompt():
    rr=rows("prompt_ai_quality");fig,aa=canvas(2,height=3.15,widths=[1.65,1]);fig.subplots_adjust(top=.80,wspace=.45)
    grouped(aa[0],QLABELS,["旧 Prompt","当前 Prompt"],[[r[q+"_mean"] for q in QFIELDS] for r in rr],colors=[ORANGE,TEAL],top=5.8)
    aa[0].set_ylabel("AI 评审均分（1–5）");panel(aa[0],"(a) 同 4 题的提示模板比较")
    bars(aa[1],["旧 Prompt","当前 Prompt"],[r["completion_tokens_mean"] for r in rr],colors=[ORANGE,TEAL],fmt=".2f")
    aa[1].set_ylabel("平均输出 Token");panel(aa[1],"(b) 描述性资源统计")
    save(fig,"fig13_prompt",["prompt_ai_quality"],"当前Prompt复用参数基线的4条回答；引用质量均分2.50到3.50仅描述这4题，不推断总体收益。")


def agent():
    rr=rows("agent");fig,aa=canvas(3,height=3.25,widths=[1,1,1.15]);fig.subplots_adjust(left=.07,right=.99,wspace=.62,bottom=.22,top=.84)
    labels=["固定 RAG","Agent"]
    bars(aa[0],labels,[r["latency_mean_ms"]/1000 for r in rr],colors=[NAVY,TEAL]);aa[0].set_ylabel("平均耗时 / s");panel(aa[0],"(a) 每组 5 次")
    grouped(aa[1],labels,["输入","输出"],[[r["prompt_tokens_total"] for r in rr],[r["completion_tokens_total"] for r in rr]],colors=[NAVY,ORANGE],top=23000,fmt=".0f",show_values=False)
    aa[1].set_ylabel("5 次累计 Token");panel(aa[1],"(b) 用量与路由")
    route=[r for r in rows("generation_records") if r["study"]=="agent" and r["condition"]=="agent"]
    names=["概念","计算","时间","关键词","论文事实"]
    matches=[int((r["expected_tools"] or [])==(r["invoked_tools"] or [])) for r in route]
    aa[2].barh(np.arange(5),matches,color=TEAL,height=.56);aa[2].set_yticks(range(5),names);aa[2].invert_yaxis();aa[2].set_xlim(0,1.23);aa[2].set_xticks([0,1],["未匹配","匹配"]);aa[2].grid(axis="x");panel(aa[2],"(c) 5 个实际路由样例")
    save(fig,"fig14_agent",["agent","generation_records"],"冻结检索输入/生成阶段与真实工具路由样例；Agent在该小样本中减少输入Token但耗时更长，路由匹配不等于答案语义正确。")


def runtime_probes():
    pp=rows("parallel");cc=rows("cache");fig,aa=canvas(2,height=3.05)
    bars(aa[0],["串行 / 1 worker","并行 / 2 workers"],[r["latency_mean_ms"] for r in pp],colors=[NAVY,TEAL],fmt=".2f")
    aa[0].set_ylabel("本地独立工具平均耗时 / ms");panel(aa[0],"(a) 每组 10 次工具测量")
    bars(aa[1],["冷生成","精确重复命中"],[r["latency_mean_ms"] for r in cc],colors=[ORANGE,TEAL],fmt=".2f")
    aa[1].set_yscale("log");aa[1].set_ylim(1,20000);aa[1].set_ylabel("耗时 / ms（对数）");panel(aa[1],"(b) 缓存各 1 次真实请求")
    save(fig,"fig15_runtime_probes",["parallel","cache"],"并行实验仅本地工具，不是LLM并发；缓存仅精确重复1对请求，不能据此推广语义缓存命中率。")


def local_run():
    mm=D["local_runtime"]["generation_metrics"];fig,aa=canvas(2,height=3.05,widths=[1.5,1]);fig.subplots_adjust(left=.18,wspace=.65)
    labels=["检索与重排","首个 Token","端到端请求"]
    vv=[mm["retrieval_ms"]/1000,mm["first_token_ms"]/1000,mm["end_to_end_ms"]/1000]
    aa[0].barh(np.arange(3),vv,color=[TEAL,ORANGE,NAVY],height=.56)
    for j,v in enumerate(vv):aa[0].text(v+.6,j,f"{v:.2f}",va="center",fontsize=10)
    aa[0].set_yticks(range(3),labels);aa[0].invert_yaxis();aa[0].set_xlim(0,43);aa[0].set_xlabel("实际耗时 / s");aa[0].grid(axis="x");panel(aa[0],"(a) 单次请求的重叠计时范围")
    bars(aa[1],["输入","输出"],[mm["input_tokens"],mm["output_tokens"]],colors=[NAVY,ORANGE],fmt=".0f");aa[1].set_ylabel("实际 Token");panel(aa[1],"(b) 非估算模型用量")
    save(fig,"fig16_local_run",["local_runtime"],"2026-10-15回环本地、独立单篇29块验收请求；耗时口径重叠，不可相加，也不是稳定均值。",{"sample_count":1,"timings_overlap":True})


def badcase():
    rr=rows("baseline_ai_scores");labels=["检索失败","生成错误","引用错误","Agent 决策错误","无明显问题"]
    counts=[sum(l.replace(" ","") in str(r["bad_case_type"]).replace(" ","") for r in rr) for l in labels]
    fig,aa=canvas(height=2.75);fig.subplots_adjust(left=.19,right=.98,bottom=.19,top=.93)
    aa[0].barh(range(len(labels)),counts,color=[BLUE,ORANGE,PURPLE,NAVY,TEAL],height=.60)
    for i,v in enumerate(counts):aa[0].text(v+.6,i,f"{v} / 60",va="center",fontsize=10)
    aa[0].set_yticks(range(len(labels)),labels);aa[0].invert_yaxis();aa[0].set_xlim(0,max(counts)+10);aa[0].set_xlabel("具有该标签的回答数量（可多选）");aa[0].grid(axis="x")
    save(fig,"fig17_badcase",["baseline_ai_scores"],"60题AI评审中的原始多标签计数，标签并非互斥；Agent决策标签为0不等于专项Agent样例无错误。",{"multilabel":True,"denominator":60,"counts":dict(zip(labels,counts))})


def ui_figures():
    source_dir=OUT/"original-ui"
    specs=[
        ("新版论文库.jpg","ui01_library",[(.63,.755,"① 文档上传与索引概况"),(.60,.49,"② 论文检索与状态过滤"),(.61,.05,"③ 下载 / 范围 / 重建 / 删除")]),
        ("新版科研问答.jpg","ui02_chat",[(.66,.86,"① 实际回答与运行指标"),(.66,.52,"② 来源编号与物理页映射"),(.66,.32,"③ 可展开执行轨迹")]),
        ("新版论文对比.jpg","ui03_comparison",[(.64,.61,"① 结果保存与导出"),(.65,.34,"② 带来源的结构化对比"),(.65,.145,"③ 条件未齐时保留比较边界")]),
        ("新版原文证据.jpg","ui04_evidence",[(.34,.84,"① 文档名与引用物理页"),(.41,.55,"② 原文 / 片段切换与选页"),(.56,.13,"③ 真实 PDF 原页预览")]),
    ]
    for fn,name,marks in specs:
        src=source_dir/fn
        if not src.exists():raise FileNotFoundError(src)
        im=Image.open(src);fig=plt.figure(figsize=(8,5.5));ax=fig.add_axes([.02,.22,.96,.76]);ax.imshow(im);ax.axis("off")
        # Original image pixels remain unchanged. Markers are drawn outside the image, with arrows.
        for j,(x,y,text) in enumerate(marks):
            tx=.18+j*.32
            ax.annotate(text,xy=(x,y),xycoords="axes fraction",xytext=(tx,-.14),textcoords="axes fraction",
                        ha="center",va="center",fontsize=9.5,color=PALETTE[j],
                        arrowprops={"arrowstyle":"-|>","lw":1.1,"color":PALETTE[j],"connectionstyle":"angle3,angleA=90,angleB=0"},
                        bbox={"boxstyle":"round,pad=.32","fc":"white","ec":GRID,"lw":.7},annotation_clip=False)
        fig.text(.5,.035,"2026-10-10 历史运行截图 · 仅外部标注功能区 · 保留原始页面内容",ha="center",fontsize=9.5,color=MUTED)
        save(fig,name,["original-ui/"+fn],"2026-10-10历史真实页面，非2026-10-15新生成回答。",{"original_image_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"annotation":"external labels and arrows; original screenshot pixels unchanged"})


def main():
    setup()
    for task in [architecture,dataset,parsers,chunking,embeddings,vectorstores,retrieval,retrieval_ci,
                 retrieval_categories,rrf,ai_quality,gen_parameters,prompt,agent,runtime_probes,local_run,badcase,ui_figures]:
        task();print(task.__name__,"done",flush=True)
    local=OUT/"ui05_local.jpg"
    if local.exists():
        MANIFEST.append({"id":"ui05_local","files":["figures/ui05_local.jpg"],"source":"reports/deployment/local-20261015/本地界面运行截图.jpg","sha256":hashlib.sha256(local.read_bytes()).hexdigest(),"note":"2026-10-15真实本地运行页面；既有历史会话不视为此次新生成。"})
    result={"created_date":"2026-10-15","builder":"build_figures.py","font_size_pt":11,
            "palette":PALETTE,"data_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "source_hashes":D.get("source_hashes",{}),"figures":MANIFEST,"bootstrap_intervals":BOOTSTRAP}
    (OUT/"figure_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Figure manifest:",len(MANIFEST),"items",flush=True)


if __name__=="__main__":main()
