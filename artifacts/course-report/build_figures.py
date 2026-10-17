"""Rebuild the report's journal-style figures from bundled observations.

No model calls, synthetic measurements or inferred experimental runs are used.
Python 3.10+, matplotlib, numpy and Pillow are sufficient.
"""
from __future__ import annotations
import hashlib
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter, LogLocator
import numpy as np
from PIL import Image
from figure_style import (setup, canvas, heading, panel, heatmap, colorbar, box_points,
    NAVY, TEAL, BLUE, OCHRE, CORAL, PURPLE, INK, MUTED, GRID, PALETTE)

BASE=Path(__file__).resolve().parent
OUT=BASE/'figures'
SOURCE=BASE/'data/report-data.json'
D=json.loads(SOURCE.read_text(encoding='utf-8'))
T=D['tables']
ORANGE=OCHRE
METRICS=['hit_at_5','mrr','recall_at_5']
MLABELS=['Hit@5','MRR','Recall@5']
MODES=['vector','hybrid','hybrid_rerank']
MODE_LABELS=['Dense','Hybrid','Hybrid + rerank']
MODE_COLORS=[NAVY,TEAL,OCHRE]
CATS=['factual','comparison','synthesis','reasoning']
CAT_LABELS=['Factual','Comparison','Synthesis','Reasoning']
QLABELS=['Correctness','Completeness','Citation','Fluency']
QFIELDS=['correctness','completeness','citation_quality','fluency']
MANIFEST=[]
BOOTSTRAP=[]

def rows(key):return T[key]['rows']

def save(fig,name,sources,note,parameters=None):
    OUT.mkdir(exist_ok=True)
    for ext in ['pdf','png','svg']:
        fig.savefig((OUT/name).with_suffix('.'+ext),dpi=400)
    MANIFEST.append({'id':name,'files':[f'figures/{name}.{e}' for e in ['pdf','png','svg']],
        'source':'data/report-data.json','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_tables_or_keys':sources,'parameters':parameters or {},'note':note,
        'native_width_inches':float(fig.get_size_inches()[0]),'native_height_inches':float(fig.get_size_inches()[1]),
        'raster_dpi':400,'visual_version':'v2'})
    plt.close(fig)

def note(fig,text,x=.075,y=.035):
    fig.text(x,y,textwrap.fill(text,width=90),fontsize=10.6,color=MUTED,va='bottom',linespacing=1.35)

def bars(ax,labels,values,colors=None,fmt='.2f',ylim=None):
    bb=ax.bar(np.arange(len(labels)),values,width=.55,color=colors or TEAL,zorder=3)
    ax.set_xticks(np.arange(len(labels)),labels)
    ax.set_ylim(*(ylim or (0,max(values)*1.25 if values else 1)))
    for b,v in zip(bb,values):
        ax.annotate(format(v,fmt),(b.get_x()+b.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=10.6)
    return bb

def point_profiles(ax, rr, keys, labels, names, colors, xlim=(0,1)):
    yy=np.arange(len(keys))
    for i,k in enumerate(keys):
        vals=[r[k] for r in rr]
        ax.plot(vals,[i]*len(vals),color=GRID,lw=2,zorder=1)
    for j,(r,name,c) in enumerate(zip(rr,names,colors)):
        ax.scatter([r[k] for k in keys],yy+(j-(len(rr)-1)/2)*.12,s=46,color=c,label=name,zorder=4)
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(*xlim);ax.grid(False);ax.grid(axis='x')

def architecture():
    fig=plt.figure(figsize=(8.1,5.0));heading(fig,'Evidence-grounded research assistant','Implementation architecture | local inference, traceable sources and bounded tool execution')
    ax=fig.add_axes([.025,.035,.95,.765]);ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,6)
    def box(x,y,w,h,title,body,c):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.045,rounding_size=.075',fc=c+'0D',ec=c+'80',lw=.85))
        ax.text(x+.13,y+h*.70,title,ha='left',va='center',fontsize=11.7,weight='bold',color=c)
        ax.text(x+.13,y+h*.29,body,ha='left',va='center',fontsize=10.6,linespacing=1.25)
    def arrow(a,b,bidir=False,c=MUTED):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='<->' if bidir else '-|>',mutation_scale=10,color=c,lw=1.0))
    xx=[.15,3.66,7.17];w=2.68;h=.95
    box(xx[0],4.85,w,h,'Streamlit interface','Library / QA / compare\nOriginal evidence',NAVY)
    box(xx[1],4.85,w,h,'FastAPI service','REST + SSE\nSessions / workspaces',NAVY)
    box(xx[2],4.85,w,h,'Observability','Latency / tokens\nTool traces',PURPLE)
    arrow((2.9,5.32),(3.56,5.32),True);arrow((6.41,5.32),(7.08,5.32))
    box(xx[0],3.30,w,h,'Agent orchestration','Intent routes / ReAct\nBounded retry',TEAL)
    box(xx[1],3.30,w,h,'Tools and memory','8 tools / sliding window\nExtractive memory',TEAL)
    box(xx[2],3.30,w,h,'Local language model','Ollama\nresearch-qwen:7b',OCHRE)
    arrow((4.92,4.78),(4.92,4.32));arrow((2.9,3.77),(3.56,3.77),True);arrow((6.41,3.77),(7.08,3.77),True)
    box(xx[0],1.75,w,h,'Grounded generation','Context / prompts / citations\nExact-query cache',TEAL)
    box(xx[1],1.75,w,h,'Hybrid retrieval','Dense + BM25 / RRF\nCross-encoder reranking',TEAL)
    box(xx[2],1.75,w,h,'Neural retrieval models','Embedding\nCandidate reranking',OCHRE)
    arrow((1.49,3.23),(1.49,2.77),True);arrow((2.9,2.22),(3.56,2.22),True);arrow((6.41,2.22),(7.08,2.22),True)
    # RAG generation calls the language model; encoding/reranking is a separate dependency.
    ax.plot([2.45,2.45,8.51],[2.77,2.98,2.98],color=OCHRE,lw=.95)
    arrow((8.51,2.98),(8.51,3.23),False,OCHRE)
    box(.15,.20,9.70,.95,'Document and index layer','PDF / DOCX / TXT / MD  →  parsing + chunking\nChroma / FAISS  +  metadata + original pages',NAVY)
    arrow((5,1.68),(5,1.22),True)
    save(fig,'fig01_architecture',['production_index','local_runtime','acceptance'],'真实模块职责与主要依赖；8个工具、抽取式记忆、检索与模型调用均对应现有实现。箭头不表示并发时序。')

def dataset():
    rr=rows('dataset');cc=rows('corpus')
    fig,aa=canvas(2,height=4.65,widths=[1,1.1],title='Evaluation dataset and frozen paper corpus',subtitle='80 questions | 20 development + 60 test | 10 version-pinned vision-transformer papers',left=.15,bottom=.15,wspace=.68)
    a,b=aa;y=np.arange(4);dev=[r['dev'] for r in rr];test=[r['test'] for r in rr]
    a.barh(y,dev,color=OCHRE,height=.57,label='Development');a.barh(y,test,left=dev,color=TEAL,height=.57,label='Test')
    for i,(dv,tv) in enumerate(zip(dev,test)):
        a.text(dv/2,i,str(dv),ha='center',va='center',color='white',fontsize=11)
        a.text(dv+tv/2,i,str(tv),ha='center',va='center',color='white',fontsize=11)
    a.set_yticks(y,CAT_LABELS);a.invert_yaxis();a.set_xlim(0,34);a.set_xlabel('Questions');a.grid(False);a.grid(axis='x')
    fig.legend(*a.get_legend_handles_labels(),frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(.15,.01),columnspacing=1.4,handlelength=1)
    panel(a,'A','Stratified split')
    page=[r['physical_pages'] for r in cc];y=np.arange(10)
    b.hlines(y,0,page,color=GRID,lw=2);b.scatter(page,y,color=BLUE,s=34,zorder=3)
    for i,v in enumerate(page):b.text(v+.8,i,str(v),va='center',fontsize=10.6)
    b.set_yticks(y,[r['paper_key'].upper() for r in cc]);b.invert_yaxis();b.set_xlim(0,max(page)+5);b.set_xlabel('Physical PDF pages');b.grid(False);b.grid(axis='x');panel(b,'B','Corpus coverage')
    save(fig,'fig02_dataset',['dataset','corpus'],'80题按题型分层划分；每篇论文页数来自实际固定版本PDF，不表示独立抽样语料。')

def parsers():
    summary=rows('parsers');raw=rows('parser_records');names=['PyMuPDF','pdfplumber','PyPDF2'];keys=[r['parser'] for r in summary]
    fig,aa=canvas(2,height=4.25,widths=[1.15,1],title='Document parsing: runtime variation and anchor recovery',subtitle='30 actual parser–document runs | identical 10 PDFs | 119 manually specified source anchors',left=.09,bottom=.22,wspace=.66)
    a,b=aa;vals=[[r['parse_ms']/1000 for r in raw if r['parser']==k] for k in keys]
    box_points(a,vals,names,[NAVY,TEAL,OCHRE],ylabel='Seconds per document',log=True)
    a.set_ylim(.1,max(max(v) for v in vals)*2.6);panel(a,'A','Per-document time')
    for i,(g,c,r) in enumerate(zip(vals,[NAVY,TEAL,OCHRE],summary)):
        a.text(i,.97,f"Σ {r['parse_ms']/1000:.2f} s",transform=a.get_xaxis_transform(),ha='center',va='top',fontsize=10.6,color=c)
    vv=[r['anchor_retention'] for r in summary];y=np.arange(3)
    b.hlines(y,0,vv,color=GRID,lw=2);b.scatter(vv,y,s=62,c=[NAVY,TEAL,OCHRE],zorder=3)
    for i,r in enumerate(summary):
        b.text(.04,i-.14,f"{r['matched_anchors']} / {r['total_anchors']}",fontsize=11,weight='bold',color=[NAVY,TEAL,OCHRE][i])
        b.text(.04,i+.17,f"{r['anchor_retention']:.1%} recovered",fontsize=10.6,color=MUTED)
    b.set_yticks(y,names);b.set_ylim(2.55,-.55);b.set_xlim(0,1.07);b.xaxis.set_major_formatter(PercentFormatter(1));b.set_xlabel('Aggregate anchor retention');b.grid(False);b.grid(axis='x');panel(b,'B','Source-anchor recovery')
    note(fig,'Dots show real document runs; boxes show median and interquartile range. Anchor recovery is a proxy, not semantic fidelity.',x=.09,y=.035)
    save(fig,'fig03_parsers',['parsers','parser_records'],'每种解析器10个真实逐文档耗时，箱体为四分位距、中心线为中位数，散点为原始测量；锚点恢复不等于公式/表格语义完全正确。',{'runs_per_parser':10,'boxplot':'median and interquartile range; 1.5 IQR whiskers; all observations drawn'})

def chunking():
    rr=[r for r in rows('chunking') if r['model']=='moka-ai/m3e-base'];labels=['Fixed 256','Fixed 512','Fixed 1024','Recursive 512','Sentence 512']
    fig,aa=canvas(4,height=4.3,widths=[1,1,1,.95],title='Chunking quality and index size under a shared encoder',subtitle='m3e-base | 20 development questions | same corpus; recursive chunking is an engineering choice',left=.20,bottom=.18,wspace=.36)
    colors=[BLUE,BLUE,BLUE,TEAL,BLUE];y=np.arange(5)
    for j,(ax,key,label) in enumerate(zip(aa,METRICS+['chunk_count'],MLABELS+['Chunks'])):
        vv=[r[key] for r in rr];ax.hlines(y,0,vv,color=GRID,lw=1.8);ax.scatter(vv,y,c=colors,s=42,zorder=3)
        ax.set_yticks(y,labels if j==0 else ['']*5);ax.invert_yaxis();ax.grid(False);ax.grid(axis='x');panel(ax,chr(65+j),label)
        ax.set_xlim((0,1100) if key=='chunk_count' else (0,.78))
        for i,v in enumerate(vv):ax.annotate(f'{v:.0f}' if key=='chunk_count' else f'{v:.3f}',(v,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=10.6,color=colors[i])
        if key=='chunk_count':ax.set_xticks([0,500,1000])
        else:ax.set_xticks([0,.3,.6])
    save(fig,'fig04_chunking',['chunking'],'相同m3e模型下五种真实分块变体，横向小多图同时呈现三项检索指标与索引规模；未将不同编码器混入。')

def embeddings():
    rr=rows('embeddings');names=['m3e-base','bge-large-zh'];colors=[TEAL,NAVY]
    fig,aa=canvas(2,height=4.1,widths=[1.15,1],title='Embedding comparison: retrieval quality and query cost',subtitle='Identical 342 chunks and 20 development queries | document-encoding totals are not compared',left=.15,bottom=.23,wspace=.65)
    point_profiles(aa[0],rr,METRICS,MLABELS,names,colors,xlim=(0,.6));aa[0].set_xlabel('Retrieval score');aa[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(-.15,-.25),ncol=2,columnspacing=.8,handletextpad=.3);panel(aa[0],'A','Shared-corpus quality')
    vv=[r['query_encoding_amortized_ms'] for r in rr];bars(aa[1],names,vv,colors,fmt='.2f');aa[1].set_ylabel('Mean query encoding / ms');panel(aa[1],'B','Actual encoding cost')
    for i,r in enumerate(rr):aa[1].text(i,.95,f"{r['dimension']} dims",ha='center',va='top',transform=aa[1].get_xaxis_transform(),fontsize=10.6,color=colors[i])
    save(fig,'fig05_embeddings',['embeddings'],'两个模型均在相同342块与20题上比较；查询编码总时间按20题摊销，不冒充逐题在线延迟。')

def vectorstores():
    rr=rows('vectorstores');names=['Chroma','FAISS'];colors=[NAVY,TEAL]
    fig,aa=canvas(2,height=4.2,widths=[1,1.2],title='Vector-store operations under identical dense representations',subtitle='359 vectors × 1024 dimensions | 20 development queries | shared embedding computation excluded',left=.11,bottom=.29,wspace=.65)
    bars(aa[0],names,[r['build_and_persist_ms'] for r in rr],colors,fmt='.1f');aa[0].set_ylabel('Build + persist / ms');panel(aa[0],'A','Index construction')
    keys=['latency_mean_ms','latency_p95_ms','restart_first_query_ms'];labels=['Query mean','Query P95','First query\nafter restart'];y=np.arange(3)
    for j,(r,c) in enumerate(zip(rr,colors)):
        vv=[r[k] for k in keys];aa[1].scatter(vv,y+(j-.5)*.13,color=c,label=names[j],s=50,zorder=3)
        for i,v in enumerate(vv):aa[1].text(v+.2,i+(j-.5)*.30,f'{v:.2f}',va='center',fontsize=10.6,color=c)
    aa[1].set_yticks(y,labels);aa[1].invert_yaxis();aa[1].set_xlim(0,10);aa[1].set_xlabel('Operation latency / ms');aa[1].grid(False);aa[1].grid(axis='x');fig.legend(*aa[1].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(.63,.012),columnspacing=1.3);panel(aa[1],'B','Query and restart')
    note(fig,'Both stores: Hit@5 = 0.350, MRR = 0.324, Recall@5 = 0.275; delete / restore probes passed.',x=.11,y=.095)
    save(fig,'fig06_vectorstores',['vectorstores'],'同一向量集合的建库、查询、重启操作计时；二者在此20题上的质量指标相同，删除恢复实测通过。')

def retrieval():
    rr=rows('retrieval_test');raw=rows('retrieval_records')
    fig,aa=canvas(2,height=4.55,widths=[1.1,1.1],title='Retrieval quality improves with a substantial reranking cost',subtitle='60 test questions | historical 359-chunk index | ranking time excludes shared query encoding',left=.13,bottom=.25,wspace=.70)
    a,b=aa;point_profiles(a,rr,METRICS,MLABELS,MODE_LABELS,MODE_COLORS,xlim=(0,.85));a.set_xlabel('Test-set mean score');panel(a,'A','Quality profile')
    vals=[[r['elapsed_ms'] for r in raw if r['split']=='test' and r['mode']==m] for m in MODES]
    box_points(b,vals,['Dense','Hybrid','Rerank'],MODE_COLORS,ylabel='Ranking time / ms',log=True,jitter=.12);b.set_ylim(1,60000);panel(b,'B','60 measured latencies / mode')
    handles=[plt.Line2D([0],[0],marker='o',ls='',color=c,label=l,markersize=6) for l,c in zip(MODE_LABELS,MODE_COLORS)]
    fig.legend(handles=handles,frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(.10,.065),columnspacing=1.5)
    save(fig,'fig07_retrieval',['retrieval_test','retrieval_records'],'左侧为60题指标均值，右侧为同60题实际排序耗时分布；重排成本明显，非全链路在线延迟。',{'observations_per_mode':60,'latency_scope':'ranking only; query encoding excluded'})

def retrieval_ci():
    raw=rows('retrieval_records');rng=np.random.default_rng(20261015)
    fig,aa=canvas(3,height=4.35,title='Question-level bootstrap intervals for retrieval metrics',subtitle='60 frozen test questions | 5,000 nonparametric resamples | 95% percentile intervals',left=.09,bottom=.30,wspace=.46)
    for ax,key,label in zip(aa,METRICS,MLABELS):
        for j,mode in enumerate(MODES):
            sub=sorted([r for r in raw if r['split']=='test' and r['mode']==mode],key=lambda r:r['case_id']);assert len(sub)==60
            v=np.array([float(r[key]) for r in sub]);idx=rng.integers(0,len(v),size=(5000,len(v)));lo,hi=np.percentile(v[idx].mean(axis=1),[2.5,97.5]);mean=v.mean()
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',markersize=7,color=MODE_COLORS[j],capsize=5,lw=1.7)
            ax.annotate(f'{mean:.3f}',(j,hi),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10.7)
            BOOTSTRAP.append({'mode':mode,'metric':key,'n':60,'mean':float(mean),'ci95_low':float(lo),'ci95_high':float(hi)})
        ax.set_xticks(range(3),['Dense','Hybrid','Rerank']);ax.tick_params(axis='x',labelrotation=30);ax.set_xlim(-.55,2.55);ax.set_ylim(0,.85);panel(ax,chr(65+MLABELS.index(label)),label)
    aa[0].set_ylabel('Mean score');note(fig,'Intervals describe question-sampling uncertainty; overlap is not a paired significance test or run-to-run stability.',x=.09,y=.035)
    save(fig,'fig08_retrieval_ci',['retrieval_records'],'逐题非参数bootstrap百分位区间；描述冻结题集的抽样不确定性，不构成显著性检验。',{'bootstrap_samples':5000,'seed':20261015,'interval':'95% percentile','resampling_unit':'question','metric_order':METRICS,'mode_order':MODES})

def retrieval_categories():
    rr=[r for r in rows('retrieval_category') if r['split']=='test']
    vals=[[next(r['hit_at_5'] for r in rr if r['mode']==m and r['category']==c) for c in CATS] for m in MODES]
    fig,aa=canvas(height=3.9,title='Retrieval success varies across question types',subtitle='Test-set Hit@5 | type-specific sample sizes are retained; no pooled uncertainty is implied',left=.22,right=.92,bottom=.20)
    im=heatmap(aa[0],vals,[f'{c}\n(n = {n})' for c,n in zip(CAT_LABELS,[22,16,11,11])],MODE_LABELS,fmt='.1%',vmax=1)
    colorbar(fig,im,aa[0],'Hit@5');panel(aa[0],'A','Question-type stratification')
    save(fig,'fig09_retrieval_categories',['retrieval_category'],'题型分层的真实Hit@5，固定色标0到1；各题型样本量不同，不能据此直接判断显著性。')

def rrf():
    rr=rows('rrf_grid');fig,aa=canvas(2,height=4.25,widths=[1.1,1],title='RRF sensitivity separates cutoff effects from fusion constants',subtitle='20 development questions | existing dense / BM25 ranks reused | no additional model calls',left=.12,bottom=.22,wspace=.70)
    vals=[[next(r['hit'] for r in rr if r['rrf_k']==k and r['metric_k']==top) for top in [3,5,10]] for k in [20,60,100]]
    im=heatmap(aa[0],vals,['Top-3','Top-5','Top-10'],['k = 20','k = 60','k = 100'],vmax=.8,fmt='.2f');colorbar(fig,im,aa[0],'Hit@K');panel(aa[0],'A','Candidate cutoff × RRF k')
    vv=[next(r['mrr_at_20'] for r in rr if r['rrf_k']==k) for k in [60,20]]
    aa[1].plot([0,1],vv,color=GRID,lw=2);aa[1].scatter([0,1],vv,s=70,c=[NAVY,TEAL],zorder=3)
    for i,v in enumerate(vv):aa[1].annotate(f'{v:.4f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10.8)
    aa[1].set_xticks([0,1],['RRF k = 60','RRF k = 20']);aa[1].set_xlim(-.35,1.35);aa[1].set_ylim(0,.43);aa[1].set_ylabel('MRR@20');panel(aa[1],'B','Same-question rank change')
    note(fig,'RRF 60 → 20: Hit@5 = 0.40 and Recall@5 = 0.35 remain unchanged; the observed MRR change is small.',x=.12,y=.055)
    save(fig,'fig10_rrf',['rrf_grid'],'复用真实检索排名；RRF60到20仅小幅改善MRR@20，Hit@5与Recall@5不变；图中不将局部坐标放大冒充大幅改进。')

def ai_quality():
    rr=rows('baseline_ai_quality');dd=rows('baseline_ai_distribution')
    fig,aa=canvas(2,height=4.55,widths=[1.25,1],title='Answer fluency exceeds factual and citation reliability',subtitle='60 frozen-retrieval short answers | AI-assisted source-checked scoring | 1–5 ordinal rubric',left=.18,bottom=.24,wspace=.70)
    a,b=aa
    for i,(q,label) in enumerate(zip(QFIELDS,QLABELS)):
        counts=[next(r['proportion'] for r in dd if r['dimension']==q+'_1_to_5' and r['score']==s) for s in range(1,6)];left=0
        for j,prop in enumerate(counts):
            a.barh(i,prop,left=left,color=['#E8EDF2','#BACBD9',BLUE,TEAL,NAVY][j],height=.60,label=str(j+1) if i==0 else None)
            if prop>=.12:a.text(left+prop/2,i,f'{prop:.0%}',ha='center',va='center',fontsize=10.6,color='white' if j>=2 else INK)
            left+=prop
    a.set_yticks(range(4),QLABELS);a.invert_yaxis();a.set_xlim(0,1);a.xaxis.set_major_formatter(PercentFormatter(1));a.set_xlabel('Share of rated answers');a.grid(False);fig.legend(*a.get_legend_handles_labels(),frameon=False,ncol=5,title='Score',loc='lower left',bbox_to_anchor=(.18,.005),columnspacing=.8,handlelength=1);panel(a,'A','Full score distribution')
    for i,r in enumerate(rr):
        b.hlines(i,1,5,color=GRID,lw=2);b.scatter(r['mean'],i,s=65,color=[NAVY,TEAL,OCHRE,PURPLE][i],zorder=3);b.text(r['mean']+.10,i,f"{r['mean']:.2f}",fontsize=10.8,va='center')
    b.set_yticks(range(4),QLABELS);b.invert_yaxis();b.set_xlim(1,5.65);b.set_xticks([1,2,3,4,5]);b.set_xlabel('Mean ordinal score');b.grid(False);b.grid(axis='x');panel(b,'B','Dimension means')
    save(fig,'fig11_ai_quality',['baseline_ai_quality','baseline_ai_distribution'],'60题AI辅助评审的档位分布与均分，不改称独立人工评分；流畅性不能替代事实与引用可靠性。')

def gen_parameters():
    rr=[r for r in rows('quality_conditions') if r['study']=='parameters'];labels=['Baseline: T=.1, p=.9, k=40','Temperature = 0.0','Temperature = 0.5','Top-p = 0.7','Top-k = 20']
    vals=[[r[q+'_mean'] for q in QFIELDS] for r in rr]
    fig,aa=canvas(height=4.2,title='Generation-parameter effects are exploratory and task-dependent',subtitle='Five single-factor settings × identical four development questions | one generation per case and condition',left=.34,right=.92,bottom=.25)
    im=heatmap(aa[0],vals,QLABELS,labels,vmin=1,vmax=5,fmt='.2f');colorbar(fig,im,aa[0],'AI-assisted score',ticks=[1,2,3,4,5]);panel(aa[0],'A','Mean rubric scores (1–5)')
    note(fig,'Fixed contexts; no fixed sampling seed or repeated trials. These means do not establish an optimal sampling setting.',x=.075,y=.025)
    save(fig,'fig12_gen_parameters',['quality_conditions'],'每个单因素配置同4题、每题仅生成一次；AI辅助评分；固定1到5色标，不能推断总体最优参数。',{'score_range':[1,5],'case_count_per_condition':4})

def prompt():
    rr=rows('prompt_ai_quality');colors=[OCHRE,TEAL]
    fig,aa=canvas(2,height=4.3,widths=[1.25,1],title='Prompt revision improves citation scores in four paired cases',subtitle='Same four questions and frozen contexts | current-prompt outputs reuse the parameter baseline',left=.18,bottom=.25,wspace=.68)
    a,b=aa
    for i,q in enumerate(QFIELDS):a.plot([r[q+'_mean'] for r in rr],[i,i],color=GRID,lw=2)
    for j,(r,c) in enumerate(zip(rr,colors)):
        a.scatter([r[q+'_mean'] for q in QFIELDS],np.arange(4)+(j-.5)*.10,color=c,s=52,label=['Original prompt','Current prompt'][j],zorder=3)
    a.set_yticks(range(4),QLABELS);a.invert_yaxis();a.set_xlim(1,5.35);a.set_xticks([1,2,3,4,5]);a.set_xlabel('Mean AI-assisted score');a.grid(False);a.grid(axis='x');panel(a,'A','Four-dimensional score profile')
    bars(b,['Original','Current'],[r['completion_tokens_mean'] for r in rr],colors,fmt='.2f');b.set_ylabel('Mean completion tokens');panel(b,'B','Observed answer length')
    fig.legend(*a.get_legend_handles_labels(),frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(.16,.075),columnspacing=2)
    save(fig,'fig13_prompt',['prompt_ai_quality'],'4题配对模板比较；当前Prompt复用参数基线回答；引用质量2.50到3.50只描述这4题，输出长度变化不等于效率因果效应。')

def agent():
    rr=rows('agent');raw=rows('generation_records');names=['Fixed RAG','Agent'];colors=[NAVY,TEAL]
    fig,aa=canvas(3,height=4.6,widths=[1,1,1.1],title='Agent routing trades lower input volume for extra execution time',subtitle='Five cases per condition | frozen-retrieval generation timing and real tool-selection probes',left=.09,bottom=.31,wspace=.75)
    vals=[[r['elapsed_ms']/1000 for r in raw if r['study']=='agent' and r['condition']==cond] for cond in ['fixed_rag','agent']]
    box_points(aa[0],vals,names,colors,ylabel='Generation time / s');aa[0].set_ylim(0,max(max(v) for v in vals)*1.18);panel(aa[0],'A','Latency')
    xx=np.arange(2)
    for j,(key,c,l) in enumerate([('prompt_tokens_total',NAVY,'Input'),('completion_tokens_total',OCHRE,'Output')]):
        aa[1].bar(xx+(j-.5)*.27,[r[key] for r in rr],width=.25,color=c,label=l)
    aa[1].set_xticks(xx,names);aa[1].set_ylabel('Total tokens / 5 cases');aa[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));fig.legend(*aa[1].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(.50,.01),columnspacing=1.2,handlelength=1);panel(aa[1],'B','Token usage')
    route=[r for r in raw if r['study']=='agent' and r['condition']=='agent'];matches=[int((r['expected_tools'] or [])==(r['invoked_tools'] or [])) for r in route]
    aa[2].scatter(matches,np.arange(5),color=TEAL,s=55,zorder=3);aa[2].set_yticks(np.arange(5),['Concept','Calculator','Clock','Keyword','Paper fact']);aa[2].invert_yaxis();aa[2].set_xlim(-.15,1.25);aa[2].set_xticks([0,1],['Mismatch','Match']);aa[2].grid(False);aa[2].grid(axis='x');panel(aa[2],'C','Route match')
    note(fig,'Route agreement measures tool selection, not answer correctness. Small heterogeneous cases do not establish a population effect.',x=.09,y=.095)
    save(fig,'fig14_agent',['agent','generation_records'],'5例固定RAG与Agent的原始耗时及Token；路由匹配是工具选择一致性，不等于答案语义正确；局部小样本不推断总体收益。')

def runtime_probes():
    raw=rows('generation_records');cc=rows('cache')
    fig,aa=canvas(2,height=4.35,widths=[1.15,1],title='Tool concurrency and exact caching address distinct costs',subtitle='Local-tool-only probes: 10 aligned cases / condition | cache: one cold and one exact-repeat request',left=.11,bottom=.25,wspace=.70)
    a,b=aa;groups=[sorted([r for r in raw if r['study']=='parallel' and r['condition']==c],key=lambda r:r['case_id']) for c in ['workers_1','workers_2']]
    assert [r['case_id'] for r in groups[0]]==[r['case_id'] for r in groups[1]]
    vv=[[r['elapsed_ms'] for r in g] for g in groups]
    for v0,v1 in zip(*vv):a.plot([0,1],[v0,v1],color=GRID,lw=.9,zorder=1)
    box_points(a,vv,['1 worker','2 workers'],[NAVY,TEAL],ylabel='Local tool execution / ms',jitter=0)
    a.set_ylim(0,max(max(v) for v in vv)*1.25);panel(a,'A','Aligned tool-probe cases')
    bars(b,['Cold request','Exact repeat'],[r['latency_mean_ms'] for r in cc],[OCHRE,TEAL],fmt='.2f');b.set_yscale('log');b.set_ylim(1,25000);b.set_ylabel('Request time / ms (log)');panel(b,'B','Single cache request pair')
    note(fig,'Parallel timing excludes LLM calls and retrieval. A single exact-repeat cache hit does not estimate semantic hit rate.',x=.11,y=.06)
    save(fig,'fig15_runtime_probes',['parallel','cache','generation_records'],'10个相同case_id的本地工具实测连线，非LLM并发；箱体为描述性分布；缓存只有1对冷生成/精确重复请求，不能推广命中率。',{'parallel_paired_by':'case_id','parallel_cases':10,'cache_pairs':1})

def local_run():
    mm=D['local_runtime']['generation_metrics']
    fig,aa=canvas(2,height=4.35,widths=[1.45,1],title='A real loopback deployment request with overlapping timers',subtitle='2026-10-15 | independent 29-chunk document workspace | n = 1; actual model-reported token counts',left=.20,bottom=.30,wspace=.70)
    a,b=aa;labels=['Retrieval + rerank','Time to first token','End-to-end request'];vv=[mm['retrieval_ms']/1000,mm['first_token_ms']/1000,mm['end_to_end_ms']/1000]
    for j,(v,c) in enumerate(zip(vv,[TEAL,OCHRE,NAVY])):
        a.plot([0,v],[j,j],lw=9,color=c,solid_capstyle='butt');a.scatter(v,j,s=35,color=c,zorder=3);a.text(v+.7,j,f'{v:.2f}',va='center',fontsize=10.8,color=c)
    a.set_yticks(range(3),labels);a.invert_yaxis();a.set_xlim(0,42);a.set_xlabel('Measured elapsed duration / s');a.grid(False);a.grid(axis='x');panel(a,'A','Overlapping timing scopes')
    bars(b,['Input','Output'],[mm['input_tokens'],mm['output_tokens']],[NAVY,OCHRE],fmt='.0f');b.set_ylabel('Actual tokens');panel(b,'B','Observed usage')
    note(fig,'Timers have overlapping scopes and are shown from zero for comparison; they are not sequential components or stable means.',x=.075,y=.025)
    save(fig,'fig16_local_run',['local_runtime'],'单次本地实际请求，区间计时相互重叠不堆叠、不相加；Token为模型实际返回，不是估算值或稳定均值。',{'sample_count':1,'timings_overlap':True,'display_origin':'normalized to zero; overlapping timer scopes, not absolute phase starts'})

def badcase():
    rr=rows('baseline_ai_scores');labels=['检索失败','生成错误','引用错误','Agent 决策错误'];en=['Retrieval','Generation','Citation','Agent']
    tag=np.array([[int(l.replace(' ','') in str(r['bad_case_type']).replace(' ','')) for l in labels] for r in rr]);counts=tag.sum(0);co=tag.T@tag;clean=sum('无明显问题' in str(r['bad_case_type']) for r in rr)
    fig,aa=canvas(2,height=4.4,widths=[1,1.15],title='Bad-case taxonomy and within-question co-occurrence',subtitle='AI-assisted multi-label review of 60 baseline answers | shared labels are descriptive, not causal',left=.16,right=.93,bottom=.33,wspace=.65)
    a,b=aa;y=np.arange(4);colors=[BLUE,OCHRE,CORAL,NAVY];a.barh(y,counts,color=colors,height=.58)
    for i,v in enumerate(counts):a.text(v+.8,i,f'{v} / 60',va='center',fontsize=10.7)
    a.set_yticks(y,en);a.invert_yaxis();a.set_xlim(0,60);a.set_xlabel('Answers carrying a tag');a.grid(False);a.grid(axis='x');panel(a,'A','Tag frequency')
    im=heatmap(b,co,en,en,vmin=0,vmax=60,fmt='.0f');b.tick_params(axis='x',labelrotation=32);colorbar(fig,im,b,'Questions',ticks=[0,20,40,60]);panel(b,'B','Tag co-occurrence')
    fig.text(.16,.135,f'No labeled error: {clean} / 60 answers',fontsize=11,weight='bold',color=NAVY)
    note(fig,'Diagonal cells are marginal counts. Agent labels are zero in this baseline review, not evidence of error-free Agent behavior.',x=.16,y=.025)
    save(fig,'fig17_badcase',['baseline_ai_scores'],'60题原始多标签频数与共同出现次数；对角为边际计数、标签非互斥。Agent标签为0仅适用于该基线评审，不能推断专项Agent无错误。',{'multilabel':True,'denominator':60,'counts':dict(zip(labels,map(int,counts))),'no_labeled_error':int(clean),'cooccurrence':co.tolist()})


def ui_figures():
    source_dir=OUT/"original-ui"
    cjk_path=Path('C:/Windows/Fonts/msyh.ttc')
    ui_font=font_manager.FontProperties(fname=str(cjk_path)) if cjk_path.exists() else font_manager.FontProperties(family=['Microsoft YaHei','Noto Sans CJK SC'])
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
                        ha="center",va="center",fontsize=9.5,fontproperties=ui_font,color=PALETTE[j],
                        arrowprops={"arrowstyle":"-|>","lw":1.1,"color":PALETTE[j],"connectionstyle":"angle3,angleA=90,angleB=0"},
                        bbox={"boxstyle":"round,pad=.32","fc":"white","ec":GRID,"lw":.7},annotation_clip=False)
        fig.text(.5,.035,"2026-10-10 历史运行截图 · 仅外部标注功能区 · 保留原始页面内容",ha="center",fontsize=9.5,fontproperties=ui_font,color=MUTED)
        save(fig,name,["original-ui/"+fn],"2026-10-10历史真实页面，非2026-10-15新生成回答。",{"original_image_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"annotation":"external labels and arrows; original screenshot pixels unchanged"})



def main():
    setup()
    OUT.mkdir(exist_ok=True)
    for task in [architecture,dataset,parsers,chunking,embeddings,vectorstores,retrieval,retrieval_ci,
                 retrieval_categories,rrf,ai_quality,gen_parameters,prompt,agent,runtime_probes,local_run,badcase,ui_figures]:
        task();print(task.__name__,"done",flush=True)
    from build_advanced_figures import build_advanced
    build_advanced(D, save)
    local=OUT/"ui05_local.jpg"
    if local.exists():
        MANIFEST.append({"id":"ui05_local","files":["figures/ui05_local.jpg"],"source":"reports/deployment/local-20261015/本地界面运行截图.jpg","sha256":hashlib.sha256(local.read_bytes()).hexdigest(),"note":"2026-10-15真实本地运行页面；既有历史会话不视为此次新生成。"})
    result={"created_date":"2026-10-17","visual_version":"v2","builder":"build_figures.py","font_size_pt":11.5,
            "palette":PALETTE,"data_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "source_hashes":D.get("source_hashes",{}),"figures":MANIFEST,"bootstrap_intervals":BOOTSTRAP}
    (OUT/"figure_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Figure manifest:",len(MANIFEST),"items",flush=True)

if __name__=="__main__":main()
