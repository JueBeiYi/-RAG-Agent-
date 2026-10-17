"""Rebuild the self-contained main.tex from the saved experimental data.

Editing main.tex directly is supported; rerunning this script replaces it.
No model calls, benchmark runs, or bibliography service are required.
"""
from pathlib import Path
import json
import re

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'data/report-data.json').read_text(encoding='utf-8'))
TABLES=D['tables']
A=json.loads((HERE/'data/advanced-statistics.json').read_text(encoding='utf-8'))


class Raw(str):pass


def esc(s):
    if isinstance(s,Raw):return s
    m={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(m.get(c,c)for c in str(s))


def table(title,label,headers,rows,fmt=None,note='',best=None):
    fmt=fmt or ('X'+'r'*(len(headers)-1))
    # Highlight observed quality maxima only within the fixed table comparison.
    # Counts, latency, and costs are not automatically treated as objectives.
    display=[list(row)for row in rows]
    if best:
        for col,direction in best.items():
            values=[float(row[col])for row in rows]
            optimum=(max if direction=='max'else min)(values)
            for index,value in enumerate(values):
                if abs(value-optimum)<1e-10:
                    display[index][col]=Raw(r'\textbf{'+esc(display[index][col])+'}')
        note+=' '+r'\textbf{粗体}为本表固定口径下的观察最优值（并列保留），不代表总体最优或显著差异。'
    body='\n'.join(' & '.join(esc(v)for v in row)+r' \\' for row in display)
    return (r'\begin{table}[htbp]\centering'+'\n'+r'\caption{'+title+r'}\label{'+label+'}\n'+
        r'\ReportTableStyle'+'\n'+
        r'\begin{tabularx}{\linewidth}{@{}'+fmt+r'@{}}\toprule'+'\n'+
        ' & '.join(esc(h)for h in headers)+r' \\ \midrule'+'\n'+body+'\n'+
        r'\bottomrule\end{tabularx}'+'\n'+(r'\par\vspace{2pt}\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}'+'\n' if note else '')+
        r'\end{table}')


def rows(key):return TABLES[key]['rows']


def f(v,n=4):return f'{v:.{n}f}'


def dynamic():
    out={}
    out['DATASET']=table('题集构成与固定划分','tab:dataset',['题型','开发集','测试集','总计'],
        [[name,D['dataset_manifest']['split_category_counts'][f'dev/{key}'],D['dataset_manifest']['split_category_counts'][f'test/{key}'],D['dataset_manifest']['category_counts'][key]] for key,name in [('factual','事实型'),('comparison','比较型'),('synthesis','综合型'),('reasoning','推理型')]]+[['合计',20,60,80]],note='开发集用于配置探索；全部标签在资料整理阶段已由助手接触，测试集结果不作为独立盲测。来源E02。')
    out['CORPUS']=table('固定版本论文语料及物理页数','tab:corpus',['论文简称','固定arXiv版本','发表信息','物理页'],
        [[Raw(esc(p['paper_key'].upper())+r'\cite{'+p['paper_key']+'}'),p['arxiv_version'],f"{p['venue']} {p['publication_year']}",p['physical_pages']]for p in D['corpus_manifest']['papers']],fmt='Xlll',note='总计10篇、169个PDF物理页。发表年份与arXiv首次提交年份分别保留；引用与页码均对应表列固定版本。来源E01。')
    out['PARSERS']=table('三种PDF解析器的实测结果','tab:parsers',['解析器','累计耗时/s','锚点保持','表格块','公式块'],
        [[p['parser'],f(p['parse_ms']/1000,3),f"{p['matched_anchors']}/{p['total_anchors']}",p['table_blocks'],p['formula_blocks']]for p in rows('parsers')],note='每个解析器处理相同10篇论文；锚点保持率是代理指标，不代表完整版面、表格或公式识别准确率。来源E03。')
    out['CHUNKING']=table('同一m3e模型下的五种分块配置比较','tab:chunking',['分块配置','片段数','Hit@5','MRR','Recall@5'],
        [[p['variant'],p['chunk_count'],f(p['hit_at_5']),f(p['mrr']),f(p['recall_at_5'])]for p in rows('chunking')if 'm3e' in p['model']],best={2:'max',3:'max',4:'max'},note='开发集20题；混合检索（Dense、BM25与RRF），每种切分使用其实际索引。分块单位为项目定义的轻量token，不等同于模型tokenizer。来源E04。')
    out['EMBEDDINGS']=table('Embedding模型质量与查询编码代价','tab:embedding',['模型','维度','Hit@5','MRR','Recall@5','编码/ms'],
        [[p['model'].split('/')[-1],p['dimension'],f(p['hit_at_5']),f(p['mrr']),f(p['recall_at_5']),f(p['query_encoding_amortized_ms'],2)]for p in rows('embeddings')],best={2:'max',3:'max',4:'max'},note='同342片段、开发集20题、递归512配置。最后一列为20个实际查询的编码总时间除以20，不包含模型加载，亦不是逐请求在线延迟。此表没有比较BGE-M3。来源E05。')
    out['VECTORS']=table('相同真实向量下的存储后端比较','tab:vectors',['后端','构建/ms','查询均时/ms','Hit@5','MRR','Recall@5'],
        [[p['backend'],f(p['build_and_persist_ms'],2),f(p['latency_mean_ms'],3),f(p['hit_at_5']),f(p['mrr']),f(p['recall_at_5'])]for p in rows('vectorstores')],best={3:'max',4:'max',5:'max'},note='历史359片段、1024维真实向量及同20个开发集查询；查询时间排除共享编码。删除与恢复探针均通过。规模有限，不能据此证明后端普遍优劣。来源E06。')
    mode={'vector':'向量','hybrid':'混合RRF','hybrid_rerank':'混合加重排'}
    out['RETRIEVAL']=table('冻结测试集上的三档检索结果','tab:retrieval',['模式','Hit@5','MRR','Recall@5','均时/ms','P95/ms'],
        [[mode[p['mode']],f(p['hit_at_5']),f(p['mrr']),f(p['recall_at_5']),f(p['latency_mean_ms'],2),f(p['latency_p95_ms'],2)]for p in rows('retrieval_test')],best={1:'max',2:'max',3:'max'},note='每模式60题，历史359片段；各模式执行错误数均为0。预计算查询编码不包含在排序耗时内；BGE-M3全部80题查询编码均摊为112.915 ms/题。来源E07。')
    paired=[]
    paired_mode={'vector':'向量','hybrid':'混合','hybrid_rerank':'混合+重排'}
    for p in A['paired_bootstrap']['effects']:
        percent=p['metric']!='mrr'
        scale=100 if percent else 1
        precision=2 if percent else 4
        metric={'hit_at_5':'Hit@5/百分点','mrr':'MRR/原量纲','recall_at_5':'Recall@5/百分点'}[p['metric']]
        paired.append([paired_mode[p['from']]+' → '+paired_mode[p['to']],metric,
                       f"{p['mean_delta']*scale:+.{precision}f}",
                       '['+f(p['ci95'][0]*scale,precision)+', '+f(p['ci95'][1]*scale,precision)+']'])
    out['PAIREDEFFECTS']=table('同题配对的检索效果差及其题目级区间','tab:paired-effects',
        ['比较（后者减前者）','指标/单位',Raw(r'$\Delta$'),'95%区间'],paired,
        fmt='p{4.1cm}p{3.5cm}rR',note='各比较使用同60题；同次抽样索引应用于全部模式。'+
        str(A['paired_bootstrap']['replicates'])+'次percentile重采样，seed='+str(A['paired_bootstrap']['seed'])+
        '。混合指RRF融合；Hit与Recall列按百分点换算，MRR不换算。区间仅刻画题目构成不确定性，不作显著性检验。来源E07及派生统计快照。')
    rrf=rows('rrf_grid')
    out['RRF']=table('开发集RRF常数与截断深度比较','tab:rrf',['RRF常数','评价K','Hit@K','Recall@K','MRR@20'],
        [[p['rrf_k'],p.get('metric_k',p.get('top_k')),f(p.get('hit',p.get('hit_at_k'))),f(p.get('recall',p.get('recall_at_k'))),f(p['mrr_at_20'],6)]for p in rrf],note='同20个开发集问题，复用实际编码和检索排名，无新增模型调用。MRR固定在候选深度20定义，不随评价K重新计算。来源E08。')
    q=D['baseline_ai']; labels={'correctness_1_to_5':'正确性','completeness_1_to_5':'完整性','citation_quality_1_to_5':'引用支撑','fluency_1_to_5':'流畅性'}
    out['AIQUALITY']=table('60条实际短答案的AI辅助质量评审','tab:quality',['维度','均分/5','1分','2分','3分','4分','5分'],
        [[labels[k],f(q['means'][k],2),*[q['score_distribution'][k].get(str(i),0)for i in range(1,6)]]for k in labels],note='每维60条，评分日期2026年10月14日；独立人工评分为0。1—5分是有序量表，不是答案准确率，也不是教师课程总分。来源E09。')
    cats={'factual':'事实型','comparison':'比较型','synthesis':'综合型','reasoning':'推理型'}
    out['QUALITYBYCATEGORY']=table('四类题目的AI辅助分维度均分','tab:quality-category',
        ['题型',Raw('$n$'),'正确性','完整性','引用支撑','流畅性'],
        [[cats[p['category']],p['n'],*[f(p[k],2)for k in ['correctness','completeness','citation_quality','fluency']]]
         for p in A['ai_quality']['category_means']],
        note='均分范围1—5，按题目等权计算；四类样本数合计60。不同题型并非同难度配对任务，不据均分差异推断题型的固有难度。评分来源与表\\ref{tab:quality}一致。来源E09及派生统计快照。')
    param=[p for p in D['generation_quality']['conditions']if p['study']=='parameters']
    out['PARAMETERS']=table('生成参数的质量与成本对比','tab:parameters',[Raw('$T/p/k_s$'),'正确','完整','引用','流畅','生成/s','输出/条'],
        [[f"{p['temperature']:.1f}/{p['top_p']:.1f}/{p['top_k_sampling']}",*[f(p[k+'_mean'],2)for k in ['correctness','completeness','citation_quality','fluency']],f(p['latency_mean_ms']/1000,3),f(p['completion_tokens_mean'],2)]for p in param],best={1:'max',2:'max',3:'max',4:'max'},note='各条件仅4道开发集题，每题单次生成，未固定采样seed；输出上限256 token。$k_s$为采样Top-k，与检索返回Top-K不同。质量来源为AI评审。来源E10。')
    prompt=[p for p in D['generation_quality']['conditions']if p['study']=='prompt']
    out['PROMPT']=table('提示词对照的分维度结果','tab:prompt',['模板','正确性','完整性','引用支撑','流畅性','生成/s'],
        [['简化对照' if 'naive' in p['condition']else '当前证据约束',*[f(p[k+'_mean'],2)for k in ['correctness','completeness','citation_quality','fluency']],f(p['latency_mean_ms']/1000,3)]for p in prompt],best={1:'max',2:'max',3:'max',4:'max'},note='每模板4条；当前模板复用参数基线回答，不是额外4次生成。新旧实验合计28个条件记录、24次历史生成调用、17个独立答案评审。来源E10。')
    out['AGENT']=table('固定RAG与Agent探针的实际资源记录','tab:agent',['路径','次数','均时/s','P95/s','输入合计','输出合计'],
        [['固定RAG' if p['condition']=='fixed_rag'else 'Agent',p['run_count'],f(p['latency_mean_ms']/1000,3),f(p['latency_p95_ms']/1000,3),p['prompt_tokens_total'],p['completion_tokens_total']]for p in rows('agent')],note='同5个探针、每路径5次；Agent路由exact-match为1.00，仅说明给定探针的工具选择，不等于答案正确率。计时为冻结检索后的生成与工具流程。来源E11。')
    out['PROBES']=table('本地工具并行与缓存功能探针','tab:probes',['条件','次数','均时/ms','输入token','输出token'],
        [[{'workers_1':'串行本地工具','workers_2':'双线程本地工具','cold_exact_query':'缓存冷请求','hot_exact_repeat':'精确重复命中'}[p['condition']],p['run_count'],f(p['latency_mean_ms'],3),p['prompt_tokens_total'],p['completion_tokens_total']]for p in rows('parallel')+rows('cache')],note='并行探针不调用LLM；缓存每条件仅1次，不能外推语义近邻命中率或稳定加速倍数。来源E12。')
    r=D['local_runtime'];m=r['generation_metrics']
    out['LOCAL']=table('2026年10月15日本地独立流程验收','tab:local',['项目','实际记录'],
        [['文档与片段','CvT原文PDF，独立工作区29片段'],['模型与来源',r['actual_model']+'；knowledge_rag'],['运行状态','33类检查通过；0失败'],['候选与结果','候选20条，返回5条'],['端到端问答',f(m['end_to_end_ms']/1000,2)+' s'],['检索与重排',f(m['retrieval_ms']/1000,2)+' s'],['首个token',f(m['first_token_ms']/1000,2)+' s'],['实际输入/输出',str(m['input_tokens'])+' / '+str(m['output_tokens'])+' token'],['自动审校',str(r['quality_review_warning_count'])+'条待核查提示']],fmt='p{3.2cm}X',note='正式操作时间为北京时间17:21:16—17:23:24；33类检查对应120条含轮询的断言记录。这里只验收一次真实流程，不作为完整题集成绩或稳定时延。来源E14。')
    pk=D['environment_snapshot']['packages'];out['ENV']=table('报告整理时记录的本地软件环境','tab:environment',['组件','版本','组件','版本'],
        [['Python',D['environment_snapshot']['python'],'PyTorch',pk['torch']],['FastAPI',pk['fastapi'],'Streamlit',pk['streamlit']],['Sentence Transformers',pk['sentence-transformers'],'Transformers',pk['transformers']],['Chroma',pk['chromadb'],'FAISS CPU',pk['faiss-cpu']],['PyMuPDF',pk['PyMuPDF'],'NumPy',pk['numpy']]],fmt='XlXl',note='此表为实际本地环境快照；依赖声明中的版本区间与本表已安装版本不是同一概念。模型权重版本另由manifest和指纹记录。')
    bib=[]
    for p in D['corpus_manifest']['papers']:
        authors='; '.join(p['authors'][:3])+('; et al'if len(p['authors'])>3 else '')
        bib.append(r'\bibitem{'+p['paper_key']+'} '+esc(authors)+'. '+esc(p['title'])+'. '+esc(f"{p['venue']}, {p['publication_year']}")+r'. arXiv:'+p['arxiv_version']+r'. \url{'+p['version_url']+'}.')
    out['CORPUSBIB']='\n\n'.join(bib)
    return out


TEX=r'''% !TeX program = xelatex
% UTF-8. Compile twice with XeLaTeX; figures and data are local to this folder.
\documentclass[UTF8,a4paper,zihao=-4,fontset=fandol]{ctexart}
\usepackage[top=2.54cm,bottom=2.54cm,left=3.175cm,right=3.175cm,headheight=16pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,longtable,array}
\usepackage[table]{xcolor}
\usepackage{colortbl,etoolbox}
\usepackage{caption,subcaption,fancyhdr,enumitem,float,placeins,setspace,listings}
\usepackage{hyperref}
\setCJKmainfont{FandolFang-Regular.otf}[BoldFont=FandolHei-Regular.otf,ItalicFont=FandolKai-Regular.otf]
\definecolor{Navy}{HTML}{17324D}
\definecolor{Teal}{HTML}{15877F}
\definecolor{Pale}{HTML}{EDF4F6}
\definecolor{TableHead}{HTML}{EDF2F5}
\hypersetup{unicode=true,colorlinks=true,linkcolor=Navy,citecolor=Teal,urlcolor=Teal,
 pdftitle={智能科研助理 基于RAG与Agent的论文知识库问答系统},pdfauthor={赵志国 林骋 林弟伟}}
\graphicspath{{figures/}}
\setstretch{1.5}
\AtBeginDocument{\zihao{4}}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.15em}
\setlength{\emergencystretch}{2em}
\setcounter{tocdepth}{2}
\ctexset{section={format=\heiti\zihao{3},name={,},number=\chinese{section},aftername=\quad},
 subsection={format=\heiti\zihao{4}},subsubsection={format=\heiti\zihao{-4}}}
\renewcommand{\contentsname}{\heiti\zihao{2}目\quad 录}
\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=6pt}
\captionsetup[table]{position=top,skip=7pt}
\setlength{\heavyrulewidth}{.65pt}
\setlength{\lightrulewidth}{.35pt}
\newcommand{\ReportTableStyle}{\fontsize{10.5}{14}\selectfont\setstretch{1.15}%
 \setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.28}}
\AtBeginEnvironment{table}{\setlength{\abovecaptionskip}{6pt}\setlength{\belowcaptionskip}{7pt}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlist[itemize]{leftmargin=2em,itemsep=1pt,topsep=4pt}
\setlist[enumerate]{leftmargin=2.5em,itemsep=1pt,topsep=4pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small 南京农业大学生产实习课程报告}
\fancyhead[R]{\small 智能科研助理}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible,keepspaces=true,
 frame=single,rulecolor=\color{black!20},backgroundcolor=\color{Pale},showstringspaces=false}
\newcommand{\reportfigure}[4][.48\textheight]{\begin{figure}[htbp]\centering
 \includegraphics[width=\linewidth,height=#1,keepaspectratio]{#2}
 \caption{#3}\label{#4}\end{figure}}
\newcommand{\status}[1]{\textbf{#1}}
\makeatletter
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{12pt}\setlength{\@fpbot}{0pt plus 1fil}
\makeatother
\begin{document}
\hypersetup{pageanchor=false}
\pagenumbering{gobble}
\begin{titlepage}\thispagestyle{empty}\centering
 {\heiti\zihao{1}南京农业大学}\par\vspace{1.1cm}
 {\heiti\zihao{2}课程考核情况表}\par\vspace{1.1cm}
 {\zihao{4}2026—2027学年第1学期}\par\vspace{.9cm}
 \renewcommand{\arraystretch}{1.9}
 \begin{tabularx}{\linewidth}{|p{3cm}|X|}\hline
 学院 & 智慧农业学院（人工智能学院）\\\hline
 课程名称 & 生产实习\\\hline
 学分 & 1\\\hline
 考核方式 & 课程报告\\\hline
 考核内容 & 智能科研助理——基于RAG与Agent的论文知识库问答系统\\\hline
 学生姓名 & 赵志国、林骋、林弟伟\\\hline
 成绩 & \rule{0pt}{1.1cm}\\\hline
 教师签名 & \rule{0pt}{1.1cm}\\\hline
 \end{tabularx}
 \vfill\raggedright{\small 实际提交日期：\underline{\hspace{4cm}}\quad（提交时填写）}
\end{titlepage}
\begin{titlepage}\thispagestyle{empty}\centering
 {\heiti\zihao{1}生产实习课程报告}\par\vspace{2cm}
 {\heiti\zihao{2}智能科研助理}\par\vspace{.5cm}
 {\heiti\zihao{3}基于RAG与Agent的论文知识库问答系统}\par\vspace{2cm}
 \renewcommand{\arraystretch}{1.6}
 \begin{tabular}{lll}\toprule 姓名 & 班级 & 学号\\\midrule
 赵志国 & 人工智能231 & 11523122\\
 林骋 & 人工智能231 & 11523121\\
 林弟伟 & 人工智能232 & 11523120\\\bottomrule\end{tabular}
 \vfill
 智慧农业学院（人工智能学院）\par
 南京农业大学\par\vspace{.4cm}
 编制日期：2026年10月15日
\end{titlepage}

\pagenumbering{Roman}\hypersetup{pageanchor=true}
\section*{摘要}\addcontentsline{toc}{section}{摘要}
论文知识库问答既需要覆盖原文证据，也需要控制生成过程中的事实归属、引用位置与工具调用。
本项目面向通用论文阅读任务，构建由文档处理、混合检索、检索增强生成、手写Agent控制及评测记录组成的科研助理。
系统支持多格式导入、三种分块策略、向量与BM25候选融合、交叉编码器重排、流式回答和物理页引用；使用原始token-ID分窗处理超长输入，并记录模型、索引和配置指纹。
在视觉Transformer专题的10篇固定版本论文上建立80题评测集，区分20题开发集与60题测试集。
历史359片段索引上的测试结果显示，向量检索、混合检索和混合加重排的Hit@5分别为0.4167、0.4333和0.6167；CPU重排同时带来较明显的时延代价。
60条实际短答案的AI辅助评审显示，流畅性均分为4.58，而完整性和引用支撑分别为2.98和3.03，表明证据覆盖与断言约束仍是主要瓶颈。
四题提示词对照中，引用支撑均分由2.50增至3.50，正确性均分不变；该结果仅作为小样本探索性观察。
本地标准模型流程完成33类实际检查，覆盖论文导入、真实生成、引用预览及记录保存。
本文分别报告历史索引、现有生产语料和独立流程验收的适用范围，不将AI评分表述为独立人工盲评，也不将单次运行或功能探针外推为总体性能结论。
\par\noindent\textbf{关键词：}检索增强生成；工具调用；论文知识库；混合检索；可追溯评测

\clearpage
\section*{Abstract}\addcontentsline{toc}{section}{Abstract}
\begin{spacing}{1.25}\small
This project develops a research assistant that combines retrieval-augmented generation with explicit tool orchestration.
The implementation preserves document locations across parsing, chunking, retrieval, generation, and citation inspection, while recording model and index fingerprints.
A benchmark of 80 source-checked questions is constructed from ten fixed-version Vision Transformer papers, with 20 development and 60 test questions.
On a historical frozen index of 359 chunks, Hit@5 increases from 0.4167 for dense retrieval to 0.6167 with hybrid retrieval and cross-encoder reranking, at a substantial CPU latency cost.
AI-assisted assessment of 60 generated short answers reveals stronger fluency than completeness or citation support.
A four-question prompt comparison improves mean citation support from 2.50 to 3.50 without changing mean correctness; the sample is insufficient for claims of general superiority.
An independent local workflow passes 33 categories of operational checks, including real PDF ingestion, model generation, citation preview, and persistence of interaction records.
Historical retrieval evaluation, generation-only studies, and current local acceptance are reported separately.
AI-assisted ratings are identified by provenance, and operational success is not interpreted as semantic verification of every answer.
\par\noindent\textbf{Keywords:} retrieval-augmented generation; tool orchestration; paper knowledge base; hybrid retrieval; traceable evaluation
\end{spacing}
\clearpage\tableofcontents\clearpage
\pagenumbering{arabic}

\section{项目概述}
\subsection{项目背景}
科研阅读需要在多篇论文之间核对方法、训练条件和实验结果。仅依赖语言模型记忆容易产生事实与来源不一致的问题，单纯关键词检索又难以直接完成归纳和比较。因此，项目将可定位的论文证据检索与受约束的生成、计算和工具协同结合，以支持可复查的科研问答。

\subsection{任务定义与项目目标}
设当前工作区论文集合为$\mathcal D$，问题为$q$，有界会话历史为$H$，系统输出为
\begin{equation}
 (a,\mathcal C,\mathcal L)=F(q,H,\mathcal D;\theta),
\end{equation}
其中$a$为答案，$\mathcal C$为绑定原文位置的来源集合，$\mathcal L$为实际执行与资源记录，$\theta$包含模型、索引和生成配置。
项目目标是使文档可导入、证据可检索、回答可引用、动作可观察、实验可复核。
检索分数不解释为事实正确性的概率，来源编号存在也不等价于该来源支持全部断言。

RAG通过外部检索证据补充生成模型的参数知识\cite{rag}；ReAct提供动作与观察交替的工具使用范式\cite{react}。
本文据此实现可检查的编排过程，而不是以现成Agent框架替代课程要求的核心控制逻辑。
系统面向通用论文知识库，评测专题限定为视觉Transformer架构、训练与数据效率，不把专题限制固化为产品使用范围。

\subsection{技术栈与实现边界}
服务层采用FastAPI，交互层采用Streamlit，状态数据保存于SQLite；文档适配器使用PyMuPDF、pdfplumber、PyPDF2及python-docx。
向量编码与交叉编码器推理由Sentence Transformers提供，向量存储适配Chroma与FAISS，稀疏检索调用BM25Okapi。
本地生成通过Ollama调用Qwen2.5-7B-Instruct的Q4\_K\_M权重，实际模型标签为\texttt{research-qwen:7b}。
项目自行实现分块来源传播、RRF融合、原token-ID窗口控制、引用规则审校、缓存准入和工具控制器。
联网搜索属于课程可选能力，未将其作为八个本地工具之一。

\section{团队分工说明}
\begin{table}[htbp]\centering
\caption{成员分工安排与约定贡献比例}\label{tab:team}
\small\renewcommand{\arraystretch}{1.35}
\begin{tabularx}{\linewidth}{@{}p{2.0cm}p{1.2cm}Yp{1.8cm}r@{}}\toprule
角色 & 姓名 & 核心职责安排 & 模块 & 比例\\\midrule
检索与RAG & 赵志国 & 文档处理、证据检索、生成与服务编排 & 文档与问答 & 40\%\\
Agent与工具 & 林骋 & 工具注册、路由、记忆及异常终止控制 & Agent & 30\%\\
交互与评测 & 林弟伟 & 页面流程、实验整理、文档与展示材料 & 集成与评测 & 30\%\\\bottomrule
\end{tabularx}
\end{table}
表\ref{tab:team}的姓名、班级、学号与贡献比例依据小组提供的信息整理，比例合计100\%。
职责细分为本报告采用的分工安排，提交前应由成员按实际工作确认。
现有仓库保存真实版本历史，但不能据此推断三位成员各自的独立提交数量；本文不以自动生成记录代替个人工作量证明。

\section{模块交付物清单}
交付状态区分代码实现、真实执行和评价来源。以下五表对应课程模块，不将“提供脚本”视为“已完成环境验收”。

\begin{table}[htbp]\centering\caption{模块一 文档处理与检索交付物}\label{tab:module1}\small
\begin{tabularx}{\linewidth}{@{}p{2.8cm}p{1.7cm}Y@{}}\toprule 交付项 & 状态 & 证据与范围\\\midrule
多格式解析与三解析器 & 已实现 & PDF、DOCX、TXT、MD；10篇论文的30次解析记录，E03\\
三类分块与长度比较 & 已实现 & 固定256/512/1024、递归、句段边界；同20题dev，E04\\
两类Embedding比较 & 已实现 & m3e与BGE-large-zh；生产另用BGE-M3，E05\\
向量库与混合重排 & 已实现 & Chroma/FAISS、BM25、手写RRF及CrossEncoder，E06—E08\\\bottomrule\end{tabularx}\end{table}
\begin{table}[htbp]\centering\caption{模块二 RAG问答交付物}\label{tab:module2}\small
\begin{tabularx}{\linewidth}{@{}p{2.8cm}p{1.7cm}Y@{}}\toprule 交付项 & 状态 & 证据与范围\\\midrule
提示词与上下文构造 & 已实现 & 预算控制、论文轮转、来源同步裁剪，E10\\
SSE与引用追溯 & 已实现 & 增量回答、物理页预览、原件哈希核对，E14\\
缓存与失败处理 & 已实现 & 指纹约束准入、精确重复探针、显式降级，E12\\
生成参数质量比较 & 已完成AI评审 & 五条件各4题；不作独立人工或显著性声明，E10\\\bottomrule\end{tabularx}\end{table}
\begin{table}[htbp]\centering\caption{模块三 Agent交付物}\label{tab:module3}\small
\begin{tabularx}{\linewidth}{@{}p{2.8cm}p{1.7cm}Y@{}}\toprule 交付项 & 状态 & 证据与范围\\\midrule
手写执行控制器 & 已实现 & schema校验、工具动作与观察、有界终止\\
八项本地工具 & 已实现 & 问答、元信息、比较、关键词、摘要、时间、计算和目录\\
记忆与错误恢复 & 已实现 & 会话隔离、窗口与摘录记忆、受限重试\\
路由与并行探针 & 已实测 & 5个路由样本；20次本地工具执行，E11、E12\\\bottomrule\end{tabularx}\end{table}
\begin{table}[htbp]\centering\caption{模块四 集成与交互交付物}\label{tab:module4}\small
\begin{tabularx}{\linewidth}{@{}p{2.8cm}p{1.7cm}Y@{}}\toprule 交付项 & 状态 & 证据与范围\\\midrule
四个工作台页面 & 已实现 & 论文库、问答、比较、实验与运行记录\\
会话及文档管理 & 已实现 & 上传进度、重试、元信息、会话搜索与导出\\
运行追踪与统计 & 已实现 & 状态、用量、模型与检索配置快照\\
本地标准模式验收 & 已通过 & 33类实际检查、0失败；单篇CvT独立工作区，E14\\\bottomrule\end{tabularx}\end{table}
\begin{table}[htbp]\centering\caption{模块五 评测与材料交付物}\label{tab:module5}\small
\begin{tabularx}{\linewidth}{@{}p{2.8cm}p{1.7cm}Y@{}}\toprule 交付项 & 状态 & 证据与范围\\\midrule
语料与评测题集 & 已建立 & 10篇固定论文、80题、四类问题，E01、E02\\
检索指标与Bad Case & 已执行 & 三档test60、四类错误、优化前后记录，E07、E15\\
答案质量评分 & 部分满足原要求 & 60题AI辅助评分已完成；独立人工评分未执行，E09\\
文档及演示材料 & 已提供 & 报告、设计、手册、实验附件与演示稿\\
Docker部署 & 文件已提供 & 本报告采用本地验收；容器构建及运行未实测\\\bottomrule\end{tabularx}\end{table}
\FloatBarrier

\section{技术实现记录}
\subsection{体系结构与数据流}
系统将文档处理路径与问答路径分开：前者形成可定位的片段和索引，后者在指定范围内检索、编排证据并产生回答。
SQLite保存文档、会话、运行与实验元数据，向量库保存可按工作区过滤的编码结果。
原文物理页与字符来源贯穿整个流程，使回答中的来源编号可以回到实际文件，而不是停留在模型生成的参考文献文字。
图\ref{fig:architecture}展示模块及其数据关系。
\reportfigure{fig01_architecture.pdf}{论文知识库与Agent系统的数据流。原文位置、配置指纹和执行记录与正文处理链路共同保存；框架能力与项目控制逻辑在实现中分开。}{fig:architecture}

\subsection{文档解析 分块与超长输入处理}
PDF解析输出统一来源块。能够获取布局信息的适配器保留物理页与边界框，阅读顺序采用全宽块划分阅读带、带内先左后右的启发式方法。
表格以逐行文本保存，公式保留可提取的Unicode及上标信息，跨页来源分别记录。
这些处理适用于具有文本层的PDF，不能表述为通用OCR、完整版面理解或无损公式重建。
DOCX按正文顺序读取段落和表格；缺失作者、年份或DOI时保留缺失状态。

固定窗口、递归边界和句段边界三类分块分别控制长度、优先保持文本边界及利用段落/句末结构。
递归策略实际调用\texttt{RecursiveCharacterTextSplitter}提出边界，再检验字符偏移和原子结构约束。
句段策略未使用Embedding模型预测语义边界，因此不将其描述为学习式语义切分。
默认长度512、重叠64，单位是项目定义的汉字、英文词或独立符号，与模型原生tokenizer分别管理。

为避免模型输入超限时静默截尾，编码器直接在原始token-ID序列上分窗，扣除特殊token预算，不进行decode再encode往返。
设第$j$个窗口含$w_j$个原始token，模型输出$f(x_j)$，片段编码为
\begin{align}
 u_j&=\frac{f(x_j)}{\max(\lVert f(x_j)\rVert_2,\epsilon)},\\
 v(d)&=\operatorname{norm}\!\left(\frac{\sum_{j=1}^{m}w_ju_j}{\sum_{j=1}^{m}w_j}\right).
\end{align}
实现覆盖全部窗口并检查输入约束；加权聚合并不等价于无限上下文编码。
模型缓存版本、窗口策略版本与窗口上限纳入指纹，以阻止不同编码策略误用已有向量。

\subsection{混合检索与重排算法}
密集检索使用归一化向量内积；Chroma的cosine距离转换为$1-\mathrm{distance}$，FAISS使用精确\texttt{IndexFlatIP}。
稀疏检索采用jieba、中文二元组与技术标识英文切分后调用BM25Okapi\cite{bm25}：
\begin{equation}
 B(q,d)=\sum_{t\in q}\widetilde{\mathrm{IDF}}(t)
 \frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/\overline{|d|})}.
\end{equation}
实际默认$k_1=1.5$、$b=0.75$；原始IDF采用$\log[(N-n_t+0.5)/(n_t+0.5)]$，负值按库实现替换为平均IDF的0.25倍。
本文据实际实现描述公式，不改写为另一种恒正IDF形式。

项目手写RRF仅使用候选名次，以避免不同检索分数的量纲影响\cite{rrf}：
\begin{equation}
 R(d)=\sum_{r\in\{\mathrm{dense},\mathrm{BM25}\}}
 \frac{\mathbb{I}[d\in L_r]}{k_0+\operatorname{rank}_r(d)}.
\end{equation}
名次从1开始，每路先去重，默认$k_0=60$；同分按片段标识稳定排序。
每路候选数默认20，融合后保留候选池，再返回5条或进入交叉编码器。
融合没有对原始分数做min--max归一化。

重排使用BGE-reranker-base联合编码问题与候选；长文档保留完整问题，以原token-ID窗口计算logit并取最大值：
\begin{equation}
 z(q,d)=\max_j g(q,d_j),\qquad s(q,d)=\frac{1}{1+\exp[-z(q,d)]}.
\end{equation}
$s$是单调阈值分数，未经概率校准；最大窗口聚合可能高估单个相关窗口的代表性。
重排失败时保留RRF顺序并记录降级原因，避免把未执行的重排记为成功。
生产环境采用BGE-M3编码\cite{bgem3}，与模块实验中的m3e和BGE-large-zh比较分别报告。

\subsection{证据约束生成 引用与缓存}
上下文构造扣除系统提示、历史、问题与预留开销后分配证据预算，按论文分组轮转以减少单篇来源挤占。
裁剪正文时同步裁剪来源及页区间；预算采用保守字符估算，模型服务返回的真实token另行记录。
默认上下文预算4096、生产输出预算768；生成对比实验为256输出token，二者不可混同。

服务器给来源分配\texttt{[S1]}等编号，绑定文档、片段、物理页和可用边界框。
生成后规则审校检查引用编号、数字对应和部分相邻页码表达，但不证明语义蕴涵。
无结果时明确标注普通模型回答，低相关度时提示核查，模型失败时保留失败状态或原文摘录降级状态。
SSE传递状态、来源、文本增量、用量与结束事件；多工具路径的综合回放与单一RAG的模型增量在事件中区分。

缓存首先匹配索引、模型、生成配置、文档范围、历史与提示词指纹，再检查问题匹配及相似度阈值。
默认阈值0.96，并约束数字、英文锚点、否定及意图一致；有待核查引用、时效或指代问题不写入缓存。
索引变化使旧指纹失效。此机制降低错误复用风险，但一次精确重复探针不足以测量总体缓存命中率。

\subsection{手写Agent与八工具协同}
注册表对工具名称、必填参数、类型和长度执行schema校验。八项工具的作用见表\ref{tab:tools}。
\begin{table}[htbp]\centering\caption{工具注册表及边界}\label{tab:tools}\small
\begin{tabularx}{\linewidth}{@{}p{4.4cm}Y@{}}\toprule 工具 & 能力与限制\\\midrule
knowledge\_qa & 指定论文范围内的证据问答\\
paper\_metadata & 读取实际保存的题名、作者、年份等字段\\
compare\_papers & 两篇论文的结构化比较；条件不齐时不直接排名\\
extract\_keywords & 文档或输入的词频候选，不把词频当论文结论\\
summarize\_paper & 依据证据生成结构化摘要\\
current\_time & 返回实际系统时间\\
calculator & 白名单算术AST解释，不执行任意Python\\
paper\_catalog\_search & 查询当前本地论文目录，不冒充互联网搜索\\\bottomrule
\end{tabularx}\end{table}

Agent的System Prompt将角色限定为基于本地论文库的科研助理，要求先调用工具查证，再依据已返回的来源回答。
工具说明及参数schema由注册表附入模型请求；原生工具调用不可用时，仅接受包含\texttt{tool\_calls}的严格JSON动作。
提示词约束论文标识来自当前范围、引用仅使用工具返回编号、独立动作可并行、依赖动作须等待观察，并禁止重复同参调用及隐瞒失败。
论文正文和工具结果被视为数据；最终回答不输出内部思维链，数值比较须说明任务、数据划分与指标条件。
这些约束与服务器的范围校验、动作去重及步数预算共同生效，不能仅凭提示词保证执行安全或内容正确。

控制器采用快捷路由与有界模型规划结合的实现。一般概念可走显式模型常识路径，单一论文事实问答进入流式RAG；其余任务由动作和工具观察交替推进。
两篇论文已被明确选定时，服务器固定比较范围，避免规划阶段编造文档标识。
独立工具可在受限线程池并行，存在依赖的动作等待前一步结果；并非每个问题都执行多轮规划。
规划不可用时采用透明的本地路由，不将快捷路径伪装为复杂Agent推理。

默认最多6步，限制单工具与请求总预算，按工具名和规范化参数检测重复动作。
仅在原调用已结束且属于临时错误时允许有界重试；超时但仍运行的线程继续占用槽位，避免推测性副本。
会话记忆由近期窗口与旧消息抽取摘录组成，默认最多8条近期消息、1400个估算token；派生记忆与原消息分开保存。
公开日志只呈现动作、参数、结果、耗时和状态，不输出模型内部思维链。

\begin{table}[htbp]\centering\caption{范围与预算约束下的证据增强执行过程}\label{tab:algorithm}\small
\begin{tabularx}{\linewidth}{@{}rY@{}}\toprule 步骤 & 执行逻辑\\\midrule
1 & 校验工作区和就绪文档范围，读取有界会话记忆。\\
2 & 判定普通模型、单一RAG或工具循环路径，并记录所选路径。\\
3 & 对知识检索生成dense与BM25候选，按名次执行RRF，可选窗口重排。\\
4 & 按论文轮转编排来源、分配编号，执行上下文与位置同步裁剪。\\
5 & 调用实际模型，输出SSE增量，审校来源编号及数字断言。\\
6 & 对多工具任务校验动作schema，执行受限并行、错误恢复与终止检查。\\
7 & 保存回答、来源、模型与索引快照及已知/未知用量，返回完整或部分结果。\\\bottomrule\end{tabularx}\end{table}

\subsection{交互实现与功能区说明}
界面由科研问答、论文库、论文对比、实验与运行记录四页组成。
文档导入呈现批量任务进度和重试，论文库提供元信息编辑、片段与原文查看；问答提供会话、范围、答案、来源和执行轨迹，运行页保留用量与实验配置。
以下四张为2026年10月10日留存的真实界面实现截图，按功能区作外部标注，保留原图内容。
它们用于说明界面结构，不作为10月15日新运行问题的回答证据；后者由独立HTTP记录和本地就绪截图支持。
\reportfigure{ui01_library.pdf}{论文库的文档管理、索引状态和操作区域。历史真实界面截图，日期为2026年10月10日；标注不改变原图内容。}{fig:ui-library}
\reportfigure{ui02_chat.pdf}{科研问答的会话与范围、答案和证据区域。截图中的历史回答不是本轮新生成结果，也不代表内容已经通过完整语义评分。}{fig:ui-chat}
\reportfigure{ui03_comparison.pdf}{论文对比的文档选择、比较维度与结构化结果区域。实验条件不完整时，界面保留不直接计算差值和排名的限制。}{fig:ui-comparison}
\reportfigure{ui04_evidence.pdf}{原文证据抽屉的来源、物理页和正文预览区域。页码依据服务器保存的原文元数据，而不是模型自行填写。}{fig:ui-evidence}
\FloatBarrier

\subsection{评测设计与指标定义}
\subsubsection{语料与题集}
选择视觉Transformer作为评测方向，收集表\ref{tab:corpus}中的10篇固定版本论文，涵盖架构、数据效率与训练策略。
语料共169个物理页，保存原始文件哈希；任务是检索论文信息，不复现论文模型的训练或分类准确率。
@@CORPUS@@
@@DATASET@@
\reportfigure{fig02_dataset.pdf}{四类问题的开发集和测试集构成。开发集20题、测试集60题；数量来自固定题集记录。}{fig:dataset}

参考答案和相关来源由助手依据本地论文逐题核对，相关性采用短文本锚点及物理页、来源段落约束。
锚点匹配便于复查，但可能遗漏语义等价证据，也可能把含有短锚点的邻近段落判为相关。
题集由开发和资料整理过程接触过的标注构成，因此本文不将测试结果描述为独立多人建立、完全不可见的盲测。

\subsubsection{指标与比较条件}
设第$i$题的标注来源证据单元集合为$G_i$，前$K$条检索结果覆盖的来源证据单元集合为$R_i^{(K)}$，首个相关项名次为$r_i$。
本项目报告
\begin{align}
 \operatorname{Hit@K}&=\frac1n\sum_{i=1}^{n}\mathbb I[R_i^{(K)}\cap G_i\ne\varnothing],\\
 \operatorname{Recall@K}&=\frac1n\sum_{i=1}^{n}\frac{|R_i^{(K)}\cap G_i|}{|G_i|},\\
 \operatorname{MRR}&=\frac1n\sum_{i=1}^{n}\frac{\mathbb I[r_i<\infty]}{r_i}.
\end{align}
实际相关项匹配遵循源记录中的文档、物理页及锚点规则；MRR使用实验候选深度内首个相关项，而不是自动视为MRR@5。
路由exact-match只比较预期工具集合与实际工具选择；用量与延迟分别保存已知token、估算标记和时间范围。
质量评分采用正确性、完整性、引用支撑与流畅性四维1—5分量表，保留AI辅助评审的真实来源。

正式三档检索和短答案生成使用历史359片段索引；分块/Embedding比较中的递归索引与10篇生产语料为342片段；本轮本地运行验收为独立CvT工作区29片段。
这些版本不能合并为一次统一端到端实验。
三档检索复用已真实编码的查询向量，生成研究复用真实命中，不能用其计时替代在线完整路径。

\subsection{解析 分块 模型与向量库实验}
@@PARSERS@@
\reportfigure{fig03_parsers.pdf}{相同10篇论文上的解析累计耗时和119个来源锚点保持率。检测块数与锚点保持均不是完整语义识别率。来源E03。}{fig:parsers}
PyPDF2的累计耗时较低，但锚点保持为114/119，且缺少块级坐标；其表格和公式块检测数量不能与布局解析器简单等价。
PyMuPDF与pdfplumber均保留全部既定锚点，也不能由此推断所有阅读顺序、表格或公式均正确。

@@CHUNKING@@
\reportfigure{fig04_chunking.pdf}{同一m3e模型和20题开发集上的分块配置比较。每个切分方案具有自己的真实片段索引。来源E04。}{fig:chunking}
固定1024在该开发集的Hit@5和MRR最高，递归512没有取得最高指标。
生产选择递归策略是来源结构、上下文预算与工程一致性的折中，不以实验数据未支持的“最优切分”解释。
长块可能增加包含相关锚点的机会，也可能稀释局部语义，因此需在更多题目与上下文预算下重复比较。

@@EMBEDDINGS@@
\reportfigure{fig05_embeddings.pdf}{一致递归索引下的模型质量与20题查询编码摊销时间。m3e和BGE-large-zh的MRR及Hit@5各有差异，不能单一排序为全面优劣。来源E05。}{fig:embedding}
m3e的Hit@5为0.4000，高于BGE-large-zh的0.3500；后者MRR较高，说明命中与排序指标需要分别解释。
文档编码吞吐统计的文本集合不同：m3e覆盖五种切分去重后的1836条，BGE-large为递归方案342条，因而不作同分布吞吐结论。

@@VECTORS@@
\reportfigure{fig06_vectorstores.pdf}{同一359片段与真实向量的后端质量和查询耗时。后端比较复用模型推理，结果仅代表当前规模与实现。来源E06。}{fig:vectors}
两个后端得到相同的检索质量，但持久化构建和查询耗时不同。
这些结果主要用于验证后端可替换性和本机行为，不足以推断大规模吞吐、召回近似误差或生产集群优势。

\subsection{三档检索消融与参数迭代}
@@RETRIEVAL@@
\reportfigure{fig07_retrieval.pdf}{同一历史359片段与60题测试集上的检索模式消融。重排提高证据覆盖，同时引入CPU推理成本。来源E07。}{fig:retrieval}
混合加重排的Hit@5较纯向量提高20.00个百分点，较混合RRF提高18.33个百分点，Recall@5也提高。
但其CPU排序均时约16.7秒，显著高于预编码向量的纯检索阶段；不能将纯向量3.20 ms写成在线问答时延。
误差区间采用对实际题目进行5000次percentile bootstrap、固定随机种子20261015。
图\ref{fig:ci}表示题目抽样不确定性，不是模型多次运行方差，也不据此宣称统计显著。
\reportfigure{fig08_retrieval_ci.pdf}{60题逐题重采样的95\% percentile区间。每个模式使用同样的实际问题集合；区间不替代重复运行或配对显著性检验。来源E07。}{fig:ci}
@@PAIREDEFFECTS@@
图\ref{fig:paired-effects}进一步使用同一题目的模式差值，三组比较的每次重采样均保留题目配对关系。
差值及其95\%区间用于描述本题集上的效果与题目构成不确定性，不作显著性判定。
\reportfigure{fig18_paired_effects.pdf}{三组检索模式比较的配对效果。横轴为后者减前者的指标差，线段为题目级配对bootstrap的95\%区间。图中均采用0—1指标的原量纲差值；Hit与Recall在表\ref{tab:paired-effects}换算为百分点，MRR不换算。来源E07及派生统计快照。}{fig:paired-effects}
\reportfigure[.58\textheight]{fig19_rank_matrix.pdf}{同60题在三种检索模式下的逐题首个相关项名次。题目按四类任务分组，颜色仅对应已检到项；null表示在保存的候选深度内未检到，不能填充为末位名次。来源E07。}{fig:rank-matrix}
\reportfigure{fig09_retrieval_categories.pdf}{按事实、比较、综合和推理问题分解的Hit@5。各类样本量不同；推理题在重排后仍保留较大漏召回空间。来源E07。}{fig:category}

@@RRF@@
\reportfigure{fig10_rrf.pdf}{RRF常数与评价截断深度的开发集网格。图中分别呈现Hit@K与MRR@20；后者按候选深度20计算，不随评价K重新定义。来源E08。}{fig:rrf}
RRF常数60改为20后，MRR@20由0.314796变为0.324717，而Hit@5为0.40、Recall@5为0.35，均未变化。
该轮实际优化改善了排名，未解决前五条的漏召回；本文保留这一有限改善而不使用“召回显著提高”的表述。

\subsection{答案质量 生成参数与提示词比较}
@@AIQUALITY@@
\reportfigure{fig11_ai_quality.pdf}{60条历史短答案的AI辅助量表评审。流畅性明显高于完整性与引用支撑；评分来源为AI，独立人工评审未执行。来源E09。}{fig:ai-quality}
四维均分分别为3.50、2.98、3.03和4.58。该分布说明语言流畅不能替代证据覆盖、数值正确及来源支持。
评分基于原回答、参考答案、实际引用映射和论文物理页，保留逐题评语，但未开展独立双人盲评或评分一致性分析。
课程原文要求的人工评分不能由AI来源标签的替换而得到满足；本文据实将该项列为评价限制。
@@QUALITYBYCATEGORY@@
\reportfigure{fig20_quality_association.pdf}{AI辅助评分的联合分布与秩相关。气泡依次表示正确性—完整性、正确性—引用支撑、流畅性—正确性的评分组合，面积为题数；热图为Spearman相关。其共同变化不证明因果联系或人工评分一致性。来源E09及派生统计快照。}{fig:quality-association}

@@PARAMETERS@@
\reportfigure{fig12_gen_parameters.pdf}{生成参数条件的四维AI评分。每条件4题，包含事实、比较、综合和推理；相同输出复用同一评分。来源E10。}{fig:parameters}
参数实验合计20条真实回答，提示词实验含旧模板4条及复用基线的当前模板4条。
28个条件记录对应24次历史调用，按问题、原回答和引用映射去重后有17个独立答案。
基线与采样Top-k为20的条件均值并列，但逐题错误构成不同，不能据此宣布某组参数稳定最优。
温度为0未消除综合题中的事实错误；较短回答的生成耗时较低，同时受输出长度、模型热身和条件执行顺序混杂。

@@PROMPT@@
\reportfigure{fig13_prompt.pdf}{简化对照与当前证据约束提示词的比较。正确性均值不变，完整性增加0.25、引用支撑增加1.00分；样本不足以支持普遍优势。来源E10。}{fig:prompt}
事实题中，旧回答正确给出ViT-Base配置，但引用位置与其自写页码不符；当前回答采用实际包含表1的来源，改善了引用支撑。
另一方面，综合题的冻结Top5均来自T2T，没有CvT原文，多个配置仍将普通ViT机制错误归给CvT。
该例表明，采样和提示词不能独立修复证据覆盖缺失，应在生成前检验所需论文是否均有可靠来源。
\reportfigure{fig21_prompt_pairs.pdf}{四题提示词比较的逐题配对分数。连接线对应同一题目，保留改善、持平与退化的实际结果；每题每模板单次生成，不由四题观测推断总体效果。来源E10及派生统计快照。}{fig:prompt-pairs}

\subsection{Agent 效率与本地运行验收}
@@AGENT@@
\reportfigure{fig14_agent.pdf}{5个探针上固定RAG与Agent的时延和token结构。路由匹配为1.00，不是答案正确率；不同路径的开销组成应分别解释。来源E11。}{fig:agent}
固定RAG与Agent各运行5次，Agent输入token合计较少而均时较高，说明减少输入并不保证端到端更快。
概念、计算、时间、关键词和论文事实五类探针只覆盖部分路由，不能据此给出全任务成功率。
@@PROBES@@
\reportfigure{fig15_runtime_probes.pdf}{本地独立工具的串行/双线程配对记录与精确重复缓存探针。缓存使用对数耗时坐标；样本量分别为10轮/方式和1次/条件。来源E12。}{fig:probes}
串行与双线程工具均时为24.125与21.161 ms，未包含LLM并发；缓存冷请求2955.515 ms，精确重复4.074 ms，后者新增模型token为0。
缓存样例说明功能生效，不作为总体命中率或稳定数百倍加速证据。
另有10次异质在线诊断请求，平均端到端52.089秒，输入/输出合计43908/4634 token，平均Agent步数1.2；任务不同，不构成同题受控负载测试。

@@LOCAL@@
\reportfigure{fig16_local_run.pdf}{独立CvT本地验收的单次问答资源记录。端到端、检索和首token计时互有包含，不能相加；该样例不代表完整题集均值。来源E14。}{fig:local}
本地标准模式实际访问前端8501、API 8000和项目模型服务11435，均绑定127.0.0.1，保留原系统11434服务。
验收核对了真实PDF上传、29片段索引、模型SSE回答、有效来源编号、原件SHA-256、引用页PNG及会话和调用保存。
33类检查均通过，轮询计入时共有120条通过断言；不能把它们写成120项独立功能。
本次回答仍有5条自动审校待核查提示，所以运行通过不表示全部语义正确。
\begin{figure}[htbp]\centering
\includegraphics[width=.62\linewidth,height=.48\textheight,keepaspectratio]{ui05_local.jpg}
\caption{2026年10月15日本地页面与模型就绪状态截图。截图仅证明页面加载状态；当日实际新问答以E14中的HTTP、SSE和调用记录为准。}\label{fig:local-ui}
\end{figure}

\subsection{Bad Case分析与优化边界}
\begin{table}[htbp]\centering\caption{四类Bad Case与工程输入窗口问题}\label{tab:badcases}\small
\begin{tabularx}{\linewidth}{@{}p{2cm}YY@{}}\toprule 类型 & 观察与根因 & 改进及剩余限制\\\midrule
检索失败 & 相近架构文本干扰，跨语言实体与跨论文覆盖不足 & 混合检索和配置探索；RRF改进只改善MRR，仍有漏召回\\
生成错误 & 自由格式输出、模型补写数值及事实归属错误 & schema与有界修复；格式可用仍不等于语义正确\\
引用错误 & 模型自写页码，来源编号与事实对应不严 & 服务器页码和来源核对；断言语义支持仍需复查\\
Agent决策 & 规划阶段编造文档标识或重复动作 & 固定明确文档范围、重复与预算终止；探针规模有限\\
超长窗口 & decode/re-encode导致长度膨胀与输入拒绝 & 原token-ID全覆盖分窗；不宣称等价于无限上下文编码\\\bottomrule\end{tabularx}\end{table}
\reportfigure{fig17_badcase.pdf}{60条AI评分记录的多标签错误分布。同一回答可同时计入检索、生成和引用错误，因此总数可超过60；“无明显问题”不等于全面语义认证。来源E09、E15。}{fig:badcase}
实际问题还包括Swin型号与数值错配、PVT下采样解释、LeViT结构与class token判断、算术错误以及训练条件归属颠倒。
这些问题不能仅依靠界面完善或降低温度消除，应优先核验引用覆盖、数值计算来源和跨论文实验条件。
除RRF同题优化外，结构化比较、服务端页码校正和窗口修复有定向回归或真实样例记录；样例关注点不完全一致时，只作诊断证据而不据其计算因果提升。
\FloatBarrier

\section{技术设计文档}
\subsection{模块接口与持久化}
API封装文档、任务、会话、运行和实验操作。文档任务状态包括queued、running、done与error；文档就绪后才能进入限定范围的检索。
回答、来源、执行事件与配置快照在同一运行记录中保存，便于从界面回到实际调用。
部分核心接口见表\ref{tab:api}，精确schema以项目实际API文档为准。
\begin{table}[htbp]\centering\caption{核心接口及数据用途}\label{tab:api}\small
\begin{tabularx}{\linewidth}{@{}p{1.2cm}p{5.4cm}Y@{}}\toprule 方法 & 路径 & 作用\\\midrule
GET & \path{/health} & 存活、模型及检索就绪\\
GET & \path{/config} & 不含API密钥的公开配置\\
POST & \path{/documents} & 批量上传及异步索引任务\\
GET & \path{/jobs/<id>} & 索引进度和实际错误\\
GET & \path{/documents/<id>/chunks} & 分页片段及位置\\
GET & \path{/documents/<id>/preview} & 论文物理页预览\\
POST & \path{/sessions/<id>/chat} & SSE问答与来源事件\\
GET & \path{/runs/<id>} & 回答、用量及执行追踪\\
GET & \path{/sessions/<id>/export} & 会话导出\\\bottomrule\end{tabularx}\end{table}

\subsection{配置与可复现控制}
配置由YAML和环境变量合并，公开配置排除云端API密钥。
应用默认上下文、输出、检索和Agent预算见表\ref{tab:config}；不同实验覆盖的值写入各自记录，不从当前默认值倒推历史配置。
\begin{table}[htbp]\centering\caption{主要配置与实验覆盖关系}\label{tab:config}\small
\begin{tabularx}{\linewidth}{@{}p{5.1cm}p{1.8cm}Y@{}}\toprule 配置项 & 取值 & 含义与范围\\\midrule
\texttt{chunk\_size/overlap} & 512/64 & 轻量分块单位；实验另有256、1024\\
\texttt{candidate\_k/result\_k} & 20/5 & 检索候选与最终来源数量\\
\texttt{rrf\_k} & 60 & 历史基线；开发集另比较20、100\\
\texttt{context\_tokens} & 4096 & 上下文预算配置，采用保守估算\\
\texttt{answer\_tokens} & 768 & 生产上限；生成研究为256\\
\texttt{temperature/top\_p} & 0.1/0.9 & 历史基线与实际条件分别记录\\
\texttt{top\_k\_sampling} & 40 & 采样参数，不是检索Top-K\\
\texttt{max\_agent\_steps} & 6 & 循环预算\\
\texttt{memory\_tokens} & 1400 & 历史记忆估算预算\\
\texttt{memory\_recent\_messages} & 8 & 近期窗口上限\\
\texttt{cache\_threshold} & 0.96 & 还需满足锚点、历史与指纹条件\\\bottomrule\end{tabularx}\end{table}

\subsection{隔离 错误与安全边界}
服务端对文档、会话、运行和实验读取进行工作区归属校验；随机标识提供本地逻辑隔离，不等于完整生产身份认证。
上传检查扩展名、空文件及大小，去除文件名路径部分，默认单文件50 MB、单批20份。
安全计算器使用白名单AST；论文正文和工具结果作为数据处理，不赋予其覆盖系统规则的权限。
未获得的失败调用用量保留未知，不填零。模型、工具或索引失败通过明确状态、警告及可检查的部分结果呈现。
本次部署采用回环地址，不据此推断互联网访问环境下的安全性或并发能力。

\subsection{扩展设计的价值与限制}
第一项扩展是版本感知的来源与运行记录：编码窗口版本、模型revision、语料哈希、索引快照与生成配置共同记录，用于识别缓存和实验是否可比。
其价值是降低跨版本误用风险，不是对所有历史执行的第三方审计认证。
第二项扩展是条件约束的结构化论文比较：对明确论文范围使用服务器标识，优先核对任务、数据集、指标及训练条件，不能确认可比时不计算差值或排名。
该设计改善结果组织，但条件抽取和事实核验仍可能出错，需以原文逐项复查。

\section{总结与展望}
\subsection{项目总结}
项目完成从多格式论文到可追溯回答的工程链路，并对文档、检索、生成、工具和运行过程建立实际记录。
实验支持的主要发现是：在冻结题集和索引条件下，交叉编码器提高前五条证据覆盖，但CPU代价明显；分块与Embedding不存在被所有指标一致支持的单一最优项；提示词引用约束的收益应按维度和题目解释。
本地标准模式的实际流程可用，并不消除完整性、数值归属和引用支撑方面的质量短板。

\subsection{困难与解决过程}
文档布局与表格公式导致来源和阅读顺序难以保持，解决思路是统一来源块并保留物理位置，而不是生成不存在的精确结构。
超长输入的decode/re-encode曾引起窗口膨胀，改用原ID分窗后完成实际模型验证。
规划动作可能编造文档标识，因而将明确论文范围交由服务器固定，并保留工具失败及恢复路径。
引用规则容易把来源存在误当语义正确，本报告将规则审校、量表质量评分和运行成功分别记录，保留未解决的例子。

\subsection{研究限制与后续工作}
评测仅覆盖一个专题10篇论文，题集及其相关性标注由助手整理，尚缺独立人工复核与评分一致性分析；因此外部有效性有限。
生成参数每条件4题、单次采样且未固定seed，存在输出长度和运行顺序混杂，不能支持稳定参数最优或显著性结论。
历史359片段的冻结成绩、342片段生产语料及29片段独立流程验收未构成统一的最新版全题集复测。
本地探针不等于并发负载测试，模型回答仍保留待核查项；容器配置已提供，但未完成容器实际验收。

后续优先检查跨论文证据覆盖，在证据缺失时补检索或明确不足，再对数值计算与逐条引用作约束。
在更大且经独立复核的题集上重复采样，固定硬件、版本和随机设置，分别评估质量、时延和资源成本。
最后再依据受控证据调整默认分块、重排候选规模和采样参数，避免以单一小样本均分替代工程决策。
\FloatBarrier

\phantomsection\addcontentsline{toc}{section}{参考文献}
\begin{spacing}{1.2}\small
\begin{thebibliography}{99}
\bibitem{rag} Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 2020. \url{https://arxiv.org/abs/2005.11401}.
\bibitem{react} Yao S, Zhao J, Yu D, et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR, 2023. \url{https://arxiv.org/abs/2210.03629v3}.
\bibitem{bm25} Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 2009, 3(4):333--389. doi:10.1561/1500000019.
\bibitem{rrf} Cormack G V, Clarke C L A, Büttcher S. Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR, 2009:758--759. doi:10.1145/1571941.1572114.
\bibitem{bgem3} Chen J, Xiao S, Zhang P, et al. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Findings of ACL, 2024:2318--2335. doi:10.18653/v1/2024.findings-acl.137. \url{https://aclanthology.org/2024.findings-acl.137/}.
@@CORPUSBIB@@
\end{thebibliography}\end{spacing}

\clearpage\section{附录}
\subsection{实际代码结构与复现环境}
\begin{lstlisting}
src/
  core/          config, schemas, storage
  data_loader/   document adapters and source blocks
  chunking/      fixed, recursive, sentence/paragraph splitters
  retrieval/     embeddings, RRF, vector stores, reranking
  generation/    context, prompts, citations, cache, LLM clients
  agent/         registry, router, executor, memory, ReAct loop
  services/      application orchestration
  api/           HTTP and SSE endpoints
  frontend/      Streamlit workspace pages
  evaluation/    datasets, metrics, reviews, exports
scripts/         preparation, execution, reporting, acceptance
tests/           unit, API, frontend, and integration checks
docker/          unverified container deployment configuration
\end{lstlisting}
@@ENV@@
本地运行环境使用Windows 11及NVIDIA GeForce RTX 4060 Laptop GPU；Embedding与重排配置为CPU，Ollama依据资源加载生成模型。
仅将软件版本和启动条件作为本机复现信息，不把不同时间的全部实验假设为相同负载或相同模型驻留状态。
仓库依赖清单、模型manifest与实验快照应联合使用；本报告文件夹只保存文档、图表和复核数据，不打包模型权重或私有工作区标识。

\subsection{本地运行与重新验收}
新环境需要Python 3.10—3.13（建议3.12）和Ollama。先在项目根目录安装模型运行依赖，并准备四种实验所需的编码与重排模型：
\begin{lstlisting}
.\scripts\setup.ps1 -Python python -Models
.venv\Scripts\python.exe scripts/prepare_models.py --models BAAI/bge-m3 BAAI/bge-large-zh-v1.5 moka-ai/m3e-base BAAI/bge-reranker-base --cache-dir data/models/hub
\end{lstlisting}
首次下载需要联网，实际权重revision写入\path{data/models/manifest.json}；严格复现时应使用清单对应的snapshot，而不是把未来同名仓库视为相同权重。
在单独终端设置Ollama监听地址后启动服务：
\begin{lstlisting}
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama serve
\end{lstlisting}
便携版路径见\path{data/tools/ollama-manifest.json}。
在应用终端指定同一模型服务地址，按固定GGUF revision下载、导入并验证生成模型：
\begin{lstlisting}
$env:RAG_OLLAMA_URL = "http://127.0.0.1:11435"
.venv\Scripts\python.exe scripts/prepare_qwen_gguf.py all --ollama-url http://127.0.0.1:11435 --model research-qwen:7b
\end{lstlisting}
该脚本核对分片SHA256并生成\path{data/models/local-llm.json}，启动器据此选择实际模型标签。
模型和缓存已就绪的环境可直接启动；在应用终端执行：
\begin{lstlisting}
.\scripts\start.ps1 -Profile standard -ApiPort 8000 -UiPort 8501 -ModelPort 11435
.venv\Scripts\python.exe scripts/verify_local_deployment.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --evidence-dir reports/local/new-run --document data/corpus/cvt_2103.15808v1.pdf --timeout 900
\end{lstlisting}
每次验收使用新随机工作区，保存实际HTTP响应、SSE、原件哈希与预览，不复用旧通过标记。
首次真实编码和模型加载可能明显慢于后续请求；记录应注明是否命中缓存及实际token，而不是直接套用表\ref{tab:local}的单次耗时。

\subsection{证据索引与课程要求对应}
\begingroup\ReportTableStyle\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{.9cm}p{2.6cm}p{9.9cm}@{}}
\caption{报告数值与实现证据索引}\label{tab:evidence}\\\toprule 编号 & 内容 & 项目中的实际位置\\\midrule\endfirsthead
\toprule 编号 & 内容 & 项目中的实际位置\\\midrule\endhead
E01 & 固定论文版本 & \path{data/corpus/manifest.json}\\
E02 & 80题与划分 & \path{reports/datasets/dataset_manifest.json}\\
E03 & 三解析器 & \path{reports/experiments/retrieval/parsers/results.json}\\
E04 & 五分块配置 & \path{reports/experiments/retrieval/chunking/m3e/}\\
E05 & 编码与模型对比 & \path{reports/experiments/retrieval/encoding/}；\path{reports/experiments/retrieval/chunking/bge-zh/}\\
E06 & 向量库 & \path{reports/experiments/retrieval/vectorstores/}\\
E07 & 三档test60 & \path{reports/experiments/retrieval/m3/test/}\\
E08 & RRF同题优化 & \path{reports/experiments/retrieval/dev-parameters/results.json}\\
E09 & 60题AI评分 & \path{reports/experiments/generation/baseline/ai_review_summary.json}及同目录逐题CSV\\
E10 & 参数与Prompt & \path{reports/experiments/generation/quality-comparison/}；原始记录在\path{reports/experiments/generation/parameters/}和\path{reports/experiments/generation/prompt/}\\
E11 & Agent探针 & \path{reports/experiments/generation/agent/results.json}\\
E12 & 并行与缓存 & \path{reports/experiments/generation/parallel/}和\path{reports/experiments/generation/cache/}\\
E13 & 异质在线记录 & \path{reports/experiments/online-functionality/results.json}\\
E14 & 本地实际验收 & \path{reports/deployment/local-20261015/runtime-record.json}及\path{reports/deployment/local-20261015/acceptance/acceptance.json}\\
E15 & Bad Case & \path{reports/experiments/bad-cases/actual-bad-cases.json}\\
E16 & 已有测试记录 & \path{reports/testing/}；完整与定向测试时间分别保存，不叠加重叠数量\\
E17 & 历史界面图 & \path{docs/screenshots/}；本文件夹保留原件及图表来源指纹\\\bottomrule
\end{longtable}\endgroup
本报告文件夹中的\path{data/report-data.json}保存原始制表与绘图快照，\path{data/advanced-statistics.json}保存同题配对效果和评分分组等派生统计，\path{figures/figure_manifest.json}记录图形及截图来源。
课程要求中的多格式解析、三分块、两编码模型、三档检索、生成参数、八工具及四类Bad Case分别在第三、四部分对应展开。
原文的独立人工评分与容器实测尚无完成证据，未因本报告采用AI评审和本地部署而改写为已完成。

\subsection{图表与源文件编译}
\path{main.tex}为完整正文源文件，参考文献内嵌，不依赖外部BibTeX；所有图像均从本文件夹的\path{figures/}相对引用。
建议使用XeLaTeX连续编译两次，或使用支持XeTeX的Tectonic；中文字体采用TeX发行版提供的Fandol仿宋，避免依赖仅存在于本机的中文字体。
图表包含矢量PDF与可预览的高分辨率PNG，原始界面图片保留在\path{figures/original-ui/}，标注只说明功能区域，不修改界面内容。
正文按模板采用仿宋四号、1.5倍行距及A4页边距；图注和结果表适度缩小，以保证图、文和数据的可读性。
\end{document}
'''


def academic_table_style(text):
    """Style numbered tables consistently, leaving the official cover grid intact."""
    def table_block(match):
        block=match.group(0)
        block=re.sub(r'\\small(?:\\renewcommand\{\\arraystretch\}\{[^}]+\})?',
                     lambda _:r'\ReportTableStyle',block,count=1)
        block=re.sub(r'(?<![A-Za-z])p\{([^{}]+)\}',r'P{\1}',block)
        return block
    text=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table_block,text,flags=re.S)
    text=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',table_block,text,flags=re.S)

    def header(match):
        cells=match.group('header').strip().split(' & ')
        return (r'\toprule\rowcolor{TableHead}'+'\n'+
                ' & '.join(r'\textbf{'+cell.strip()+'}'for cell in cells)+
                r' \\ \midrule')
    # Apply the same pale header and emphasis to both longtable head versions.
    return re.sub(r'\\toprule\s*(?P<header>.*?)\\\\\s*\\midrule',header,text,flags=re.S)


def main():
    text=TEX
    for key,value in dynamic().items():text=text.replace('@@'+key+'@@',value)
    if '@@' in text:raise ValueError('Unresolved TeX template placeholder')
    text=academic_table_style(text)
    (HERE/'main.tex').write_text(text,encoding='utf-8')
    print(json.dumps({'tex':str(HERE/'main.tex'),'characters':len(text),'figures':len(re.findall(r'\\reportfigure(?:\[[^\]]+\])?\{',text))+text.count('{ui05_local.jpg}'),'numbered_tables':text.count('\\begin{table}')+text.count('\\begin{longtable}')},ensure_ascii=False))


if __name__=='__main__':main()
