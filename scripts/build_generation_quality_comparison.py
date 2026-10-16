"""Export source-checked AI judgments for existing generation experiments.

This script aggregates recorded judgments; it does not call an LLM or pretend
that these judgments were supplied by a human. Source answers are immutable.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/generation"
OUT = BASE / "quality-comparison"
SCORE_KEYS = ("correctness", "completeness", "citation_quality", "fluency")

# Scores were assigned by Codex after reading every distinct answer and the
# cited snippets, then checking the relevant physical pages of local PDFs.
# Keys select an existing answer, rather than generating a surrogate answer.
JUDGMENTS = {
    ("parameters", 0): ((5, 5, 5, 5), "四项配置均与 ViT 物理页5表1一致，S3包含该表；没有额外不受支持的结论。"),
    ("parameters", 1): ((3, 3, 3, 5), "抓住DeiT的数据效率与蒸馏，但遗漏ImageNet-21k/JFT和仅ImageNet的具体对照；把DeiT结论中针对CNN架构搜索的论述扩大到ViT。S3仅支持其中部分断言。"),
    ("parameters", 2): ((4, 3, 2, 5), "两种核心思路基本正确，但未明确CvT的卷积token嵌入和注意力投影；引用S4实际为T2T论文，不能支撑CvT断言，冻结上下文缺少CvT原文。"),
    ("parameters", 3): ((5, 5, 4, 5), "196和197及计算过程正确。S1为BEiT物理页3，支持HW/P²与14×14网格，但该引用片段未直接说明class token；完整证据可见DeiT物理页5的N+1。"),
    ("parameters", 5): ((3, 3, 3, 5), "DeiT训练配方和蒸馏方向正确，仍遗漏具体数据集；将CNN架构搜索与过拟合概括到ViT，超出S3原文适用对象，引用只能部分支持。"),
    ("parameters", 6): ((2, 2, 2, 4), "T2T步骤大体正确，但将CvT误写为固定分块且不强调局部结构，与CvT卷积嵌入、投影及层级设计相反；全篇只引T2T的S1，且末段存在概括上的自相矛盾。"),
    ("parameters", 8): ((5, 5, 5, 5), "四项数值正确且完整，S3对应ViT物理页4—5并包含表1；措辞简洁。"),
    ("parameters", 9): ((3, 2, 2, 5), "只保留DeiT不依赖特别大数据这一方向；把ViT一侧改为需要优化，没有回答其大规模预训练依赖，也未解释蒸馏及训练配方；S1不能完整支持数据依赖比较。"),
    ("parameters", 10): ((2, 2, 2, 5), "T2T方向正确，但将CvT解释为固定长度分块、未说明局部先验，漏掉其关键卷积机制；S1是T2T而非CvT证据。"),
    ("parameters", 13): ((3, 2, 2, 5), "DeiT数据效率方向正确，但用优化架构和长时间训练代替ViT的大规模数据依赖，比较不完整；没有具体训练集、蒸馏说明，S1证据覆盖不足。"),
    ("parameters", 14): ((2, 2, 2, 5), "T2T的重构、软分割叙述较具体，但CvT部分仍被写成普通ViT式固定分块加全局Transformer，未给出卷积嵌入与投影；T2T来源不能支撑CvT机制。"),
    ("parameters", 17): ((4, 3, 3, 5), "较清楚给出ViT较依赖数据、DeiT靠训练和蒸馏提升效率的方向，但缺少ImageNet与ImageNet-21k/JFT的具体条件。引用S1可支持训练策略差异，却不足以独立证明全部数据依赖结论。"),
    ("parameters", 18): ((3, 3, 2, 5), "正确区分T2T重塑/软分割和CvT卷积，但声称CvT不进行分层token处理，与其三阶段层级设计相反；S4仍是T2T论文，CvT事实未由所引来源支持。"),
    ("prompt", 0): ((5, 5, 1, 5), "四项数值本身正确，但S4实际为ViT物理页14，既不含表1也不是声称的页5；属于来源编号与页码错配，而非仅格式问题。"),
    ("prompt", 1): ((4, 3, 2, 5), "两者数据依赖的方向及DeiT蒸馏描述基本正确，缺具体数据集与公平比较条件；ViT断言引用S4实际为CaiT物理页3，却写第4页，不能完整追溯该断言。"),
    ("prompt", 2): ((3, 2, 2, 5), "T2T步骤正确，但CvT仅写特定设计，未说明任何卷积机制，未实质回答第二种办法；S2依然为T2T论文，不能支持CvT结论。"),
    ("prompt", 3): ((5, 5, 5, 5), "196、197及推导完整正确；S2确为DeiT物理页5，同时包含14×14分块和class token形成N+1的说明，证据链完整。"),
}

EVIDENCE = {
    "VT-F01": [
        {"path": "data/corpus/vit_2010.11929v2.pdf", "physical_pages": [5], "finding": "表1 ViT-Base: Layers 12; Hidden size 768; Heads 12; Params 86M。"},
        {"path": "data/corpus/vit_2010.11929v2.pdf", "physical_pages": [14], "finding": "该页为微调/自监督训练等内容，不是模型变体表1；旧Prompt的S4指向此页。"},
    ],
    "VT-C01": [
        {"path": "data/corpus/vit_2010.11929v2.pdf", "physical_pages": [4, 5], "finding": "ViT比较ImageNet、ImageNet-21k及JFT等预训练规模，后续迁移评估。"},
        {"path": "data/corpus/deit_2012.12877v2.pdf", "physical_pages": [1, 8, 17, 18], "finding": "DeiT以ImageNet训练、数据增强/正则和蒸馏提升效率。第18页架构搜索的历史论述对象是CNN，不是ViT。"},
    ],
    "VT-S03": [
        {"path": "data/corpus/t2t_2101.11986v3.pdf", "physical_pages": [3, 4], "finding": "图3及3.1节：重构、重叠软分割、聚合邻近token。"},
        {"path": "data/corpus/cvt_2103.15808v1.pdf", "physical_pages": [1, 3, 4], "finding": "卷积token嵌入和卷积注意力投影；多阶段层级结构，逐级减少token数量。该论文未出现在本题冻结Top5中。"},
    ],
    "VT-R01": [
        {"path": "data/corpus/vit_2010.11929v2.pdf", "physical_pages": [3], "finding": "N=HW/P²，附加分类token。"},
        {"path": "data/corpus/beit_2106.08254v2.pdf", "physical_pages": [3], "finding": "224×224图像，16×16 patch，14×14网格；仅该引用片段未明确N+1。"},
        {"path": "data/corpus/deit_2012.12877v2.pdf", "physical_pages": [5], "finding": "明确N=14×14，以及class token使Transformer处理N+1个token；196与197由公式计算。"},
    ],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_key(record: dict) -> str:
    citations = [{k: c.get(k) for k in ("citation_id", "name", "page_start", "page_end", "text")} for c in record["citations"]]
    body = {"question": record["question"], "answer": record["answer"], "citations": citations}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    studies = {s: json.loads((BASE / s / "results.json").read_text(encoding="utf-8")) for s in ("parameters", "prompt")}
    judgments = {}
    for (study, i), (scores, notes) in JUDGMENTS.items():
        record = studies[study]["records"][i]
        judgments[review_key(record)] = {**dict(zip(SCORE_KEYS, scores)), "notes": notes}
    rows, first_seen = [], {}
    for study, data in studies.items():
        jsonl_rows = [json.loads(x) for x in (BASE / study / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert all(review_key(r) == review_key(data["records"][i]) for i, r in enumerate(jsonl_rows))
        for i, record in enumerate(data["records"]):
            key = review_key(record)
            judgment = judgments[key]
            reused = first_seen.get(key, "")
            row_id = f"{study}:{record['condition']}:{record['case_id']}"
            first_seen.setdefault(key, row_id)
            row = {
                "study": study, "condition": record["condition"], "case_id": record["case_id"],
                "question": record["question"], "answer": record["answer"], "reference_answer": record["reference_answer"],
                **{k: judgment[k] for k in SCORE_KEYS},
                "reviewer": "Codex AI辅助评审", "grading_method": "ai_assisted", "grading_date": "2026-10-15",
                "evidence": EVIDENCE[record["case_id"]], "notes": judgment["notes"],
                "review_id": key, "score_reused_from": reused,
                "answer_sha256": hashlib.sha256(record["answer"].encode()).hexdigest(),
                "source_record_index": i, "context_hash": record["context_hash"],
                "cited_source_mapping": [{k: c.get(k) for k in ("citation_id", "name", "page_start", "page_end")} for c in record["citations"]],
                "temperature": record["config"]["temperature"], "top_p": record["config"]["top_p"],
                "top_k_sampling": record["config"]["top_k_sampling"],
                "latency_ms": record["elapsed_ms"], "completion_tokens": record["completion_tokens"],
            }
            assert all(isinstance(row[k], int) and 1 <= row[k] <= 5 for k in SCORE_KEYS)
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["condition"])].append(row)
    conditions = []
    for (study, condition), group in groups.items():
        conditions.append({
            "study": study, "condition": condition, "n": len(group),
            **{f"{k}_mean": mean(r[k] for r in group) for k in SCORE_KEYS},
            **{k: group[0][k] for k in ("temperature", "top_p", "top_k_sampling")},
            "latency_mean_ms": mean(r["latency_ms"] for r in group),
            "completion_tokens_mean": mean(r["completion_tokens"] for r in group),
            "quality_three_dimension_mean": mean(r[k] for r in group for k in SCORE_KEYS[:3]),
        })
    sources = [{"path": str((BASE / s / f).relative_to(ROOT)).replace("\\", "/"), "sha256": sha(BASE / s / f)} for s in studies for f in ("results.json", "records.jsonl")]
    pdfs = sorted({item["path"] for values in EVIDENCE.values() for item in values})
    sources += [{"path": p, "sha256": sha(ROOT / p)} for p in pdfs]
    summary = {
        "created_date": "2026-10-15", "grading_method": "ai_assisted", "reviewer": "Codex AI辅助评审",
        "score_scale": {"min": 1, "max": 5, "anchor": {"1": "错误或缺失严重", "2": "局部正确但核心机制/证据存在严重问题", "3": "部分正确，存在重要遗漏或错误", "4": "基本正确，有次要遗漏或证据缺口", "5": "正确完整且该维度没有实质问题"}},
        "rubric": {"correctness": "事实和推导是否正确；对机制误判与过度概括扣分。", "completeness": "是否覆盖问题要求及关键条件；不同于文字长度。", "citation_quality": "实际引用编号/页码/片段是否支持相应断言；来源存在不等于语义支持。", "fluency": "语言是否清楚连贯；不替代事实正确性。"},
        "metric_definitions": {"quality_three_dimension_mean": "正确性、完整性、引用质量三维等权算术平均，仅作描述性汇总，不是课程官方总分。", "latency_mean_ms": "已有单次生成记录elapsed_ms的条件内算术平均；不含在线检索重算。", "completion_tokens_mean": "模型返回completion_tokens的条件内算术平均。"},
        "scope": {"case_count_per_condition": 4, "case_ids": ["VT-F01", "VT-C01", "VT-S03", "VT-R01"], "split": "dev", "runs_per_case_condition": 1, "frozen_chunk_count": 359, "answer_token_limit": 256, "context_token_budget": 4096, "model": "research-qwen:7b / Qwen2.5-7B-Instruct Q4_K_M", "measurement": "frozen-retrieval / generation-only; no online embedding or reranking", "sampling_seed": None, "run_order": "condition-major", "condition_record_count": len(rows), "historical_model_call_count": 24, "unique_answer_review_count": len(first_seen), "prompt_current_reused_count": 4},
        "conditions": conditions, "source_files": sources,
        "findings": ["当前Prompt修正了事实题S4错引，引用质量均分从2.50提高至3.50；正确性均分均为4.25，不能声称所有维度同步提高。", "参数条件正确性均分为3.75—4.25；温度0.5样本的输出更短、记录耗时更低，但不能据此归因参数导致速度改善或确立最优参数。", "综合题冻结Top5没有CvT，所有条件都受到证据覆盖缺口影响；调采样参数无法弥补检索缺失。"],
        "limitations": ["每条件仅4题，每题一次，未固定seed；同一题跨条件配对但不是独立重复试验。", "人工评审人数为0，本文件记录的是AI辅助评审；用户采用该评分用于展示不改变其来源。", "Prompt当前条件复用参数基线，不是4次新增调用；去重评分复用不增加样本量。", "延迟为串行历史生成阶段实测值，受模型热身、上下文缓存和运行顺序影响；不能当作可控端到端速度排名。", "不计算显著性、置信区间或总体准确率；不外推到当前342片段索引或生产768-token配置。"],
    }
    (OUT / "ai_quality_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "ai_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "ai_quality_rows.csv", rows)
    write_csv(OUT / "ai_quality_condition_summary.csv", conditions)
    print(json.dumps({"records": len(rows), "unique_reviews": len(first_seen), "conditions": conditions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
