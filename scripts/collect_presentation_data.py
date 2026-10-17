"""Collect measured course-study data for reproducible presentation artifacts.

This script never invokes a model, alters an experimental record or invents a
missing quality score. Source paths are relative and all exported CSVs use BOM.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/presentation/source-data"
TABLES: dict = {}
SOURCE_CONFIGS: dict = {}
SOURCES: dict = {}


def sanitize(value, key=""):
    if re.search(r"^(api_key|api_token|access_token|authorization|password|secret|workspace_key)$", key, re.I):
        return "[redacted]" if value else value
    if isinstance(value, dict):
        return {k: sanitize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = value.replace(str(ROOT) + "\\", "").replace(str(ROOT) + "/", "")
        value = value.replace(str(ROOT), ".")
        if re.match(r"^[A-Za-z]:[\\/]", value):
            return "[external-local-path]/" + re.split(r"[\\/]", value)[-1]
        if re.match(r"^(data|reports|scripts|src|docs|artifacts)[\\/]", value):
            value = value.replace("\\", "/")
        return value
    return value


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    name = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    SOURCES[name] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    obj = json.loads(raw.decode("utf-8-sig"))
    if isinstance(obj, dict) and "config" in obj:
        # Keep only metadata needed by experiment_config, not duplicate full
        # result payloads (the original measured records remain in reports/).
        SOURCE_CONFIGS[name] = sanitize({
            "study": obj.get("study", obj.get("kind")), "source_path": name,
            "created_at": obj.get("created_at"), "index_version": obj.get("index_version"),
            "config_hash": obj.get("config_hash"), "dataset_hash": obj.get("dataset_hash"),
            "corpus_hash": obj.get("corpus_hash"), "config": obj.get("config"),
            "models": obj.get("models"), "scope": obj.get("measurement_scope", obj.get("split")),
        })
    return obj


def rows_csv(path):
    path = ROOT / path
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table(key, title, rows, notes, sources, columns=None):
    rows = sanitize(rows)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    TABLES[key] = {"title": title, "columns": columns, "rows": rows,
                   "notes": notes, "source_paths": sorted(set(str(p).replace("\\", "/") for p in sources))}


def flat_metrics(metrics):
    return {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}


def scalar_stats(values):
    values = [float(v) for v in values if v is not None and str(v) != ""]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path = "data/corpus/manifest.json"
    corpus = read(corpus_path)
    table("corpus", "固定版本论文语料", [{k: p.get(k) for k in ["paper_key", "title", "arxiv_version", "publication_year", "venue", "physical_pages", "byte_count", "pdf_url", "local_file", "sha256"]} for p in corpus["papers"]],
          ["10篇论文，共169个PDF物理页；页码以固定版本原文为准。", "产品为通用科研助理，Vision Transformers为本次评测专题。"], [corpus_path])
    dataset_path = "reports/datasets/dataset_manifest.json"
    ds = read(dataset_path)
    table("dataset", "题集组成与开发测试划分", [
        {"category": cat, "total": ds["category_counts"][cat], "dev": ds["split_category_counts"].get("dev/" + cat), "test": ds["split_category_counts"].get("test/" + cat)}
        for cat in ds["category_counts"]], ["80题=20 dev+60 test；仅dev用于调参。", "标签为助手核对原文生成，不冒充多人共识标注。"], [dataset_path])
    current = read("reports/requirements/current-standard-index.json")
    frozen = read("reports/experiments/retrieval/frozen/metadata.json")
    parser_path = "reports/experiments/retrieval/parsers/results.json"
    parsers = read(parser_path)
    table("parsers", "PDF解析器对比", parsers["summary"], parsers["notes"], [parser_path])
    table("parser_records", "30次逐篇解析记录", parsers["records"], ["每种解析器每篇仅运行一次；总计3×10次。", "首次调用计入导入时间，锚点保留率为代理指标而非通用解析准确率。"], [parser_path])
    metric_notes = ["比例指标范围0–1；延迟单位毫秒。", "检索计算耗时排除共享查询编码，不能当作在线端到端耗时。"]
    retrieval_records, category_rows = [], []
    retrieval_sources = []
    for split, modes in [("test", ["vector", "hybrid", "hybrid_rerank"]), ("dev", ["vector", "hybrid"])]:
        rows, paths = [], []
        for mode in modes:
            name = f"reports/experiments/retrieval/m3/{split}/{mode}/results.json"
            x = read(name); paths.append(name); retrieval_sources.append(name)
            rows.append({"mode": mode, "split": split, "chunk_count": 359, **flat_metrics(x["metrics"])})
            for cat, m in x["metrics"].get("by_category", {}).items():
                category_rows.append({"mode": mode, "split": split, "category": cat, "case_count": ds["split_category_counts"].get(f"{split}/{cat}"), **m})
            for r in x["records"]:
                retrieval_records.append({"mode": mode, "split": split, **{k: r.get(k) for k in ["case_id", "category", "question", "error", "hit_at_5", "mrr", "recall_at_5", "evidence_count", "recalled_evidence_count", "first_relevant_rank", "top1_score", "elapsed_ms", "vector_db_ms", "bm25_ms", "fusion_ms", "rerank_ms", "amortized_query_encoding_ms", "latency_scope"]}})
        table("retrieval_" + split, f"三档检索正式测试（{split}）" if split == "test" else "开发集检索", rows,
              metric_notes + ["历史359片段持久索引；正式test配置冻结candidate=20/RRF=60。", "未执行dev rerank，不用缺失值伪造零。"], paths)
    table("retrieval_category", "按题型拆分的检索指标", category_rows, metric_notes + ["与对应split相同的历史359片段索引。"], retrieval_sources)
    table("retrieval_records", "逐题检索指标与耗时", retrieval_records, metric_notes, retrieval_sources)
    chunk_rows, embedding_rows, batches, chunk_sources, encoding_sources = [], [], [], [], []
    for model_key in ["m3e", "bge-zh"]:
        ep = f"reports/experiments/retrieval/encoding/{model_key}/results.json"
        enc = read(ep); encoding_sources.append(ep)
        defs = {d["variant"]: d for d in enc["variant_definitions"]}
        for b in enc["document_batches"]:
            batches.append({"model": enc["embedding_model"], "batch_index": 1 + sum(r["model"] == enc["embedding_model"] for r in batches), **b})
        for variant in enc["variant_names"]:
            rp = f"reports/experiments/retrieval/chunking/{model_key}/{variant}/results.json"
            x = read(rp); chunk_sources.append(rp)
            row = {"model": enc["embedding_model"], "variant": variant, **defs[variant], **flat_metrics(x["metrics"])}
            chunk_rows.append(row)
            if variant == "recursive-512":
                embedding_rows.append({"model": enc["embedding_model"], **{k: enc.get(k) for k in ["dimension", "cpu_threads", "model_batch_size", "window_token_cap", "query_count", "query_encoding_total_ms", "query_encoding_amortized_ms", "model_load_ms", "unique_document_text_count", "document_encoding_total_ms", "document_texts_per_second", "artifact_revision", "embedding_fingerprint"]}, "chunk_count": 342, **flat_metrics(x["metrics"])})
    table("chunking", "分块策略与嵌入的开发集实验", chunk_rows, metric_notes + ["m3e下五种分块策略；bge-large-zh仅recursive-512。", "chunk_size与overlap为配置单位；实际拆分为递归字符提议及Token窗口编码。"], chunk_sources + encoding_sources)
    table("embeddings", "嵌入模型公平查询工作量比较", embedding_rows, metric_notes + ["同一20 dev查询、同一342片段recursive-512比较检索质量及查询编码时间。", "批量文档编码集合不同（m3e1836条、BGE342条），其总耗时/TPS不可用于公平模型速度排名。"], encoding_sources + chunk_sources)
    table("embedding_batches", "逐批嵌入编码记录", batches, ["elapsed_ms单位毫秒；两模型bulk输入总集合不同。"], encoding_sources)
    vpath = "reports/experiments/retrieval/vectorstores/results.json"
    vx = read(vpath)
    table("vectorstores", "Chroma与FAISS存储比较", vx["records"], metric_notes + ["同20 dev查询、同359片段1024维真实向量；模型推理复用。", "构建、重启后首查、删除及恢复分别计时；一次测量仅说明本机运行。"], [vpath])
    rpath = "reports/experiments/retrieval/dev-parameters/results.json"; rrf = read(rpath)
    table("rrf_grid", "RRF常数与Top-k开发集消融", rrf["summary"], rrf["notes"] + ["mrr_at_20针对固定前20排名；metric_k仅决定Hit/Recall截断。"], [rpath])
    table("rrf_records", "RRF开发集逐题记录", rrf["records"], rrf["notes"], [rpath])
    conditions, generation_records, gen_paths, configs = [], [], [], []
    for study in ["baseline", "parameters", "prompt", "agent", "parallel", "cache"]:
        name = f"reports/experiments/generation/{study}/results.json"
        x = read(name); gen_paths.append(name)
        cond_configs = dict(x.get("config", {}).get("conditions", []))
        for cond, m in x.get("condition_metrics", {}).items():
            row = {"study": study, "condition": cond, **cond_configs.get(cond, {}), **flat_metrics(m), "measurement_scope": x.get("measurement_scope")}
            if study == "agent" and cond == "agent":
                row["routing_exact_match"] = x["metrics"].get("routing_exact_match")
            conditions.append(row)
        for r in x["records"]:
            row = {"study": study, **{k: r.get(k) for k in ["condition", "case_id", "category", "split", "question", "answer", "reference_answer", "elapsed_ms", "first_token_ms", "prompt_tokens", "completion_tokens", "success", "error", "truncated", "expected_tools", "invoked_tools", "cache_hit", "actual_model_calls", "workers", "local_tools_only", "llm_called"]}}
            generation_records.append(row)
        configs.append({"study": study, "source_path": name, "created_at": x.get("created_at"), "index_version": x.get("index_version"), "config_hash": x.get("config_hash"), "dataset_hash": x.get("dataset_hash"), "corpus_hash": x.get("corpus_hash"), "config": x.get("config"), "models": x.get("models"), "scope": x.get("measurement_scope")})
        if study in ["agent", "parallel", "cache"]:
            specific = {"agent": ["5个路由功能探针，两种条件各5次；精确工具路由率非通用Agent答案质量。"], "parallel": ["10轮×2条件，共20次；仅计算器和关键词本地独立工具，无LLM/检索。"], "cache": ["仅1个问题的空缓存与完全重复查询各1次。", "模型已常驻；不是模型冷启动，也不证明语义改写缓存命中效果。"]}[study]
            table(study, {"agent": "Agent路由与固定RAG探针", "parallel": "独立本地工具串并行", "cache": "空缓存与完全重复查询"}[study], [r for r in conditions if r["study"] == study], specific + [x.get("measurement_scope", "")], [name])
    table("generation_conditions", "全部生成及工具实验条件", conditions, ["生成质量缺失值保持null；不得把自动规则通过率当答案质量。", "parameters每条件4题单次运行、prompt每条件4题；不宣称统计显著。", "Prompt新模板4条与parameters baseline4条复用，不能算作独立重复实验。", "回答上限256 token、4096上下文；历史359片段冻结检索输入。"], gen_paths)
    table("generation_records", "逐次生成与工具记录", generation_records, ["同一case/condition记录为真实输出；Prompt复用基线不算独立生成。", "elapsed_ms单位毫秒；不同study的测量范围不同。"], gen_paths)
    apath = "reports/experiments/generation/baseline/ai_review_summary.json"; ai = read(apath)
    spath = "reports/experiments/generation/baseline/human_scoring_60.csv"; scores = rows_csv(spath)
    dimensions = list(ai["means"])
    for r in scores:
        for k in dimensions:
            r[k] = int(r[k]) if r[k] else None
    quality = [{"dimension": k, **scalar_stats(r[k] for r in scores), "scale_min": 1, "scale_max": 5, "grading_method": "ai_assisted", "grading_date": ai["grading_date"]} for k in dimensions]
    table("baseline_ai_quality", "60题答案质量评审", quality, [ai["scope"], "评分来源为Codex AI辅助评分；用户接纳其作为项目评审依据，保留实际评分方法。", "std为60题样本标准差，不代表多次独立评审的不确定度。"], [apath, spath])
    table("baseline_ai_distribution", "60题四维评分分布", [{"dimension": k, "score": s, "count": int(ai["score_distribution"][k].get(str(s), 0)), "proportion": int(ai["score_distribution"][k].get(str(s), 0)) / ai["case_count"]} for k in dimensions for s in range(1, 6)], ["1–5分有序量表；零频次来自完整分布，不是缺失值替代。"], [apath])
    table("baseline_ai_category", "按题型的答案质量", [{"category": cat, "dimension": k, **scalar_stats(r[k] for r in scores if r["category"] == cat)} for cat in ds["category_counts"] for k in dimensions], [ai["scope"], "评分方法AI辅助；不同题型样本量不同。"], [apath, spath])
    table("baseline_ai_scores", "60题逐题评分与评审依据", scores, ["保留真实问答、评分人、方法、日期、Bad Case、根因与纸面证据。", "AI评审没有伪标为真人评审。"], [spath])
    online_path = "reports/experiments/online-functionality/results.json"; online = read(online_path)
    table("online", "真实在线功能样本汇总", [online["metrics"]], online["notes"] + ["10次请求问题/配置异构，不能作为受控同题配置对比。"], [online_path])
    table("online_records", "10次异构在线请求", online["records"], online["notes"], [online_path])
    bcpath = "reports/experiments/bad-cases/actual-bad-cases.json"
    table("bad_cases", "实际Bad Case根因与改进", read(bcpath), ["改进状态和结果保留原文；未测得的提升不填造数字。"], [bcpath])
    auditpath = "reports/requirements/course-audit.json"; audit = read(auditpath)
    table("acceptance", "课程逐条验收证据快照", [{k: r.get(k) for k in ["id", "source_paragraph_id", "module", "requirement", "source_text", "mandatory", "implementation_status", "execution_status", "evidence_paths", "interpretation", "pending"]} for r in audit["requirements"]], ["读取源验收表时的状态快照，不能作为最终完成率。", "可选/不适用项不能计入强制任务完成率；AI评分仍保留来源属性。"], [auditpath])
    table("acceptance_summary", "验收证据状态分布", [{"execution_status": k, "requirement_count": v} for k, v in Counter(r["execution_status"] for r in audit["requirements"]).items()], ["108条记录包含背景解释、可选项和交付要求，不是108个同权重评分项。", "状态为源验收表快照，生成参数评分补充结果另列。"], [auditpath])
    known_configs = {r["source_path"] for r in configs}
    configs.extend(value for name, value in SOURCE_CONFIGS.items() if name not in known_configs)
    table("experiment_config", "实验参数、模型和版本", configs, ["完整真实配置与模型revision保留，路径已去除本机根目录。", "此表声明的是历史实验配置，不宣称等同于最新生产默认值。"], [r["source_path"] for r in configs])
    quality_summary_path = "reports/experiments/generation/quality-comparison/ai_quality_summary.json"
    quality_rows_path = "reports/experiments/generation/quality-comparison/ai_quality_rows.json"
    if (ROOT / quality_summary_path).exists() and (ROOT / quality_rows_path).exists():
        qs, qr = read(quality_summary_path), read(quality_rows_path)
        qnotes = ["评分方法AI辅助，用户接纳为本项目评审依据；保留实际方法，不伪造真人身份。",
                  "每条件同4题dev、每题单次生成，无固定seed；探索性比较，不作显著性或普遍最优结论。",
                  "历史359片段冻结检索，256输出token、4096上下文，仅生成阶段。",
                  "28条条件记录来自24次历史模型调用，新Prompt复用4条基线；17个独立答案进行内容评审。",
                  "quality_three_dimension_mean为正确性、完整性、引用支撑三维等权描述性均分，非课程官方评分。"]
        table("quality_conditions", "生成参数和Prompt回答质量对比", qs["conditions"], qnotes, [quality_summary_path, quality_rows_path])
        table("quality_records", "28条条件回答的逐题质量与证据", qr, qnotes, [quality_rows_path])
        table("prompt_ai_quality", "Prompt改进质量比较", [r for r in qs["conditions"] if r["study"] == "prompt"], qnotes, [quality_summary_path])
        for r in TABLES["generation_conditions"]["rows"]:
            match = next((q for q in qs["conditions"] if (q["study"], q["condition"]) == (r["study"], r["condition"])), None)
            if match:
                r.update({"ai_" + k: match[k] for k in ["correctness_mean", "completeness_mean", "citation_quality_mean", "fluency_mean", "quality_three_dimension_mean"]})
                r["answer_quality_grading_method"] = "ai_assisted"
        TABLES["generation_conditions"]["columns"] = list(dict.fromkeys(k for row in TABLES["generation_conditions"]["rows"] for k in row))
        TABLES["generation_conditions"]["source_paths"].extend([quality_summary_path, quality_rows_path])
        TABLES["generation_conditions"]["notes"][0] = "原始human_*指标仍保留null；新增ai_*为实际AI辅助评分，不能将自动规则通过率或AI评审伪写为真人数据。"
    scopes = [
        {"scope_id": "current_production", "name": "最新生产索引", "paper_count": current["document_count"], "chunk_count": current["chunks"], "index_version": current["index_version"], "embedding_fingerprint": current["embedding_fingerprint"], "measurement": "current index metadata; not a full 60-case current online benchmark"},
        {"scope_id": "legacy_test", "name": "历史359片段正式检索", "paper_count": 10, "chunk_count": 359, "index_version": frozen["index_version"], "case_count": 60, "split": "test", "measurement": "real retrieval compute; excludes shared query encoding and LLM"},
        {"scope_id": "dev_chunking", "name": "分块/嵌入开发集", "case_count": 20, "split": "dev", "chunk_count_by_variant": {r["variant"]: r["chunk_count"] for r in chunk_rows}, "measurement": "m3e five chunk variants; BGE-zh same recursive342 input"},
        {"scope_id": "generation", "name": "历史冻结检索生成研究", "chunk_count": 359, "baseline_case_count": 60, "parameter_cases_per_condition": 4, "prompt_cases_per_condition": 4, "answer_tokens": 256, "context_tokens": 4096, "measurement": "generation-only with frozen real retrieval input; current output settings may differ"},
        {"scope_id": "online", "name": "在线异构功能样本", "sample_count": 10, "measurement": "actual request end-to-end, heterogeneous questions/configurations, no controlled superiority claim"},
        {"scope_id": "parallel", "name": "本地独立工具并行", "run_count": 20, "case_count": 10, "measurement": "calculator + extract_keywords; excludes LLM and retrieval"},
        {"scope_id": "cache", "name": "完全重复缓存探针", "case_count": 1, "run_count": 2, "measurement": "empty-cache generation vs exact-repeat lookup; resident model; not semantic paraphrase recall"},
    ]
    report = {"schema_version": 1, "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "policy": {"no_new_model_runs": True, "missing_values": "null", "scoring_method": "ai_assisted", "user_accepts_ai_as_project_evaluation": True, "actual_human_review_not_claimed": True},
              "tables": TABLES, "experimental_scopes": scopes, "source_hashes": list(SOURCES.values())}
    (OUT / "experiments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, t in TABLES.items():
        with (OUT / f"{key}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=t["columns"], extrasaction="ignore")
            writer.writeheader()
            for row in t["rows"]:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    print(json.dumps({"tables": len(TABLES), "rows": sum(len(t["rows"]) for t in TABLES.values()), "sources": len(SOURCES), "output": OUT.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
