"""Join the exact source-linked implementation audit with real delivery evidence."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    base = read(ROOT / "reports/requirements/implementation-audit.json")
    source = {row["paragraph_id"]: row["text"] for row in read(ROOT / "reports/requirements/course-source.json")}
    results = []
    for path in (ROOT / "reports/experiments").rglob("results.json"):
        item = read(path)
        if item.get("status") in {"running", "error", "failed"}:
            continue
        results.append((str(path.relative_to(ROOT)).replace("\\", "/"), item))
    artifacts = ROOT / "artifacts/delivery-final"
    entries = list(base["requirements"])
    quality_path = ROOT / "reports/experiments/generation/quality-comparison/ai_quality_summary.json"
    quality_summary = read(quality_path) if quality_path.exists() else {}

    def evidence(prefix):
        return [name for name, _ in results if prefix in name]

    upgrades = {
        124: evidence("retrieval/parsers/"), 127: evidence("retrieval/chunking/"),
        130: evidence("retrieval/chunking/"), 134: evidence("retrieval/chunking/"),
        135: evidence("retrieval/vectorstores/"), 142: evidence("retrieval/m3/test/"),
        145: evidence("retrieval/m3/test/"), 154: evidence("generation/parameters/"),
        159: evidence("generation/cache/"), 166: evidence("retrieval/m3/test/"),
        170: evidence("generation/prompt/"), 191: evidence("generation/parallel/")}
    for entry in entries:
        measured = upgrades.get(entry["source_paragraph_id"], [])
        if measured:
            entry["latest_experiment_evidence"] = measured
            entry["execution_status"] = "real_experiment_executed"
            entry["pending"] = [text for text in entry.get("pending", []) if not text.startswith("Complete formal experiments")]
        output_name = {132: "分块策略对比实验报告.docx", 146: "分块策略对比实验报告.docx",
                       170: "Prompt模板设计与实验记录.docx"}.get(entry["source_paragraph_id"])
        if output_name and (artifacts / output_name).exists():
            entry["implementation_status"] = "implemented"
            entry["execution_status"] = "completed"
            entry["evidence_paths"] = [str((artifacts / output_name).relative_to(ROOT)).replace("\\", "/")]
        study = {154: "parameters", 170: "prompt"}.get(entry["source_paragraph_id"])
        if study:
            studies = [item for name, item in results if f"generation/{study}/" in name]
            quality_keys = ("human_correctness_mean", "human_completeness_mean", "human_citation_quality_mean")
            conditions = [metrics for item in studies for metrics in item.get("condition_metrics", {}).values()]
            if studies and (not conditions or any(any(metrics.get(key) is None for key in quality_keys) for metrics in conditions)):
                entry["execution_status"] = "quality_comparison_pending"
                entry["pending"] = [*entry.get("pending", []), "补充各条件实际回答的质量评价及对比结论；已有真实回答可复用，无需重跑生成。"]
                entry["interpretation"] = "真实生成与耗时/Token记录已完成；引用编号合法率和数字规则检查不能替代答案正确性、完整性与引用支撑程度的评价。"
            reviewed = [item for item in quality_summary.get("conditions", []) if item.get("study") == study]
            if reviewed and all(item.get("n", 0) > 0 and all(item.get(key + "_mean") is not None for key in ("correctness", "completeness", "citation_quality")) for item in reviewed):
                entry["execution_status"] = "ai_quality_comparison_completed"
                entry["latest_quality_evidence"] = [str(quality_path.relative_to(ROOT)).replace("\\", "/"), "docs/生成参数与Prompt质量比较.md"]
                entry["pending"] = [item for item in entry.get("pending", []) if not item.startswith("补充各条件实际回答")]
                entry["interpretation"] = "现有真实回答已完成统一AI质量评审及条件比较；每条件4题、单次生成，仅作探索性结论。AI评审来源保留，不计入真人评分。"

    def add(p, name, files, status="completed", note=""):
        existing = [file for file in files if (ROOT / file).exists()]
        entries.append({"id": f"p{p}.delivery", "source_paragraph_id": p, "module": 5,
                        "source_text": source[p], "requirement": name,
                        "implementation_status": status if existing else "pending",
                        "execution_status": status if existing else "pending",
                        "evidence_paths": existing, "interpretation": note})

    add(239, "具体研究方向与10篇真实论文", ["data/corpus/manifest.json"], note="视觉Transformer架构与数据效率，原文和版本哈希已固定。")
    add(240, "80题评测集与20dev/60test划分", ["reports/datasets/dataset_manifest.json"])
    for p in (241, 242, 243, 244):
        add(p, source[p], ["reports/datasets/dataset_manifest.json"])
    tests = evidence("retrieval/m3/test/")
    modes = {item.get("mode") for name, item in results if "retrieval/m3/test/" in name}
    add(247, "60题Hit@5 MRR Recall三档", tests, "completed" if {"vector", "hybrid", "hybrid_rerank"} <= modes else "partial")
    add(248, "工具路由准确率与推理轮次", evidence("generation/agent/"))
    add(249, "真实在线平均延迟与Token", ["reports/experiments/online-functionality/results.json"], note="10次真实线上样本；冻结检索生成实验另列，不能冒充在线端到端。")
    forms = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "reports/experiments/generation").rglob("human_scoring_60.csv")]
    ai_summary_path = ROOT / "reports/experiments/generation/baseline/ai_review_summary.json"
    ai_summary = read(ai_summary_path) if ai_summary_path.exists() else {}
    baseline = next((item for name, item in results if name == "reports/experiments/generation/baseline/results.json"), {})
    human_count = baseline.get("metrics", {}).get("human_scoring_completed_count", 0)
    case_count = baseline.get("metrics", {}).get("case_count", 60)
    add(250, "实际答案人工评分", forms + ([str(ai_summary_path.relative_to(ROOT)).replace("\\", "/")] if ai_summary else []),
        "completed" if human_count >= case_count else "human_scoring_pending",
        f"{ai_summary.get('case_count', 0)}题AI辅助评分已完成；真实人工评分{human_count}/{case_count}题。AI评分单独标注，不计入课程要求的人工评分。")
    add(251, "真实参数性能图表", evidence("retrieval/dev-parameters/") + evidence("generation/parameters/"))
    for p in (254, 255, 256, 257, 258):
        add(p, source[p], ["reports/experiments/bad-cases/actual-bad-cases.json"])
    add(259, "至少一轮真实优化前后记录", ["reports/experiments/retrieval/dev-parameters/results.json"] + evidence("generation/prompt/"), note="同20dev题RRF60→20，Hit@5与Recall不变，MRR小幅变化；不夸大提升，不用不同题集作因果比较。")
    names = {260: "Bad Case分析报告.docx", 262: "系统评测报告.docx", 263: "技术设计文档.docx",
             296: "系统评测报告.docx", 299: "技术设计文档.docx", 302: "演示PPT.pptx", 305: "用户手册.docx"}
    for p, name in names.items():
        matches = list(artifacts.glob(name)) if artifacts.exists() else []
        if not matches and name == "演示PPT.pptx" and artifacts.exists():
            matches = list(artifacts.glob("*.pptx"))
        add(p, source[p], [str(match.relative_to(ROOT)).replace("\\", "/") for match in matches])
    add(294, "代码仓库 README 依赖 配置与真实Git历史", ["README.md", "requirements.txt", "docs/模型与API配置.md", "reports/requirements/git-history.txt"])
    add(84, "Docker一键部署文件", ["docker/Dockerfile", "docker/docker-compose.yml", "scripts/deploy_docker.ps1"],
        "provided_runtime_unverified", "部署文件与Compose配置校验通过，WSL/Docker已安装但容器未实测；2026-10-15用户取消Docker验收，改为本地127运行留证，不将本地运行写成容器验收。")
    local_runtime_path = ROOT / "reports/deployment/local-20261015/runtime-record.json"
    local_runtime = read(local_runtime_path) if local_runtime_path.exists() else {}
    report = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "source": base["source"],
              "requirements": entries, "summary": dict(Counter(item["execution_status"] for item in entries)),
              "course_evaluation_complete": human_count >= case_count and not any(item["execution_status"] == "quality_comparison_pending" for item in entries),
              "pending_tasks": [item["requirement"] for item in entries if item["execution_status"] in {"human_scoring_pending", "quality_comparison_pending", "provided_runtime_unverified"}],
              "human_fields_pending": ["实际成员职责确认（现为拟定，40/30/30比例已由用户提供）", "实际提交日期待填写"],
              "unverified_environment": ["Docker Engine容器实机运行"],
              "local_runtime_acceptance": {"status": local_runtime.get("status", "not_recorded"), "evidence_path": str(local_runtime_path.relative_to(ROOT)).replace("\\", "/"),
                                           "captured_at": local_runtime.get("captured_at"), "ui_url": local_runtime.get("ui_url"), "api_url": local_runtime.get("api_url"),
                                           "model": local_runtime.get("actual_model"), "unique_check_count": local_runtime.get("unique_check_count"),
                                           "scope": "用户选择本地127验收；Docker验收已取消，本地通过不替代课程原文中的容器实测证明。"},
              "artifact_scope": "Word/PPT保留2026-10-11生成快照；2026-10-14的AI评分已另存，最终提交文档需同步后续结果。本次仅核对证据，不重跑模型、全量测试或容器。",
              "scope": "仅题目一；原文其他三道题为替代选题，不是额外必做项目。"}
    target = ROOT / "reports/requirements/course-audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 南京农业大学课程实践逐条验收", "", f"更新时间：{report['generated_at']}。", "",
             "逐条依据项目原DOCX，区分源码、真实执行、人工评分和环境验收。题目一的应用保持通用，评测专题选择视觉Transformer。", "",
             "| 原文段落 | 要求 | 实现状态 | 执行状态 | 证据 |", "| --- | --- | --- | --- | --- |"]
    labels = {"implemented": "已实现", "artifact_generation_supported": "已提供生成器", "optional_not_selected": "可选未启用",
              "implemented_offline_optional_search_unselected": "离线已实现，联网可选", "targeted_automated_executed": "定向验证通过",
              "real_experiment_executed": "真实实验已执行", "real_functional_executed": "真实功能已验证", "previous_automated_executed": "已有验证通过",
              "artifact_pending_external_delivery_task": "等待最终材料", "not_applicable_unselected_optional": "未选择可选项",
              "completed": "已完成", "partial": "部分完成", "pending": "待完成", "human_scoring_pending": "人工评分待完成",
              "quality_comparison_pending": "实验已执行，质量对比待补",
              "ai_quality_comparison_completed": "AI质量对比已完成",
              "provided_runtime_unverified": "文件已提供，实机未验"}
    for entry in entries:
        paths = entry.get("evidence_paths", entry.get("latest_experiment_evidence", []))
        if not paths:
            paths = [item["path"] for item in entry.get("code", [])][:2]
        description = entry["source_text"].replace("|", "／").replace("\n", " ")
        lines.append(f"| P{entry['source_paragraph_id']} | {description} | {labels.get(entry['implementation_status'], entry['implementation_status'])} | {labels.get(entry['execution_status'], entry['execution_status'])} | {'; '.join(paths)} |")
    lines += ["", f"60题基线已有{ai_summary.get('case_count', 0)}题AI辅助评分，真实人工评分{human_count}/{case_count}题；AI辅助评分、自动引用/数字检查均不等同于课程要求的人工评分。参数与Prompt质量比较以表中状态及latest_quality_evidence为准。Docker一键部署文件已提供，未做容器实机启动。实际提交日不从截止日期推定。", "",
              "Word/PPT是2026-10-11快照；2026-10-14 AI评分另存，最终提交材料需同步。详细结论见docs/剩余任务与验收结论.md。", "",
              f"本地127实际运行：{local_runtime.get('status', 'not_recorded')}；记录见reports/deployment/local-20261015/运行记录.md。Docker验收已按用户要求取消，不将本地运行视为容器实测。", "",
              "现有359片段实验属于旧持久索引，最新生产索引状态以reports/requirements/current-standard-index.json为准。历史引用仍按原始PDF物理页追溯。"]
    (ROOT / "docs/课程要求逐条验收.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"requirements": len(entries), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
