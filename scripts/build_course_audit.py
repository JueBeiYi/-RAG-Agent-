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
    add(250, "实际答案人工评分表", forms, "human_scoring_pending", "用户明确要求先生成可填写表；评分人、正确性、完整性、引用质量不代填。")
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
        "provided_runtime_unverified", "Compose与启动脚本静态校验通过；本机没有Docker Engine，不声明容器已实机启动。")
    report = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "source": base["source"],
              "requirements": entries, "summary": dict(Counter(item["execution_status"] for item in entries)),
              "user_selected_pending": ["人工评分先提供可填写表", "实际提交日期待填写"],
              "unverified_environment": ["Docker Engine容器实机运行"],
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
              "completed": "已完成", "partial": "部分完成", "pending": "待完成", "human_scoring_pending": "评分表已提供，人工待填",
              "provided_runtime_unverified": "文件已提供，实机未验"}
    for entry in entries:
        paths = entry.get("evidence_paths", entry.get("latest_experiment_evidence", []))
        if not paths:
            paths = [item["path"] for item in entry.get("code", [])][:2]
        description = entry["source_text"].replace("|", "／").replace("\n", " ")
        lines.append(f"| P{entry['source_paragraph_id']} | {description} | {labels.get(entry['implementation_status'], entry['implementation_status'])} | {labels.get(entry['execution_status'], entry['execution_status'])} | {'; '.join(paths)} |")
    lines += ["", "人工评分按用户选择保留待填；自动引用/数字检查不等同于正确性评分。Docker一键部署文件已提供，本机未做容器实机启动。实际提交日不从截止日期推定。", "",
              "现有359片段实验属于旧持久索引，最新生产索引状态以reports/requirements/current-standard-index.json为准。历史引用仍按原始PDF物理页追溯。"]
    (ROOT / "docs/课程要求逐条验收.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"requirements": len(entries), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
