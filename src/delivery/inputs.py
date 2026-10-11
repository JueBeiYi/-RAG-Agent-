from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_TITLE = "智能科研助理 基于RAG与Agent的论文知识库问答系统"
COURSE_SECTIONS = ["一 项目概述", "二 团队分工说明", "三 模块交付物清单", "四 实现记录",
                   "五 技术设计文档", "六 总结与展望", "七 附录"]
STATUS_LABELS = {"implemented": "已实现并验证", "partial": "部分实现", "not_implemented": "未实现",
                 "untested": "已提供源码 尚未验证", "not_executed": "尚未执行", "passed": "验证通过"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass
class DeliveryInputs:
    config: dict
    root: Path
    output: Path
    results: list[dict]
    implementation: dict
    screenshots: list[dict]
    bad_cases: list[dict]
    source_files: list[dict]

    @property
    def project(self) -> dict:
        return self.config.get("project", {})

    @property
    def title(self) -> str:
        return self.project.get("title", PROJECT_TITLE)

    @property
    def formal_results(self) -> list[dict]:
        return [result for result in self.results if not is_demo(result)]

    @property
    def pending_fields(self) -> list[str]:
        pending = []
        for field in ("college", "course", "term", "submission_date"):
            if self.project.get(field) in (None, "", "待填写"):
                pending.append("project." + field)
        members = self.project.get("members", [])
        if not members:
            pending.append("project.members")
        for i, member in enumerate(members):
            for field in ("name", "student_id", "class_name", "role", "contribution_percent"):
                if member.get(field) in (None, "", "待填写"):
                    pending.append(f"project.members[{i}].{field}")
        if not self.config.get("human_scoring_completed", False):
            pending.append("human_review.answer_quality_scores")
        if not self.config.get("human_dataset_review_completed", False):
            pending.append("human_review.question_and_evidence_confirmation")
        return pending


def is_demo(result: dict) -> bool:
    config = result.get("config", {})
    return bool(result.get("demo") or result.get("fixture") or
                config.get("embedding_backend") == "hash" or config.get("llm_backend") == "extractive")


def load_inputs(config_path: str | Path) -> DeliveryInputs:
    path = Path(config_path).resolve()
    config = read_json(path)
    root = Path(config.get("project_root", path.parent)).resolve()
    output = Path(config.get("output_dir", root / "artifacts" / "delivery")).resolve()

    def resolve(value):
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    sources, results = [], []
    for entry in config.get("results", []):
        source = resolve(entry if isinstance(entry, str) else entry["path"])
        value = read_json(source)
        if "result" in value:
            value = value["result"]
        if isinstance(value, dict) and "document_encoding_total_ms" in value and "query_encoding_total_ms" in value:
            value = {**value, "kind": "embedding_encoding", "config": {
                "embedding_model": value.get("embedding_model"), "embedding_backend": value.get("embedding_backend")},
                "metrics": {key: value.get(key) for key in ("unique_document_text_count", "document_encoding_total_ms", "document_texts_per_second",
                    "query_count", "query_encoding_total_ms", "query_encoding_amortized_ms", "short_encode_max_abs_deviation")}}
        if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
            raise ValueError(f"无有效真实指标：{source}")
        source_sha256 = digest(source)
        # Reports use aggregate measurements and provenance. Full per-case
        # rankings/answers remain in source JSON instead of multiplying memory.
        if value.get("kind") != "vector_database_comparison":
            value.pop("records", None)
        value = {**value, "source_path": str(source), "source_sha256": source_sha256}
        if isinstance(entry, dict):
            value["delivery_label"] = entry.get("label", source.parent.name)
            value["study"] = entry.get("study", value.get("kind", "retrieval"))
            value["delivery_scope"] = entry.get("scope", "开发集实验" if value.get("split") == "dev" else "实际测量")
        results.append(value)
        sources.append({"path": str(source), "sha256": source_sha256, "kind": "experiment", "demo": is_demo(value)})
    for entry in config.get("evidence_files", []):
        source = resolve(entry if isinstance(entry, str) else entry["path"])
        if not source.is_file():
            raise ValueError(f"验证证据不存在：{source}")
        sources.append({"path": str(source), "sha256": digest(source), "kind": "verification"})
    implementation = {}
    manifest = config.get("implementation_manifest")
    if manifest:
        manifest_path = resolve(manifest)
        implementation = read_json(manifest_path)
        sources.append({"path": str(manifest_path), "sha256": digest(manifest_path), "kind": "implementation"})
    screenshots = []
    for entry in config.get("screenshots", []):
        source = resolve(entry["path"])
        if not source.is_file():
            raise ValueError(f"界面截图不存在：{source}")
        if entry.get("kind") != "real_application":
            raise ValueError("界面截图必须明确 kind=real_application，不能将界面设计稿当作运行截图")
        screenshots.append({**entry, "path": str(source), "sha256": digest(source)})
        sources.append({"path": str(source), "sha256": digest(source), "kind": "screenshot"})
    bad_cases = config.get("bad_cases", [])
    if isinstance(bad_cases, str):
        bad_case_path = resolve(bad_cases)
        bad_cases = read_json(bad_case_path)
        sources.append({"path": str(bad_case_path), "sha256": digest(bad_case_path), "kind": "bad_case"})
    if not isinstance(bad_cases, list):
        raise ValueError("bad_cases 必须为记录数组或JSON文件路径")
    contributions = [member.get("contribution_percent") for member in config.get("project", {}).get("members", [])]
    if contributions and all(value is not None for value in contributions):
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 100 for value in contributions):
            raise ValueError("贡献比例必须为0至100的实际数值")
        if abs(sum(contributions) - 100) > .01:
            raise ValueError("已填写的团队贡献比例总和必须为100%")
    return DeliveryInputs(config, root, output, results, implementation, screenshots, bad_cases, sources)


def module_rows(inputs: DeliveryInputs) -> list[list[str]]:
    modules = inputs.implementation.get("modules", [])
    if modules:
        rows = []
        for module in modules:
            verified = module.get("verified") is True
            status = module.get("status", "untested")
            if status == "implemented" and not verified:
                status = "untested"
            evidence = module.get("evidence", module.get("evidence_files", []))
            if isinstance(evidence, str):
                evidence = [evidence]
            rows.append([module.get("name", "未命名模块"), STATUS_LABELS.get(status, status),
                         "\n".join(str(item) for item in evidence) or "待补充验证记录"])
        return rows
    return [[name, "验证记录待提供", path] for name, path in [
        ("文档处理与检索", "src/data_loader src/chunking src/retrieval"),
        ("RAG生成", "src/generation"), ("Agent决策", "src/agent"),
        ("系统与前端", "src/api src/services src/frontend"), ("评测与交付", "src/evaluation src/delivery")]]


def metric_rows(inputs: DeliveryInputs, kind: str | None = None) -> list[list[str]]:
    rows = []
    for result in inputs.results:
        if kind and result.get("kind") != kind:
            continue
        label = result.get("delivery_label", result.get("mode", result.get("experiment_id", "实验")))
        status = "功能演示数据" if is_demo(result) else "实际模型运行数据"
        for metric, value in result["metrics"].items():
            if isinstance(value, (dict, list)):
                continue
            display = "未测量" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            rows.append([label, metric, display, status])
    return rows


def deck_payload(inputs: DeliveryInputs) -> dict:
    retrieval = sorted([r for r in inputs.results if r.get("kind") == "retrieval" and not is_demo(r)],
                       key=lambda r: (r.get("metrics", {}).get("case_count", 0), "test" in r.get("split", [])), reverse=True)
    # Show a matched group. Different corpora/questions are never silently mixed.
    selected = []
    if retrieval:
        first = retrieval[0]
        def experiment_identity(result):
            cfg = result.get("config", {})
            return (result.get("dataset_hash"), result.get("corpus_hash"), str(result.get("split")),
                    result.get("chunk_snapshot_hash"), cfg.get("embedding_model"), cfg.get("chunk_strategy"),
                    cfg.get("chunk_size"), cfg.get("chunk_overlap"), cfg.get("pdf_parser"))
        identity = experiment_identity(first)
        seen = set()
        for result in retrieval:
            if experiment_identity(result) == identity and result.get("mode") not in seen:
                seen.add(result.get("mode"))
                selected.append({key: result.get(key) for key in ("mode", "kind", "metrics", "config", "source_path", "delivery_scope", "dataset_hash", "corpus_hash", "split")})
    return {
        "project": {"title": inputs.title, **inputs.project}, "modules": module_rows(inputs),
        "screenshots": inputs.screenshots[:5], "retrieval": selected,
        "generation": [{key: r.get(key) for key in ("kind", "metrics", "config", "source_path", "delivery_scope")}
                       for r in inputs.formal_results if r.get("kind") == "generation"][:3],
        "online": [{key: r.get(key) for key in ("kind", "metrics", "source_path", "delivery_scope")}
                   for r in inputs.formal_results if r.get("kind") in {"online_functionality", "online_functional"}],
        "experiments": [{"label": r.get("delivery_label", r.get("mode", "实验")), "kind": r.get("kind"),
                         "study": r.get("study"), "config": r.get("config", {}), "metrics": r["metrics"],
                         "demo": is_demo(r), "source_path": r["source_path"]} for r in inputs.results],
        "pending_fields": inputs.pending_fields, "bad_cases": inputs.bad_cases,
        "validation_summary": inputs.config.get("validation_summary", []),
        "evaluation_note": inputs.config.get("evaluation_note", "人工答案质量评分尚待完成"),
        "source_files": inputs.source_files, "source_count": len(inputs.results),
        "formal_result_count": len(inputs.formal_results), "demo_result_count": sum(is_demo(r) for r in inputs.results),
    }
