from __future__ import annotations

import csv
import json
from pathlib import Path

from .generation import summarise_generation

SCORE_FIELDS = ("correctness_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5", "fluency_1_to_5")


def export_review_sheet(result: dict, path: Path):
    fields = ["case_id", "repeat", "question", "reference_answer", "answer", "reviewer", *SCORE_FIELDS,
              "bad_case_type", "root_cause", "improvement", "status", "grading_method", "review_notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in result.get("records", []):
            review = record.get("human_review", {})
            method = "human"
            if not review.get("reviewer") and record.get("ai_review", {}).get("reviewer"):
                review = record["ai_review"]
                method = "ai_assisted"
            writer.writerow({key: method if key == "grading_method" else record.get(key, review.get(key, "")) for key in fields})


def import_reviews(result: dict, path: Path) -> dict:
    updated = json.loads(json.dumps(result, ensure_ascii=False))
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    seen = set()
    index = {(str(r["case_id"]), str(r.get("repeat", 1))): r for r in updated.get("records", [])}
    for row in rows:
        key = (row["case_id"], row.get("repeat") or "1")
        if key in seen or key not in index:
            raise ValueError(f"评分表包含重复或未知样本：{key}")
        seen.add(key)
        if not row.get("reviewer", "").strip():
            continue
        method = row.get("grading_method", "human").strip() or "human"
        if "AI辅助评分" in row.get("reviewer", ""):
            method = "ai_assisted"
        if method not in {"human", "ai_assisted"}:
            raise ValueError(f"{key} 未知评分方式：{method}")
        review_key = "ai_review" if method == "ai_assisted" else "human_review"
        review = dict(index[key].get(review_key, {}))
        for field in SCORE_FIELDS:
            value = row.get(field, "").strip()
            if value:
                score = float(value)
                if not 1 <= score <= 5:
                    raise ValueError(f"{key} {field} 评分应为1–5")
                review[field] = score
        for field in ("reviewer", "bad_case_type", "root_cause", "improvement", "status", "review_notes", "grading_date", "grading_evidence"):
            if row.get(field):
                review[field] = row[field]
        review["grading_method"] = method
        index[key][review_key] = review
    updated["metrics"] = {**updated.get("metrics", {}), **summarise_generation(updated["records"])}
    ai_reviews = [r["ai_review"] for r in updated["records"] if r.get("ai_review", {}).get("reviewer")]
    for field in SCORE_FIELDS:
        values = [review[field] for review in ai_reviews if field in review]
        updated["metrics"]["ai_" + field + "_mean"] = sum(values) / len(values) if values else None
        updated["metrics"]["ai_" + field + "_count"] = len(values)
    updated["metrics"]["ai_scoring_completed_count"] = sum(all(field in review for field in SCORE_FIELDS[:3]) for review in ai_reviews)
    updated["review_status"]["human_answer_review_pending"] = any(
        not r.get("human_review", {}).get("reviewer") or
        any(r.get("human_review", {}).get(field) is None for field in SCORE_FIELDS[:3])
        for r in updated["records"])
    updated["review_status"]["ai_answer_scoring_completed"] = bool(updated["records"]) and updated["metrics"]["ai_scoring_completed_count"] == len(updated["records"])
    return updated
