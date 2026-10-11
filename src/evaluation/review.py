from __future__ import annotations

import csv
import json
from pathlib import Path

from .generation import summarise_generation

SCORE_FIELDS = ("correctness_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5", "fluency_1_to_5")


def export_review_sheet(result: dict, path: Path):
    fields = ["case_id", "repeat", "question", "reference_answer", "answer", "reviewer", *SCORE_FIELDS,
              "bad_case_type", "root_cause", "improvement", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in result.get("records", []):
            writer.writerow({key: record.get(key, record.get("human_review", {}).get(key, "")) for key in fields})


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
        review = dict(index[key].get("human_review", {}))
        for field in SCORE_FIELDS:
            value = row.get(field, "").strip()
            if value:
                score = float(value)
                if not 1 <= score <= 5:
                    raise ValueError(f"{key} {field} 评分应为1–5")
                review[field] = score
        for field in ("reviewer", "bad_case_type", "root_cause", "improvement", "status"):
            if row.get(field):
                review[field] = row[field]
        index[key]["human_review"] = review
    updated["metrics"] = summarise_generation(updated["records"])
    updated["review_status"]["human_answer_review_pending"] = any(
        not r.get("human_review", {}).get("reviewer") or
        any(r.get("human_review", {}).get(field) is None for field in SCORE_FIELDS[:3])
        for r in updated["records"])
    return updated
