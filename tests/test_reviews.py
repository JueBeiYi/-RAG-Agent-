import csv

import pytest

from src.evaluation.generation import summarise_generation
from src.evaluation.review import export_review_sheet, import_reviews


def result():
    return {"review_status": {"human_answer_review_pending": True}, "records": [
        {"case_id": "a", "repeat": 1, "success": True, "elapsed_ms": 100, "prompt_tokens": 20,
         "completion_tokens": 10, "metrics": {"agent_steps": 2}, "human_review": {}, "tool_call_count": 2},
        {"case_id": "b", "repeat": 1, "success": False, "elapsed_ms": 200, "prompt_tokens": None,
         "completion_tokens": None, "metrics": {}, "human_review": {}, "tool_call_count": 1, "tool_error_count": 1}]
    }


def test_summary_keeps_unknown_values_unknown():
    summary = summarise_generation(result()["records"])
    assert summary["average_agent_steps"] == 2
    assert summary["agent_steps_known_count"] == 1
    assert summary["prompt_tokens_total"] == 20
    assert summary["token_usage_known_count"] == 1
    assert summary["human_correctness_1_to_5_mean"] is None


def test_review_import_only_named_reviewers_and_valid_scores(tmp_path):
    path = tmp_path / "review.csv"
    export_review_sheet(result(), path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows[0].update(reviewer="Actual reviewer", correctness_1_to_5="4", completeness_1_to_5="3")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reviewed = import_reviews(result(), path)
    assert reviewed["metrics"]["human_correctness_1_to_5_mean"] == 4
    assert reviewed["review_status"]["human_answer_review_pending"] is True
    rows[0]["correctness_1_to_5"] = "99"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="评分"):
        import_reviews(result(), path)


def test_reviewer_name_without_required_scores_is_still_pending(tmp_path):
    data = result()
    path = tmp_path / "review.csv"
    export_review_sheet(data, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for row in rows:
        row["reviewer"] = "Reviewer"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    assert import_reviews(data, path)["review_status"]["human_answer_review_pending"] is True
