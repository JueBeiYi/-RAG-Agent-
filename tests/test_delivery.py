import json

import pytest

from src.delivery.inputs import deck_payload, is_demo, load_inputs, metric_rows, module_rows


def config(tmp_path, **overrides):
    value = {"project_root": str(tmp_path), "output_dir": str(tmp_path / "delivery"), **overrides}
    target = tmp_path / "config.json"
    target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return target


def result(tmp_path, *, demo=False, mode="vector", corpus="corpus1"):
    path = tmp_path / (mode + ".json")
    path.write_text(json.dumps({"kind": "retrieval", "mode": mode, "demo": demo, "corpus_hash": corpus,
                               "dataset_hash": "data1", "split": ["test"], "metrics": {"case_count": 80, "hit_at_5": .75, "mrr": .6, "recall_at_5": .5}}), encoding="utf-8")
    return str(path)


def test_missing_identity_remains_explicit(tmp_path):
    inputs = load_inputs(config(tmp_path))
    assert "project.members" in inputs.pending_fields
    assert inputs.results == []
    assert all(row[1] == "验证记录待提供" for row in module_rows(inputs))


def test_demo_results_never_enter_formal_presentation_chart(tmp_path):
    inputs = load_inputs(config(tmp_path, results=[result(tmp_path, demo=True)]))
    assert inputs.formal_results == []
    assert deck_payload(inputs)["retrieval"] == []
    assert metric_rows(inputs)[0][-1] == "功能演示数据"


def test_mixed_corpora_are_not_combined_in_one_chart(tmp_path):
    paths = [result(tmp_path, mode="vector"), result(tmp_path, mode="hybrid", corpus="different")]
    payload = deck_payload(load_inputs(config(tmp_path, results=paths)))
    assert [entry["mode"] for entry in payload["retrieval"]] == ["vector"]


def test_user_team_contributions_are_preserved(tmp_path):
    members = [{"name": name, "student_id": sid, "class_name": group, "role": "拟定职责", "contribution_percent": contribution}
               for name, sid, group, contribution in [("赵志国", "11523122", "人工智能231", 40),
                   ("林骋", "11523121", "人工智能231", 30), ("林弟伟", "11523120", "人工智能232", 30)]]
    inputs = load_inputs(config(tmp_path, project={"members": members}))
    assert inputs.project["members"] == members
    assert not any(field.startswith("project.members") for field in inputs.pending_fields)


def test_invalid_contributions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="100"):
        load_inputs(config(tmp_path, project={"members": [{"contribution_percent": 20}, {"contribution_percent": 30}]}))


def test_unverified_implementation_cannot_claim_passed(tmp_path):
    manifest = tmp_path / "implementation.json"
    manifest.write_text(json.dumps({"modules": [{"name": "检索", "status": "implemented", "verified": False}]}), encoding="utf-8")
    rows = module_rows(load_inputs(config(tmp_path, implementation_manifest=str(manifest))))
    assert rows[0][1] == "已提供源码 尚未验证"


def test_design_image_is_rejected_as_execution_evidence(tmp_path):
    image = tmp_path / "mock.png"; image.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="real_application"):
        load_inputs(config(tmp_path, screenshots=[{"path": str(image), "kind": "mockup"}]))


def test_hash_backend_marked_demo_even_when_result_flag_missing():
    assert is_demo({"config": {"embedding_backend": "hash"}})
    assert is_demo({"fixture": True})
