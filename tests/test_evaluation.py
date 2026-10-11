import json

import pytest

from src.core.config import Settings
from src.core.schemas import Chunk, Document, RetrievalHit, SourceBlock
from src.evaluation.dataset import EvaluationCase, RelevantSource, load_cases
from src.evaluation.metrics import evidence_matches_chunk, map_evidence_to_chunks, retrieval_metrics
from src.evaluation.runner import run_retrieval, serialize_export


def make_chunk(cid, text="模型在独立测试集上的准确率为百分之八十。", doc="d1", page=3):
    return Chunk(chunk_id=cid, document_id=doc, text=text, page_start=page, page_end=page)


def test_metrics_multiple_sources_do_not_count_duplicate_chunks():
    # Two fragments from evidence A should not outweigh missing evidence B.
    value = retrieval_metrics(["a1", "a2", "noise"], [{"a1", "a2"}, {"b1"}])
    assert value["hit_at_5"] == 1
    assert value["mrr"] == 1
    assert value["recall_at_5"] == 0.5
    assert value["recalled_evidence_count"] == 1


def test_mrr_uses_full_ranking_but_hit5_does_not():
    result = retrieval_metrics(["a", "b", "c", "d", "e", "answer"], [{"answer"}])
    assert result["hit_at_5"] == 0
    assert result["recall_at_5"] == 0
    assert result["mrr"] == pytest.approx(1 / 6)


def test_evidence_requires_correct_page_and_quote():
    evidence = RelevantSource(document_id="d1", page=3, quote="准确率为百分之八十")
    assert evidence_matches_chunk(evidence, make_chunk("right"))
    assert not evidence_matches_chunk(evidence, make_chunk("wrong-page", page=2))
    assert not evidence_matches_chunk(evidence, make_chunk("wrong-doc", doc="d2"))
    assert not evidence_matches_chunk(evidence, make_chunk("wrong-text", text="使用自适应优化器进行训练。"))


def test_quote_normalises_pdf_whitespace():
    evidence = RelevantSource(document_id="d1", page=3, quote="Accuracy is 80 percent.")
    assert evidence_matches_chunk(evidence, make_chunk("right", text="ACCURACY\nis 80   percent."))


def test_original_source_block_must_not_label_unrelated_chunk():
    chunk = make_chunk("noise", text="这段仅说明训练设备。")
    chunk.sources = [SourceBlock(text="模型在测试集的准确率为百分之八十。", page=3)]
    source = RelevantSource(document_id="d1", page=3, quote="准确率为百分之八十")
    assert not evidence_matches_chunk(source, chunk)


def test_ground_truth_survives_changed_chunk_ids_and_boundaries():
    quote = "0123456789" * 12
    source = RelevantSource(document_id="d1", page=3, quote=quote)
    before = map_evidence_to_chunks([source], [make_chunk("old", text=quote)])
    after = map_evidence_to_chunks([source], [make_chunk("new1", text=quote[:60]), make_chunk("new2", text=quote[60:])])
    assert retrieval_metrics(["old"], before)["recall_at_5"] == 1
    assert retrieval_metrics(["new1"], after)["recall_at_5"] == 1


def test_dataset_review_guard_and_duplicate_ids(tmp_path):
    record = {"case_id": "f1", "question": "结果如何", "category": "factual", "split": "dev", "reviewed": False}
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="来源核对"):
        load_cases(path)
    assert len(load_cases(path, require_reviewed=False)) == 1
    path.write_text(json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_cases(path, require_reviewed=False)


class Store:
    def __init__(self, chunks):
        self.chunks = chunks
    def list_chunks(self, scope):
        return self.chunks
    def index_version(self, scope):
        return "version1"
    def list_documents(self, scope):
        return [Document(document_id="d1", name="paper.pdf", content_hash="hash1", format="pdf")]


def case():
    return EvaluationCase(case_id="f1", question="准确率如何", category="factual", reviewed=True,
                          reference_answer="百分之八十", relevant_sources=[RelevantSource(document_id="d1", page=3, quote="准确率为百分之八十")])


def test_runner_refuses_unmapped_evidence():
    with pytest.raises(ValueError, match="未映射"):
        run_retrieval(None, Store([make_chunk("bad", page=7)]), [case()], "test-scope")


def test_runner_real_ranking_and_demo_label():
    class Retriever:
        def search(self, question, scope, mode, top_k):
            return [RetrievalHit(chunk=make_chunk("noise", text="训练设备说明。"), score=0.8),
                    RetrievalHit(chunk=make_chunk("right"), score=0.7)]
    settings = Settings(embedding_backend="hash", llm_backend="extractive")
    result = run_retrieval(Retriever(), Store([make_chunk("right")]), [case()], "test-scope", settings=settings)
    assert result["metrics"]["mrr"] == 0.5
    assert result["demo"] is True
    assert result["formal_models"] is False
    assert result["dataset_requirement_checks"]["at_least_50_reviewed_cases"] is False
    assert result["dataset_requirement_checks"]["human_corpus_confirmation"] == "not_recorded"
    assert len(result["config_hash"]) == 64
    assert result["index_version"] == "version1"


def test_runner_failures_count_zero_quality():
    class BrokenRetriever:
        def search(self, *args, **kwargs):
            raise RuntimeError("模型暂不可用")
    result = run_retrieval(BrokenRetriever(), Store([make_chunk("right")]), [case()], "test-scope")
    assert result["metrics"]["hit_at_5"] == 0
    assert result["metrics"]["error_count"] == 1
    assert "模型暂不可用" in result["records"][0]["error"]


def test_json_csv_exports_preserve_chinese_and_missing_values():
    result = {"records": [{"case_id": "中文题目", "prompt_tokens": None, "evidence": ["来源A"]}]}
    raw = serialize_export(result, "json")
    assert json.loads(raw)["records"][0]["prompt_tokens"] is None
    csv_data = serialize_export(result, "csv").decode("utf-8-sig")
    assert "中文题目" in csv_data and "来源A" in csv_data
    with pytest.raises(ValueError):
        serialize_export(result, "unknown")
