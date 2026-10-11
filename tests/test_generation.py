import json

import httpx
import numpy as np
import pytest

from src.core.config import Settings
from src.core.schemas import Chunk, Document, RetrievalHit, SourceBlock
from src.core.storage import Storage
from src.generation.cache import SemanticCache
from src.generation.context import audit_citations, build_context, estimate_tokens
from src.generation.llm_client import LLMClient, ModelUnavailable
from src.generation.rag_pipeline import RAGEngine
from src.generation.context import normalise_page_references


class Encoder:
    fingerprint = "revision-1"

    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts])


def test_semantic_cache_blocks_negative_intent_and_revision(tmp_path):
    store = Storage(tmp_path)
    settings = Settings(embedding_backend="hash", llm_backend="extractive")
    cache = SemanticCache(settings, store, Encoder())
    fingerprint = cache.fingerprint("u", [], None, [])
    cache.put("该方法有什么优点", "u", fingerprint, {"answer": "优点", "audit": {}})
    assert cache.get("该方法有哪些优点", "u", fingerprint)["answer"] == "优点"
    assert cache.get("该方法有什么缺点", "u", fingerprint) is None
    assert cache.get("该方法没有什么优点", "u", fingerprint) is None
    cache.embedder.fingerprint = "revision-2"
    assert cache.fingerprint("u", [], None, []) != fingerprint
    store.bump_version("u")
    assert cache.get("该方法有什么优点", "u", fingerprint) is None


def test_numeric_evidence_does_not_accept_substrings():
    audit = audit_citations("精度是8.2%。[S1]\n结果为9个。[S1]", [{"citation_id": "S1", "text": "精度88.2%，90个样本。"}])
    assert audit["needs_review"]
    assert "8.2" in audit["claims"][0]["missing_numbers"]


def test_cross_page_context_and_budget_preserve_true_location():
    blocks = [SourceBlock(text="第一页研究背景与方法。" * 8, page=1), SourceBlock(text="第二页关键实验结果。" * 8, page=2)]
    text = "\n".join(b.text for b in blocks)
    hit = RetrievalHit(chunk=Chunk(chunk_id="x", document_id="d", text=text, name="d.pdf", page_start=1, page_end=2, sources=blocks))
    context, citations = build_context([hit], 500)
    assert "第1–2页" in context and "原文第2页" in context
    context, citations = build_context([hit], 100)
    assert estimate_tokens(context) <= 100
    assert citations[0].page_end == 1
    assert not any(b.page == 2 for b in citations[0].sources)


@pytest.mark.parametrize("lines", [
    [{"message": {"content": "half"}}],
    [{"error": "out of memory"}],
])
def test_provider_partial_eof_and_errors_fail(monkeypatch, lines):
    original = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content="\n".join(json.dumps(x) for x in lines)))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    client = LLMClient(Settings())
    with pytest.raises(ModelUnavailable):
        list(client.stream([{"role": "user", "content": "question"}]))


def test_ollama_stream_terminal_usage_and_length_warning(monkeypatch):
    original = httpx.Client
    lines = [{"message": {"content": "答案"}}, {"done": True, "done_reason": "length", "prompt_eval_count": 12, "eval_count": 3}]
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content="\n".join(json.dumps(x) for x in lines)))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    events = list(LLMClient(Settings()).stream([{"role": "user", "content": "question"}]))
    assert any(e.get("warning") for e in events)
    assert [e for e in events if e["type"] == "usage"] == [{"type": "usage", "input_tokens": 12, "output_tokens": 3, "estimated": False}]


def test_optional_cache_failure_does_not_drop_completed_answer(tmp_path):
    store = Storage(tmp_path)
    doc = Document(document_id="d", scope="u", name="paper.txt", format="txt", content_hash="h")
    chunk = Chunk(chunk_id="c", document_id="d", scope="u", text="研究方法采用 Transformer", name="paper.txt", paragraph=1)
    store.commit_document(doc, [chunk])
    class Retriever:
        embedder = Encoder()
        last_search = {}
        def search(self, *args):
            return [RetrievalHit(chunk=chunk, score=1.)]
    class LLM:
        def stream(self, messages):
            yield {"type": "answer_delta", "text": "采用 Transformer。[S1]"}
            yield {"type": "usage", "input_tokens": 12, "output_tokens": 5, "estimated": False}
    engine = RAGEngine(Settings(embedding_backend="hash", llm_backend="extractive"), store, Retriever(), LLM())
    engine.cache.get = lambda *args: (_ for _ in ()).throw(OSError("read failed"))
    engine.cache.put = lambda *args: (_ for _ in ()).throw(OSError("write failed"))
    events = list(engine.stream("研究方法", "u"))
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "采用 Transformer。[S1]"
    assert len(events[-1]["warnings"]) == 2


def test_atomic_commit_publishes_version_and_invalidates(tmp_path):
    store = Storage(tmp_path)
    store.put("cache", "old", "u", {"answer": "old"})
    doc = Document(document_id="d", scope="u", name="d.txt", format="txt", content_hash="hash")
    version = store.commit_document(doc, [Chunk(chunk_id="c", document_id="d", scope="u", text="text")])
    assert store.get_document("d", "u").status == "ready"
    assert store.list_chunks("u")[0].index_version == store.index_version("u") == version
    assert store.list("cache", "u") == []


def test_page_claims_use_server_location_and_keep_unknown_sources_visible():
    answer, corrections = normalise_page_references("据[S1] 第4页，窗口移位产生跨窗口联系。另[S2] 第3页。", [
        {"citation_id": "S1", "page_start": 2, "page_end": 2},
        {"citation_id": "S2", "page_start": 3, "page_end": 5}])
    assert "[S1]（物理页2）" in answer
    assert "[S2]（物理页3至5）" in answer
    assert corrections[0]["server_location"] == "物理页2"
    unknown, _ = normalise_page_references("[S99]第4页", [])
    assert unknown == "[S99]第4页"
    numeric_fact, _ = normalise_page_references("[S1] 95 percent of samples.", [{"citation_id": "S1", "page_start": 2}])
    assert numeric_fact == "[S1] 95 percent of samples."


def test_authoritative_location_numbers_are_not_result_claims():
    citations = [{"citation_id": "S1", "text": "窗口移位产生跨窗口联系。"}]
    assert not audit_citations("窗口移位产生跨窗口联系。[S1]（物理页4至5）", citations)["needs_review"]
    assert audit_citations("性能提高了4%。[S1]（物理页4至5）", citations)["needs_review"]


def test_numeric_audit_recognises_sentence_period_without_loosening_unknown_values():
    citations = [{"citation_id": "S1", "text": "Window size is 7. Accuracy is 90.5."}]
    assert not audit_citations("窗口尺寸为7，准确率为90.5[S1]。", citations)["needs_review"]
    assert audit_citations("窗口尺寸为8，准确率为90.6[S1]。", citations)["needs_review"]


def test_structured_completion_requests_provider_json_grammar(monkeypatch):
    original = httpx.Client
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"rows": []}'}, "prompt_eval_count": 12, "eval_count": 5})
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    result = LLMClient(Settings(llm_backend="ollama")).complete_json([{"role": "user", "content": "返回JSON rows"}])
    assert captured[0]["format"] == "json" and captured[0]["stream"] is False
    assert result["content"] == '{"rows": []}' and result["usage"]["output_tokens"] == 5
    schema = {"type": "object", "properties": {"rows": {"type": "array", "maxItems": 2}}, "required": ["rows"]}
    LLMClient(Settings(llm_backend="ollama")).complete_json([{"role": "user", "content": "返回两行JSON"}], schema=schema)
    assert captured[1]["format"] == schema
