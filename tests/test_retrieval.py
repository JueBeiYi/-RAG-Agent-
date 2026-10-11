from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.chunking.strategies import _ranges, count_tokens, split_blocks
from src.core.config import Settings
from src.core.schemas import Chunk, Document, SourceBlock
from src.core.storage import Storage
from src.retrieval.embeddings import Embedder
from src.retrieval.engine import Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.tokenization import tokenize


def settings(tmp_path):
    return Settings(data_dir=str(tmp_path), embedding_backend="hash", vector_backend="sqlite", reranker_backend="lexical")


def add_document(store, retriever, identifier, text, scope="alice"):
    document = Document(document_id=identifier, scope=scope, name=identifier + ".md",
                        content_hash=identifier, format="md", status="ready")
    chunks = split_blocks(document, [SourceBlock(text=text, paragraph=1)], size=64, overlap=8)
    retriever.index(chunks)
    store.replace_chunks(identifier, scope, chunks)
    store.put_document(document)
    store.bump_version(scope)
    return chunks


def test_rrf_combines_independent_ranks_and_deduplicates():
    fusion = reciprocal_rank_fusion({"vector": ["a", "a", "b"], "bm25": ["b", "c"]}, k=10)
    assert fusion[0][0] == "b"
    assert dict(fusion)["a"] == pytest.approx(1 / 11)
    assert dict(fusion)["b"] == pytest.approx(1 / 12 + 1 / 11)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_chunking_keeps_tail_and_real_multpage_provenance(strategy):
    document = Document(document_id="article", scope="u", name="article.pdf", content_hash="abc", format="pdf")
    blocks = [SourceBlock(text="第一段 " + "token " * 45, page=1, paragraph=1, bbox=[1, 2, 40, 60], section="方法"),
              SourceBlock(text="| dataset | score |\n| --- | --- |\n| ImageNet | 88.2 |", page=2, paragraph=2,
                          bbox=[1, 2, 60, 70], kind="table"),
              SourceBlock(text="末尾重要证据 x²+y²=z²。", page=3, paragraph=3, bbox=[1, 2, 80, 100], kind="formula")]
    chunks = split_blocks(document, blocks, strategy=strategy, size=24, overlap=5)
    assert chunks and any("末尾重要证据" in c.text for c in chunks)
    assert chunks[-1].text.endswith("z²。")
    assert chunks[-1].page_end == 3
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(count_tokens(c.text) <= 24 for c in chunks)
    for source in blocks:
        fragments = "".join(s.text for c in chunks for s in c.sources if s.paragraph == source.paragraph)
        assert all(token.group() in fragments for token in re.finditer(r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]", source.text))
    assert all(c.bbox is None for c in chunks if c.page_start != c.page_end)
    assert all(s.page in {1, 2, 3} for c in chunks for s in c.sources)
    if strategy != "fixed":
        assert any(blocks[-1].text in c.text for c in chunks)


def test_recursive_splitter_candidates_preserve_all_source_positions_and_budget(monkeypatch):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    original = RecursiveCharacterTextSplitter.create_documents
    calls = []
    def observed(self, texts, *args, **kwargs):
        calls.append(texts)
        return original(self, texts, *args, **kwargs)
    monkeypatch.setattr(RecursiveCharacterTextSplitter, "create_documents", observed)
    text = ("重复段落 containing longer English words and punctuation.\n\n" * 15
            + "末尾唯一证据 END-OF-SOURCE。").strip()
    ranges = _ranges(text, "recursive", 32, 7)
    assert calls == [[text]]
    covered = {position for start, end in ranges for position in range(start, end)}
    assert all(position in covered for position, char in enumerate(text) if not char.isspace())
    assert all(count_tokens(text[start:end]) <= 32 for start, end in ranges)
    assert ranges[-1][1] == len(text)


def test_recursive_proposals_do_not_split_short_table_rows_or_formula():
    document = Document(document_id="atomic", scope="u", name="atomic.pdf", content_hash="atomic", format="pdf")
    table = "| 数据集 | 准确率 |\n| ImageNet-1K | 88.2% |"
    formula = "E = mc² + x²。"
    blocks = [SourceBlock(text="背景 " + "context " * 27, page=1, paragraph=1),
              SourceBlock(text=table, page=2, paragraph=2, kind="table"),
              SourceBlock(text=formula, page=3, paragraph=3, kind="formula"),
              SourceBlock(text="后文 " + "discussion " * 24, page=3, paragraph=4)]
    chunks = split_blocks(document, blocks, "recursive", size=32, overlap=8)
    assert any("| ImageNet-1K | 88.2% |" in chunk.text for chunk in chunks)
    assert any(formula in chunk.text for chunk in chunks)
    assert all(count_tokens(chunk.text) <= 32 for chunk in chunks)
    assert all(source.page in {1, 2, 3} for chunk in chunks for source in chunk.sources)


def test_search_modes_persistence_and_scope_isolation(tmp_path):
    config = settings(tmp_path)
    store = Storage(tmp_path)
    retrieval = Retriever(config, store)
    good = add_document(store, retrieval, "vit", "Vision Transformer ViT-B/16 achieved 88.2 accuracy on ImageNet using attention.")
    add_document(store, retrieval, "wheat", "小麦表型识别使用农业图像数据集与卷积网络。")
    add_document(store, retrieval, "private", "Vision Transformer SECRET 100 percent accuracy.", scope="bob")
    for mode in ("vector", "hybrid", "hybrid_rerank", "bm25"):
        hits = retrieval.search("ViT-B/16 ImageNet accuracy", "alice", mode=mode)
        assert hits[0].chunk.document_id == "vit"
        assert hits[0].score > .15
        assert all(h.chunk.scope == "alice" for h in hits)
        assert all(0 <= h.score <= 1 for h in hits)
        assert "SECRET" not in "".join(h.chunk.text for h in hits)
    reopened = Retriever(config, Storage(tmp_path))
    assert reopened.search("ImageNet", "alice", mode="vector")[0].chunk.chunk_id == good[0].chunk_id
    assert reopened.health()["demo"] is True
    assert reopened.search("ImageNet", "unknown") == []
    assert {h.chunk.document_id for h in reopened.search("ViT", "alice", document_ids=["wheat"])} <= {"wheat"}


def test_incremental_index_reuses_vectors_and_replacement_removes_old_rows(tmp_path, monkeypatch):
    config = settings(tmp_path)
    store, retrieval = Storage(tmp_path), Retriever(config, Storage(tmp_path))
    chunks = add_document(store, retrieval, "paper", "UniqueTerm AlphaMethod experiment 91.4 on DatasetAlpha.")
    calls = []
    original = retrieval.embedder.encode
    monkeypatch.setattr(retrieval.embedder, "encode", lambda texts: (calls.append(texts), original(texts))[1])
    retrieval.index(chunks)
    assert calls == []
    document = store.get_document("paper", "alice")
    replacement = split_blocks(document, [SourceBlock(text="BetaMethod experiment 90.0 on DatasetBeta.")], size=64, overlap=8)
    retrieval.index(replacement)
    store.replace_chunks("paper", "alice", replacement)
    store.bump_version("alice")
    assert len(calls) == 1
    assert {row[0] for row in retrieval.vectors.rows("alice")} == {c.chunk_id for c in replacement}
    assert all("AlphaMethod" not in h.chunk.text for h in retrieval.search("AlphaMethod", "alice"))


def test_deletion_and_not_ready_records_cannot_leak(tmp_path):
    config = settings(tmp_path)
    store = Storage(tmp_path)
    retriever = Retriever(config, store)
    add_document(store, retriever, "same", "用户甲独有证据 Alpha。")
    add_document(store, retriever, "same", "用户乙独有证据 Beta。", scope="bob")
    doc = store.get_document("same", "alice")
    doc.status = "indexing"
    store.put_document(doc)
    assert retriever.search("Alpha", "alice") == []
    retriever.delete("same", "alice")
    store.delete_document("same", "alice")
    assert retriever.search("Alpha", "alice") == []
    assert retriever.search("Beta", "bob")[0].chunk.scope == "bob"
    assert len(retriever.vectors.rows("alice")) == 0
    assert len(retriever.vectors.rows("bob")) > 0


def test_stale_vectors_are_filtered_by_authoritative_metadata(tmp_path):
    store = Storage(tmp_path)
    retriever = Retriever(settings(tmp_path), store)
    add_document(store, retriever, "paper", "Unlisted secret alpha content")
    store.delete_document("paper", "alice")  # simulate crash before vector delete
    assert retriever.vectors.rows("alice")
    assert retriever.search("secret", "alice") == []


def test_rerank_failure_returns_hybrid_results_and_visible_warning(tmp_path, monkeypatch):
    store = Storage(tmp_path)
    retriever = Retriever(settings(tmp_path), store)
    add_document(store, retriever, "paper", "ImageNet Transformer accuracy is 88.2.")
    def fail(*_):
        raise RuntimeError("test unavailable model")
    monkeypatch.setattr(retriever.reranker, "rerank", fail)
    hits = retriever.search("ImageNet accuracy", "alice", mode="hybrid_rerank")
    assert hits and hits[0].rrf_score is not None
    assert "降级" in retriever.last_search["warning"]
    assert "unavailable" in retriever.health()["warning"]


def test_tokenization_preserves_bilingual_identifiers():
    tokens = tokenize("Vision Transformer ViT-B/16 植物表型识别 DOI 10.1234/abcd.2024")
    assert "vit-b/16" in tokens and "transformer" in tokens
    assert "10.1234/abcd.2024" in tokens
    assert any("表型" in t for t in tokens)


def test_real_embedding_window_policy_does_not_drop_tail(tmp_path):
    torch = pytest.importorskip("torch")
    class Tokenizer:
        model_max_length = 8
        @staticmethod
        def num_special_tokens_to_add(pair=False):
            return 2
        @staticmethod
        def encode(text, **_):
            return [int(t.removeprefix("t")) for t in text.split()]
        @staticmethod
        def decode(ids, **_):
            raise AssertionError("Original IDs must never decode/re-encode")
        @staticmethod
        def prepare_for_model(ids, **_):
            return {"input_ids": [1000, *ids, 1001], "attention_mask": [1] * (len(ids) + 2)}
        @staticmethod
        def pad(rows, **_):
            width = max(len(row["input_ids"]) for row in rows)
            return {key: torch.tensor([row[key] + [0] * (width - len(row[key])) for row in rows])
                    for key in ("input_ids", "attention_mask")}

    class Model:
        max_seq_length = 8
        tokenizer = Tokenizer()
        seen = []
        device = "cpu"
        def eval(self):
            return self
        def __call__(self, features):
            rows = features["input_ids"].tolist()
            self.seen.extend([i for i in row[1:row.index(1001)]] for row in rows)
            return {"sentence_embedding": torch.tensor([[1, 0, int(19 in row)] for row in rows], dtype=torch.float32)}

    embedder = Embedder(Settings(data_dir=str(tmp_path), embedding_backend="sentence_transformers"))
    embedder.model = Model()
    vector = embedder.encode([" ".join(f"t{i}" for i in range(20))])
    seen = [i for row in embedder.model.seen for i in row]
    assert seen == list(range(20))
    assert vector[0, 2] > 0
    assert embedder.long_input_count == 1
    assert embedder.last_window_count == 4


def test_embedding_fingerprint_separates_models(tmp_path):
    hash_model = Embedder(settings(tmp_path))
    real_model = Embedder(Settings(data_dir=str(tmp_path), embedding_model="example/bge"))
    another = Embedder(Settings(data_dir=str(tmp_path), embedding_model="example/m3e"))
    assert len({hash_model.fingerprint, real_model.fingerprint, another.fingerprint}) == 3


def test_search_timing_and_warnings_are_thread_local(tmp_path):
    store = Storage(tmp_path)
    retriever = Retriever(settings(tmp_path), store)
    add_document(store, retriever, "a", "Alpha Transformer method.", scope="alice")
    add_document(store, retriever, "b", "Beta Transformer method.", scope="bob")
    def search(scope, mode):
        hits = retriever.search("Transformer", scope, mode=mode)
        return retriever.last_search.copy(), hits
    with ThreadPoolExecutor(max_workers=2) as executor:
        alice = executor.submit(search, "alice", "vector")
        bob = executor.submit(search, "bob", "hybrid")
        a_log, a_hits = alice.result()
        b_log, b_hits = bob.result()
    assert a_log["scope"] == "alice" and a_log["mode"] == "vector"
    assert b_log["scope"] == "bob" and b_log["mode"] == "hybrid"
    assert a_hits[0].chunk.scope == "alice" and b_hits[0].chunk.scope == "bob"
    assert "last_search" not in retriever.health()


@pytest.mark.parametrize("backend,module", [("chroma", "chromadb"), ("faiss", "faiss")])
def test_optional_real_vector_adapter_persistence_filter_update_delete(tmp_path, backend, module):
    pytest.importorskip(module)
    config = settings(tmp_path)
    config.vector_backend = backend
    store, retriever = Storage(tmp_path), Retriever(config, Storage(tmp_path))
    a = add_document(store, retriever, "a", "ImageNet Transformer accuracy 88.2.")
    b = add_document(store, retriever, "b", "Wheat plants use an agricultural dataset.")
    add_document(store, retriever, "a", "ImageNet SECRET private 100 percent.", scope="bob")
    assert retriever.search("ImageNet", "alice", mode="vector")[0].chunk.document_id == "a"
    filtered = retriever.search("ImageNet", "alice", document_ids=["b"], mode="vector")
    assert {hit.chunk.chunk_id for hit in filtered} == {c.chunk_id for c in b}
    assert retriever.vectors.existing(a)[("alice", a[0].chunk_id)]
    restarted = Retriever(config, Storage(tmp_path))
    assert restarted.search("ImageNet", "alice", mode="vector")[0].chunk.chunk_id == a[0].chunk_id
    doc = store.get_document("a", "alice")
    replacement = split_blocks(doc, [SourceBlock(text="CIFAR diffusion FID 2.0.")], size=64, overlap=8)
    restarted.index(replacement)
    store.replace_chunks("a", "alice", replacement)
    store.bump_version("alice")
    updated = restarted.search("ImageNet", "alice", document_ids=["a"], mode="vector")
    assert {h.chunk.chunk_id for h in updated} == {c.chunk_id for c in replacement}
    restarted.delete("a", "alice")
    store.delete_document("a", "alice")
    assert restarted.search("CIFAR", "alice", document_ids=["a"], mode="vector") == []
    assert restarted.search("SECRET", "bob", mode="vector")[0].chunk.scope == "bob"
    assert restarted.health()["vectors"]["backend"] == backend
    assert restarted.health()["vectors"]["ready"] is True
