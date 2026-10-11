import pytest
from src.core.config import Settings, load_settings
from src.core.schemas import Document, Chunk
from src.core.storage import Storage
from src.generation.context import audit_citations, build_context


def test_storage_scope_and_cache_invalidation(tmp_path):
    store = Storage(tmp_path)
    doc = Document(document_id="a", scope="alice", name="a.md", content_hash="abc", format="md")
    store.put_document(doc)
    store.replace_chunks("a", "alice", [Chunk(chunk_id="c", document_id="a", scope="alice", text="fact")])
    store.put("cache", "x", "alice", {"answer": "old"})
    assert store.get_document("a", "bob") is None
    assert not store.list_chunks("bob")
    assert store.list_chunks("alice")[0].text == "fact"
    first = store.index_version("alice")
    store.delete_document("a", "alice")
    assert store.index_version("alice") != first
    assert not store.list("cache", "alice")
    assert not store.list_chunks("alice")


def test_session_memory_isolation(tmp_path):
    store = Storage(tmp_path)
    session = store.create_session("alice")
    store.add_message(session["id"], "alice", "user", "private")
    store.set_memory(session["id"], "alice", {"summary": "private"})
    assert not store.list_messages(session["id"], "bob")
    assert not store.get_memory(session["id"], "bob")


def test_citation_existence_and_numeric_review():
    audit = audit_citations("准确率95%。依据 [S99]。另一个事实 [S1]。", [{"citation_id": "S1"}])
    assert audit["invalid_ids"] == ["S99"]
    assert audit["needs_review"]


def test_offline_config_rejects_cloud(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("offline: true\nllm_backend: cloud\n", encoding="utf-8")
    monkeypatch.delenv("RAG_PROFILE", raising=False)
    with pytest.raises(ValueError, match="离线"):
        load_settings(config)
