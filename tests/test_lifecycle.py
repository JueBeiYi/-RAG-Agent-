from concurrent.futures import ThreadPoolExecutor

from src.core.config import Settings
from src.core.schemas import ChatRequest
from src.services.application import Application


def settings(path):
    return Settings(data_dir=str(path), embedding_backend="hash", vector_backend="sqlite",
                    reranker_backend="lexical", llm_backend="extractive")


def test_explicit_empty_selection_clears_previous_scope_and_log_failure_unlocks(tmp_path):
    app = Application(settings(tmp_path))
    try:
        session = app.store.create_session("u")
        session["document_ids"] = ["previously-deleted"]
        app.store.put("sessions", session["id"], "u", session)
        app.logger.write = lambda *args: (_ for _ in ()).throw(OSError("log unwritable"))
        events = list(app.chat(session["id"], "u", ChatRequest(message="1+2", document_ids=[])))
        assert events[-1]["type"] == "done"
        assert not app._session_lock(session["id"], "u").locked()
        assert app.store.get("sessions", session["id"], "u")["document_ids"] == []
        assert list(app.chat(session["id"], "u", ChatRequest(message="2+3", document_ids=[])))[-1]["type"] == "done"
    finally:
        app.close()


def test_upload_concurrency_deduplicates_and_profile_change_requires_reindex(tmp_path):
    app = Application(settings(tmp_path))
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: app.upload("paper.md", b"# Paper\nTransformer method with unique evidence.", "u"), range(2)))
        assert len(app.store.list_documents("u")) == 1
        assert sum(r["duplicate"] for r in results) == 1
        app.worker.shutdown(wait=True)
        doc = app.store.list_documents("u")[0]
        assert doc.status == "ready"
    finally:
        app.close()
    modified = settings(tmp_path)
    modified.chunk_size = 256
    app = Application(modified)
    try:
        doc = app.store.list_documents("u")[0]
        assert doc.status == "error" and "配置已变更" in doc.error
        app.retry(doc.document_id, "u")
        app.worker.shutdown(wait=True)
        assert app.store.get_document(doc.document_id, "u").status == "ready"
    finally:
        app.close()
