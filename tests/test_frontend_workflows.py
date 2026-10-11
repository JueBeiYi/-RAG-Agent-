"""Real HTTP + Streamlit AppTest regression of complete user workflows.

Every test uses an unpredictable private workspace token. With no explicit
RAG_UI_TEST_API, a dedicated ephemeral-port demo API is started under
data/runtime/ui-tests. Existing services and real research data are untouched.
AppTest does not emulate an OS file chooser/download manager: fixture uploads
and export payload validation use the same real HTTP endpoints as the UI.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import fitz
import httpx
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "frontend" / "app.py"


@dataclass
class Backend:
    url: str
    owned: bool
    service: object | None = None


@pytest.fixture(scope="module")
def backend():
    external = os.getenv("RAG_UI_TEST_API", "").strip()
    if external:
        parsed = urlparse(external)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("UI workflow tests only allow an explicitly chosen local demo API")
        response = httpx.get(external.rstrip("/") + "/health", timeout=5, trust_env=False)
        response.raise_for_status()
        assert response.json().get("demo") is True, "Use a dedicated demo API, not a real research/model service"
        yield Backend(external.rstrip("/"), False)
        return
    from src.api.main import create_app
    from src.core.config import Settings
    runtime = ROOT / "data" / "runtime" / "ui-tests" / ("api-" + uuid.uuid4().hex)
    runtime.mkdir(parents=True, exist_ok=False)
    settings = Settings(data_dir=str(runtime), embedding_backend="hash", vector_backend="sqlite",
                        reranker_backend="lexical", llm_backend="extractive", chunk_size=512,
                        chunk_overlap=64, tool_timeout_seconds=10)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    application = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True, name="ui-test-api")
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "Dedicated UI test API did not start"
    try:
        yield Backend(url, True, application.state.service)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@dataclass
class Workspace:
    backend: Backend
    token: str
    client: httpx.Client
    documents: list[dict]
    failed_document: dict | None = None

    def get(self, path, **kwargs):
        response = self.client.get(path, **kwargs)
        response.raise_for_status()
        return response.json()

    def app(self):
        at = AppTest.from_file(str(APP), default_timeout=45)
        at.session_state["api_url"] = self.backend.url
        at.session_state["workspace_token"] = self.token
        at.run()
        no_exception(at)
        return at


@pytest.fixture
def workspace(backend):
    token = "ui-test-" + uuid.uuid4().hex
    with httpx.Client(base_url=backend.url, headers={"X-Workspace-Token": token}, timeout=30, trust_env=False) as client:
        ws = Workspace(backend, token, client, [])
        assert ws.get("/documents") == []
        try:
            yield ws
        finally:
            # Enumerating and deleting uses only the token created in this fixture.
            # No global, user, imported corpus, or unrelated workspace is touched.
            for table in ("runs", "sessions", "experiments", "documents"):
                response = client.get("/" + table)
                if not response.is_success:
                    continue
                for row in response.json():
                    key = row.get("id") or row.get("document_id")
                    if key:
                        client.delete(f"/{table}/{key}")


def pdf_bytes(title: str, dataset: str, result: int) -> bytes:
    document = fitz.open()
    for page_number in (1, 2):
        page = document.new_page(width=595, height=842)
        if page_number == 1:
            text = (title + "\nBackground: We study the vision classification problem.\n"
                    "Method: We propose a Transformer image classification method.\n"
                    f"Dataset: We evaluate on {dataset}.\n" + "Evidence context. " * 8)
        else:
            text = (f"Results: The experimental accuracy is {result} percent.\n"
                    "Conclusion: We conclude that this method supports vision classification.\n"
                    "Limitations: Small training data restricts generalization.\n" + "Evaluation details. " * 8)
        page.insert_textbox(fitz.Rect(60, 60, 530, 730), text, fontsize=12)
    content = document.tobytes()
    document.close()
    return content


def wait_job(ws, job_id):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        job = ws.get(f"/jobs/{job_id}")
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.03)
    raise AssertionError(f"Import/reindex job {job_id} did not complete")


def wait_documents(ws):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        docs = ws.get("/documents")
        if all(doc["status"] not in {"queued", "processing"} for doc in docs):
            return docs
        time.sleep(0.03)
    raise AssertionError("Document processing did not finish")


@pytest.fixture
def papers(workspace):
    for title, dataset, accuracy in (("UI Alpha Study", "AlphaSet", 90), ("UI Beta Study", "BetaSet", 85)):
        response = workspace.client.post("/documents", files=[("files", (title + ".pdf", pdf_bytes(title, dataset, accuracy), "application/pdf"))])
        response.raise_for_status()
        body = response.json()
        assert wait_job(workspace, body["jobs"][0]["job_id"])["status"] == "done"
        doc = body["documents"][0]
        metadata = workspace.client.post(f"/documents/{doc['document_id']}/metadata",
                                         json={"title": title, "authors": ["Test Scientist"], "year": 2024})
        metadata.raise_for_status()
        workspace.documents.append(metadata.json())
    failed = workspace.client.post("/documents", files=[("files", ("UI broken document.pdf", b"not a PDF fixture", "application/pdf"))])
    failed.raise_for_status()
    body = failed.json()
    assert wait_job(workspace, body["jobs"][0]["job_id"])["status"] == "error"
    workspace.failed_document = body["documents"][0]
    return workspace


def no_exception(at):
    assert not at.exception, "\n".join(str(item.message) for item in at.exception)


def widget(at, kind: str, key: str, *alternatives: str):
    items = getattr(at, kind)
    for candidate in (key, *alternatives):
        for item in items:
            if item.key == candidate:
                return item
    if key == "compare_history":
        for item in items:
            if item.key and item.key.startswith("compare_history_"):
                return item
    raise AssertionError(f"Missing {kind} key {key}; available={[(item.key, item.label) for item in items]}")


def click(at, key, *alternatives):
    widget(at, "button", key, *alternatives).click().run()
    no_exception(at)
    return at


def navigate(at, label):
    widget(at, "radio", "nav_page").set_value(label).run()
    no_exception(at)
    return at


def texts(at):
    return "\n".join(str(item.value) for kind in ("markdown", "caption", "info", "warning", "error", "success")
                     for item in getattr(at, kind))


def trace_visible(at):
    return any("过程" in item.label or "轨迹" in item.label for item in at.expander)


def connect_workspace(at, url, token):
    click(at, "workspace_settings")
    widget(at, "text_input", "workspace_api_input").set_value(url)
    widget(at, "text_input", "workspace_token_input").set_value(token)
    click(at, "workspace_apply")
    return at


def chunk_rows(payload):
    return payload.get("chunks", payload.get("items", [])) if isinstance(payload, dict) else payload


def evaluation_case(ws, *, reviewed=True, case_id="ui-case-1", split="dev"):
    return {"case_id": case_id, "question": "What method does UI Alpha Study use?", "category": "factual",
            "split": split, "reference_answer": "A Transformer image classification method.",
            "reviewed": reviewed, "relevant_sources": [{"document_id": ws.documents[0]["document_id"], "page": 1,
            "quote": "We propose a Transformer image classification method."}]}


def wait_experiments(ws, count):
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        experiments = ws.get("/experiments")
        if len(experiments) >= count and all(exp["status"] in {"done", "error"} for exp in experiments):
            return experiments
        time.sleep(0.05)
    raise AssertionError("Experiment jobs did not finish")


def download_controls(at):
    return at.get("download_button")


def assert_downloads(at, minimum=1):
    assert len(download_controls(at)) >= minimum, "Direct download controls must remain after reruns"


def assert_png(response):
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def current_session(at):
    return widget(at, "selectbox", "session_selector").value


def confirm_delete(at):
    # Every destructive action is restricted to this test's random workspace.
    for item in at.checkbox:
        if item.key in {"confirm_delete_ack", "delete_ack", "confirm_delete_checkbox"}:
            item.set_value(True)
    click(at, "confirm_delete")


def test_session_create_rename_search_switch_exports_and_delete(workspace):
    at = workspace.app()
    click(at, "new_session")
    first = current_session(at)
    click(at, "new_session")
    second = current_session(at)
    assert first != second
    widget(at, "text_input", "rename_title", f"rename_title_{second}").set_value("UI renamed session")
    click(at, "rename_session")
    assert workspace.get(f"/sessions/{second}")["title"] == "UI renamed session"
    widget(at, "text_input", "session_search").set_value("renamed").run()
    no_exception(at)
    assert current_session(at) == second
    widget(at, "text_input", "session_search").set_value("").run()
    widget(at, "selectbox", "session_selector").set_value(first).run()
    no_exception(at)
    assert current_session(at) == first
    widget(at, "selectbox", "session_selector").set_value(second).run()
    assert_downloads(at, 2)
    for fmt in ("json", "md"):
        exported = workspace.client.get(f"/sessions/{second}/export", params={"format": fmt})
        exported.raise_for_status()
        assert "UI renamed session" in exported.text
    click(at, "delete_session")
    confirm_delete(at)
    assert workspace.client.get(f"/sessions/{second}").status_code == 404
    assert current_session(at) != second


def test_documents_metadata_chunks_preview_persistence_reindex_retry_download_and_delete(papers):
    ws = papers
    at = navigate(ws.app(), "论文库")
    document_id = ws.documents[0]["document_id"]
    widget(at, "selectbox", "doc_detail_selector").set_value(document_id).run()
    no_exception(at)
    widget(at, "text_input", f"meta_title_{document_id}").set_value("UI Alpha corrected title")
    widget(at, "text_area", f"meta_authors_{document_id}").set_value("Alice Test\nBob Test")
    click(at, f"meta_save_{document_id}")
    saved = ws.get(f"/documents/{document_id}")
    assert saved["title"] == "UI Alpha corrected title"
    assert set(saved["authors"]) == {"Alice Test", "Bob Test"}
    widget(at, "text_input", "document_search").set_value("corrected").run()
    no_exception(at)
    assert widget(at, "selectbox", "doc_detail_selector").value == document_id
    widget(at, "text_input", "document_search").set_value("").run()
    widget(at, "text_input", f"meta_year_{document_id}").set_value("")
    click(at, f"meta_save_{document_id}")
    assert ws.get(f"/documents/{document_id}")["year"] is None, "Blank year must clear an incorrect previously saved year"
    chunks = ws.get(f"/documents/{document_id}/chunks")
    rows = chunk_rows(chunks)
    assert rows and all(row["document_id"] == document_id for row in rows)
    widget(at, "selectbox", f"doc_chunk_{document_id}").set_value(rows[0]["chunk_id"]).run()
    no_exception(at)
    widget(at, "number_input", f"doc_page_{document_id}").set_value(2).run()
    click(at, f"doc_preview_{document_id}")
    assert at.image, "A PDF page should be visible after requesting preview"
    at.run()
    no_exception(at)
    assert at.image, "A harmless rerun must not discard the prepared preview"
    assert_png(ws.client.get(f"/documents/{document_id}/preview", params={"page": 2}))
    assert_downloads(at)
    widget(at, "download_button", f"doc_download_{document_id}").click().run()
    no_exception(at)
    assert at.image and widget(at, "selectbox", "doc_detail_selector").value == document_id
    original = ws.client.get(f"/documents/{document_id}/file")
    original.raise_for_status()
    assert original.content.startswith(b"%PDF")
    old_versions = {row.get("index_version") for row in rows}
    click(at, f"doc_reindex_{document_id}")
    wait_documents(ws)
    at.run()
    no_exception(at)
    fresh = ws.get(f"/documents/{document_id}/chunks")
    fresh_rows = chunk_rows(fresh)
    assert {row.get("index_version") for row in fresh_rows} != old_versions
    assert ws.get(f"/documents/{document_id}")["title"] == "UI Alpha corrected title"
    failed_id = ws.failed_document["document_id"]
    widget(at, "selectbox", "doc_detail_selector").set_value(failed_id).run()
    click(at, f"doc_retry_{failed_id}")
    wait_documents(ws)
    at.run()
    no_exception(at)
    assert ws.get(f"/documents/{failed_id}")["status"] == "error"
    assert at.error, "A repeated parse failure must remain visible instead of masquerading as complete"
    click(at, f"doc_delete_{failed_id}")
    confirm_delete(at)
    assert ws.client.get(f"/documents/{failed_id}").status_code == 404


def test_question_document_selection_evidence_history_navigation_and_token_isolation(papers):
    ws = papers
    at = ws.app()
    document_id = ws.documents[0]["document_id"]
    widget(at, "multiselect", "doc_selection").set_value([document_id]).run()
    at.chat_input[0].set_value("Explain the Transformer method and cite its source.").run()
    no_exception(at)
    session_id = current_session(at)
    session = ws.get(f"/sessions/{session_id}")
    answer = next(message for message in reversed(session["messages"]) if message["role"] == "assistant")
    assert answer["citations"] and {citation["document_id"] for citation in answer["citations"]} == {document_id}
    assert trace_visible(at)
    assert any("knowledge_qa" in str(item.value) for item in at.json)
    request_id = answer["request_id"]
    click(at, f"citation_view_{request_id}_0")
    assert at.image, "Opening a real citation should display its original PDF page"
    assert_downloads(at)
    navigate(at, "论文库")
    navigate(at, "科研问答")
    assert widget(at, "multiselect", "doc_selection").value == [document_id]
    assert trace_visible(at), "Historical answer trace must survive leaving and returning to chat"
    assert any("knowledge_qa" in str(item.value) for item in at.json), "Restored trace must contain actual recorded tool events"
    widget(at, "multiselect", "doc_selection").set_value([]).run()
    at.chat_input[0].set_value("计算 `1+2`").run()
    no_exception(at)
    assert ws.get(f"/sessions/{session_id}")["document_ids"] == []
    alternate = "ui-test-" + uuid.uuid4().hex
    connect_workspace(at, ws.backend.url, alternate)
    no_exception(at)
    assert widget(at, "multiselect", "doc_selection").value == []
    assert widget(at, "multiselect", "doc_selection").options == []
    assert current_session(at) != session_id
    assert "UI Alpha Study" not in texts(at)
    other_headers = {"X-Workspace-Token": alternate}
    assert httpx.get(ws.backend.url + f"/sessions/{session_id}", headers=other_headers, trust_env=False).status_code == 404
    # Remove only auto-created sessions in the second random token we own.
    with httpx.Client(base_url=ws.backend.url, headers=other_headers, trust_env=False) as other:
        for own_session in other.get("/sessions").json():
            other.delete(f"/sessions/{own_session['id']}")
    connect_workspace(at, ws.backend.url, ws.token)
    no_exception(at)
    assert current_session(at) in {session["id"] for session in ws.get("/sessions")}


def test_comparison_persists_across_pages_and_exports_both_paper_evidences(papers):
    ws = papers
    at = navigate(ws.app(), "论文对比")
    ids = [doc["document_id"] for doc in ws.documents]
    widget(at, "selectbox", "compare_a").set_value(ids[0])
    widget(at, "selectbox", "compare_b").set_value(ids[1])
    click(at, "compare_generate")
    comparison_id = widget(at, "selectbox", "compare_history").value
    record = ws.get(f"/sessions/{comparison_id}")
    answer = next(message for message in reversed(record["messages"]) if message["role"] == "assistant")
    assert {citation["document_id"] for citation in answer["citations"]} == set(ids)
    navigate(at, "论文库")
    navigate(at, "论文对比")
    assert widget(at, "selectbox", "compare_history").value == comparison_id
    assert "UI Alpha Study" in texts(at) and "UI Beta Study" in texts(at)
    assert_downloads(at, 2)
    for fmt in ("json", "md"):
        exported = ws.client.get(f"/sessions/{comparison_id}/export", params={"format": fmt})
        exported.raise_for_status()
        assert "UI Alpha Study" in exported.text and "UI Beta Study" in exported.text


def test_new_browser_session_restores_server_document_scope(papers):
    ws = papers
    first = ws.app()
    click(first, "new_session")
    session_id = current_session(first)
    document_id = ws.documents[0]["document_id"]
    widget(first, "multiselect", "doc_selection").set_value([document_id]).run()
    assert ws.get(f"/sessions/{session_id}")["document_ids"] == [document_id]
    # A fresh AppTest has no previous widget/session state, like a browser refresh.
    restored = ws.app()
    assert current_session(restored) == session_id
    assert widget(restored, "multiselect", "doc_selection").value == [document_id]


def test_experiments_reject_unreviewed_data_run_three_modes_details_and_exports(papers):
    ws = papers
    at = navigate(ws.app(), "实验与运行记录")
    widget(at, "radio", "experiment_input_mode").set_value("粘贴 JSONL").run()
    case = evaluation_case(ws, reviewed=False)
    widget(at, "text_area", "experiment_dataset_text").set_value(json.dumps(case)).run()
    no_exception(at)
    click(at, "experiment_run_all")
    assert ws.get("/experiments") == []
    assert "reviewed" in texts(at) or "核对" in texts(at)
    case["reviewed"] = True
    widget(at, "text_area", "experiment_dataset_text").set_value(json.dumps(case)).run()
    no_exception(at)
    click(at, "experiment_run_all")
    experiments = wait_experiments(ws, 3)
    assert len(experiments) == 3 and all(exp["status"] == "done" for exp in experiments), experiments
    assert {exp["mode"] for exp in experiments} == {"vector", "hybrid", "hybrid_rerank"}
    at.run()
    no_exception(at)
    ids = [exp["id"] for exp in experiments]
    widget(at, "multiselect", "experiment_selection").set_value(ids).run()
    no_exception(at)
    assert at.dataframe
    assert at.get("arrow_vega_lite_chart") or at.get("vega_lite_chart"), "Comparable experiments should produce the method comparison chart"
    assert "功能" in texts(at), "Demo evaluation must not be presented as formal model-quality evidence"
    assert_downloads(at, 3)
    detail_id = ids[0]
    widget(at, "selectbox", "experiment_detail").set_value(detail_id).run()
    no_exception(at)
    widget(at, "selectbox", "experiment_case_selector").set_value("ui-case-1").run()
    no_exception(at)
    assert "A Transformer image classification method." in texts(at)
    for experiment_id in ids:
        detail = ws.get(f"/experiments/{experiment_id}")
        assert detail["result"]["records"][0]["case_id"] == "ui-case-1"
        for fmt in ("json", "csv", "xlsx"):
            response = ws.client.get(f"/experiments/{experiment_id}/export", params={"format": fmt})
            response.raise_for_status()
            if fmt == "xlsx":
                assert response.content.startswith(b"PK")
            else:
                assert "ui-case-1" in response.text
    click(at, "delete_experiment")
    confirm_delete(at)
    assert ws.client.get(f"/experiments/{detail_id}").status_code == 404
    surviving = [identifier for identifier in ids if identifier != detail_id][0]
    widget(at, "selectbox", "experiment_detail").set_value(surviving).run()
    click(at, f"case_source_{surviving}_ui-case-1_1")
    assert at.image, "Single-case evidence inspection should open the real original PDF"


def test_run_replay_export_delete_and_service_offline_recovery(workspace):
    ws = workspace
    at = ws.app()
    at.chat_input[0].set_value("计算 `3.14*2.56`").run()
    no_exception(at)
    runs = ws.get("/runs")
    assert len(runs) == 1
    run_id = runs[0]["id"]
    navigate(at, "实验与运行记录")
    widget(at, "selectbox", "run_selector").set_value(run_id).run()
    no_exception(at)
    assert "8.0384" in texts(at)
    assert trace_visible(at)
    assert any("calculator" in str(item.value) for item in at.json)
    assert_downloads(at)
    exported = ws.client.get(f"/runs/{run_id}/export", params={"format": "json"})
    exported.raise_for_status()
    assert any(event["type"] == "tool_end" for event in exported.json()["events"])
    click(at, "run_open_session")
    assert current_session(at) == runs[0]["session_id"]
    navigate(at, "实验与运行记录")
    widget(at, "text_input", "run_search").set_value("3.14").run()
    widget(at, "selectbox", "run_status_filter").set_value("已完成").run()
    no_exception(at)
    assert widget(at, "selectbox", "run_selector").value == run_id
    click(at, "delete_run", f"run_delete_{run_id}")
    confirm_delete(at)
    assert ws.client.get(f"/runs/{run_id}").status_code == 404
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    offline_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    # Keep the socket bound, without listening, so the test address cannot be reused.
    try:
        connect_workspace(at, offline_url, ws.token)
        no_exception(at)
        assert "未连接" in texts(at) or "无法连接" in texts(at)
        connect_workspace(at, ws.backend.url, ws.token)
        no_exception(at)
        assert "API 已连接" in texts(at) or "已连接" in texts(at)
    finally:
        sock.close()


def test_private_demo_http_contract_supports_ui_actions_without_user_data(papers):
    """Separately verify upload/export/preview payloads AppTest cannot download."""
    ws = papers
    document_id = ws.documents[0]["document_id"]
    detail = ws.get(f"/documents/{document_id}")
    assert detail["metadata"]["page_count"] == 2
    rows = chunk_rows(ws.get(f"/documents/{document_id}/chunks"))
    assert rows
    assert_png(ws.client.get(f"/documents/{document_id}/preview", params={"page": 2, "chunk_id": rows[0]["chunk_id"]}))
    assert ws.client.get(f"/documents/{document_id}/preview", params={"page": 3}).status_code == 400
    broken_id = ws.failed_document["document_id"]
    broken_detail = ws.client.get(f"/documents/{broken_id}")
    assert broken_detail.status_code == 200
    assert broken_detail.json()["metadata"]["preview_available"] is False
    assert ws.client.get(f"/documents/{broken_id}/preview").status_code == 400
    cleared = ws.client.post(f"/documents/{document_id}/metadata", json={"year": None})
    cleared.raise_for_status()
    assert cleared.json()["year"] is None
    created = ws.client.post("/sessions", json={"title": "UI API contract"})
    created.raise_for_status()
    session_id = created.json()["id"]
    patched = ws.client.patch(f"/sessions/{session_id}", json={"title": "UI API renamed", "document_ids": [document_id]})
    patched.raise_for_status()
    assert patched.json()["title"] == "UI API renamed"
    events_response = ws.client.post(f"/sessions/{session_id}/chat", json={"message": "计算 `1+2`", "document_ids": [], "agent": True})
    events_response.raise_for_status()
    events = [json.loads(line[6:]) for line in events_response.text.splitlines() if line.startswith("data: ")]
    final = next(item for item in events if item["type"] == "done")
    assert "**3**" in final["answer"]
    assert ws.get(f"/sessions/{session_id}")["document_ids"] == []
    assert ws.get(f"/sessions/{session_id}")["title"] == "UI API renamed", "First question must preserve a manually edited title"
    run_id = ws.get("/runs")[0]["id"]
    for table, identifier in (("sessions", session_id), ("runs", run_id)):
        for fmt in ("json", "md"):
            exported = ws.client.get(f"/{table}/{identifier}/export", params={"format": fmt})
            exported.raise_for_status()
            assert "attachment" in exported.headers["content-disposition"]
            assert "3" in exported.text
    other = "ui-test-" + uuid.uuid4().hex
    headers = {"X-Workspace-Token": other}
    assert httpx.get(ws.backend.url + f"/documents/{document_id}/file", headers=headers, trust_env=False).status_code == 404
    unreviewed = evaluation_case(ws, reviewed=False)
    rejected = ws.client.post("/experiments", files={"file": ("unreviewed.jsonl", json.dumps(unreviewed), "application/jsonl")}, data={"mode": "vector", "split": "dev"})
    assert rejected.status_code == 400 and "reviewed" in rejected.text
    assert ws.get("/experiments") == []
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        result = ws.client.post("/experiments", files={"file": ("reviewed.jsonl", json.dumps(evaluation_case(ws)), "application/jsonl")}, data={"mode": mode, "split": "dev"})
        result.raise_for_status()
    experiments = wait_experiments(ws, 3)
    assert all(experiment["status"] == "done" for experiment in experiments)
    for experiment in experiments:
        identifier = experiment["id"]
        for fmt in ("json", "csv", "xlsx"):
            exported = ws.client.get(f"/experiments/{identifier}/export", params={"format": fmt})
            exported.raise_for_status()
            assert (exported.content.startswith(b"PK") if fmt == "xlsx" else "ui-case-1" in exported.text)
        assert ws.client.delete(f"/experiments/{identifier}").status_code == 200
    assert ws.client.delete(f"/runs/{run_id}").status_code == 200
    assert ws.client.delete(f"/sessions/{session_id}").status_code == 200


def test_private_http_cancellation_is_scoped_and_releases_session_lock(workspace):
    """A controlled pause verifies real HTTP cancellation without timing races."""
    ws = workspace
    if not ws.backend.owned:
        pytest.skip("A bounded controlled-stream pause requires the automatically owned test API; no hooks are installed in an existing server")
    service = ws.backend.service
    original = service.rag.stream
    entered, release = threading.Event(), threading.Event()
    def paused_stream(*args, **kwargs):
        yield {"type": "status", "text": "Controlled UI regression pause"}
        entered.set()
        release.wait(timeout=5)
        yield {"type": "answer_delta", "text": "This output must not appear after cancellation"}
        yield {"type": "done", "answer": "unexpected", "citations": [], "metrics": {}, "origin": "demo_extractive"}
    session = ws.client.post("/sessions", json={"title": "UI cancellation fixture"}).json()
    service.rag.stream = paused_stream
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pending = pool.submit(ws.client.post, f"/sessions/{session['id']}/chat",
                              json={"message": "Controlled cancellation test", "agent": False, "document_ids": []})
        assert entered.wait(timeout=5)
        active = ws.get(f"/sessions/{session['id']}")["active_request"]
        assert active and active["status"] == "running"
        other_headers = {"X-Workspace-Token": "ui-test-" + uuid.uuid4().hex}
        assert httpx.post(ws.backend.url + f"/runs/{active['id']}/cancel", headers=other_headers, trust_env=False).status_code == 404
        assert httpx.get(ws.backend.url + f"/runs/{active['id']}", headers=other_headers, trust_env=False).status_code == 404
        own_cancel = ws.client.post(f"/runs/{active['id']}/cancel")
        own_cancel.raise_for_status()
        assert own_cancel.json()["status"] == "cancelling"
        release.set()
        response = pending.result(timeout=10)
        response.raise_for_status()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert any(item["type"] == "cancelled" for item in events)
        assert not any(item["type"] == "answer_delta" for item in events)
        assert ws.get(f"/runs/{active['id']}")["status"] == "cancelled"
        assert ws.get(f"/sessions/{session['id']}")["active_request"] is None
        assert not service._session_lock(session["id"], ws.token).locked()
        service.rag.stream = original
        followup = ws.client.post(f"/sessions/{session['id']}/chat", json={"message": "计算 `2+2`", "agent": True, "document_ids": []})
        followup.raise_for_status()
        assert '"type": "done"' in followup.text and "**4**" in followup.text
    finally:
        release.set()
        service.rag.stream = original
        pool.shutdown(wait=True)


def test_ui_can_cancel_active_request_then_retry_without_duplicate_question(workspace):
    ws = workspace
    if not ws.backend.owned:
        pytest.skip("This UI timing regression uses only the automatically owned test API")
    service = ws.backend.service
    original = service.rag.stream
    entered, release = threading.Event(), threading.Event()
    def paused_stream(*args, **kwargs):
        yield {"type": "status", "text": "Controlled bounded test request"}
        entered.set()
        release.wait(timeout=15)
        yield {"type": "answer_delta", "text": "unexpected after cancellation"}
        yield {"type": "done", "answer": "unexpected", "citations": [], "metrics": {}}
    session = ws.client.post("/sessions", json={"title": "UI running request"}).json()
    service.rag.stream = paused_stream
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pending = pool.submit(ws.client.post, f"/sessions/{session['id']}/chat",
                              json={"message": "Explain a paper without evidence", "agent": False, "document_ids": []})
        assert entered.wait(timeout=5)
        at = ws.app()
        assert current_session(at) == session["id"]
        assert at.chat_input[0].disabled is True
        click(at, "chat_cancel_existing")
        assert ws.get(f"/sessions/{session['id']}")["active_request"]["status"] == "cancelling"
        release.set()
        assert '"type": "cancelled"' in pending.result(timeout=10).text
        service.rag.stream = original
        at.run()
        no_exception(at)
        assert at.chat_input[0].disabled is False
        click(at, "retry_last")
        messages = ws.get(f"/sessions/{session['id']}")["messages"]
        assert sum(message["role"] == "user" for message in messages) == 1
        assert messages[-1]["role"] == "assistant"
    finally:
        release.set()
        service.rag.stream = original
        pool.shutdown(wait=True)


@pytest.mark.parametrize("difference", ["dataset", "corpus", "split"])
def test_incomparable_experiments_warn_and_refuse_comparison_chart(papers, difference):
    ws = papers
    first = evaluation_case(ws)
    result = ws.client.post("/experiments", files={"file": ("baseline.jsonl", json.dumps(first), "application/jsonl")}, data={"mode": "vector", "split": "dev"})
    result.raise_for_status()
    wait_experiments(ws, 1)
    second = dict(first)
    requested_split = "dev"
    if difference == "dataset":
        second["case_id"] = "ui-different-case"
        second["question"] = "Which image method was proposed by UI Alpha Study?"
    elif difference == "split":
        second["split"] = requested_split = "test"
    else:
        uploaded = ws.client.post("/documents", files=[("files", ("UI new private corpus.md", b"# UI new private paper\nAn unrelated corpus change for the comparison regression.", "text/markdown"))])
        uploaded.raise_for_status()
        assert wait_job(ws, uploaded.json()["jobs"][0]["job_id"])["status"] == "done"
    result = ws.client.post("/experiments", files={"file": ("changed.jsonl", json.dumps(second), "application/jsonl")}, data={"mode": "hybrid", "split": requested_split})
    result.raise_for_status()
    experiments = wait_experiments(ws, 2)
    assert all(record["status"] == "done" for record in experiments)
    at = navigate(ws.app(), "实验与运行记录")
    widget(at, "multiselect", "experiment_selection").set_value([record["id"] for record in experiments]).run()
    no_exception(at)
    assert at.warning and any("不能" in str(item.value) or "不同" in str(item.value) or "不可" in str(item.value) for item in at.warning)
    assert not at.get("arrow_vega_lite_chart") and not at.get("vega_lite_chart"), "Incomparable scores must not be drawn as a method superiority chart"


def test_duplicate_paper_titles_remain_unambiguous_in_document_and_chat_selectors(papers):
    ws = papers
    ids = [document["document_id"] for document in ws.documents]
    for identifier in ids:
        response = ws.client.post(f"/documents/{identifier}/metadata", json={"title": "UI duplicate paper title"})
        response.raise_for_status()
    at = navigate(ws.app(), "论文库")
    selector = widget(at, "selectbox", "doc_detail_selector")
    assert len(selector.options) == len(set(selector.options))
    for identifier in ids:
        widget(at, "selectbox", "doc_detail_selector").set_value(identifier).run()
        no_exception(at)
        at.run()
        no_exception(at)
        assert widget(at, "selectbox", "doc_detail_selector").value == identifier
        assert widget(at, "text_input", f"meta_title_{identifier}").value == "UI duplicate paper title"
    navigate(at, "科研问答")
    selected = widget(at, "multiselect", "doc_selection")
    assert len(selected.options) == len(set(selected.options))
    selected.set_value(ids).run()
    no_exception(at)
    at.run()
    assert widget(at, "multiselect", "doc_selection").value == ids


def test_comparison_disables_identical_papers_or_empty_dimensions_without_creating_runs(papers):
    ws = papers
    at = navigate(ws.app(), "论文对比")
    ids = [document["document_id"] for document in ws.documents]
    widget(at, "selectbox", "compare_a").set_value(ids[0])
    widget(at, "selectbox", "compare_b").set_value(ids[0]).run()
    no_exception(at)
    assert widget(at, "button", "compare_generate").disabled
    widget(at, "selectbox", "compare_b").set_value(ids[1]).run()
    widget(at, "multiselect", "compare_aspects").set_value([]).run()
    no_exception(at)
    assert widget(at, "button", "compare_generate").disabled
    assert ws.get("/runs") == [] and ws.get("/sessions") == []
    widget(at, "multiselect", "compare_aspects").set_value(["方法与创新"]).run()
    no_exception(at)
    assert not widget(at, "button", "compare_generate").disabled


def test_chat_retrieval_and_agent_controls_change_real_execution(papers):
    ws = papers
    at = ws.app()
    widget(at, "multiselect", "doc_selection").set_value([ws.documents[0]["document_id"]]).run()
    widget(at, "selectbox", "chat_mode").set_value("vector")
    widget(at, "toggle", "chat_agent_enabled").set_value(False).run()
    at.chat_input[0].set_value("What Transformer method is described in the paper?").run()
    no_exception(at)
    session_id = current_session(at)
    messages = ws.get(f"/sessions/{session_id}")["messages"]
    first_request = messages[-1]["request_id"]
    run = ws.get(f"/runs/{first_request}")
    assert not any(event["type"] == "tool_start" for event in run["events"])
    assert run["metrics"]["retrieval_stages"]["mode"] == "vector"
    widget(at, "selectbox", "chat_mode").set_value("hybrid_rerank")
    widget(at, "toggle", "chat_agent_enabled").set_value(True).run()
    at.chat_input[0].set_value("计算 `2+2`").run()
    no_exception(at)
    messages = ws.get(f"/sessions/{session_id}")["messages"]
    second_request = messages[-1]["request_id"]
    second = ws.get(f"/runs/{second_request}")
    assert any(event["type"] == "tool_start" and event.get("tool") == "calculator" for event in second["events"])
    assert "**4**" in second["answer"]
