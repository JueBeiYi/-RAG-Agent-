import json
import time

from fastapi.testclient import TestClient
from src.api.main import create_app
from src.core.config import Settings


def settings_at(path):
    return Settings(data_dir=str(path), embedding_backend="hash", vector_backend="sqlite",
                    reranker_backend="lexical", llm_backend="extractive")


def wait_job(client, job_id, headers):
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError("import job did not finish")


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_full_upload_chat_cache_delete_and_scope(tmp_path):
    headers = {"X-Workspace-Token": "workspace-alice-12345"}
    other = {"X-Workspace-Token": "workspace-bob-678901"}
    with TestClient(create_app(settings_at(tmp_path))) as client:
        assert client.get("/health").json()["demo"] is True
        assert client.get("/documents").status_code == 401
        content = "# Vision Study\n作者：王明\n2024\n本研究采用 PlantVillage 数据集，使用 Transformer 图像分类方法，实验准确率 95%。\n"
        result = client.post("/documents", headers=headers, files=[("files", ("vision.md", content.encode(), "text/markdown"))])
        assert result.status_code == 200, result.text
        document_id = result.json()["documents"][0]["document_id"]
        job = wait_job(client, result.json()["jobs"][0]["job_id"], headers)
        assert job["status"] == "done", job
        assert client.get("/documents", headers=other).json() == []
        assert client.get(f"/documents/{document_id}/file", headers=other).status_code == 404
        duplicate = client.post("/documents", headers=headers, files=[("files", ("again.md", content.encode(), "text/plain"))]).json()
        assert duplicate["documents"][0]["duplicate"]
        session = client.post("/sessions", headers=headers, json={"title": "研究"}).json()
        assert "id" in session, session
        request = {"message": "PlantVillage 数据集对应的方法是什么？", "document_ids": [document_id], "agent": False}
        stream = sse(client.post(f"/sessions/{session['id']}/chat", headers=headers, json=request))
        final = next((e for e in stream if e["type"] == "done"), None)
        assert final is not None, stream
        assert final["citations"][0]["document_id"] == document_id
        assert "PlantVillage" in final["answer"]
        assert any(e["type"] == "answer_delta" for e in stream)
        assert client.get(f"/sessions/{session['id']}", headers=other).status_code == 404
        assert client.get("/runs", headers=headers).json()[0]["status"] == "done"
        invalid = client.post(f"/sessions/{session['id']}/chat", headers=headers, json={**request, "document_ids": ["missing"]})
        assert any(e["type"] == "error" for e in sse(invalid))
        assert client.delete(f"/documents/{document_id}", headers=headers).status_code == 200
        assert client.get("/documents", headers=headers).json() == []
        assert client.app.state.service.store.list("cache", headers["X-Workspace-Token"]) == []


def test_agent_calculator_and_empty_knowledge_base(tmp_path):
    headers = {"X-Workspace-Token": "workspace-test-12345"}
    with TestClient(create_app(settings_at(tmp_path))) as client:
        session = client.post("/sessions", headers=headers, json={}).json()
        events = sse(client.post(f"/sessions/{session['id']}/chat", headers=headers,
                                json={"message": "3.14*2.56", "agent": True}))
        final = next((e for e in events if e["type"] == "done"), None)
        assert final is not None, events
        assert "8.0384" in final["answer"]
        assert any(e["type"] == "tool_end" for e in events)
        events = sse(client.post(f"/sessions/{session['id']}/chat", headers=headers,
                                json={"message": "某论文的精确实验数值是什么？", "agent": False}))
        final = next(e for e in events if e["type"] == "done")
        assert final["citations"] == []
        assert "未找到" in final["answer"]
