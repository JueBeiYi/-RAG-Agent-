import json

from fastapi.testclient import TestClient
from src.api.main import create_app
from src.core.config import Settings


def test_retry_preserves_original_rag_and_retrieval_selection(tmp_path):
    settings = Settings(data_dir=str(tmp_path), embedding_backend="hash", vector_backend="sqlite",
                        reranker_backend="lexical", llm_backend="extractive")
    headers = {"X-Workspace-Token": "retry-private-workspace-20261009"}
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", headers=headers, json={"title": "Retry modes"}).json()
        options = {"message": "知识库的研究方法是什么？", "agent": False, "retrieval_mode": "vector", "document_ids": []}
        client.post(f"/sessions/{session['id']}/chat", headers=headers, json=options)
        response = client.post(f"/sessions/{session['id']}/retry-last", headers=headers)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert any(e["type"] == "done" for e in events)
        assert not any(e["type"] in {"tool_start", "tool_end"} for e in events)
        messages = client.get(f"/sessions/{session['id']}", headers=headers).json()["messages"]
        users = [m for m in messages if m["role"] == "user"]
        assert users[-1]["agent"] is False and users[-1]["retrieval_mode"] == "vector"
