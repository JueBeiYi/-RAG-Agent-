"""Native widget-selection regressions with an isolated demo metadata API.

These tests do not send chat/generation requests or touch a user workspace.
The explicit protobuf value update is the browser sync contract being tested.
"""
import uuid
import json
import time

import pytest

from test_frontend_workflows import backend, workspace, papers, click, navigate, widget, no_exception, current_session, evaluation_case, wait_experiments


def assert_synced(selector, identifier):
    assert selector.value == identifier
    assert selector.proto.set_value is True
    assert identifier[:8] in str(selector.proto.raw_value)


def test_new_session_explicitly_updates_browser_value_and_stays_selected(workspace):
    at = workspace.app()
    click(at, "new_session")
    first = current_session(at)
    before = {record["id"] for record in workspace.get("/sessions")}
    click(at, "new_session")
    created = {record["id"] for record in workspace.get("/sessions")} - before
    assert len(created) == 1
    second = created.pop()
    assert second != first
    assert_synced(widget(at, "selectbox", "session_selector"), second)
    at.run()
    no_exception(at)
    assert_synced(widget(at, "selectbox", "session_selector"), second)
    assert at.session_state["active_session_id"] == second
    assert workspace.get(f"/sessions/{second}")["messages"] == []
    assert workspace.get("/runs") == []


def test_library_search_pushes_new_valid_document_value_to_browser(papers):
    ws = papers
    at = navigate(ws.app(), "论文库")
    alpha = ws.documents[0]["document_id"]
    beta = ws.documents[1]["document_id"]
    widget(at, "selectbox", "doc_detail_selector").set_value(alpha).run()
    widget(at, "text_input", "document_search").set_value("Beta").run()
    no_exception(at)
    assert_synced(widget(at, "selectbox", "doc_detail_selector"), beta)
    assert any("UI Beta Study" in element.value for element in at.subheader)
    at.run()
    no_exception(at)
    assert_synced(widget(at, "selectbox", "doc_detail_selector"), beta)
    assert ws.get("/runs") == []


def test_run_navigation_forces_target_session_instead_of_previous_selection(workspace):
    ws = workspace
    if not ws.backend.owned:
        pytest.skip("Inject a metadata fixture only into the test-owned API")
    at = ws.app()
    click(at, "new_session")
    target = current_session(at)
    click(at, "new_session")
    previous = current_session(at)
    assert previous != target
    identifier = uuid.uuid4().hex
    ws.backend.service.store.put("runs", identifier, ws.token, {
        "id": identifier, "request_id": identifier, "session_id": target,
        "created_at": "2026-10-10T00:00:00+00:00", "status": "done",
        "question": "Native navigation metadata fixture", "answer": "Metadata fixture only, no model was called.",
        "citations": [], "metrics": {}, "events": []})
    navigate(at, "实验与运行记录")
    widget(at, "selectbox", "run_selector").set_value(identifier).run()
    click(at, "run_open_session")
    assert_synced(widget(at, "selectbox", "session_selector"), target)
    at.run()
    assert current_session(at) == target
    assert ws.get(f"/sessions/{target}")["messages"] == []


def test_second_comparison_focuses_new_history_and_new_answer(papers):
    ws = papers
    if not ws.backend.owned:
        pytest.skip("Controlled answer fixtures only modify the test-owned demo API")
    service = ws.backend.service
    original = service.agent.stream
    count = [0]

    def controlled_comparison(*args, **kwargs):
        count[0] += 1
        text = f"CONTROLLED COMPARISON RESULT {count[0]}"
        yield {"type": "status", "text": "UI comparison fixture only"}
        time.sleep(.5)  # Exercise the asynchronous stream and its UI heartbeat.
        yield {"type": "answer_delta", "text": text}
        yield {"type": "done", "answer": text, "citations": [], "metrics": {}, "origin": "demo_extractive"}

    service.agent.stream = controlled_comparison
    try:
        at = navigate(ws.app(), "论文对比")
        click(at, "compare_generate")
        first_widget = widget(at, "selectbox", "compare_history")
        first = first_widget.value
        first_key = first_widget.key
        assert "CONTROLLED COMPARISON RESULT 1" in "\n".join(str(item.value) for item in at.markdown)
        click(at, "compare_generate")
        selector = widget(at, "selectbox", "compare_history")
        second = selector.value
        assert second != first
        assert selector.key != first_key
        assert_synced(selector, second)
        text = "\n".join(str(item.value) for item in at.markdown)
        assert "CONTROLLED COMPARISON RESULT 2" in text
        assert "CONTROLLED COMPARISON RESULT 1" not in text
        assert ws.get(f"/sessions/{second}")["messages"][-1]["content"] == "CONTROLLED COMPARISON RESULT 2"
        at.run()
        assert_synced(widget(at, "selectbox", "compare_history"), second)
        widget(at, "selectbox", "compare_history").set_value(first).run()
        no_exception(at)
        assert widget(at, "selectbox", "compare_history").value == first
        assert "CONTROLLED COMPARISON RESULT 1" in "\n".join(str(item.value) for item in at.markdown)
    finally:
        service.agent.stream = original


def test_document_multiselect_does_not_push_unchanged_values(papers):
    ws = papers
    at = ws.app()
    ids = [doc["document_id"] for doc in ws.documents]
    widget(at, "multiselect", "doc_selection").set_value(ids[:1]).run()
    picker = widget(at, "multiselect", "doc_selection")
    assert picker.value == ids[:1] and picker.proto.set_value is False
    picker.set_value(ids).run()
    picker = widget(at, "multiselect", "doc_selection")
    assert picker.value == ids and picker.proto.set_value is False
    at.run()
    picker = widget(at, "multiselect", "doc_selection")
    assert picker.value == ids and picker.proto.set_value is False
    if ws.backend.owned:
        doc = ws.backend.service.store.get_document(ids[1], ws.token)
        doc.status = "error"  # test-owned readiness fixture; no user file is modified/deleted
        ws.backend.service.store.put_document(doc)
        click(at, "refresh_connection")
        picker = widget(at, "multiselect", "doc_selection")
        assert picker.value == ids[:1] and picker.proto.set_value is True


def test_experiment_multiselect_preserves_additions_and_only_pushes_cleanup(papers):
    ws = papers
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        response = ws.client.post("/experiments", files={"file": (
            "ui-multiselect-fixture.jsonl", json.dumps(evaluation_case(ws)), "application/jsonl")},
                                  data={"mode": mode, "split": "dev"})
        response.raise_for_status()
    experiments = wait_experiments(ws, 3)
    ids = {record["mode"]: record["id"] for record in experiments}
    at = navigate(ws.app(), "实验与运行记录")
    chosen = [ids["hybrid_rerank"]]
    widget(at, "multiselect", "experiment_selection").set_value(chosen).run()
    chosen.append(ids["vector"])
    widget(at, "multiselect", "experiment_selection").set_value(chosen).run()
    chosen.append(ids["hybrid"])
    widget(at, "multiselect", "experiment_selection").set_value(chosen).run()
    picker = widget(at, "multiselect", "experiment_selection")
    assert picker.value == chosen and picker.proto.set_value is False
    at.run()
    picker = widget(at, "multiselect", "experiment_selection")
    assert picker.value == chosen and picker.proto.set_value is False
    widget(at, "selectbox", "experiment_mode_filter").set_value("vector").run()
    picker = widget(at, "multiselect", "experiment_selection")
    assert picker.value == [ids["vector"]] and picker.proto.set_value is True
