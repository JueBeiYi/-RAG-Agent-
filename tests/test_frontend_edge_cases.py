from unittest.mock import Mock
from contextlib import nullcontext

import pytest

from src.frontend.client import APIError
from src.frontend import sections_chat, sections_experiments, sections_library, ui


class State(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def state(monkeypatch):
    value = State(pending_upload_jobs=[], completed_upload_jobs=[{"id": "old-error", "status": "error"}],
                  active_session_id="previous-session", nav_page="实验与运行记录", session_search="旧筛选",
                  session_selector="previous-session", selected_document_ids=["paper-a"], doc_selection=["paper-a"],
                  api_cache={"old": "cached"}, preview_cache={"old": b"image"})
    monkeypatch.setattr(sections_chat.st, "session_state", value)
    return value


def test_restored_workspace_tracks_server_imports_through_completion(state):
    first = [{"job_id": "j1", "status": "queued"}, {"id": "j2", "status": "running"},
             {"id": "older", "status": "done"}]
    sections_library._sync_pending_imports(first)
    assert state.pending_upload_jobs == ["j1", "j2"]
    assert state.completed_upload_jobs == []
    # A later completed job stays pending until the progress renderer handles
    # it, rather than disappearing from the completion/refresh condition.
    changed = [{"job_id": "j1", "status": "done"}, {"id": "j2", "status": "running"}]
    sections_library._sync_pending_imports(changed)
    assert state.pending_upload_jobs == ["j1", "j2"]
    sections_library._sync_pending_imports(changed)
    assert state.pending_upload_jobs.count("j2") == 1


def test_deleted_run_session_stays_on_replay_page(state, monkeypatch):
    client = Mock()
    client.get.side_effect = APIError("记录不存在", 404)
    monkeypatch.setattr(sections_experiments, "api", lambda: client)
    sections_experiments._open_session("deleted-session")
    client.get.assert_called_once_with("/sessions/deleted-session")
    assert state.nav_page == "实验与运行记录"
    assert state.active_session_id == "previous-session"
    assert state.session_search == "旧筛选"
    assert "会话已删除" in state.flash["message"]


def test_existing_run_session_opens_target_and_clears_stale_search(state, monkeypatch):
    client = Mock()
    client.get.return_value = {"id": "target-session", "messages": []}
    monkeypatch.setattr(sections_experiments, "api", lambda: client)
    sections_experiments._open_session("target-session")
    assert state.active_session_id == "target-session"
    assert state.nav_page == "科研问答"
    assert state.session_search == ""
    assert state.session_selector == "target-session"


def test_busy_session_rejects_clear_without_changing_selected_papers(state, monkeypatch):
    client = Mock()
    client.patch.side_effect = APIError("会话正在执行，请稍后修改", 409)
    monkeypatch.setattr(sections_chat, "api", lambda: client)
    sections_chat._clear_docs()
    assert state.doc_selection == ["paper-a"]
    assert state.selected_document_ids == ["paper-a"]
    assert state.api_cache == {"old": "cached"}
    assert state.flash["kind"] == "warning"


def test_clear_range_updates_local_state_after_server_commit(state, monkeypatch):
    client = Mock()
    client.patch.return_value = {"id": "previous-session", "document_ids": []}
    monkeypatch.setattr(sections_chat, "api", lambda: client)
    sections_chat._clear_docs()
    client.patch.assert_called_once_with("/sessions/previous-session", json={"document_ids": []})
    assert state.doc_selection == [] and state.selected_document_ids == []
    assert state.api_cache == {} and state.preview_cache == {}


def test_new_session_overwrites_previous_browser_selection_before_render(state, monkeypatch):
    client = Mock()
    client.post.return_value = {"id": "new-session"}
    monkeypatch.setattr(sections_chat, "api", lambda: client)
    sections_chat._new_session()
    assert state.session_selector == state.active_session_id == "new-session"
    assert state.session_search == ""
    assert state.last_selected_session == "new-session"
    assert state.selected_document_ids == state.doc_selection == []
    assert all(call.args[0] == "/sessions" for call in client.post.call_args_list)


def test_new_comparison_selects_new_history_without_starting_model(state, monkeypatch):
    state.update(compare_a="a", compare_b="b", compare_aspects=["数据集"], compare_focus="")
    client = Mock()
    client.post.return_value = {"id": "new-comparison"}
    monkeypatch.setattr(sections_chat, "api", lambda: client)
    monkeypatch.setattr(sections_chat, "documents", lambda: [
        {"document_id": "a", "title": "A", "status": "ready"},
        {"document_id": "b", "title": "B", "status": "ready"}])
    sections_chat._prepare_comparison()
    assert state.compare_history == state.comparison_session_id == "new-comparison"
    assert state.comparison_history_widget_key == "compare_history_1"
    assert state.pending_comparison_request["session_id"] == "new-comparison"
    assert state.pending_comparison_request["document_ids"] == ["a", "b"]
    assert all(call.args[0] == "/sessions" for call in client.post.call_args_list)


def test_restored_workspace_discovers_running_experiments_without_duplicates(state):
    state.pending_experiments = []
    records = [{"id": "running", "status": "running"}, {"id": "queued", "status": "queued"},
               {"id": "older", "status": "done"}]
    sections_experiments._sync_pending_experiments(records)
    sections_experiments._sync_pending_experiments(records)
    assert state.pending_experiments == ["running", "queued"]
    sections_experiments._sync_pending_experiments([{"id": "running", "status": "done"}])
    assert "running" in state.pending_experiments  # completion renderer performs removal and refresh


@pytest.mark.parametrize("key", ["needs_review", "unsupported_numbers", "invalid_ids", "warnings"])
def test_audit_numeric_or_reference_review_is_visible(monkeypatch, key):
    fake = Mock()
    fake.expander.return_value = nullcontext()
    monkeypatch.setattr(ui, "st", fake)
    value = {key: ["check this claim"], "page_reference_corrections": [{"from": 4, "to": 2}]}
    ui.audit(value)
    fake.warning.assert_called_once()
    assert "需要核对" in fake.expander.call_args.args[0]
    assert any("校正" in call.args[0] for call in fake.caption.call_args_list)
    fake.json.assert_called_once_with(value, expanded=False)


def test_retrieval_metrics_use_grouped_bars_without_adding_scores(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(sections_experiments, "st", fake)
    results = [{"dataset_hash": "same", "corpus_hash": "same", "split": ["dev"], "kind": "retrieval",
                "demo": True, "mode": mode, "metrics": {"hit_at_5": .25, "mrr": .2, "recall_at_5": .15}}
               for mode in ["vector", "hybrid"]]
    sections_experiments._comparison(results)
    kwargs = fake.bar_chart.call_args.kwargs
    assert kwargs["stack"] is False
    assert "0至1" in kwargs["y_label"]
    assert fake.bar_chart.call_args.args[0]["MRR"].tolist() == [.2, .2]
