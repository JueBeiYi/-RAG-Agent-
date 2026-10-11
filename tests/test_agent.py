from __future__ import annotations

import threading
import time
import json
from datetime import datetime

import pytest
import httpx

from src.agent.executor import ToolExecutor
from src.agent.memory import ConversationMemory
from src.agent.react_loop import ResearchAgent, parse_tool_calls
from src.agent.registry import ToolCall, ToolContext, ToolRegistry, ToolSpec, object_schema
from src.agent.router import route, use_model_knowledge
from src.agent.catalog import resolve_reference
from src.agent.tools.local import create_registry, safe_calculate
from src.core.config import Settings
from src.core.schemas import Chunk, Document, RetrievalHit, SourceBlock
from src.generation.context import build_context
from src.core.storage import Storage


class FakeLLM:
    def __init__(self, responses=None, fail=False):
        self.responses = list(responses or [])
        self.calls = []
        self.fail = fail

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("offline")
        return self.responses.pop(0) if self.responses else {"content": "已完成。", "tool_calls": [], "usage": {}}


class FakeRAG:
    def __init__(self, store):
        self.store, self.calls = store, []
        self.retrieve_calls = []

    def retrieve(self, query, scope, document_ids=None, mode=None):
        self.retrieve_calls.append({"query": query, "scope": scope, "document_ids": document_ids})
        return [RetrievalHit(chunk=chunk, score=0.8) for chunk in self.store.list_chunks(scope, document_ids)]

    def ask(self, query, scope, document_ids=None, mode=None, history=None):
        self.calls.append({"query": query, "scope": scope, "document_ids": document_ids, "history": history})
        chunks = self.store.list_chunks(scope, document_ids)
        citations = [{"citation_id": f"S{i+1}", "document_id": chunk.document_id,
                      "chunk_id": chunk.chunk_id, "name": chunk.name, "text": chunk.text, "page_start": chunk.page_start}
                     for i, chunk in enumerate(chunks)]
        answer = "\n".join(f"{chunk.text} [S{i+1}]" for i, chunk in enumerate(chunks)) if chunks else "没有证据。"
        return {"answer": answer, "citations": citations, "origin": "knowledge_rag", "metrics": {"input_tokens": 10, "output_tokens": 20}, "warnings": []}

    def stream(self, query, scope, document_ids=None, mode=None, history=None):
        result = self.ask(query, scope, document_ids, mode, history)
        for citation in result["citations"]:
            yield {"type": "citation", "citation": citation}
        # Deliberately split references into separate transport tokens.
        for char in result["answer"]:
            yield {"type": "answer_delta", "text": char}
        yield {"type": "usage", "input_tokens": 10, "output_tokens": 20, "estimated": False}
        yield {"type": "done", **result}


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=str(tmp_path), llm_backend="extractive", memory_tokens=180, memory_recent_messages=3,
                        tool_timeout_seconds=0.2, max_tool_workers=2)
    store = Storage(tmp_path)
    for scope, ids in (("alice", ["a", "b"]), ("bob", ["secret"])):
        for paper in ids:
            doc = Document(document_id=paper, scope=scope, name=f"Paper {paper}.pdf", title=f"Transformer Paper {paper}",
                           authors=["A. Scientist"], year=2024, doi="", format="pdf", content_hash=paper, status="ready")
            store.put_document(doc)
            chunks = [Chunk(chunk_id=f"{scope}-{paper}-1", document_id=paper, scope=scope, name=doc.name,
                            page_start=2, page_end=2, text=f"We propose a Transformer method for vision. The dataset is Sample{paper}. The result is accuracy 90%. A limitation is small training data.")]
            store.replace_chunks(paper, scope, chunks)
    session = store.create_session("alice")
    llm, rag = FakeLLM(), FakeRAG(store)
    context = ToolContext(settings, store, llm, rag, "alice", session["id"])
    return settings, store, session, llm, rag, context


def test_registry_has_eight_callable_tools(env):
    *_, context = env
    registry = create_registry()
    assert len(registry.schemas()) == 8
    arguments = {
        "knowledge_qa": {"query": "Transformer", "document_ids": ["a"]},
        "paper_metadata": {"document_ids": ["a"]},
        "compare_papers": {"document_ids": ["a", "b"]},
        "extract_keywords": {"document_ids": ["a"], "count": 5},
        "summarize_paper": {"document_ids": ["a"]},
        "current_time": {}, "calculator": {"expression": "sqrt(9)+(3/2)"},
        "paper_catalog_search": {"query": "Transformer"},
    }
    for name, args in arguments.items():
        spec = registry.validate(ToolCall(name, args))
        result = spec.handler(context, args)
        assert result["content"] and not result["error"]
    assert "secret" not in registry.get("paper_catalog_search").handler(context, {"query": ""})["content"]


def test_keywords_accept_question_without_documents_or_fake_citations(env):
    *_, context = env
    context.scope = "empty-library"
    registry = create_registry()
    call = ToolCall("extract_keywords", {"query": "Transformer retrieval Transformer evaluation", "count": 3})
    result = registry.validate(call).handler(context, call.arguments)
    assert result["data"]["origin"] == "user_input" and not result["citations"]
    assert {item["term"]: item["count"] for item in result["data"]["keywords"]}["transformer"] == 2
    routed = route("请提取这个问题的关键词：Transformer retrieval evaluation", context)
    assert routed[0].name == "extract_keywords"
    assert routed[0].arguments["query"] == "Transformer retrieval evaluation"


def test_general_definition_route_keeps_selected_papers_and_references_grounded(env):
    *_, context = env
    assert use_model_knowledge("深度学习是什么？", context)
    assert not use_model_knowledge("这篇论文的深度学习是什么？", context)
    assert not use_model_knowledge("最新的深度学习方法是什么？", context)
    assert not use_model_knowledge("Transformer Paper a是什么？", context)
    context.document_ids = ["a"]
    assert not use_model_knowledge("深度学习是什么？", context)


def test_general_definition_streams_model_without_retrieval_and_labels_origin(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    class DefinitionLLM:
        def stream(self, messages):
            assert "没有检索" in messages[0]["content"]
            yield {"type": "answer_delta", "text": "深度学习使用多层神经网络学习数据表示。"}
            yield {"type": "usage", "input_tokens": 30, "output_tokens": 12, "estimated": False}
    agent = ResearchAgent(settings, store, DefinitionLLM(), rag)
    try:
        events = list(agent.stream("深度学习是什么？", "alice", session["id"]))
        final = events[-1]
        assert not rag.calls and not rag.retrieve_calls
        assert not any(event["type"] == "tool_start" for event in events)
        assert final["origin"] == "model_knowledge" and not final["citations"]
        assert "未检索当前知识库" in final["answer"]
        assert final["metrics"]["input_tokens"] == 30
    finally:
        agent.close()


def test_fast_rag_records_actual_tokens_per_tool_call(env):
    settings, store, session, llm, rag, _ = env
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("论文采用的数据集是什么？", "alice", session["id"], ["a"]))
        tool = next(event for event in events if event["type"] == "tool_end")
        final = events[-1]
        assert tool["usage"]["input_tokens"] == 10 and tool["usage"]["output_tokens"] == 20
        assert final["metrics"]["tool_token_usage"]["knowledge_qa"]["calls"] == 1
        assert final["metrics"]["tool_token_usage"]["knowledge_qa"]["unknown_calls"] == 0
        assert final["metrics"]["planner_usage"]["input_tokens"] == 0
    finally:
        agent.close()


@pytest.mark.parametrize("expression", ["__import__('os').system('echo unsafe')", "(1).__class__", "[1]*1000", "2**1000000", "exp(1000)", "True+1", "10/0", "round(1.2,100000)", "1e101"])
def test_calculator_rejects_unsafe_or_unbounded_code(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        safe_calculate(expression)


def test_calculator_returns_real_arithmetic():
    assert safe_calculate("round((0.9-0.8)/0.8*100,2)") == 12.5
    assert safe_calculate("min(2,3)+sqrt(16)") == 6


def test_tool_parameters_reject_unknown_type_and_scope(env):
    *_, context = env
    registry = create_registry()
    with pytest.raises(ValueError):
        registry.validate(ToolCall("calculator", {"expression": "1+2", "shell": "powershell"}))
    with pytest.raises(ValueError):
        registry.validate(ToolCall("extract_keywords", {"count": True}))
    with pytest.raises(ValueError):
        registry.get("paper_metadata").handler(context, {"document_id": "secret"})
    context.document_ids = ["a"]
    with pytest.raises(ValueError):
        registry.get("paper_metadata").handler(context, {"document_id": "b"})


def test_comparison_requires_distinct_papers_and_retains_both_evidences(env):
    *_, context = env
    tool = create_registry().get("compare_papers")
    with pytest.raises(ValueError):
        tool.handler(context, {"document_ids": ["a", "a"]})
    result = tool.handler(context, {"document_ids": ["a", "b"]})
    assert {citation["document_id"] for citation in result["citations"]} == {"a", "b"}
    assert result["data"]["comparable_metrics"] is False
    assert "不计算" in result["content"]
    assert "[S1]" in result["content"] and "[S2]" in result["content"]


def test_metadata_does_not_fill_unknown_doi(env):
    *_, context = env
    result = create_registry().get("paper_metadata").handler(context, {"document_id": "a"})
    assert result["data"]["papers"][0]["doi"] == ""
    assert "DOI：未识别" in result["content"]


def test_current_time_uses_real_clock_and_timezone(env):
    *_, context = env
    tool = create_registry().get("current_time")
    result = tool.handler(context, {})
    now = datetime.fromisoformat(result["data"]["time"])
    assert abs(now.timestamp() - time.time()) < 3
    assert now.utcoffset().total_seconds() == 8 * 3600
    with pytest.raises(ValueError):
        tool.handler(context, {"timezone": "Not/AZone"})


def test_memory_compression_budget_and_scope_isolation(env):
    settings, store, session, *_ = env
    memory = ConversationMemory(settings, store)
    for i in range(20):
        store.add_message(session["id"], "alice", "user" if i % 2 == 0 else "assistant", f"对话{i} " + "科研证据" * 80)
    store.add_message(session["id"], "alice", "user", "当前问题")
    view = memory.load(session["id"], "alice", "当前问题")
    assert view.tokens <= settings.memory_tokens
    assert view.compressed_messages > 0 and view.summary
    assert all(message["content"] != "当前问题" for message in view.messages)
    assert memory.load(session["id"], "bob", "当前问题").messages == []
    assert store.get_memory(session["id"], "bob") == {}
    # The original history is preserved, and Agent does not save messages itself.
    assert len(store.list_messages(session["id"], "alice")) == 21


def test_router_resolves_paper_titles_and_independent_tasks(env):
    *_, context = env
    assert route("请总结 Transformer Paper a", context)[0].arguments["document_ids"] == ["a"]
    calls = route("当前时间；计算 `3*7`", context)
    assert {call.name for call in calls} == {"current_time", "calculator"}
    assert route("比较两篇论文", context)[0].name == "paper_catalog_search"
    assert route("为什么使用Transformer？", context)[0].name == "knowledge_qa"


def test_executor_dispatches_independent_tasks_concurrently(env):
    settings, *_, context = env
    registry = ToolRegistry()
    barrier = threading.Barrier(2)
    def task(ctx, args):
        barrier.wait(timeout=0.15)
        return {"content": "done"}
    registry.register(ToolSpec("parallel", "parallel", object_schema({}), task))
    executor = ToolExecutor(registry, settings)
    try:
        results = executor.batch(context, [ToolCall("parallel", {}), ToolCall("parallel", {})])
        assert all(result["content"] == "done" and not result["error"] for result in results)
    finally:
        executor.close()


def test_timeout_preserves_worker_limit_and_error_is_visible(env):
    settings, *_, context = env
    settings.max_tool_workers, settings.tool_timeout_seconds = 1, 0.02
    registry = ToolRegistry()
    entered, release = threading.Event(), threading.Event()
    def task(ctx, args):
        entered.set()
        release.wait(timeout=1)
        return {"content": "late"}
    registry.register(ToolSpec("slow", "slow", object_schema({}), task))
    executor = ToolExecutor(registry, settings)
    try:
        result = executor.batch(context, [ToolCall("slow", {})])[0]
        assert entered.is_set() and result["error"] and result["data"]["worker_pending"]
        occupied = executor.batch(context, [ToolCall("slow", {})])[0]
        assert "资源已占用" in occupied["error"]
    finally:
        release.set()
        executor.close()


def test_native_and_json_tool_call_parsing():
    assert parse_tool_calls({"tool_calls": [{"function": {"name": "calculator", "arguments": '{"expression":"2+2"}'}}]})[0].arguments == {"expression": "2+2"}
    assert parse_tool_calls({"content": '```json\n{"tool_calls":[{"name":"current_time","arguments":{}}]}\n```'})[0].name == "current_time"
    assert parse_tool_calls({"content": "任意解释 {bad json}"}) == []
    with pytest.raises(ValueError):
        parse_tool_calls({"tool_calls": "not a list"})


def test_agent_streams_rag_tokens_and_preserves_history_without_resaving(env):
    settings, store, session, llm, rag, _ = env
    store.add_message(session["id"], "alice", "user", "之前的问题")
    store.add_message(session["id"], "alice", "assistant", "之前的回答")
    store.add_message(session["id"], "alice", "user", "为什么使用Transformer？")
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("为什么使用Transformer？", "alice", session["id"], ["a"]))
        done = events[-1]
        text = "".join(event["text"] for event in events if event["type"] == "answer_delta")
        assert text == done["answer"] and "[S1]" in text
        assert sum(event["type"] == "answer_delta" for event in events) > 10
        assert done["metrics"]["input_tokens"] == 10
        assert "audit" in done and not done["audit"]["invalid_ids"]
        assert rag.calls[0]["history"][-1]["content"] == "之前的回答"
        assert len(store.list_messages(session["id"], "alice")) == 3
        assert llm.calls == []
    finally:
        agent.close()


def test_agent_recovers_missing_selection_with_scoped_catalog(env):
    settings, store, session, llm, rag, _ = env
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("请总结论文", "alice", session["id"]))
        done = events[-1]
        assert done["metrics"]["tool_calls"] == 2
        assert done["metrics"]["tool_successes"] == 1
        assert "需要明确选择" in done["answer"] and "Transformer Paper a" in done["answer"]
        assert "secret" not in done["answer"]
    finally:
        agent.close()


def test_agent_stops_repeated_model_action(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    repeat = {"tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}], "usage": {"input_tokens": 4, "output_tokens": 2}}
    llm = FakeLLM([repeat, repeat])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("计算 `2+2`", "alice", session["id"]))[-1]
        assert done["metrics"]["tool_calls"] == 1
        assert any("终止重复调用" in warning for warning in done["metrics"]["warnings"])
        assert "4" in done["answer"]
        assert done["metrics"]["input_tokens"] == 8
    finally:
        agent.close()


def test_agent_merges_colliding_citations_and_refuses_unknown_sources(env):
    settings, store, session, _, rag, context = env
    settings.llm_backend = "ollama"
    llm = FakeLLM([
        {"tool_calls": [{"name": "summarize_paper", "arguments": {"document_ids": ["a"]}},
                        {"name": "summarize_paper", "arguments": {"document_ids": ["b"]}}]},
        {"content": "论文A优于论文B [S99]。", "tool_calls": []},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("请总结两篇论文", "alice", session["id"], ["a", "b"]))[-1]
        assert [c["citation_id"] for c in done["citations"]] == ["S1", "S2"]
        assert "[S1]" in done["answer"] and "[S2]" in done["answer"] and "[S99]" not in done["answer"]
        assert "优于" not in done["answer"]
        refs, known = [], {}
        result = agent._merge_sources({"content": "秘密 [S1]", "citations": [{"citation_id": "S1", "document_id": "secret", "chunk_id": "bob-secret-1", "text": "forged"}]}, refs, known, context)
        assert refs == [] and "引用未核实" in result["content"]
    finally:
        agent.close()


def test_llm_failure_returns_tools_and_discloses_fallback(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    agent = ResearchAgent(settings, store, FakeLLM(fail=True), rag)
    try:
        done = list(agent.stream("计算 `12/3`", "alice", session["id"]))[-1]
        assert "**4**" in done["answer"]
        assert any("模型规划不可用" in warning for warning in done["metrics"]["warnings"])
    finally:
        agent.close()


def test_native_tool_call_response_protocol_and_bounded_steps(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend, settings.max_agent_steps = "ollama", 1
    llm = FakeLLM([
        {"tool_calls": [{"id": "provided-id", "name": "calculator", "arguments": {"expression": "3+4"}}]},
        {"tool_calls": [{"name": "calculator", "arguments": {"expression": "4+5"}}]},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("计算 `3+4`", "alice", session["id"]))[-1]
        assert done["metrics"]["tool_calls"] == 1
        assert "7" in done["answer"]
        assert any("最大步骤数" in warning for warning in done["metrics"]["warnings"])
        second_prompt = llm.calls[1]
        assistant_call = next(message for message in second_prompt if message.get("tool_calls"))
        result_message = next(message for message in second_prompt if message["role"] == "tool")
        assert assistant_call["tool_calls"][0]["id"] == result_message["tool_call_id"] == "provided-id"
        assert result_message["name"] == "calculator"
    finally:
        agent.close()


def test_disabled_routing_uses_model_instead_of_rag_shortcut(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    settings.agent_routing_enabled = False
    llm = FakeLLM([
        {"tool_calls": [{"name": "calculator", "arguments": {"expression": "5*5"}}]},
        {"content": "结果为 25。", "tool_calls": []},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("请完成这个任务", "alice", session["id"]))
        assert len(llm.calls) == 2 and rag.calls == []
        assert events[-1]["answer"] == "结果为 25。"
        assert any(event.get("tool") == "calculator" for event in events)
    finally:
        agent.close()


def test_compare_failure_recovers_each_scoped_paper_and_merges_evidence(env):
    settings, store, session, llm, rag, _ = env
    agent = ResearchAgent(settings, store, llm, rag)
    def broken(context, args):
        raise RuntimeError("comparison temporarily unavailable")
    agent.registry.get("compare_papers").handler = broken
    try:
        events = list(agent.stream("比较这两篇论文的方法与结果", "alice", session["id"], ["a", "b"]))
        done = events[-1]
        assert done["metrics"]["tool_calls"] == 3
        assert done["metrics"]["tool_successes"] == 2
        assert sorted(call["document_ids"] for call in rag.calls) == [["a"], ["b"]]
        assert {c["document_id"] for c in done["citations"]} == {"a", "b"}
        assert "逐篇检索" in done["answer"] and "[S1]" in done["answer"] and "[S2]" in done["answer"]
        assert "部分工具未完成" in done["answer"]
        assert "secret" not in done["answer"]
    finally:
        agent.close()


def test_metadata_failure_uses_catalog_without_crossing_selection(env):
    settings, store, session, llm, rag, _ = env
    agent = ResearchAgent(settings, store, llm, rag)
    def broken(context, args):
        raise ValueError("metadata unavailable")
    agent.registry.get("paper_metadata").handler = broken
    try:
        events = list(agent.stream("这篇论文的作者是谁", "alice", session["id"], ["a"]))
        assert any(event.get("tool") == "paper_catalog_search" for event in events)
        assert "Transformer Paper a" in events[-1]["answer"]
        assert "Transformer Paper b" not in events[-1]["answer"]
        assert events[-1]["metrics"]["retry_count"] == 0
    finally:
        agent.close()


def test_completed_transient_error_retries_twice_then_recovers(env):
    settings, store, session, llm, rag, _ = env
    settings.max_tool_workers = 1
    agent = ResearchAgent(settings, store, llm, rag)
    original, attempts = agent.registry.get("calculator").handler, []
    def transient(context, args):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary connection failure")
        return original(context, args)
    agent.registry.get("calculator").handler = transient
    try:
        events = list(agent.stream("计算 `4*5`", "alice", session["id"]))
        done = events[-1]
        assert len(attempts) == 3 and done["metrics"]["retry_count"] == 2
        assert done["metrics"]["recovered_tools"] == 1
        assert done["metrics"]["tool_calls"] == 3 and done["metrics"]["tool_successes"] == 1
        assert "20" in done["answer"] and "部分工具未完成" not in done["answer"]
        starts = [event for event in events if event["type"] == "tool_start"]
        assert [event["attempt"] for event in starts] == [1, 2, 3]
        assert starts[1]["retry_of"] == starts[0]["call_id"]
    finally:
        agent.close()


def test_retryable_result_stops_after_three_total_attempts(env):
    settings, store, session, llm, rag, _ = env
    agent = ResearchAgent(settings, store, llm, rag)
    attempts = []
    def temporary(context, args):
        attempts.append(1)
        return {"error": "transient response", "retryable": True, "content": "", "data": {}}
    agent.registry.get("calculator").handler = temporary
    try:
        done = list(agent.stream("计算 `4*5`", "alice", session["id"]))[-1]
        assert len(attempts) == 3 and done["metrics"]["retry_count"] == 2
        assert "transient response" in done["answer"]
        assert done["metrics"]["recovered_tools"] == 0
    finally:
        agent.close()


def test_agent_never_retries_task_still_running_after_timeout(env):
    settings, store, session, llm, rag, _ = env
    settings.max_tool_workers, settings.tool_timeout_seconds = 1, 0.01
    agent = ResearchAgent(settings, store, llm, rag)
    release, attempts = threading.Event(), []
    def slow(context, args):
        attempts.append(1)
        release.wait(timeout=1)
        return {"error": "temporary", "retryable": True, "data": {}}
    agent.registry.get("calculator").handler = slow
    try:
        events = list(agent.stream("计算 `4*5`", "alice", session["id"]))
        assert len(attempts) == 1 and events[-1]["metrics"]["retry_count"] == 0
        failed = next(event for event in events if event["type"] == "tool_end")
        assert failed["data"]["worker_pending"] is True
        assert failed["retryable"] is False
    finally:
        release.set()
        agent.close()


def test_transient_http_response_can_retry_but_unknown_server_read_timeout_cannot():
    request = httpx.Request("GET", "http://example.test")
    error = httpx.HTTPStatusError("busy", request=request, response=httpx.Response(503, request=request))
    assert ToolExecutor._temporary_error(error)
    assert not ToolExecutor._temporary_error(httpx.ReadTimeout("server work may still run", request=request))
    assert not ToolExecutor._temporary_error(ValueError("bad arguments"))


def test_calculator_formats_float_noise_but_preserves_numeric_data(env):
    *_, context = env
    result = create_registry().get("calculator").handler(context, {"expression": "3.14*2.56"})
    assert result["content"] == "`3.14*2.56` = **8.0384**"
    assert result["data"]["value"] == 3.14 * 2.56
    assert result["data"]["display_value"] == "8.0384"


def test_true_rag_stream_retries_completed_transient_errors_before_answer(env):
    settings, store, session, llm, rag, _ = env
    original, attempts = rag.stream, []
    def transient(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("service not ready")
        yield from original(*args, **kwargs)
    rag.stream = transient
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("Transformer采用了什么方法？", "alice", session["id"], ["a"]))
        done = events[-1]
        assert len(attempts) == 3 and done["metrics"]["retry_count"] == 2
        assert done["metrics"]["recovered_tools"] == 1
        assert done["metrics"]["failed_attempt_usage_unknown"]
        assert done["metrics"]["tool_token_usage"]["knowledge_qa"]["unknown_calls"] == 2
        failed = [event for event in events if event["type"] == "tool_end" and event.get("error")]
        assert len(failed) == 2 and all(event["usage"]["unknown"] and event["usage"]["input_tokens"] is None for event in failed)
        assert "".join(event["text"] for event in events if event["type"] == "answer_delta") == done["answer"]
    finally:
        agent.close()


def test_partial_rag_answer_is_never_automatically_replayed(env):
    settings, store, session, llm, rag, _ = env
    attempts = []
    def interrupted(*args, **kwargs):
        attempts.append(1)
        yield {"type": "answer_delta", "text": "partial output"}
        raise ConnectionError("transport interrupted after output")
    rag.stream = interrupted
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("Transformer采用了什么方法？", "alice", session["id"], ["a"]))
        assert len(attempts) == 1 and events[-1]["origin"] == "agent_error"
        assert events[-1]["metrics"]["tool_calls"] == 1
        failed = next(event for event in events if event["type"] == "tool_end")
        assert failed["retryable"] is False
    finally:
        agent.close()


def test_agent_preserves_budget_cropped_cross_page_sources_in_true_rag_stream(env):
    settings, store, session, llm, rag, context = env
    first = SourceBlock(text="A" * 300 + " methodology", page=1, bbox=[0, 0, 100, 100])
    second = SourceBlock(text="Z" * 300 + " accuracy 100", page=2, bbox=[0, 0, 100, 100])
    chunk = Chunk(chunk_id="alice-a-cross", document_id="a", scope="alice", name="Paper a.pdf",
                  text=first.text + "\n" + second.text, page_start=1, page_end=2, sources=[first, second])
    store.replace_chunks("a", "alice", [chunk])
    _, limited = build_context([RetrievalHit(chunk=chunk, score=0.9)], budget=100)
    raw = limited[0].model_dump(mode="json")
    assert raw["page_start"] == raw["page_end"] == 1
    assert raw["text"] != chunk.text and "Z" not in raw["text"]
    def limited_stream(*args, **kwargs):
        yield {"type": "citation", "citation": raw}
        yield {"type": "answer_delta", "text": "方法证据 [S1]"}
        yield {"type": "done", "answer": "方法证据 [S1]", "citations": [raw], "metrics": {}, "origin": "knowledge_rag"}
    rag.stream = limited_stream
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("研究采用什么方法？", "alice", session["id"], ["a"]))
        returned = events[-1]["citations"][0]
        assert returned["text"] == raw["text"]
        assert returned["sources"] == raw["sources"]
        assert returned["page_start"] == returned["page_end"] == 1
        assert "Z" not in returned["text"]
        assert {source["page"] for source in returned["sources"]} == {1}
    finally:
        agent.close()


def test_agent_rejects_forged_excerpt_and_distinguishes_two_excerpts_of_one_chunk(env):
    settings, store, session, llm, rag, context = env
    agent = ResearchAgent(settings, store, llm, rag)
    chunk = store.list_chunks("alice", ["a"])[0]
    def raw(text):
        return {"citation_id": "S1", "document_id": "a", "chunk_id": chunk.chunk_id,
                "text": text, "page_start": 999, "sources": [{"text": "invented", "page": 999}]}
    try:
        citations, known = [], {}
        forged = agent._merge_sources({"content": "模型编造 [S1]", "citations": [raw("not present in the chunk")]}, citations, known, context)
        assert citations == [] and "引用未核实" in forged["content"]
        first_text, second_text = chunk.text[:50], chunk.text[50:100]
        first = agent._merge_sources({"content": "片段甲 [S1]", "citations": [raw(first_text)]}, citations, known, context)
        second = agent._merge_sources({"content": "片段乙 [S1]", "citations": [raw(second_text)]}, citations, known, context)
        assert first["citations"][0]["text"] == first_text
        assert second["citations"][0]["text"] == second_text
        assert first["content"].endswith("[S1]") and second["content"].endswith("[S2]")
        assert {citation["page_start"] for citation in citations} == {2}
    finally:
        agent.close()


def _summary_response(result="90%", conclusion="作者认为该方法有效 [S1]"):
    import json
    return {"content": json.dumps({
        "背景": "研究视觉任务 [S1]", "方法": "提出 Transformer 方法 [S1]",
        "数据与实验条件": "使用 Samplea 数据集 [S1]", "结果": "准确率为 " + result + " [S1]",
        "结论": conclusion, "局限": "训练数据较少 [S1]",
    }, ensure_ascii=False), "tool_calls": [], "usage": {"input_tokens": 42, "output_tokens": 66, "estimated": False}}


def _add_explicit_conclusion(store):
    chunk = store.list_chunks("alice", ["a"])[0]
    chunk.text += " We conclude that the method is effective for vision."
    store.replace_chunks("a", "alice", [chunk])


def test_standard_summary_uses_llm_four_required_sections_real_sources_and_usage(env):
    settings, store, _, _, _, context = env
    settings.llm_backend = "ollama"
    _add_explicit_conclusion(store)
    context.llm = FakeLLM([_summary_response()])
    result = create_registry().get("summarize_paper").handler(context, {"document_id": "a"})
    assert result["data"]["summary_method"] == "llm_grounded"
    assert {"背景", "方法", "结果", "结论"} <= set(result["data"]["papers"][0]["sections"])
    assert result["data"]["metrics"]["input_tokens"] == 42
    assert result["data"]["metrics"]["output_tokens"] == 66
    assert len(context.llm.calls) == 1 and result["data"]["warnings"] == []
    assert all(citation["text"] in store.list_chunks("alice", ["a"])[0].text for citation in result["citations"])
    assert all("作者认为" not in citation["text"] for citation in result["citations"])
    assert "基于原文证据的模型摘要" in result["content"]


@pytest.mark.parametrize("mode", ["demo", "unavailable", "bad_number", "invented_conclusion", "bad_format"])
def test_summary_fallback_is_explicit_and_does_not_invent_conclusion(env, mode):
    settings, store, _, _, _, context = env
    settings.llm_backend = "extractive" if mode == "demo" else "ollama"
    if mode == "unavailable":
        context.llm = FakeLLM(fail=True)
    elif mode == "bad_number":
        _add_explicit_conclusion(store)
        context.llm = FakeLLM([_summary_response("95%")])
    elif mode == "bad_format":
        context.llm = FakeLLM([{"content": "没有结论结构的普通文字 [S1]", "usage": {"input_tokens": 42, "output_tokens": 66}}])
    else:
        context.llm = FakeLLM([_summary_response()])
    result = create_registry().get("summarize_paper").handler(context, {"document_id": "a"})
    assert result["data"]["summary_method"] == "evidence_excerpts"
    assert "降级模式" in result["content"] and result["data"]["warnings"]
    assert "95%" not in result["content"]
    if mode != "bad_number":
        assert result["data"]["papers"][0]["sections"]["结论"] == []
        assert "作者认为该方法有效" not in result["content"]
    if mode in {"bad_number", "invented_conclusion", "bad_format"}:
        assert result["data"]["metrics"]["input_tokens"] == 42
        assert result["data"]["metrics"]["output_tokens"] == 66
    if mode == "demo":
        assert context.llm.calls == []


def test_agent_accounts_for_summary_llm_tokens_in_addition_to_planning(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    _add_explicit_conclusion(store)
    llm = FakeLLM([
        {"tool_calls": [{"name": "summarize_paper", "arguments": {"document_ids": ["a"]}}],
         "usage": {"input_tokens": 8, "output_tokens": 2}},
        _summary_response(),
        {"content": "", "tool_calls": [], "usage": {"input_tokens": 4, "output_tokens": 3}},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("总结Transformer Paper a", "alice", session["id"]))[-1]
        assert done["metrics"]["input_tokens"] == 54
        assert done["metrics"]["output_tokens"] == 71
        assert done["metrics"]["tool_token_usage"]["summarize_paper"]["input_tokens"] == 42
        assert done["metrics"]["planner_usage"]["input_tokens"] == 12
        assert "**背景**" in done["answer"] and "**结论**" in done["answer"]
        assert all(citation["text"] in store.list_chunks("alice", ["a"])[0].text for citation in done["citations"])
    finally:
        agent.close()


def test_title_alias_and_unique_title_fragment_resolve_without_ui_ids(env):
    *_, context = env
    store = context.store
    doc = store.get_document("a", "alice")
    doc.title = "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    doc.metadata["aliases"] = ["ViT", "Vision Transformer"]
    store.put_document(doc)
    other = store.get_document("b", "alice")
    other.title = "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows"
    other.metadata["aliases"] = ["Swin"]
    store.put_document(other)
    calls = route("比较ViT和Swin的方法", context)
    assert calls[0].name == "compare_papers" and set(calls[0].arguments["document_ids"]) == {"a", "b"}
    assert route("《An Image Is Worth 16x16 Words》的作者是谁？", context)[0].arguments["document_ids"] == ["a"]
    assert route("总结 `ViT`", context)[0].name == "summarize_paper"
    assert route("总结 `ViT`", context)[0].arguments["document_ids"] == ["a"]
    metadata = create_registry().get("paper_metadata").handler(context, {"document_id": "ViT"})
    assert metadata["data"]["papers"][0]["document_id"] == "a"
    catalog = create_registry().get("paper_catalog_search").handler(context, {"query": "ViT"})
    assert [row["document_id"] for row in catalog["data"]["papers"]] == ["a"]


def test_ambiguous_alias_requires_catalogue_selection_and_scope_is_enforced(env):
    *_, context = env
    for paper in ("a", "b"):
        doc = context.store.get_document(paper, "alice")
        doc.metadata["aliases"] = ["ViT"]
        context.store.put_document(doc)
    assert set(resolve_reference("ViT", context)) == {"a", "b"}
    with pytest.raises(ValueError, match="歧义"):
        create_registry().get("paper_metadata").handler(context, {"document_id": "ViT"})
    context.document_ids = ["a"]
    assert resolve_reference("ViT", context) == ["a"]
    assert resolve_reference("secret", context) == []
    result = create_registry().get("paper_metadata").handler(context, {"document_id": "ViT"})
    assert [paper["document_id"] for paper in result["data"]["papers"]] == ["a"]


def test_standard_summary_can_mark_missing_conclusion_without_fabrication(env):
    settings, store, _, _, _, context = env
    settings.llm_backend = "ollama"
    context.llm = FakeLLM([_summary_response(conclusion="所选证据未包含明确结论，无法确认。")])
    result = create_registry().get("summarize_paper").handler(context, {"document_id": "a"})
    assert result["data"]["summary_method"] == "llm_grounded"
    assert result["data"]["papers"][0]["sections"]["结论"] == ["所选证据未包含明确结论，无法确认。"]
    assert all(citation["text"] in store.list_chunks("alice", ["a"])[0].text for citation in result["citations"])


def test_summary_selects_front_and_conclusion_by_source_order_not_chunk_hash(env):
    *_, context = env
    chunks = [Chunk(chunk_id="hash-fff-front", document_id="a", scope="alice", name="a.pdf", page_start=1,
                    text="Background: The vision research problem motivates this study.", section="Background"),
              Chunk(chunk_id="hash-000-conclusion", document_id="a", scope="alice", name="a.pdf", page_start=8,
                    text="Conclusion: We conclude that the proposed approach is effective.", section="")]
    chunks.extend(Chunk(chunk_id=f"hash-{i:03d}", document_id="a", scope="alice", name="a.pdf", page_start=2+i//3,
                        text="Experimental passage without a concluding claim.") for i in range(20))
    context.store.replace_chunks("a", "alice", chunks)
    result = create_registry().get("summarize_paper").handler(context, {"document_id": "a"})
    assert {citation["page_start"] for citation in result["citations"]} >= {1, 8}
    assert result["data"]["papers"][0]["sections"]["背景"]
    assert any("We conclude" in excerpt for excerpt in result["data"]["papers"][0]["sections"]["结论"])


def test_agent_final_synthesis_cannot_drop_required_summary_structure(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    _add_explicit_conclusion(store)
    llm = FakeLLM([
        {"tool_calls": [{"name": "summarize_paper", "arguments": {"document_ids": ["a"]}}]},
        _summary_response(),
        {"content": "Transformer 方法的准确率为90% [S1]。", "tool_calls": []},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("请总结Transformer Paper a", "alice", session["id"]))[-1]
        assert all("**" + section + "**" in done["answer"] for section in ("背景", "方法", "结果", "结论"))
        assert done["metrics"]["summary_generated_papers"] == 1
        assert done["metrics"]["summary_fallback_papers"] == 0
        assert any("结构或证据" in warning for warning in done["metrics"]["warnings"])
    finally:
        agent.close()


def _comparison_response(*, first_source="S1", second_source="S2", accuracy="90", drop_second=False):
    import json
    rows = [{"aspect": "方法与创新", "cells": {"P1": f"采用Transformer视觉方法 [{first_source}]。", "P2": f"提出Transformer视觉模型 [{second_source}]。"}},
            {"aspect": "性能指标", "cells": {"P1": f"文内准确率为{accuracy}% [{first_source}]。", "P2": f"文内准确率为90% [{second_source}]。"}}]
    if drop_second:
        for row in rows:
            row["cells"].pop("P2")
    return {"content": json.dumps({"rows": rows, "note": "尚未完整核验实验条件"}, ensure_ascii=False),
            "tool_calls": [], "usage": {"input_tokens": 120, "output_tokens": 85, "estimated": False}}


def test_standard_comparison_produces_requested_chinese_rows_real_sources_and_usage(env):
    settings, store, _, _, _, context = env
    settings.llm_backend = "ollama"
    context.llm = FakeLLM([_comparison_response()])
    result = create_registry().get("compare_papers").handler(context, {
        "document_ids": ["a", "b"], "question": "比较方法与性能，不计算不同条件的差值",
        "aspects": ["方法与创新", "性能指标"]})
    assert result["data"]["comparison_method"] == "llm_grounded"
    assert [row["aspect"] for row in result["data"]["rows"]] == ["方法与创新", "性能指标"]
    assert result["data"]["metrics"]["input_tokens"] == 120
    assert result["data"]["metrics"]["output_tokens"] == 85
    assert result["data"]["fallback_reason"] == "" and result["data"]["warnings"] == []
    assert "中文对比" in result["content"] and "不计算跨论文" in result["content"]
    assert {citation["document_id"] for citation in result["citations"]} == {"a", "b"}
    for citation in result["citations"]:
        actual = next(chunk for chunk in store.list_chunks("alice", [citation["document_id"]]) if chunk.chunk_id == citation["chunk_id"])
        assert citation["text"] in actual.text and "采用Transformer视觉方法" not in citation["text"]
    assert len(context.llm.calls) == 1
    assert "方法与创新" in context.llm.calls[0][-1]["content"]


@pytest.mark.parametrize("failure", ["unavailable", "bad_format", "bad_number", "bad_ref", "cross_paper", "missing_paper", "uncited_sentence", "ranking", "authors"])
def test_standard_comparison_failures_are_explicit_incomplete_evidence_fallbacks(env, failure):
    settings, store, _, _, _, context = env
    settings.llm_backend = "ollama"
    response = _comparison_response()
    if failure == "unavailable":
        context.llm = FakeLLM(fail=True)
    else:
        if failure == "bad_format":
            response["content"] = "只有一句不完整比较 [S1]"
        elif failure == "bad_number":
            response = _comparison_response(accuracy="95")
        elif failure == "bad_ref":
            response = _comparison_response(first_source="S99")
        elif failure == "cross_paper":
            response = _comparison_response(first_source="S2", second_source="S1")
        elif failure == "missing_paper":
            response = _comparison_response(drop_second=True)
        else:
            parsed = json.loads(response["content"])
            parsed["rows"][0]["cells"]["P1"] = {
                "uncited_sentence": "采用Transformer [S1]。还具有不曾引用的新事实。",
                "ranking": "性能优于另一篇论文 [S1]。",
                "authors": "作者equal contribution作为论文结论 [S1]。",
            }[failure]
            response["content"] = json.dumps(parsed, ensure_ascii=False)
        context.llm = FakeLLM([response])
    result = create_registry().get("compare_papers").handler(context, {
        "document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert result["data"]["comparison_method"] == "evidence_excerpts"
    assert "原文摘录降级" in result["content"] and "对比不完整" in result["content"]
    assert result["data"]["warnings"] and result["data"]["fallback_reason"]
    assert {citation["document_id"] for citation in result["citations"]} == {"a", "b"}
    assert "95%" not in result["content"] and "equal contribution" not in result["content"]
    if failure != "unavailable":
        assert result["data"]["metrics"]["input_tokens"] == 120
        assert result["data"]["metrics"]["output_tokens"] == 85
    else:
        assert result["data"]["metrics"]["failed_attempt_usage_unknown"]


def test_comparison_filters_bibliography_and_equal_contribution_without_affecting_summary_helper(env):
    settings, store, _, _, rag, context = env
    for paper in ("a", "b"):
        chunks = [Chunk(chunk_id=f"{paper}-0-references", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=9, section="References",
                        text="References\n[1] A. Author. Training data. Proceedings CVPR, 2020.\n[2] B. Author. Method comparison. ICCV, 2021."),
                  Chunk(chunk_id=f"{paper}-1-authors", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=1,
                        text="Several Authors and several institutions. Equal contribution. Corresponding author: email@example.test."),
                  Chunk(chunk_id=f"{paper}-9-body", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=2,
                        text="Abstract\nWe propose a Transformer method. The experimental accuracy is 90%. Dataset: Sample. We conclude the approach is effective.")]
        store.replace_chunks(paper, "alice", chunks)
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["数据集", "结论"]})
    assert all(citation["chunk_id"].endswith("body") for citation in result["citations"])
    assert "Proceedings" not in result["content"] and "Equal contribution" not in result["content"]
    assert "数据集" in result["content"] and "结论" in result["content"]
    assert sorted(call["document_ids"] for call in rag.retrieve_calls) == [["a"], ["b"]]
    assert all("数据集" in call["query"] and call["scope"] == "alice" for call in rag.retrieve_calls)


def test_comparison_rejects_papers_with_only_bibliography_or_authors(env):
    *_, context = env
    context.store.replace_chunks("a", "alice", [Chunk(chunk_id="a-ref", document_id="a", scope="alice", name="a.pdf",
        text="References\n[1] An Author. A benchmark method. Proceedings CVPR 2020.\n[2] Other Author. Dataset methods. ICCV 2021.", section="References")])
    with pytest.raises(ValueError, match="正文证据"):
        create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"]})


def test_comparison_accepts_grounded_markdown_table_and_extracts_ui_question_dimensions(env):
    settings, _, _, _, _, context = env
    settings.llm_backend = "ollama"
    response = _comparison_response()
    response["content"] = "| 维度 | 第一篇 | 第二篇 |\n|---|---|---|\n| 方法与创新 | 提出Transformer方法 [S1]。 | 采用Transformer方法 [S2]。 |\n| 性能指标 | 文内准确率90% [S1]。 | 文内准确率90% [S2]。 |"
    context.llm = FakeLLM([response])
    result = create_registry().get("compare_papers").handler(context, {
        "document_ids": ["a", "b"], "question": "请比较。比较维度：方法与创新、性能指标。用表格组织。"})
    assert result["data"]["aspects"] == ["方法与创新", "性能指标"]
    assert result["data"]["comparison_method"] == "llm_grounded"


def test_grounded_comparison_agent_preserves_tool_content_and_avoids_extra_synthesis(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    llm = FakeLLM([
        _comparison_response(),
        {"content": "This expensive synthesis must not run", "tool_calls": []},
    ])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        done = list(agent.stream("比较所选两篇论文的方法与性能", "alice", session["id"], ["a", "b"]))[-1]
        assert len(llm.calls) == 1
        assert "中文对比" in done["answer"] and "原文摘录降级" not in done["answer"]
        assert done["metrics"]["input_tokens"] == 120 and done["metrics"]["output_tokens"] == 85
        assert done["metrics"]["comparison_method"] == "llm_grounded"
        assert {citation["document_id"] for citation in done["citations"]} == {"a", "b"}
        assert not done["metrics"]["warnings"]
    finally:
        agent.close()


def _qualitative_comparison_response(first="采用Transformer视觉方法 [S1]。", second="采用Transformer视觉方法 [S2]。"):
    return {"content": json.dumps({"rows": [{"aspect": "方法与创新", "cells": {"P1": first, "P2": second}},
            {"aspect": "实验条件", "cells": {"P1": "论文在所述数据集上训练评估 [S1]。", "P2": "论文使用所述训练数据进行评估 [S2]。"}}]}, ensure_ascii=False),
            "tool_calls": [], "usage": {"input_tokens": 80, "output_tokens": 60, "estimated": False}}


def test_selected_two_paper_comparison_bypasses_planner_doi_and_uses_fixed_scope(env):
    settings, store, session, _, rag, _ = env
    settings.llm_backend = "ollama"
    response = _qualitative_comparison_response()
    response["tool_calls"] = [{"name": "compare_papers", "arguments": {"document_ids": ["10.1145/fabricated", "foreign"]}}]
    llm = FakeLLM([response])
    agent = ResearchAgent(settings, store, llm, rag)
    try:
        events = list(agent.stream("比较所选两篇的方法与实验条件，论文中有DOI 10.1145/test", "alice", session["id"], ["a", "b"]))
        starts = [event for event in events if event["type"] == "tool_start"]
        assert len(starts) == 1 and starts[0]["tool"] == "compare_papers"
        assert starts[0]["arguments"]["document_ids"] == ["a", "b"]
        assert len(llm.calls) == 1 and "tool_calls" not in llm.calls[0][0]
        assert "中文对比" in events[-1]["answer"]
        assert not events[-1]["metrics"]["warnings"]
    finally:
        agent.close()


def test_comparison_one_bounded_repair_preserves_rejected_candidate_and_all_usage(env):
    settings, _, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", 90
    first = _comparison_response(accuracy="95")
    second = _comparison_response()
    second["usage"] = {"input_tokens": 140, "output_tokens": 90, "estimated": False}
    context.llm = FakeLLM([first, second])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert result["data"]["comparison_method"] == "llm_grounded"
    metrics = result["data"]["metrics"]
    assert metrics["comparison_repaired"] and metrics["comparison_generation_attempts"] == 2
    assert metrics["input_tokens"] == 260 and metrics["output_tokens"] == 175
    attempts = result["data"]["diagnostics"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["accepted"] is False and attempts[1]["accepted"] is True
    assert "95%" in attempts[0]["model_candidate"] and "95" in attempts[0]["validation_reason"]
    assert attempts[0]["validation_details"]["paper_key"] == "P1"
    assert "95" in context.llm.calls[1][-1]["content"] and "同一份证据" in context.llm.calls[1][-1]["content"]
    assert "95%" not in result["content"]


def test_comparison_repair_never_weakens_numeric_guard_or_runs_more_than_twice(env):
    settings, _, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", 90
    context.llm = FakeLLM([_comparison_response(accuracy="95"), _comparison_response(accuracy="99"), _comparison_response()])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert len(context.llm.calls) == 2
    assert result["data"]["comparison_method"] == "evidence_excerpts"
    assert result["data"]["metrics"]["input_tokens"] == 240
    diagnostics = result["data"]["diagnostics"]
    assert len(diagnostics["attempts"]) == 2 and not any(attempt["accepted"] for attempt in diagnostics["attempts"])
    assert "95" in diagnostics["attempts"][0]["validation_reason"] and "99" in diagnostics["attempts"][1]["validation_reason"]
    assert not diagnostics["candidate_is_accepted_answer"]
    assert "95%" not in result["content"] and "99%" not in result["content"]


def test_comparison_repair_is_not_started_without_remaining_budget(env):
    settings, _, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", .1
    context.llm = FakeLLM([_comparison_response(accuracy="95"), _comparison_response()])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert len(context.llm.calls) == 1
    assert result["data"]["diagnostics"]["repair_skipped_reason"]
    assert result["data"]["comparison_method"] == "evidence_excerpts"


def test_qualitative_comparison_repair_removes_even_supported_unrequested_sizes(env):
    settings, store, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", 90
    chunk = store.list_chunks("alice", ["a"])[0]
    # Keep the supported size inside the actual bounded evidence window.
    chunk.text = "Window size is 7. " + chunk.text
    store.replace_chunks("a", "alice", [chunk])
    context.llm = FakeLLM([_qualitative_comparison_response(first="采用尺寸为7的窗口 [S1]。"), _qualitative_comparison_response()])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "实验条件"]})
    assert result["data"]["comparison_method"] == "llm_grounded"
    first = result["data"]["diagnostics"]["attempts"][0]
    assert "定性" in first["validation_reason"] and "7" in first["model_candidate"]
    assert "尺寸为7" not in result["content"]


def test_qualitative_method_evidence_uses_abstract_body_and_excludes_table_noise(env):
    settings, store, _, _, _, context = env
    for paper in ("a", "b"):
        abstract = Chunk(chunk_id=f"{paper}-abstract", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=1, section="Abstract",
                         text="Abstract\nWe propose a Transformer vision method. It is trained and evaluated on the stated dataset.")
        method = Chunk(chunk_id=f"{paper}-method", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=3, section="Method",
                       text="Method: The architecture uses self-attention to represent visual information and combine image features.")
        table = Chunk(chunk_id=f"{paper}-0-table", document_id=paper, scope="alice", name=f"{paper}.pdf", page_start=2, section="Results",
                      text="Table 1\nModel size 7\nParameter count 28\nAccuracy 81\nResolution 224",
                      sources=[SourceBlock(text="Parameter table with numeric method results", kind="table", page=2)])
        store.replace_chunks(paper, "alice", [table, abstract, method])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "实验条件"]})
    assert all(not citation["chunk_id"].endswith("table") for citation in result["citations"])
    assert {citation["document_id"] for citation in result["citations"]} == {"a", "b"}
    assert any(citation["chunk_id"].endswith("abstract") for citation in result["citations"])
    assert any(citation["chunk_id"].endswith("method") for citation in result["citations"])


def test_grouped_comparison_citations_keep_per_paper_provenance_checks(env):
    settings, _, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", 90
    context.llm = FakeLLM([_comparison_response(first_source="S1, S1")])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert result["data"]["comparison_method"] == "llm_grounded"
    assert "[S1, S1]" not in result["content"]
    context.llm = FakeLLM([_comparison_response(first_source="S1, S2"), _comparison_response(first_source="S1, S2")])
    rejected = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "性能指标"]})
    assert rejected["data"]["comparison_method"] == "evidence_excerpts"
    assert "不属于本列" in rejected["data"]["fallback_reason"]


def test_qualitative_comparison_keeps_source_checked_dataset_identifiers(env):
    settings, store, _, _, _, context = env
    settings.llm_backend, settings.tool_timeout_seconds = "ollama", 90
    chunk = store.list_chunks("alice", ["a"])[0]
    chunk.text = "Evaluated on CIFAR-100. " + chunk.text
    store.replace_chunks("a", "alice", [chunk])
    context.llm = FakeLLM([_qualitative_comparison_response(first="在CIFAR-100进行评估 [S1]。")])
    result = create_registry().get("compare_papers").handler(context, {"document_ids": ["a", "b"], "aspects": ["方法与创新", "实验条件"]})
    assert result["data"]["comparison_method"] == "llm_grounded" and "CIFAR-100" in result["content"]
