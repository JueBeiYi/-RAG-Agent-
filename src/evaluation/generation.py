from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from statistics import mean

from src.core.schemas import ChatRequest
from .dataset import EvaluationCase
from .metrics import evidence_matches_chunk, map_evidence_to_chunks
from .runner import _fingerprint, _model_identity, package_versions


def capture_llm_identity(settings) -> dict:
    identity = {"backend": settings.llm_backend, "model": settings.llm_model, "digest": None}
    if settings.llm_backend == "extractive":
        identity["implementation"] = "extractive-functional-demo"
    elif settings.llm_backend == "ollama":
        import httpx
        try:
            response = httpx.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=5.0, trust_env=False)
            response.raise_for_status()
            requested = settings.llm_model if ":" in settings.llm_model else settings.llm_model + ":latest"
            model = next((item for item in response.json().get("models", [])
                          if item.get("name", item.get("model")) == requested), None)
            if model:
                identity.update(digest=model.get("digest"), details=model.get("details"), modified_at=model.get("modified_at"))
            else:
                identity["version_note"] = "模型标签未在Ollama本地列表找到，未记录digest"
        except Exception as exc:
            identity["version_note"] = f"无法读取本地模型版本：{type(exc).__name__}"
    else:
        identity["version_note"] = "云端标签可能可变，应保存服务商提供的实际模型版本"
    return identity


def run_generation(application, cases: list[EvaluationCase], scope: str, *, agent: bool = True,
                   mode: str = "hybrid_rerank", repeats: int = 1, clear_cache: bool = True) -> dict:
    """Measure real answer generation and tool routing; leave quality to reviewers.

    Every question/repetition starts a new session to avoid memory contamination.
    Cache is cleared before every case, but repeated copies within that case can
    measure cold/hot differences. Original corpus and source evidence are logged.
    """
    if repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    if not cases:
        raise ValueError("生成评测集为空")
    before = application.store.index_version(scope)
    llm_before = capture_llm_identity(application.settings)
    chunks = {chunk.chunk_id: chunk for chunk in application.store.list_chunks(scope)}
    for case in cases:
        if not case.reviewed or not case.relevant_sources or not case.reference_answer.strip():
            raise ValueError(f"{case.case_id} 尚未人工核对或缺少来源/答案")
        if any(not ids for ids in map_evidence_to_chunks(case.relevant_sources, chunks.values())):
            raise ValueError(f"{case.case_id} 来源证据未映射到当前知识库，请先核对标注")
    records = []
    for case in cases:
        if clear_cache:
            application.store.clear_cache(scope)
        for repeat in range(repeats):
            session = application.store.create_session(scope, f"评测 {case.case_id} · {repeat + 1}")
            started = time.perf_counter()
            events, first_output_ms, first_output_replay = [], None, False
            for event in application.chat(session["id"], scope, ChatRequest(
                    message=case.question, retrieval_mode=mode, agent=agent)):
                events.append(event)
                if first_output_ms is None and event.get("type") == "answer_delta" and event.get("text"):
                    first_output_ms = (time.perf_counter() - started) * 1000
                    first_output_replay = bool(event.get("replay"))
            done = next((event for event in reversed(events) if event.get("type") == "done"), {})
            tools = [event for event in events if event.get("type") == "tool_end"]
            # Agent tool_end contains the result directly or under result.
            invoked = [event.get("tool", event.get("tool_name", "")) for event in tools]
            expected = set(case.expected_tools)
            tool_errors = [event for event in tools if event.get("error") or event.get("result", {}).get("error")]
            citations = done.get("citations", [])
            matched_evidence = sum(any(
                citation.get("chunk_id") in chunks and evidence_matches_chunk(source, chunks[citation["chunk_id"]])
                for citation in citations) for source in case.relevant_sources)
            errors = [event.get("text", event.get("error", "执行失败")) for event in events if event.get("type") == "error"]
            usage = done.get("metrics", {}).get("usage", done.get("metrics", {}))
            records.append({
                "case_id": case.case_id, "question": case.question, "category": case.category, "split": case.split,
                "repeat": repeat + 1, "session_id": session["id"], "request_id": done.get("request_id"),
                "answer": done.get("answer", ""), "reference_answer": case.reference_answer,
                "citations": citations, "events": events, "expected_tools": sorted(expected),
                "invoked_tools": invoked, "tool_call_count": len(tools), "tool_error_count": len(tool_errors),
                "routing_recall": len(expected & set(invoked)) / len(expected) if expected else None,
                "routing_exact_match": float(expected == set(invoked)) if expected else None,
                "citation_evidence_recall": matched_evidence / len(case.relevant_sources),
                "success": bool(done) and not errors, "error": " | ".join(errors),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "first_output_ms": first_output_ms, "first_output_is_result_replay": first_output_replay,
                "review_method": case.review_method, "human_review_pending": case.human_review_pending,
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "metrics": done.get("metrics", {}), "audit": done.get("audit", {}), "origin": done.get("origin"),
                "human_review": {"correctness_1_to_5": None, "completeness_1_to_5": None,
                                 "citation_quality_1_to_5": None, "fluency_1_to_5": None,
                                 "bad_case_type": None, "root_cause": "", "reviewer": ""},
            })
    if application.store.index_version(scope) != before:
        raise ValueError("生成评测期间语料发生变更，结果作废")
    llm_after = capture_llm_identity(application.settings)
    if llm_before.get("digest") and llm_after.get("digest") and llm_before["digest"] != llm_after["digest"]:
        raise ValueError("生成评测期间Ollama模型digest发生变更，结果作废")
    config = application.settings.public_dict()
    tool_total = sum(row["tool_call_count"] for row in records)
    routing = [row["routing_recall"] for row in records if row["routing_recall"] is not None]
    exact_routing = [row["routing_exact_match"] for row in records if row["routing_exact_match"] is not None]
    import numpy as np
    latency = [row["elapsed_ms"] for row in records]
    corpus = [{"document_id": d.document_id, "content_hash": d.content_hash, "name": d.name}
              for d in application.store.list_documents(scope)]
    return {
        "experiment_id": str(uuid.uuid4()), "kind": "generation", "mode": mode, "agent": agent,
        "created_at": datetime.now(timezone.utc).isoformat(), "split": sorted({case.split for case in cases}),
        "demo": application.settings.llm_backend == "extractive" or application.settings.embedding_backend == "hash",
        "formal_models": application.settings.llm_backend != "extractive" and application.settings.embedding_backend != "hash",
        "clear_cache_before_case": clear_cache, "config": config,
        "config_hash": _fingerprint(config), "dataset_hash": _fingerprint([case.model_dump() for case in cases]),
        "corpus_hash": _fingerprint(sorted(corpus, key=lambda d: d["document_id"])),
        "index_version": before, "package_versions": package_versions(), "repeats": repeats,
        "models": {**_model_identity(application.retriever, application.settings), "llm_actual": llm_after},
        "metrics": {"case_count": len(cases), "run_count": len(records),
                    "answer_success_rate": mean(float(row["success"]) for row in records) if records else 0,
                    "tool_success_rate": 1 - sum(row["tool_error_count"] for row in records) / tool_total if tool_total else None,
                    "routing_recall": mean(routing) if routing else None,
                    "routing_exact_match": mean(exact_routing) if exact_routing else None,
                    "citation_evidence_recall": mean(row["citation_evidence_recall"] for row in records) if records else 0,
                    "latency_mean_ms": mean(latency) if latency else 0,
                    "latency_p50_ms": float(np.percentile(latency, 50)) if latency else 0,
                    "latency_p95_ms": float(np.percentile(latency, 95)) if latency else 0},
        "records": records,
        "notes": ["工具路由只对有 expected_tools 的题统计。额外工具调用不必然是错误，需人工判定。",
                  "来源覆盖率不等于答案正确率；正确性、完整性、引用质量与流畅性保留人工评分。",
                  "Token 缺失保持 null，不补造统计；repeat=1 为冷缓存，后续为热缓存候选。"],
    } | {"metrics": summarise_generation(records), "review_status": {
            "source_checked_cases": sum(case.reviewed for case in cases),
            "human_source_reviewed_cases": sum(case.review_method == "human" and not case.human_review_pending for case in cases),
            "human_answer_review_pending": any(not row["human_review"].get("reviewer") for row in records)}}


def summarise_generation(records: list[dict]) -> dict:
    """Aggregate known measurements only; missing/provider-unknown values stay null."""
    import numpy as np
    def known(values):
        return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    latency = known([r.get("elapsed_ms") for r in records])
    tokens_in = known([r.get("prompt_tokens") for r in records])
    tokens_out = known([r.get("completion_tokens") for r in records])
    rounds = known([r.get("metrics", {}).get("agent_steps") for r in records])
    tool_total = sum(r.get("tool_call_count", 0) for r in records)
    result = {"case_count": len({r.get("case_id") for r in records}), "run_count": len(records),
              "answer_success_rate": mean(bool(r.get("success")) for r in records) if records else None,
              "tool_success_rate": 1-sum(r.get("tool_error_count", 0) for r in records)/tool_total if tool_total else None,
              "average_agent_steps": mean(rounds) if rounds else None, "agent_steps_known_count": len(rounds),
              "prompt_tokens_total": sum(tokens_in) if tokens_in else None,
              "completion_tokens_total": sum(tokens_out) if tokens_out else None,
              "prompt_tokens_mean": mean(tokens_in) if tokens_in else None,
              "completion_tokens_mean": mean(tokens_out) if tokens_out else None,
              "token_usage_known_count": min(len(tokens_in), len(tokens_out)),
              "latency_mean_ms": mean(latency) if latency else None,
              "latency_p50_ms": float(np.percentile(latency, 50)) if latency else None,
              "latency_p95_ms": float(np.percentile(latency, 95)) if latency else None}
    for key in ("routing_recall", "routing_exact_match", "citation_evidence_recall", "first_output_ms"):
        values = known([r.get(key) for r in records])
        result[key + "_mean" if key == "first_output_ms" else key] = mean(values) if values else None
    for key in ("correctness_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5", "fluency_1_to_5"):
        values = known([r.get("human_review", {}).get(key) for r in records if r.get("human_review", {}).get("reviewer")])
        result["human_" + key + "_mean"] = mean(values) if values else None
        result["human_" + key + "_count"] = len(values)
    return result
