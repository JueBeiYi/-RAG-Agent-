from __future__ import annotations

import time
from src.core.schemas import event
from src.generation.cache import SemanticCache
from src.generation.context import SYSTEM_PROMPT, audit_citations, build_context, estimate_tokens, normalise_page_references


class RAGEngine:
    def __init__(self, settings, store, retriever, llm):
        self.settings, self.store, self.retriever, self.llm = settings, store, retriever, llm
        self.cache = SemanticCache(settings, store, retriever.embedder)

    def retrieve(self, query, scope, document_ids=None, mode=None):
        if document_ids and len(document_ids) > 1:
            hits = []
            for document_id in document_ids:
                hits.extend(self.retriever.search(query, scope, [document_id], mode))
            return hits
        return self.retriever.search(query, scope, document_ids, mode)

    def ask(self, query, scope, document_ids=None, mode=None, history=None):
        result = None
        for item in self.stream(query, scope, document_ids, mode, history):
            if item["type"] == "done":
                result = {k: v for k, v in item.items() if k != "type"}
        if result is None:
            raise RuntimeError("RAG 未产生完整结果")
        return result

    def stream(self, query, scope, document_ids=None, mode=None, history=None):
        started, first_token = time.perf_counter(), None
        snapshot_version = self.store.index_version(scope)
        fingerprint = self.cache.fingerprint(scope, document_ids, mode, history)
        cache_warnings = []
        try:
            cached = self.cache.get(query, scope, fingerprint)
        except Exception as exc:
            cached = None
            cache_warnings.append(f"缓存读取失败，继续正常检索 ({type(exc).__name__})")
        if self.store.index_version(scope) != snapshot_version:
            cached = None
        if cached:
            yield event("status", text="已命中当前索引版本的语义缓存")
            for citation in cached.get("citations", []):
                yield event("citation", citation=citation)
            yield event("answer_delta", text=cached["answer"])
            result = {**cached, "metrics": {**cached.get("metrics", {}), "cache_hit": True,
                      "input_tokens": 0, "output_tokens": 0, "elapsed_ms": (time.perf_counter()-started)*1000}}
            yield event("done", **result)
            return
        yield event("status", text="正在检索论文证据")
        retrieval_start = time.perf_counter()
        for attempt in range(3):
            snapshot_version = self.store.index_version(scope)
            hits = self.retrieve(query, scope, document_ids, mode)
            if self.store.index_version(scope) == snapshot_version:
                break
        else:
            raise ValueError("知识库正在持续更新，请等待导入完成后重试")
        fingerprint = self.cache.fingerprint(scope, document_ids, mode, history)
        retrieval_ms = (time.perf_counter()-retrieval_start)*1000
        warnings = cache_warnings
        search_metadata = dict(getattr(self.retriever, "last_search", {}))
        if search_metadata.get("warning"):
            warnings.append(search_metadata["warning"])
        history = [{"role": m["role"], "content": m["content"]} for m in (history or []) if m.get("role") in {"user", "assistant"}]
        while history and sum(estimate_tokens(m["content"]) for m in history) > self.settings.memory_tokens:
            history.pop(0)
        prompt_cost = estimate_tokens(SYSTEM_PROMPT + query) + sum(estimate_tokens(m["content"]) for m in history)
        budget = self.settings.context_tokens - self.settings.answer_tokens - prompt_cost - 150
        if budget < 100:
            raise ValueError("问题或对话超过模型上下文预算，请缩短问题或新建会话")
        context, citations = build_context(hits, budget)
        if not citations:
            warnings.append("当前知识库中未找到相关文档；以下是普通模型回答，不具有知识库证据")
            user = "当前知识库中未找到相关文档。可以解释一般知识，但不要猜测特定论文的数字或结论。问题：" + query
            origin = "model_fallback"
        else:
            origin = "knowledge_rag"
            if max((h.score for h in hits), default=0) < self.settings.minimum_relevance:
                warnings.append("候选证据相关度较低，请先核查来源；无法确认的内容应保留不确定性")
            user = "证据（来源编号只能从这里选）：\n" + context + "\n\n用户问题：" + query
        if self.settings.llm_backend == "extractive":
            origin = "demo_extractive"
        for warning in warnings:
            yield event("status", text=warning, warning=True)
        citation_dicts = [c.model_dump() for c in citations]
        for citation in citation_dicts:
            yield event("citation", citation=citation)
        yield event("status", text="正在依据证据生成回答")
        answer, usage = "", {"input_tokens": 0, "output_tokens": 0, "estimated": False}
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user}]
        for item in self.llm.stream(messages):
            if item["type"] == "answer_delta":
                if first_token is None:
                    first_token = time.perf_counter()
                answer += item["text"]
            elif item["type"] == "usage":
                usage = {k: item[k] for k in usage if k in item}
            elif item.get("warning"):
                warnings.append(item.get("text", "模型回答可能不完整"))
            yield item
        answer, page_corrections = normalise_page_references(answer, citation_dicts)
        audit = audit_citations(answer, citation_dicts)
        audit["page_reference_corrections"] = page_corrections
        if audit["invalid_ids"]:
            warnings.append("模型引用了不存在的编号，已标为来源待核实")
            for source in audit["invalid_ids"]:
                answer = answer.replace(f"[{source}]", "〔来源待核实〕")
        if warnings and not citations:
            answer = "当前知识库中未找到相关文档；以下是普通模型回答，不具有知识库证据。\n\n" + answer
        metrics = {**usage, "retrieval_ms": retrieval_ms, "elapsed_ms": (time.perf_counter()-started)*1000,
                   "first_token_ms": (first_token-started)*1000 if first_token else None,
                   "cache_hit": False, "retrieved_count": len(hits), "top1_score": hits[0].score if hits else None,
                   "index_version": snapshot_version, "model": self.settings.llm_model,
                   "demo": self.settings.llm_backend == "extractive"}
        metrics["retrieval_stages"] = {k: v for k, v in search_metadata.items() if k != "scope"}
        result = {"answer": answer, "citations": citation_dicts, "metrics": metrics,
                  "hits": [h.model_dump() for h in hits], "warnings": warnings, "origin": origin, "audit": audit}
        if self.store.index_version(scope) != snapshot_version:
            warnings.append("回答生成期间知识库已变更，本回答使用检索时的证据快照，请核查当前原件")
            current_ids = {c.chunk_id for c in self.store.list_chunks(scope)}
            for citation in citation_dicts:
                citation["verified"] = citation["chunk_id"] in current_ids
        elif citations:
            try:
                self.cache.put(query, scope, fingerprint, result)
            except Exception as exc:
                warnings.append(f"缓存写入失败，回答仍已完成 ({type(exc).__name__})")
        yield event("done", **result)
