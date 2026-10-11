from __future__ import annotations

import hashlib
import json
import re
import time
import uuid

import numpy as np


class SemanticCache:
    def __init__(self, settings, store, embedder):
        self.settings, self.store, self.embedder = settings, store, embedder

    def fingerprint(self, scope, document_ids, mode, history):
        value = {"index": self.store.index_version(scope), "documents": sorted(document_ids or []),
                 "mode": mode, "config": self.settings.public_dict(), "history": history or [], "prompt_version": "4",
                 "embedding_fingerprint": getattr(self.embedder, "fingerprint", "unknown")}
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def eligible(self, query):
        return self.settings.cache_enabled and not re.search(r"今天|现在|时间|最新|它|这篇|上述|之前|联网|\d+\s*[+*/]", query)

    @staticmethod
    def intent_signature(text):
        return tuple(bool(re.search(pattern, text, re.I)) for pattern in
                     [r"不|无|未|没|否|not|never|without", r"为什么|原因|why", r"优点|优势|benefit|advantage",
                      r"缺点|局限|风险|limitation|risk", r"对比|区别|compare", r"作者|author", r"年份|year",
                      r"数据集|dataset", r"方法|method", r"结果|性能|result|performance"])

    def get(self, query, scope, fingerprint):
        if not self.eligible(query):
            return None
        candidates = [c for c in self.store.list("cache", scope, 200)
                      if c["fingerprint"] == fingerprint and time.time()-c["saved_at"] < self.settings.cache_ttl_seconds]
        best, best_similarity = None, self.settings.cache_threshold
        for c in candidates:
            if c["query"].strip() == query.strip():
                return c["result"]
        if not candidates:
            return None
        query_vector = self.embedder.encode([query])[0]
        for c in candidates:
            other = np.asarray(c["vector"])
            norm = np.linalg.norm(other) * np.linalg.norm(query_vector)
            similarity = float(np.dot(other, query_vector)/norm) if norm else 0
            # Same numeric/quoted anchors prevents dangerous near-neighbor reuse.
            anchors = lambda t: set(re.findall(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_-]{2,}", t.lower()))
            if (similarity >= best_similarity and anchors(query) == anchors(c["query"])
                    and self.intent_signature(query) == self.intent_signature(c["query"])):
                best, best_similarity = c["result"], similarity
        return best

    def put(self, query, scope, fingerprint, result):
        if (not self.eligible(query) or result.get("warnings") or result.get("audit", {}).get("invalid_ids")
                or result.get("audit", {}).get("needs_review")):
            return
        vector = self.embedder.encode([query])[0].tolist()
        self.store.put("cache", uuid.uuid4().hex, scope, {"query": query, "fingerprint": fingerprint,
                         "vector": vector, "result": result, "saved_at": time.time()})
