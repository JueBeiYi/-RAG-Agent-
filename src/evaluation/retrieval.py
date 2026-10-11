"""Staged, reusable retrieval experiments over a frozen real corpus.

Encoding, database/ranking, and reranking are separately timed. Reusing vectors
does not masquerade as fresh online inference. Persisted reports never include
workspace credentials. Model stages run in separate serial processes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.config import Settings
from src.core.schemas import Chunk, Document, RetrievalHit, SourceBlock
from src.core.storage import Storage
from src.evaluation.dataset import EvaluationCase, load_cases
from src.evaluation.metrics import map_evidence_to_chunks, normalise_quote, retrieval_metrics, summarise_records
from src.evaluation.runner import MATCHING_RULE, export_result, package_versions
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.tokenization import tokenize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "experiments" / "retrieval"
DEFAULT_CACHE = ROOT / "data" / "runtime" / "benchmarks" / "retrieval-cache"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item.model_dump() if hasattr(item, "model_dump") else item,
                                         ensure_ascii=False) for item in records) + "\n", encoding="utf-8")


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for item in records for key in item})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in item.items()} for item in records)


def sanitise_chunk(chunk):
    # Source IDs and page coordinates are needed by generation replay, the
    # workspace scope string is a credential and must not enter reports/caches.
    return chunk.model_copy(update={"scope": "benchmark"})


def read_snapshot(cache):
    cache = Path(cache)
    snapshot = json.loads((cache / "snapshot.json").read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate_json(line) for line in (cache / "standard_chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    cases = load_cases(cache / "cases.jsonl")
    return snapshot, chunks, cases


def snapshot_corpus(settings: Settings, scope: str, cases_path: Path, cache=DEFAULT_CACHE):
    from src.retrieval.embeddings import Embedder
    from src.retrieval.vectorstores import ChromaVectors, scoped_id

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source = Storage(settings.path)
    docs = source.list_documents(scope)
    chunks = source.list_chunks(scope)
    cases = load_cases(cases_path)
    if len(docs) != 10 or any(doc.status != "ready" for doc in docs) or not chunks:
        raise ValueError("本次基准需要已就绪的10篇真实论文与完整持久化分块")
    if len(cases) != 80 or Counter(case.split for case in cases) != {"dev": 20, "test": 60}:
        raise ValueError("本次固定评测集必须80题(20dev/60test)")
    evidence = [map_evidence_to_chunks(case.relevant_sources, chunks) for case in cases]
    if any(not matches for group in evidence for matches in group):
        raise ValueError("源知识库存在未映射证据，请先validate-index")
    index_before = source.index_version(scope)
    embedder = Embedder(settings)
    vectors = ChromaVectors(settings.path / "indices", embedder.fingerprint)
    collection = vectors._load()
    found = collection.get(ids=[scoped_id(scope, chunk.chunk_id) for chunk in chunks], include=["embeddings", "metadatas"])
    by_id = {meta["chunk_id"]: np.asarray(vector, dtype=np.float32)
             for meta, vector in zip(found["metadatas"], found["embeddings"])
             if meta["scope"] == scope}
    if any(chunk.chunk_id not in by_id for chunk in chunks):
        raise ValueError("生产Chroma中缺少当前Embedding fingerprint对应的向量")
    matrix = np.stack([by_id[chunk.chunk_id] for chunk in chunks])
    if source.index_version(scope) != index_before:
        raise ValueError("快照过程中索引发生变化，请重跑")
    corpus = [{"document_id": doc.document_id, "name": doc.name, "content_hash": doc.content_hash,
               "path": doc.path, "title": doc.title, "authors": doc.authors, "year": doc.year} for doc in docs]
    code_hashes = {name: file_hash(ROOT / name) for name in
                   ("src/retrieval/embeddings.py", "src/retrieval/engine.py", "src/retrieval/rrf.py", "src/evaluation/metrics.py")}
    snapshot = {"created_at": timestamp(), "scope_fingerprint": hashlib.sha256(scope.encode()).hexdigest()[:24],
                "index_version": index_before, "corpus": corpus,
                "corpus_hash": fingerprint(sorted([{k: d[k] for k in ("document_id", "name", "content_hash")} for d in corpus], key=lambda d: d["document_id"])),
                "dataset_hash": fingerprint([case.model_dump() for case in cases]),
                "cases_file_sha256": file_hash(cases_path), "standard_chunk_hash": fingerprint([sanitise_chunk(c).model_dump() for c in chunks]),
                "standard_chunk_count": len(chunks), "embedding_fingerprint": embedder.fingerprint,
                "embedding_artifact_revision": embedder.revision,
                "settings": settings.public_dict(), "implementation_hashes": code_hashes,
                "review_method": "assistant_source_checked", "human_review_pending": True,
                "retrieval_label_limit": "Short page-local literal anchors may have several matching chunks; these are source-checked draft labels, not human consensus."}
    save_json(cache / "snapshot.json", snapshot)
    save_jsonl(cache / "standard_chunks.jsonl", [sanitise_chunk(c) for c in chunks])
    save_jsonl(cache / "cases.jsonl", cases)
    np.savez_compressed(cache / "standard_m3_documents.npz", vectors=matrix,
                        ids=np.asarray([chunk.chunk_id for chunk in chunks]), fingerprint=embedder.fingerprint)
    print(f"Frozen {len(docs)} real papers, {len(chunks)} persistent chunks, {len(cases)} cases; credentials excluded", flush=True)
    return snapshot


def parser_comparison(cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    from src.data_loader.loaders import load_blocks
    import pymupdf

    cache, output = Path(cache), Path(output) / "parsers"
    snapshot, _, cases = read_snapshot(cache)
    records = []
    for parser in ("pymupdf", "pdfplumber", "pypdf2"):
        all_blocks, parser_start = {}, time.perf_counter()
        for doc in snapshot["corpus"]:
            start = time.perf_counter()
            blocks = load_blocks(doc["path"], parser)
            elapsed = (time.perf_counter() - start) * 1000
            all_blocks[doc["document_id"]] = blocks
            sources = [s for case in cases for s in case.relevant_sources if s.document_id == doc["document_id"]]
            matched = sum(any(block.page == source.page and normalise_quote(source.quote) in normalise_quote(block.text)
                              for block in blocks) for source in sources)
            with pymupdf.open(doc["path"]) as pdf:
                physical_pages = len(pdf)
            row = {"parser": parser, "document_id": doc["document_id"], "name": doc["name"],
                   "content_hash": doc["content_hash"], "parse_ms": round(elapsed, 3),
                   "physical_pages": physical_pages, "pages_with_text": len({b.page for b in blocks}),
                   "block_count": len(blocks), "text_characters": sum(len(b.text) for b in blocks),
                   "table_blocks": sum(b.kind == "table" for b in blocks),
                   "formula_blocks": sum(b.kind == "formula" for b in blocks),
                   "blocks_with_bbox": sum(b.bbox is not None for b in blocks),
                   "source_anchor_count": len(sources), "source_anchors_retained": matched,
                   "anchor_retention": matched / len(sources) if sources else None}
            records.append(row)
            print(f"{parser}: {doc['name']}; {elapsed:.0f}ms; anchors {matched}/{len(sources)}", flush=True)
        save_json(cache / f"blocks_{parser}.json", {key: [b.model_dump() for b in blocks] for key, blocks in all_blocks.items()})
        print(f"Parser {parser} complete: {time.perf_counter()-parser_start:.1f}s", flush=True)
    summary = []
    for parser in ("pymupdf", "pdfplumber", "pypdf2"):
        rows = [r for r in records if r["parser"] == parser]
        total, matched = sum(r["source_anchor_count"] for r in rows), sum(r["source_anchors_retained"] for r in rows)
        summary.append({"parser": parser, "paper_count": len(rows), "parse_ms": sum(r["parse_ms"] for r in rows),
                        "anchor_retention": matched / total, "matched_anchors": matched, "total_anchors": total,
                        "table_blocks": sum(r["table_blocks"] for r in rows), "formula_blocks": sum(r["formula_blocks"] for r in rows),
                        "pages_with_text": sum(r["pages_with_text"] for r in rows), "bbox_blocks": sum(r["blocks_with_bbox"] for r in rows)})
    result = {"kind": "pdf_parser_comparison", "created_at": timestamp(), "status": "completed",
              "demo": False, "formal_models": False, "language_model_called": False,
              "corpus_hash": snapshot["corpus_hash"], "dataset_hash": snapshot["dataset_hash"],
              "metrics": {"paper_count": 10, "parser_count": 3, "annotated_evidence_count": sum(len(c.relevant_sources) for c in cases)},
              "summary": summary, "records": records, "package_versions": package_versions(),
              "parser_implementation_sha256": file_hash(ROOT / "src/data_loader/loaders.py"),
              "notes": ["Each original PDF parsed once per backend. Timing includes backend import for its first document.",
                        "Anchor retention/page coverage/bounding boxes are measurable proxies, not a universal parser accuracy score.",
                        "A detected table/formula count is not proof of correct table reading or mathematical meaning.",
                        "Source-checked anchors retain human_review_pending=true."]}
    save_json(output / "results.json", result)
    write_csv(output / "records.csv", records)
    write_csv(output / "comparison.csv", summary)
    return result


def chunk_variants(cache=DEFAULT_CACHE):
    from src.chunking.strategies import split_blocks

    cache = Path(cache)
    snapshot, _, _ = read_snapshot(cache)
    raw = json.loads((cache / "blocks_pymupdf.json").read_text(encoding="utf-8"))
    variants = [("fixed", 256), ("fixed", 512), ("fixed", 1024), ("recursive", 512), ("semantic", 512)]
    manifest = []
    source_hash = file_hash(ROOT / "src/chunking/strategies.py")
    for strategy, size in variants:
        started = time.perf_counter()
        chunks = []
        for doc in snapshot["corpus"]:
            document = Document(document_id=doc["document_id"], name=doc["name"], content_hash=doc["content_hash"],
                                title=doc["title"], format="pdf", status="ready", scope="benchmark")
            blocks = [SourceBlock.model_validate(block) for block in raw[doc["document_id"]]]
            chunks.extend(split_blocks(document, blocks, strategy=strategy, size=size, overlap=64))
        key = f"{strategy}-{size}"
        save_jsonl(cache / "variants" / (key + ".jsonl"), chunks)
        manifest.append({"variant": key, "strategy": strategy, "chunk_size": size, "chunk_overlap": 64,
                         "chunk_count": len(chunks), "split_ms": (time.perf_counter()-started)*1000,
                         "chunks_hash": fingerprint([c.model_dump() for c in chunks]),
                         "implementation_sha256": source_hash})
        print(f"Frozen chunk variant {key}: {len(chunks)} chunks", flush=True)
    if file_hash(ROOT / "src/chunking/strategies.py") != source_hash:
        raise ValueError("分块源码在变体准备期间改变，请冻结后重跑")
    save_json(cache / "chunk_variants.json", manifest)
    return manifest


def get_variant(cache, name):
    return [Chunk.model_validate_json(line) for line in (Path(cache) / "variants" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if line]


def encode_token_windows(embedder, texts):
    """Exact original tokenizer IDs avoid lossy decode/re-encode at word edges.

    This benchmark policy is independently fingerprinted. Every token, including
    UNK IDs, is passed to the same SentenceTransformer modules; no byte/word
    boundary can silently lose text or exceed the model's window limit.
    """
    model = embedder._load()
    tokenizer = model.tokenizer
    limit = min(int(model.max_seq_length or tokenizer.model_max_length), embedder.settings.embedding_max_tokens)
    from src.retrieval.token_windows import single_windows, sentence_vectors
    windows = single_windows(tokenizer, texts, limit)
    embedder.long_input_count += windows.long_input_count
    embedder.last_window_count = len(windows.features)
    return sentence_vectors(model, windows, embedder.settings.embedding_batch_size)


def encode_queries(settings: Settings, cache=DEFAULT_CACHE):
    from src.retrieval.embeddings import Embedder

    cache = Path(cache)
    snapshot, _, cases = read_snapshot(cache)
    embedder = Embedder(settings)
    model_start = time.perf_counter()
    embedder._load()
    load_ms = (time.perf_counter()-model_start)*1000
    started = time.perf_counter()
    vectors = embedder.encode([case.question for case in cases])
    query_ms = (time.perf_counter()-started)*1000
    np.savez_compressed(cache / "standard_m3_queries.npz", vectors=vectors, ids=np.asarray([c.case_id for c in cases]))
    info = {"created_at": timestamp(), "model": settings.embedding_model, "fingerprint": embedder.fingerprint,
            "artifact_revision": embedder.revision, "model_load_ms": load_ms, "query_encoding_total_ms": query_ms,
            "query_count": len(cases), "encoding_batch_size": settings.embedding_batch_size,
            "cpu_threads": settings.cpu_threads, "encoding_window_tokens": settings.embedding_max_tokens,
            "complete_window_count": embedder.last_window_count, "dimension": vectors.shape[1],
            "dataset_hash": snapshot["dataset_hash"], "query_encoding_amortized_ms": query_ms / len(cases),
            "amortized_time_is_not_individual_online_latency": True}
    save_json(cache / "standard_m3_query_encoding.json", info)
    print(f"M3 queries encoded once: {len(cases)}, {query_ms/1000:.1f}s; dim={vectors.shape[1]}", flush=True)
    return info


def lexical_model(chunks):
    from rank_bm25 import BM25Okapi
    import jieba
    import logging
    jieba.setLogLevel(logging.WARNING)
    documents = [tokenize(chunk.text) or ["__empty__"] for chunk in chunks]
    return BM25Okapi(documents), documents


def lexical_rank(query, chunks, bm25, documents, candidate_k=20):
    terms = set(tokenize(query))
    if not terms:
        return [], {}
    raw = bm25.get_scores(tokenize(query))
    coverage = {chunk.chunk_id: len(terms & set(documents[i])) / len(terms) for i, chunk in enumerate(chunks)}
    order = sorted((i for i, c in enumerate(chunks) if coverage[c.chunk_id] > 0),
                   key=lambda i: (-float(raw[i]), -coverage[chunks[i].chunk_id], chunks[i].chunk_id))[:candidate_k]
    return [(chunks[i].chunk_id, float(raw[i])) for i in order], coverage


def fuse_hits(chunks, vector, keyword, coverage, mode="hybrid", candidate_k=20, rrf_k=60):
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    vector_scores, keyword_scores = dict(vector), dict(keyword)
    if mode == "vector":
        keyword_scores = {}
    vector_ranks = {key: rank for rank, (key, _) in enumerate(vector, 1)}
    keyword_ranks = {key: rank for rank, (key, _) in enumerate(keyword, 1)} if mode != "vector" else {}
    fused = reciprocal_rank_fusion({"vector": list(vector_scores), "bm25": list(keyword_scores)}, rrf_k)
    rrf_scores = dict(fused)
    order = list(vector_scores) if mode == "vector" else [item[0] for item in fused]
    hits = []
    for identifier in order[:candidate_k]:
        v = vector_scores.get(identifier)
        relevance = max(0.0, min(1.0, v or 0.0))
        if mode != "vector":
            relevance = max(relevance, coverage.get(identifier, 0.0))
        ranks = {}
        if identifier in vector_ranks:
            ranks["vector"] = vector_ranks[identifier]
        if identifier in keyword_ranks:
            ranks["bm25"] = keyword_ranks[identifier]
        hits.append(RetrievalHit(chunk=by_id[identifier], score=relevance, vector_score=v,
                                 bm25_score=keyword_scores.get(identifier),
                                 rrf_score=rrf_scores.get(identifier) if mode != "vector" else None, ranks=ranks))
    return hits


def exact_vector_rank(chunks, vectors, query, candidate_k=20):
    scores = vectors @ query
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id))[:candidate_k]
    return [(chunks[i].chunk_id, float(scores[i])) for i in order]


def case_record(case, hits, chunks, elapsed_ms, extra=None):
    mapping = map_evidence_to_chunks(case.relevant_sources, chunks)
    if any(not item for item in mapping):
        return {"case_id": case.case_id, "question": case.question, "category": case.category, "split": case.split,
                "reference_answer": case.reference_answer, "error": "Evidence not mapped to this variant",
                "hit_at_5": 0., "mrr": 0., "recall_at_5": 0., "retrieved": [], "elapsed_ms": elapsed_ms}
    raw = []
    for hit in hits:
        payload = hit.model_dump()
        payload["chunk"]["scope"] = ""
        raw.append(payload)
    identifiers = [hit.chunk.chunk_id for hit in hits]
    return {"case_id": case.case_id, "question": case.question, "category": case.category, "split": case.split,
            "reference_answer": case.reference_answer,
            "relevant_sources": [source.model_dump() for source in case.relevant_sources],
            "expected_tools": case.expected_tools, "review_method": case.review_method,
            "human_review_pending": case.human_review_pending, "error": "",
            **retrieval_metrics(identifiers, mapping, k=5), "retrieved": raw,
            "retrieved_chunk_ids": identifiers, "matching_chunk_ids": [sorted(ids) for ids in mapping],
            "top1_score": hits[0].score if hits else None, "elapsed_ms": elapsed_ms, **(extra or {})}


def result_bundle(snapshot, chunks, cases, records, mode, model, settings, notes=None):
    effective_config = {**settings.public_dict(), "retrieval_mode": mode}
    return {"experiment_id": fingerprint({"corpus": snapshot["corpus_hash"], "cases": [c.case_id for c in cases],
                                         "mode": mode, "model": model, "chunks": [c.chunk_id for c in chunks]})[:32],
            "kind": "retrieval", "mode": mode, "status": "completed", "created_at": timestamp(),
            "demo": False, "formal_models": True, "split": sorted({case.split for case in cases}),
            "config": effective_config, "config_hash": fingerprint(effective_config),
            "dataset_hash": fingerprint([case.model_dump() for case in cases]), "corpus_hash": snapshot["corpus_hash"],
            "corpus": [{key: doc[key] for key in ("document_id", "name", "content_hash")} for doc in snapshot["corpus"]],
            "index_version": snapshot["index_version"], "chunk_snapshot_hash": fingerprint([c.model_dump() for c in chunks]),
            "models": model, "package_versions": package_versions(), "matching_rule": MATCHING_RULE,
            "metrics": summarise_records(records), "records": records,
            "dataset_requirement_checks": {"document_count": len(snapshot["corpus"]), "document_count_10_to_20": True,
                                           "at_least_50_reviewed_cases": len(cases) >= 50,
                                           "all_four_categories": len({c.category for c in cases}) == 4,
                                           "human_corpus_confirmation": "assistant_source_checked_human_pending"},
            "review_method": "assistant_source_checked", "human_review_pending": True,
            "llm_called": False, "notes": [snapshot["retrieval_label_limit"], *(notes or [])]}


def embedding_study(settings, model_key, cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    from src.retrieval.embeddings import Embedder

    cache, output = Path(cache), Path(output)
    snapshot, _, all_cases = read_snapshot(cache)
    cases = [case for case in all_cases if case.split == "dev"]
    manifest = json.loads((cache / "chunk_variants.json").read_text(encoding="utf-8"))
    variants = manifest if model_key == "m3e" else [item for item in manifest if item["variant"] == "recursive-512"]
    chunks_by_variant = {item["variant"]: get_variant(cache, item["variant"]) for item in variants}
    unique_texts = sorted({chunk.text for chunks in chunks_by_variant.values() for chunk in chunks})
    # A shared full-text embedding is safe: encoding depends on text, never page,
    # scope or chunk ID. Overlapping but different texts are still encoded anew.
    embedder = Embedder(settings)
    started = time.perf_counter()
    embedder._load()
    model_load_ms = (time.perf_counter()-started)*1000
    # One meaningful adapter check: complete token features must reproduce short
    # normal encode vectors, before we rely on them for longer pooled inputs.
    check_texts = ["视觉Transformer论文分类实验。", "Vision Transformer and image classification."]
    reference = np.asarray(embedder.model.encode(check_texts, prompt="", batch_size=1,
                                               normalize_embeddings=True, convert_to_numpy=True,
                                               show_progress_bar=False), dtype=np.float32)
    direct = encode_token_windows(embedder, check_texts)
    deviation = float(np.max(np.abs(reference-direct)))
    if deviation > 1e-4:
        raise ValueError(f"直接token窗口与标准短文本encode不一致: {deviation}")
    vectors, measurements = [], []
    group_size = 64  # orchestration only; actual model encoding batch remains 1.
    checkpoint_path = cache / (model_key + "_encoding_checkpoint.npz")
    checkpoint_meta = cache / (model_key + "_encoding_checkpoint.json")
    cache_identity = fingerprint({"fingerprint": embedder.fingerprint, "texts": [hashlib.sha256(t.encode()).hexdigest() for t in unique_texts],
                                  "helper": file_hash(ROOT / "src/retrieval/token_windows.py")})
    completed = 0
    if checkpoint_path.exists() and checkpoint_meta.exists():
        previous = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if previous.get("cache_identity") == cache_identity:
            loaded = np.load(checkpoint_path)["vectors"]
            completed = len(loaded)
            vectors.append(loaded)
            measurements = previous["measurements"]
            print(f"{model_key}: validated prior partial encoding reused {completed}/{len(unique_texts)}", flush=True)
    encode_start = time.perf_counter()
    for offset in range(completed, len(unique_texts), group_size):
        group = unique_texts[offset:offset + group_size]
        started = time.perf_counter()
        vectors.append(encode_token_windows(embedder, group))
        measurements.append({"text_count": len(group), "window_count": embedder.last_window_count,
                             "elapsed_ms": (time.perf_counter()-started)*1000, "measured_at": timestamp()})
        np.savez_compressed(checkpoint_path, vectors=np.concatenate(vectors))
        save_json(checkpoint_meta, {"cache_identity": cache_identity, "completed": min(offset+group_size,len(unique_texts)),
                                   "total": len(unique_texts), "measurements": measurements})
        print(f"{model_key}: actual document texts {min(offset+group_size,len(unique_texts))}/{len(unique_texts)}", flush=True)
    document_ms = sum(row["elapsed_ms"] for row in measurements)
    matrix = np.concatenate(vectors)
    text_rows = {text: i for i, text in enumerate(unique_texts)}
    queries = [case.question for case in cases]
    if "bge-large-zh" in settings.embedding_model.lower():
        queries = ["为这个句子生成表示以用于检索相关文章：" + query for query in queries]
    started = time.perf_counter()
    query_vectors = encode_token_windows(embedder, queries)
    query_ms = (time.perf_counter()-started)*1000
    np.savez_compressed(cache / (model_key + "_document_vectors.npz"), vectors=matrix,
                        text_hashes=np.asarray([hashlib.sha256(text.encode()).hexdigest() for text in unique_texts]))
    np.savez_compressed(cache / (model_key + "_dev_queries.npz"), vectors=query_vectors,
                        ids=np.asarray([case.case_id for case in cases]))
    identity = {"embedding_model": settings.embedding_model, "artifact_revision": embedder.revision,
                "embedding_fingerprint": embedder.fingerprint,
                "product_embedding_fingerprint": embedder.fingerprint, "embedding_backend": "sentence_transformers",
                "dimension": matrix.shape[1], "cpu_threads": settings.cpu_threads,
                "model_batch_size": 1, "window_token_cap": 512, "complete_windows_preserve_tail": True,
                "window_policy": "original-token-id-windows-v3 shared production helper; normalized weighted mean",
                "short_encode_max_abs_deviation": deviation, "short_adapter_check_count": 2}
    measurement = {"created_at": timestamp(), **identity, "model_load_ms": model_load_ms,
                   "unique_document_text_count": len(unique_texts), "document_encoding_total_ms": document_ms,
                   "document_texts_per_second": len(unique_texts) / (document_ms / 1000),
                   "query_count": len(cases), "query_encoding_total_ms": query_ms,
                   "query_encoding_amortized_ms": query_ms / len(cases), "document_batches": measurements,
                   "checkpoint_texts_reused": completed,
                   "document_speed_comparison_warning": "Models encode different total variant text counts; compare their identical20-query timing, not bulk doc TPS as a fair same-workload rank.",
                   "corpus_hash": snapshot["corpus_hash"], "dataset_hash": fingerprint([case.model_dump() for case in cases]),
                   "variant_names": list(chunks_by_variant), "variant_definitions": variants}
    save_json(output / "encoding" / model_key / "results.json", measurement)
    baseline_rows = []
    for variant, chunks in chunks_by_variant.items():
        doc_vectors = matrix[[text_rows[chunk.text] for chunk in chunks]]
        bm25, token_documents = lexical_model(chunks)
        rows = []
        for case, query_vector in zip(cases, query_vectors):
            started = time.perf_counter()
            vector = exact_vector_rank(chunks, doc_vectors, query_vector)
            keyword, coverage = lexical_rank(case.question, chunks, bm25, token_documents)
            hits = fuse_hits(chunks, vector, keyword, coverage)
            elapsed = (time.perf_counter()-started)*1000
            rows.append(case_record(case, hits, chunks, elapsed,
                                    {"latency_scope": "ranking_only_precomputed_vectors",
                                     "amortized_query_encoding_ms": query_ms / len(cases), "query_vector_reused": True}))
        variant_definition = next(item for item in variants if item["variant"] == variant)
        variant_settings = replace(settings, chunk_strategy=variant_definition["strategy"],
                                   chunk_size=variant_definition["chunk_size"], chunk_overlap=variant_definition["chunk_overlap"])
        result = result_bundle(snapshot, chunks, cases, rows, "hybrid", identity, variant_settings,
                               ["Dev-only fixed comparison; actual unique texts encoded once and reused across chunk variants.",
                                "elapsed_ms measures only exact vector+BM25+handwritten RRF ranking; shared encoding is reported separately."])
        result["encoding_measurements"] = measurement
        result["chunking_variant"] = variant_definition
        export_result(result, output / "chunking" / model_key / variant)
        baseline_rows.append({"model": settings.embedding_model, "variant": variant,
                              "query_count": len(cases), **{k: v for k, v in result["metrics"].items() if not isinstance(v, dict)}})
        print(f"{model_key}/{variant}: Hit@5={result['metrics']['hit_at_5']:.3f}; Recall@5={result['metrics']['recall_at_5']:.3f}", flush=True)
    save_json(output / "encoding" / model_key / "comparison.json", baseline_rows)
    return measurement


def vectorstore_study(settings, cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    from src.retrieval.vectorstores import ChromaVectors, FaissVectors

    cache, output = Path(cache), Path(output) / "vectorstores"
    snapshot, chunks, all_cases = read_snapshot(cache)
    cases = [case for case in all_cases if case.split == "dev"]
    data = np.load(cache / "standard_m3_documents.npz")
    documents = data["vectors"]
    query_data = np.load(cache / "standard_m3_queries.npz")
    query_by_id = dict(zip(query_data["ids"].tolist(), query_data["vectors"]))
    records = []
    for name, cls in (("chroma", ChromaVectors), ("faiss", FaissVectors)):
        directory = cache / "databases" / name
        backend = cls(directory, snapshot["embedding_fingerprint"])
        started = time.perf_counter()
        backend.upsert(chunks, documents)
        for doc_id in {c.document_id for c in chunks}:
            backend.retain("benchmark", doc_id, [c.chunk_id for c in chunks if c.document_id == doc_id])
        build_ms = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        reopened = cls(directory, snapshot["embedding_fingerprint"])
        first = reopened.query(query_by_id[cases[0].case_id], "benchmark", {c.chunk_id for c in chunks}, 20)
        restart_ms = (time.perf_counter()-started)*1000
        mode_rows, rankings = [], {}
        for case in cases:
            started = time.perf_counter()
            found = reopened.query(query_by_id[case.case_id], "benchmark", {c.chunk_id for c in chunks}, 20)
            elapsed = (time.perf_counter()-started)*1000
            hits = [RetrievalHit(chunk=next(c for c in chunks if c.chunk_id == key), score=max(0., score), vector_score=score)
                    for key, score in found]
            rankings[case.case_id] = [key for key, _ in found]
            mode_rows.append(case_record(case, hits, chunks, elapsed, {"latency_scope": "database_query_only_precomputed_vectors"}))
        # One real delete/re-add probe uses already encoded vectors; no duplicate
        # model inference is needed to measure the adapters' mutable behavior.
        doc_id = chunks[0].document_id
        started = time.perf_counter()
        reopened.delete(doc_id, "benchmark")
        delete_ms = (time.perf_counter()-started)*1000
        subset = [i for i, c in enumerate(chunks) if c.document_id == doc_id]
        survivors = reopened.query(query_by_id[cases[0].case_id], "benchmark", {chunks[i].chunk_id for i in subset}, 20)
        if survivors:
            raise ValueError(f"{name}: deleted document remains searchable")
        started = time.perf_counter()
        reopened.upsert([chunks[i] for i in subset], documents[subset])
        reopened.retain("benchmark", doc_id, [chunks[i].chunk_id for i in subset])
        restore_ms = (time.perf_counter()-started)*1000
        row = {"backend": name, "query_count": len(cases), "chunk_count": len(chunks), "dimensions": documents.shape[1],
               "build_and_persist_ms": build_ms, "restart_first_query_ms": restart_ms,
               "delete_ms": delete_ms, "restore_ms": restore_ms, "delete_probe_passed": True,
               **{k: v for k, v in summarise_records(mode_rows).items() if not isinstance(v, dict)}}
        records.append(row)
        result = result_bundle(snapshot, chunks, cases, mode_rows, "vector", {"embedding_model": "BAAI/bge-m3", "vector_backend": name}, settings)
        result["database_measurements"] = row
        result["notes"].append("Same already encoded documents/queries; build/delete/restart measured without model inference. Isolated benchmark database only.")
        export_result(result, output / name)
        save_json(cache / ("rankings_" + name + ".json"), rankings)
        print(f"{name}: {len(cases)} actual queries; mean {row['latency_mean_ms']:.1f}ms", flush=True)
    save_json(output / "results.json", {"kind": "vector_database_comparison", "status": "completed", "created_at": timestamp(),
                                        "corpus_hash": snapshot["corpus_hash"], "dataset_hash": fingerprint([c.model_dump() for c in cases]),
                                        "metrics": {"query_count_per_backend": 20, "chunk_count": len(chunks)}, "records": records,
                                        "demo": False, "model_inference_reused": True})
    write_csv(output / "comparison.csv", records)


def frozen_rows(snapshot, cases, records, query_by_id, mode, models, query_encoding):
    """Generation receives actual observed rankings, not a fake fresh search."""
    by_case = {record["case_id"]: record for record in records}
    for case in cases:
        record = by_case[case.case_id]
        yield {"case_id": case.case_id, "query": case.question, "question": case.question,
               "split": case.split, "category": case.category, "reference_answer": case.reference_answer,
               "relevant_sources": [source.model_dump() for source in case.relevant_sources],
               "review_method": case.review_method, "human_review_pending": case.human_review_pending,
               "inference_basis": case.inference_basis, "expected_tools": case.expected_tools,
               "hits": record["retrieved"], "vector": query_by_id[case.case_id].tolist(),
               "mode": mode, "models": models, "corpus_hash": snapshot["corpus_hash"],
               "dataset_hash": snapshot["dataset_hash"], "index_version": snapshot["index_version"],
               "timings": {"study_rank_ms": record["elapsed_ms"],
                           "encoding_amortized_ms": query_encoding["query_encoding_amortized_ms"],
                           "encoding_amortized_is_estimate": True,
                           "retrieval_observed_at": record.get("observed_at", ""),
                           "not_fresh_online_end_to_end_latency": True},
               "error": record.get("error", ""), "legacy_persisted_chunk_snapshot": True}


def rank_study(settings, cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    from src.retrieval.vectorstores import ChromaVectors

    cache, output = Path(cache), Path(output)
    snapshot, chunks, cases = read_snapshot(cache)
    vector_data = np.load(cache / "standard_m3_documents.npz")
    queries = np.load(cache / "standard_m3_queries.npz")
    query_by_id = dict(zip(queries["ids"].tolist(), queries["vectors"]))
    encoding = json.loads((cache / "standard_m3_query_encoding.json").read_text(encoding="utf-8"))
    backend = ChromaVectors(cache / "databases/chroma", snapshot["embedding_fingerprint"])
    if backend._load().count() != len(chunks):
        raise ValueError("隔离Chroma数据库未完整建立，请先stores阶段")
    allowed = {chunk.chunk_id for chunk in chunks}
    bm25, token_documents = lexical_model(chunks)
    mode_rows = {"vector": [], "hybrid": []}
    cheap_dev, ranking_cache = [], []
    models = {"embedding_model": "BAAI/bge-m3", "embedding_fingerprint": snapshot["embedding_fingerprint"],
              "embedding_artifact_revision": snapshot["embedding_artifact_revision"],
              "vector_backend": "chroma", "reranker_backend": "not_called", "query_encoding": encoding}
    for number, case in enumerate(cases, 1):
        start = time.perf_counter()
        vector = backend.query(query_by_id[case.case_id], "benchmark", allowed, 20)
        vector_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        keyword, coverage = lexical_rank(case.question, chunks, bm25, token_documents)
        lexical_ms = (time.perf_counter()-start)*1000
        for mode in ("vector", "hybrid"):
            start = time.perf_counter()
            hits = fuse_hits(chunks, vector, keyword, coverage, mode=mode)
            fusion_ms = (time.perf_counter()-start)*1000
            elapsed = vector_ms + (lexical_ms + fusion_ms if mode == "hybrid" else fusion_ms)
            mode_rows[mode].append(case_record(case, hits, chunks, elapsed,
                                  {"observed_at": timestamp(), "vector_db_ms": vector_ms,
                                   "bm25_ms": lexical_ms if mode == "hybrid" else 0.0,
                                   "fusion_ms": fusion_ms,
                                   "latency_scope": "ranking_only_shared_query_encoding_excluded",
                                   "query_vector_reused": True,
                                   "amortized_query_encoding_ms": encoding["query_encoding_amortized_ms"]}))
        ranking_cache.append({"case_id": case.case_id, "vector": vector, "keyword": keyword, "coverage": coverage,
                              "vector_ms": vector_ms, "bm25_ms": lexical_ms})
        if case.split == "dev":
            for constant in (20, 60, 100):
                ranked = fuse_hits(chunks, vector, keyword, coverage, rrf_k=constant)
                for k in (3, 5, 10):
                    measured = retrieval_metrics([h.chunk.chunk_id for h in ranked],
                                                  map_evidence_to_chunks(case.relevant_sources, chunks), k=k)
                    cheap_dev.append({"case_id": case.case_id, "rrf_k": constant, "metric_k": k,
                                      "hit": measured[f"hit_at_{k}"], "recall": measured[f"recall_at_{k}"],
                                      "mrr_at_20": measured["mrr"], "model_calls": 0})
        if number % 10 == 0:
            print(f"Actual Chroma/vector+BM25 rankings {number}/{len(cases)}; no extra query encodes", flush=True)
    save_json(cache / "standard_rank_candidates.json", ranking_cache)
    for mode, rows in mode_rows.items():
        for split in ("dev", "test"):
            selected = [case for case in cases if case.split == split]
            records = [row for row in rows if row["split"] == split]
            result = result_bundle(snapshot, chunks, selected, records, mode, models, settings,
                                   ["Legacy persisted production snapshot of 359 chunks; not the revised recursive splitter's 342-chunk dev variants.",
                                    "Actual database, BM25 and RRF work timed once; query vectors were encoded once for all80 and reused.",
                                    "elapsed_ms excludes shared query encoding and is not online end-to-end latency."])
            result["encoding_measurements"] = encoding
            result["legacy_persisted_chunk_snapshot"] = True
            export_result(result, output / "m3" / split / mode)
        save_jsonl(output / "frozen" / (mode + ".jsonl"),
                   frozen_rows(snapshot, cases, rows, query_by_id, mode, models, encoding))
    summary = []
    for rrf in (20, 60, 100):
        for k in (3, 5, 10):
            group = [row for row in cheap_dev if row["rrf_k"] == rrf and row["metric_k"] == k]
            summary.append({"rrf_k": rrf, "metric_k": k, "query_count": len(group),
                            "hit": float(np.mean([row["hit"] for row in group])),
                            "recall": float(np.mean([row["recall"] for row in group])),
                            "mrr_at_20": float(np.mean([row["mrr_at_20"] for row in group])), "additional_model_calls": 0})
    save_json(output / "dev-parameters" / "results.json", {"kind": "dev_parameter_ablation", "status": "completed",
              "created_at": timestamp(), "split": ["dev"], "corpus_hash": snapshot["corpus_hash"],
              "dataset_hash": fingerprint([case.model_dump() for case in cases if case.split == "dev"]),
              "metrics": {"query_count": 20, "rrf_values": [20, 60, 100], "top_k_values": [3, 5, 10], "extra_model_calls": 0},
              "summary": summary, "records": cheap_dev,
              "notes": ["Actual observed vector/BM25 rank lists reused; only deterministic fusion/metric cutoffs vary.",
                        "Test60 config remains fixed to candidate20/RRF60; no best dev setting was retroactively applied to test."]})
    write_csv(output / "dev-parameters" / "comparison.csv", summary)
    save_json(output / "frozen" / "metadata.json", {"created_at": timestamp(), "corpus_hash": snapshot["corpus_hash"],
              "dataset_hash": snapshot["dataset_hash"], "index_version": snapshot["index_version"],
              "split_dataset_hashes": {split: fingerprint([case.model_dump() for case in cases if case.split == split]) for split in ("dev", "test")},
              "case_count": 80, "split_counts": {"dev": 20, "test": 60}, "candidate_count": 20,
              "chunks_hash": snapshot["standard_chunk_hash"], "legacy_persisted_chunk_snapshot": True,
              "scope_policy": "All public chunk.scope fields are empty; workspace credentials are absent.",
              "query_vectors_npz": str((cache / "standard_m3_queries.npz").resolve()), "query_encoding": encoding,
              "models": models, "ranking_is_frozen_not_fresh_online": True})
    print("Vector and hybrid true top20 frozen for 80 cases; dev cheap ablation complete", flush=True)


def rerank_study(settings, cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    from src.retrieval.reranking import Reranker
    import torch
    torch.set_num_threads(max(1, settings.cpu_threads))

    cache, output = Path(cache), Path(output)
    snapshot, chunks, cases = read_snapshot(cache)
    cases = [case for case in cases if case.split == "test"]
    queries = np.load(cache / "standard_m3_queries.npz")
    query_by_id = dict(zip(queries["ids"].tolist(), queries["vectors"]))
    encoding = json.loads((cache / "standard_m3_query_encoding.json").read_text(encoding="utf-8"))
    observed = {row["case_id"]: row for row in json.loads((cache / "standard_rank_candidates.json").read_text(encoding="utf-8"))}
    checkpoint_path = cache / "rerank_checkpoint.json"
    checkpoint_identity = fingerprint({"rankings": file_hash(cache / "standard_rank_candidates.json"),
                                       "helper": file_hash(ROOT / "src/retrieval/token_windows.py"),
                                       "model": settings.reranker_model, "batch": settings.embedding_batch_size,
                                       "cases": fingerprint([case.model_dump() for case in cases]), "chunks": snapshot["standard_chunk_hash"]})
    rows = []
    if checkpoint_path.exists():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if prior.get("cache_identity") == checkpoint_identity:
            rows = prior["records"]
            if [row["case_id"] for row in rows] != [case.case_id for case in cases[:len(rows)]]:
                raise ValueError("重排checkpoint题目顺序不一致")
            print(f"Validated actual rerank checkpoint reused {len(rows)}/{len(cases)}", flush=True)
    reranker = Reranker(settings)
    started = time.perf_counter()
    reranker._load()
    load_ms = (time.perf_counter()-started)*1000
    # Verify that native original-ID classifier inference matches predict on a
    # short pair, using this already-loaded model (no second model instance).
    from src.retrieval.token_windows import pair_windows, classifier_scores, WINDOW_POLICY
    import inspect
    check_pair = ("Image classification accuracy?", "The ImageNet classifier accuracy is 81.3 percent.")
    activation = "activation_fn" if "activation_fn" in inspect.signature(reranker.model.predict).parameters else "activation_fct"
    reference = float(np.asarray(reranker.model.predict([check_pair], batch_size=1, show_progress_bar=False,
                                                         **{activation: torch.nn.Identity()})).reshape(-1)[0])
    limit = int(getattr(reranker.model, "max_seq_length", None) or getattr(reranker.model, "max_length", None)
                or reranker.model.tokenizer.model_max_length)
    checked_windows = pair_windows(reranker.model.tokenizer, check_pair[0], [check_pair[1]], limit)
    direct = float(classifier_scores(reranker.model.model, reranker.model.tokenizer, checked_windows, 1)[0])
    if abs(reference-direct) > 1e-4:
        raise ValueError(f"CrossEncoder原token-ID logits短pair parity不一致: {reference} vs {direct}")
    # No result is synthesized if model work fails: the case is marked failed.
    for number, case in enumerate(cases[len(rows):], len(rows) + 1):
        candidates = observed[case.case_id]
        hits = fuse_hits(chunks, candidates["vector"], candidates["keyword"], candidates["coverage"])
        started = time.perf_counter()
        try:
            hits = reranker.rerank(case.question, hits)
            elapsed = (time.perf_counter()-started)*1000
            row = case_record(case, hits, chunks, elapsed + candidates["vector_ms"] + candidates["bm25_ms"],
                              {"observed_at": timestamp(), "rerank_ms": elapsed, "vector_db_ms": candidates["vector_ms"],
                               "bm25_ms": candidates["bm25_ms"], "reranker_batch_size": settings.embedding_batch_size,
                               "latency_scope": "measured_rerank_plus_frozen_rank_compute_shared_encoding_excluded",
                               "amortized_query_encoding_ms": encoding["query_encoding_amortized_ms"]})
        except Exception as exc:
            row = {"case_id": case.case_id, "question": case.question, "category": case.category, "split": case.split,
                   "reference_answer": case.reference_answer, "relevant_sources": [s.model_dump() for s in case.relevant_sources],
                   "expected_tools": case.expected_tools, "review_method": case.review_method,
                   "human_review_pending": case.human_review_pending, "error": f"{type(exc).__name__}: {exc}",
                   "hit_at_5": 0., "mrr": 0., "recall_at_5": 0., "retrieved": [],
                   "elapsed_ms": (time.perf_counter()-started)*1000}
        rows.append(row)
        save_json(checkpoint_path, {"cache_identity": checkpoint_identity, "completed": number,
                                   "total": len(cases), "records": rows})
        print(f"True rerank {number}/{len(cases)} {case.case_id}: {row['elapsed_ms']:.0f}ms; error={bool(row['error'])}", flush=True)
    model = {"embedding_model": "BAAI/bge-m3", "embedding_fingerprint": snapshot["embedding_fingerprint"],
             "embedding_artifact_revision": snapshot["embedding_artifact_revision"], "vector_backend": "chroma",
             "reranker_model": settings.reranker_model, "reranker_backend": "cross_encoder",
             "reranker_batch_size": settings.embedding_batch_size, "reranker_model_load_ms": load_ms,
             "reranker_window_policy": WINDOW_POLICY, "short_pair_parity_max_abs_deviation": abs(reference-direct),
             "token_window_helper_sha256": file_hash(ROOT / "src/retrieval/token_windows.py"),
             "reranker_artifact_revision": __import__("src.retrieval.embeddings", fromlist=["local_model_revision"]).local_model_revision(settings.reranker_model),
             "reranker_long_input_policy": "Every token window scored; document score is maximum, no silent tail truncation"}
    for split in ("test",):
        selected = [case for case in cases if case.split == split]
        records = [row for row in rows if row["split"] == split]
        result = result_bundle(snapshot, chunks, selected, records, "hybrid_rerank", model, settings,
                               ["Exactly20 real hybrid candidates reranked once per case; complete token windows retained.",
                                "Embedding and LLM weights were not loaded in this process. Shared frozen query encoding excluded from elapsed_ms.",
                                "Legacy persisted359-chunk snapshot; revised recursive dev variants are separately identified."])
        result["encoding_measurements"] = encoding
        result["legacy_persisted_chunk_snapshot"] = True
        export_result(result, output / "m3" / split / "hybrid_rerank")
    save_jsonl(output / "frozen" / "hybrid_rerank.jsonl", frozen_rows(snapshot, cases, rows, query_by_id,
                                                                    "hybrid_rerank", model, encoding))
    metadata_path = output / "frozen" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mode_case_counts"] = {"vector": 80, "hybrid": 80, "hybrid_rerank": 60}
    metadata["hybrid_rerank_note"] = "Exactly test60 reranked once; dev20 uses its already observed hybrid context, no dev rerank claimed."
    save_json(metadata_path, metadata)
    print("Real reranker stage complete; frozen60 test top20 export saved", flush=True)


def summarise_suite(cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    cache, output = Path(cache), Path(output)
    snapshot, _, cases = read_snapshot(cache)
    groups = {"m3_test": [], "chunking_dev": [], "embedding_dev": []}
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        path = output / "m3/test" / mode / "results.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            groups["m3_test"].append({"mode": mode, "query_count": result["metrics"]["case_count"],
                                     **{key: value for key, value in result["metrics"].items() if not isinstance(value, dict)}})
    for key in ("m3e", "bge-zh"):
        path = output / "encoding" / key / "comparison.json"
        if path.exists():
            comparison = json.loads(path.read_text(encoding="utf-8"))
            if key == "m3e":
                groups["chunking_dev"] = comparison
            groups["embedding_dev"].extend(row for row in comparison if row["variant"] == "recursive-512")
    result = {"kind": "minimal_course_retrieval_suite", "created_at": timestamp(), "status": "completed",
              "corpus_hash": snapshot["corpus_hash"], "dataset_hash": snapshot["dataset_hash"],
              "groups": groups, "metrics": {"paper_count": 10, "test_count": 60, "dev_count": 20},
              "source_labels_review_method": "assistant_source_checked", "human_review_pending": True,
              "expected_paths": {"parsers": "parsers/results.json", "vectorstores": "vectorstores/results.json",
                                 "parameters": "dev-parameters/results.json", "frozen": "frozen/metadata.json"},
              "notes": ["Every recorded result came from real model/database work. No demo/hash embedding or lexical reranker substituted.",
                        "Query encoding reused across modes/databases and separately timed; compute latency is not full online end-to-end latency.",
                        "New recursive342 chunks are used for dev chunking/model comparisons; production359 chunks are a separately frozen legacy snapshot.",
                        "Labels still await human source review; retrieval numbers do not establish generated answer quality."]}
    save_json(output / "results.json", result)
    for key, rows in groups.items():
        write_csv(output / (key + ".csv"), rows)
    lines = ["# 真实检索实验结果", "", "10篇真实论文；20dev和60test。模型串行加载，已编码向量复用。"]
    for key, rows in groups.items():
        lines.extend(["", "## " + key, "", "| 配置 | 查询数 | Hit@5 | MRR | Recall@5 |", "| --- | --- | --- | --- | --- |"])
        for row in rows:
            name = row.get("mode") or row.get("model", "") + "/" + row.get("variant", "")
            lines.append(f"| {name} | {row.get('query_count',row.get('case_count'))} | {row['hit_at_5']:.3f} | {row['mrr']:.3f} | {row['recall_at_5']:.3f} |")
    lines.extend(["", "耗时列仅记录检索/排名/精排计算，批量编码单独列出，不能作为完整在线问答延迟。",
                  "短页内证据锚点已由助理核对，人工审核仍待完成；不能以此宣称课程成绩或答案质量。"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
