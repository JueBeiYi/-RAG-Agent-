"""Real local generation studies using immutable, real frozen retrieval inputs.

No embedding/reranking model is imported or loaded. Run `prepare` while the
retrieval study owns memory, then execute generation stages serially after it
exits. This is explicitly generation-only, not online end-to-end latency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD_PROMPT = "你是科研助手。请根据提供的论文资料回答问题，尽量给出完整解释。"
PARAMETERS = [
    ("baseline_current", {"temperature": .1, "top_p": .9, "top_k_sampling": 40}),
    ("temperature_0.0", {"temperature": .0, "top_p": .9, "top_k_sampling": 40}),
    ("temperature_0.5", {"temperature": .5, "top_p": .9, "top_k_sampling": 40}),
    ("top_p_0.7", {"temperature": .1, "top_p": .7, "top_k_sampling": 40}),
    ("top_k_20", {"temperature": .1, "top_p": .9, "top_k_sampling": 20}),
]
SCOPE = "independent-generation-evaluation"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def public(value):
    """Scope never doubles as a publicly copied API credential."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: "" if key == "scope" else public(item) for key, item in value.items()
                if key not in {"workspace_token", "cloud_api_key", "authorization", "X-Workspace-Token"}}
    if isinstance(value, (list, tuple)):
        return [public(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(public(value), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def process_alive(pid):
    if os.name == "nt":
        # Windows os.kill(pid, 0) is not a harmless Unix-style probe: never use it.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is not proof of exit.
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def exclusive_study(output):
    """Prevent accidental simultaneous model studies or concurrent CSV writes."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".generation.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = json.loads(path.read_text(encoding="utf-8"))
        if not process_alive(int(owner["pid"])):
            path.unlink()
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError("A generation study is already active; do not duplicate real requests")
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump({"pid": os.getpid(), "started_at": stamp()}, file)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid():
            path.unlink()


def free_memory_gib():
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available / 1024**3
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return None


class FrozenInputs:
    def __init__(self, args):
        from src.evaluation.dataset import load_cases
        from src.core.schemas import Chunk, RetrievalHit
        from src.generation.context import SYSTEM_PROMPT, build_context, estimate_tokens
        self.args = args
        self.rows = jsonl(args.frozen)
        self.by_id = {row["case_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("Frozen retrieval contains duplicated case IDs")
        self.cases = load_cases(args.cases)
        missing = [case.case_id for case in self.cases if case.case_id not in self.by_id]
        if missing:
            raise ValueError(f"Missing real frozen retrieval rows: {missing}")
        self.chunks = [Chunk.model_validate(public(row)).model_copy(update={"scope": SCOPE}) for row in jsonl(args.chunks)]
        self.snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8-sig"))
        self.source_files = [{"path": str(Path(path).resolve()), "sha256": digest(path)}
                             for path in (args.frozen, args.cases, args.chunks, args.snapshot, Path(__file__),
                                          ROOT / "src/generation/context.py", ROOT / "src/generation/llm_client.py")]
        self.contexts = {}
        self.vectors = {}
        self.settings = make_settings(args)
        context_signature = fingerprint({"frozen_sha": digest(args.frozen), "cases_sha": digest(args.cases),
            "context_code_sha": digest(ROOT / "src/generation/context.py"), "prompt": SYSTEM_PROMPT,
            "budget_settings": [self.settings.context_tokens, self.settings.answer_tokens, self.settings.result_k]})
        context_cache = Path(args.output) / "fixed_contexts.json"
        cached = json.loads(context_cache.read_text(encoding="utf-8")) if context_cache.exists() else {}
        if cached.get("signature") == context_signature:
            self.contexts = cached["contexts"]
            for value in self.contexts.values():
                if value["context_hash"] != fingerprint({"context": value["context"], "citations": value["citations"]}):
                    raise ValueError("Frozen context cache content/hash mismatch")
        for case in self.cases:
            raw = self.by_id[case.case_id]
            if raw.get("error"):
                raise ValueError(f"Frozen retrieval case failed: {case.case_id}: {raw['error']}")
            if raw.get("mode") and raw["mode"] != args.mode:
                raise ValueError("Frozen ranking mode differs from the declared generation study mode")
            if raw.get("corpus_hash") and raw["corpus_hash"] != self.snapshot["corpus_hash"]:
                raise ValueError("Frozen ranks and stored chunk snapshot have different corpus hashes")
            if case.case_id not in self.contexts:
                values = raw.get("hits", raw.get("retrieved", []))
                hits = [RetrievalHit.model_validate(public(item)).model_copy(
                    update={"chunk": Chunk.model_validate(public(item["chunk"])).model_copy(update={"scope": SCOPE})}) for item in values]
                budget = self.settings.context_tokens - self.settings.answer_tokens - estimate_tokens(SYSTEM_PROMPT + case.question) - 150
                context, citations = build_context(hits[:self.settings.result_k], budget)
                self.contexts[case.case_id] = {"context": context, "citations": public(citations),
                    "context_hash": fingerprint({"context": context, "citations": public(citations)}),
                    "frozen_hit_ids": [hit.chunk.chunk_id for hit in hits], "sent_hit_ids": [citation.chunk_id for citation in citations]}
            if raw.get("vector"):
                self.vectors[case.question] = raw["vector"]
        save(context_cache, {"signature": context_signature, "contexts": self.contexts})
        if not self.vectors and Path(args.query_vectors).exists():
            import numpy as np
            with np.load(args.query_vectors, allow_pickle=False) as arrays:
                vector_by_id = dict(zip(arrays["ids"].tolist(), arrays["vectors"]))
                self.vectors = {case.question: vector_by_id[case.case_id].tolist() for case in self.cases if case.case_id in vector_by_id}
            self.source_files.append({"path": str(Path(args.query_vectors).resolve()), "sha256": digest(args.query_vectors)})
        self.dataset_hash = fingerprint([case.model_dump() for case in self.cases])
        self.corpus_hash = self.snapshot["corpus_hash"]
        self.models = self.rows[0].get("models", {})
        self.dev = []
        for category in ("factual", "comparison", "synthesis", "reasoning"):
            self.dev.append(next(case for case in self.cases if case.split == "dev" and case.category == category))
        self.test = [case for case in self.cases if case.split == "test"]
        if len(self.test) != 60:
            raise ValueError(f"Expected 60 real test cases, got {len(self.test)}")

    def context(self, case_id):
        return self.contexts[case_id]


def make_settings(args, **changes):
    from src.core.config import load_settings
    settings = load_settings(args.config)
    values = {"llm_backend": "ollama", "ollama_url": args.ollama_url, "llm_model": args.model,
              "tool_timeout_seconds": args.timeout, "request_timeout_seconds": max(args.timeout, 300),
              "offline": True, "network_search_enabled": False,
              "temperature": .1, "top_p": .9, "top_k_sampling": 40, **changes}
    if args.answer_tokens:
        values["answer_tokens"] = args.answer_tokens
    return replace(settings, **values)


def model_identity(args):
    import httpx
    response = httpx.get(args.ollama_url.rstrip("/") + "/api/tags", timeout=5, trust_env=False)
    response.raise_for_status()
    item = next((item for item in response.json().get("models", []) if item.get("name", item.get("model")) == args.model), None)
    if not item:
        raise ValueError(f"Prepared model {args.model} not found at the dedicated project Ollama")
    return {"backend": "ollama", "model": args.model, "digest": item.get("digest"), "details": item.get("details")}


def percentile(values, p):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return None
    position = (len(values) - 1) * p / 100
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metrics(records):
    latency = [row.get("elapsed_ms") for row in records]
    inputs = [row["prompt_tokens"] for row in records if isinstance(row.get("prompt_tokens"), (int, float))]
    outputs = [row["completion_tokens"] for row in records if isinstance(row.get("completion_tokens"), (int, float))]
    return {"case_count": len({row["case_id"] for row in records}), "run_count": len(records),
        "generation_protocol_success_rate": mean(bool(row.get("success")) for row in records) if records else None,
        "latency_mean_ms": mean(value for value in latency if isinstance(value, (int, float))) if records else None,
        "latency_p50_ms": percentile(latency, 50), "latency_p95_ms": percentile(latency, 95),
        "prompt_tokens_total": sum(inputs) if inputs else None,
        "completion_tokens_total": sum(outputs) if outputs else None,
        "token_usage_known_count": sum(row.get("prompt_tokens") is not None and row.get("completion_tokens") is not None for row in records),
        "invalid_citation_run_count": sum(bool(row.get("audit", {}).get("invalid_ids")) for row in records),
        "numeric_rule_review_run_count": sum(bool(row.get("audit", {}).get("needs_review")) for row in records),
        "answers_with_source_markers": sum(bool(row.get("audit", {}).get("used_ids")) for row in records),
        "output_truncation_count": sum(bool(row.get("truncated")) for row in records),
        "human_correctness_mean": None, "human_completeness_mean": None, "human_citation_quality_mean": None,
        "human_scoring_completed_count": 0}


def bundle(inputs, args, study, records, config, status="completed", **extra):
    case_by_id = {case.case_id: case for case in inputs.cases}
    unique_ids = list(dict.fromkeys(record["case_id"] for record in records))
    if all(identifier in case_by_id for identifier in unique_ids):
        measured_dataset = [case.model_dump() for case in inputs.cases if case.case_id in set(unique_ids)]
    else:
        measured_dataset = [{key: record.get(key) for key in ("case_id", "question", "category", "frozen_source_case_id")}
                            for record in records if record.get("condition") in {"agent", "workers_1", "cold_exact_query"}]
    conditions = list(dict.fromkeys(row.get("condition", "baseline_current") for row in records))
    return {"experiment_id": fingerprint({"study": study, "config": config, "sources": inputs.source_files})[:32],
        "created_at": stamp(), "kind": "generation", "study": study, "mode": args.mode,
        "status": status, "demo": False, "formal_models": True,
        "scope": "", "split": sorted({row.get("split", "dev") for row in records}),
        "dataset_hash": fingerprint(measured_dataset), "source_dataset_hash": inputs.dataset_hash, "corpus_hash": inputs.corpus_hash,
        "index_version": inputs.snapshot.get("index_version"), "config": public(config), "config_hash": fingerprint(config),
        "models": {"retrieval_models": inputs.models, "llm_actual": getattr(inputs, "llm_identity", {})},
        "source_files": inputs.source_files, "metrics": metrics(records),
        "condition_metrics": {condition: metrics([row for row in records if row.get("condition", "baseline_current") == condition])
                              for condition in conditions}, "records": public(records),
        "measurement_scope": "frozen-retrieval / generation-only; no online embedding or reranking",
        "review_status": {"source_checked": True, "source_review_method": "assistant_source_checked", "human_source_review_pending": True,
                          "human_answer_review_pending": True, "human_scoring_completed": False},
        "notes": ["Frozen real retrieval inputs remain identical across parameter and Prompt conditions.",
                  "Latency includes local model generation/stream handling only, not online retrieval or end-to-end browser time.",
                  "Citation/number checks are automated rules, not correctness/completeness scores; derived arithmetic can trigger numeric review.",
                  "Human reviewers and all answer quality scores are intentionally blank/pending."], **extra}


def generate(inputs, case, settings, *, prompt=None, context=None):
    from src.generation.llm_client import LLMClient
    from src.generation.context import SYSTEM_PROMPT, audit_citations, normalise_page_references
    supplied = context or inputs.context(case.case_id)
    user = "证据（来源编号只能从这里选）：\n" + supplied["context"] + "\n\n用户问题：" + case.question
    if not supplied["citations"]:
        user = "固定检索未提供可用证据。请说明证据不足，不猜测论文数字。问题：" + case.question
    if inputs.args.answer_character_target:
        user += (f"\n短答案实验要求：尽量用100–{inputs.args.answer_character_target}个汉字简洁作答，"
                 "优先保留直接回答所需事实与来源编号；证据不足应明确说明，不为凑长度添加内容。")
    llm = LLMClient(settings)
    started, first_token, answer = time.perf_counter(), None, ""
    usage = {"input_tokens": None, "output_tokens": None, "estimated": None}
    events, warnings, error = [], [], ""
    try:
        for event in llm.stream([{"role": "system", "content": prompt or SYSTEM_PROMPT}, {"role": "user", "content": user}]):
            if event["type"] == "answer_delta":
                if first_token is None:
                    first_token = (time.perf_counter()-started)*1000
                answer += event["text"]
            if event["type"] == "usage":
                usage = {key: event.get(key) for key in usage}
            if event.get("warning"):
                warnings.append(event.get("text", ""))
            events.append(event)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    answer, page_corrections = normalise_page_references(answer, supplied["citations"])
    audit = audit_citations(answer, supplied["citations"])
    return {"case_id": case.case_id, "question": case.question, "category": case.category, "split": case.split,
        "answer": answer, "reference_answer": case.reference_answer, "relevant_sources": public(case.relevant_sources),
        "review_method": case.review_method, "human_review_pending": case.human_review_pending,
        "source_checked": bool(case.reviewed), "human_answer_review_pending": True,
        "human_review": {"reviewer": "", "correctness_1_to_5": None, "completeness_1_to_5": None,
                         "citation_quality_1_to_5": None, "fluency_1_to_5": None, "notes": ""},
        "citations": supplied["citations"], "context_hash": supplied["context_hash"],
        "frozen_hit_ids": supplied.get("frozen_hit_ids", []), "sent_hit_ids": supplied.get("sent_hit_ids", []),
        "events": events, "warnings": warnings, "error": error, "success": bool(answer.strip()) and not error,
        "truncated": bool(warnings), "elapsed_ms": round((time.perf_counter()-started)*1000, 3), "first_token_ms": first_token,
        "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"], "usage_estimated": usage["estimated"],
        "audit": audit, "page_corrections": page_corrections, "config": {
            **{key: getattr(settings, key) for key in ("temperature", "top_p", "top_k_sampling", "answer_tokens", "context_tokens", "llm_model")},
            "answer_character_target": inputs.args.answer_character_target, "short_answer_instruction_is_soft_target": True}}


class Checkpoint:
    def __init__(self, path, identity, resume=True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        signature = self.path / "identity.json"
        if signature.exists() and json.loads(signature.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Resume identity differs; choose a new output directory: {path}")
        save(signature, identity)
        if not resume and (self.path / "records.jsonl").exists() and (self.path / "records.jsonl").stat().st_size:
            raise ValueError("--no-resume requires a new output directory; existing real records will not be erased or duplicated")
        self.records = jsonl(self.path / "records.jsonl") if resume and (self.path / "records.jsonl").exists() else []
        self.done = {(row["case_id"], row.get("condition", "baseline_current")) for row in self.records}

    def add(self, row):
        row = public(row)
        with (self.path / "records.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
        self.records.append(row)
        self.done.add((row["case_id"], row.get("condition", "baseline_current")))


def scoring_csv(path, records):
    columns = ["case_id", "category", "split", "question", "answer", "reference_answer", "relevant_sources", "citations",
               "success", "error", "source_checked", "review_method", "human_review_pending", "reviewer",
               "correctness_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5", "fluency_1_to_5", "bad_case_type", "root_cause", "review_notes", "grading_method", "grading_date", "grading_evidence"]
    review_fields = ("reviewer", "correctness_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5", "fluency_1_to_5", "bad_case_type", "root_cause", "review_notes", "grading_method", "grading_date", "grading_evidence")
    existing = {}
    if Path(path).exists():
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            existing = {row["case_id"]: row for row in csv.DictReader(file)}
    for record in records:
        previous = existing.get(record["case_id"], {})
        if any(previous.get(key) for key in review_fields) and previous.get("answer") != record.get("answer"):
            raise ValueError("Human scoring belongs to a different answer; retain the existing worksheet and choose a new study output")
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in columns}
            row["relevant_sources"] = json.dumps(public(row["relevant_sources"]), ensure_ascii=False)
            # Keep the editable worksheet readable and below spreadsheet cell
            # limits. Full immutable evidence stays in baseline/results.json.
            row["citations"] = "；".join(
                f"[{citation.get('citation_id', '')}] {citation.get('name', '')} · 物理页{citation.get('page_start', '?')}"
                + (f"至{citation['page_end']}" if citation.get('page_end') and citation.get('page_end') != citation.get('page_start') else "")
                for citation in (record.get("citations") or []))
            previous = existing.get(record["case_id"], {})
            for key in review_fields:
                row[key] = previous.get(key, "")
            writer.writerow(row)
    os.replace(temporary, path)


def generation_stage(inputs, args, stage):
    from src.generation.context import SYSTEM_PROMPT
    if stage == "baseline":
        cases, configurations = inputs.test, PARAMETERS[:1]
    elif stage == "parameters":
        cases, configurations = inputs.dev, PARAMETERS
    else:
        cases, configurations = inputs.dev, [("naive_old_prompt", PARAMETERS[0][1])]
    output = Path(args.output) / stage
    study_config = {"conditions": configurations, "llm_backend": "ollama", "retrieval": "frozen-real-hits",
        "answer_tokens": inputs.settings.answer_tokens, "context_tokens": inputs.settings.context_tokens,
        "answer_character_target": args.answer_character_target, "short_answer_instruction_is_soft_target": True,
        "context_selection": f"Top{inputs.settings.result_k} from real frozen Top20, cropped by application budget",
        "system_prompt": OLD_PROMPT if stage == "prompt" else SYSTEM_PROMPT,
        "system_prompt_sha256": fingerprint(OLD_PROMPT if stage == "prompt" else SYSTEM_PROMPT),
        "sampling_seed": None, "sampling_seed_note": "No fixed seed was requested; one run per case/condition is exploratory.",
        "run_order": "condition-major, canonical source case order"}
    checkpoint = Checkpoint(output, {"sources": inputs.source_files, "stage": stage, "model": inputs.llm_identity,
        "parameters": configurations, "prompt_hash": fingerprint(OLD_PROMPT if stage == "prompt" else SYSTEM_PROMPT),
        "settings": {**{key: getattr(inputs.settings, key) for key in ("answer_tokens", "context_tokens", "result_k")},
                     "answer_character_target": args.answer_character_target}}, args.resume)
    for condition, parameters in configurations:
        settings = make_settings(args, **parameters)
        for case in cases:
            if (case.case_id, condition) in checkpoint.done:
                continue
            row = generate(inputs, case, settings, prompt=OLD_PROMPT if stage == "prompt" else SYSTEM_PROMPT)
            row["condition"] = condition
            checkpoint.add(row)
            save(output / "results.json", bundle(inputs, args, stage, checkpoint.records, study_config, "running"))
            if stage == "baseline":
                scoring_csv(output / "human_scoring_60.csv", checkpoint.records)
            print(f"{stage}: {condition} {case.case_id} success={row['success']} tokens={row['completion_tokens']} seconds={row['elapsed_ms']/1000:.2f}", flush=True)
    records = checkpoint.records
    if stage == "prompt":
        parameters = Path(args.output) / "parameters" / "results.json"
        if not parameters.exists():
            raise ValueError("Run parameters first; its four current-Prompt baseline answers are reused, not regenerated")
        current = [row for row in json.loads(parameters.read_text(encoding="utf-8"))["records"] if row["condition"] == "baseline_current"]
        if len(current) != 4:
            raise ValueError("Current-Prompt baseline must contain four categories")
        records = [*records, *[{**row, "condition": "current_prompt_reused", "reused_from": str(parameters)} for row in current]]
    expected = 60 if stage == "baseline" else 20 if stage == "parameters" else 8
    if stage == "baseline":
        scoring_csv(output / "human_scoring_60.csv", records)
    if stage == "prompt":
        study_config["comparison_system_prompt"] = SYSTEM_PROMPT
        study_config["comparison_system_prompt_sha256"] = fingerprint(SYSTEM_PROMPT)
        study_config["naive_prompt_is_control_not_historical_source_version"] = True
    save(output / "results.json", bundle(inputs, args, stage, records, study_config, "completed" if len(records) == expected else "partial",
        expected_record_count=expected, prompt_baseline_reused=stage == "prompt"))
    return output / "results.json"


class FrozenVectors:
    def __init__(self, inputs):
        self.vectors = inputs.vectors
        self.fingerprint = fingerprint(inputs.models or inputs.snapshot.get("embedding_fingerprint"))

    def encode(self, queries):
        import numpy as np
        missing = [query for query in queries if query not in self.vectors]
        if missing:
            raise ValueError("No real frozen query vectors for these exact questions; hash/fake vectors are forbidden")
        return np.asarray([self.vectors[query] for query in queries], dtype=np.float32)


def independent_store(inputs, args):
    from src.core.storage import Storage
    from src.core.schemas import Document
    store = Storage(Path(args.output) / ".runtime")
    corpus = {row["document_id"]: row for row in inputs.snapshot.get("corpus", [])}
    groups = {}
    for chunk in inputs.chunks:
        groups.setdefault(chunk.document_id, []).append(chunk)
    for identifier, chunks in groups.items():
        metadata = corpus.get(identifier, {})
        name = metadata.get("name", chunks[0].name)
        alias = Path(name).stem.split("_")[0]
        doc = Document(document_id=identifier, scope=SCOPE, name=name, title=metadata.get("title", name),
            content_hash=metadata.get("content_hash", "frozen-source"), format="pdf", status="ready", chunks=len(chunks),
            authors=metadata.get("authors", []), year=metadata.get("year"), metadata={"aliases": [alias], "source": "frozen-corpus-snapshot"})
        store.put_document(doc)
        store.replace_chunks(identifier, SCOPE, chunks)
    store.put("versions", "index", SCOPE, {"version": inputs.snapshot.get("index_version", "frozen")})
    return store


class FrozenRAG:
    """Real model generation on a named frozen context; no live retrieval claim."""
    def __init__(self, inputs, args, settings, fallback_case):
        self.inputs, self.args, self.settings, self.fallback_case = inputs, args, settings, fallback_case
        self.calls = []
        self.by_query = {case.question: case for case in inputs.cases}

    def ask(self, query, scope, document_ids=None, mode=None, history=None):
        case = self.by_query.get(query, self.fallback_case)
        supplied = self.inputs.context(case.case_id)
        row = generate(self.inputs, case.model_copy(update={"question": query}), self.settings, context=supplied)
        row["frozen_source_case_id"] = case.case_id
        self.calls.append(row)
        if row["error"]:
            raise RuntimeError("Frozen-context model generation failed: " + row["error"])
        return {"answer": row["answer"], "citations": row["citations"], "audit": row["audit"],
                "metrics": {"input_tokens": row["prompt_tokens"], "output_tokens": row["completion_tokens"],
                            "estimated": row["usage_estimated"], "elapsed_ms": row["elapsed_ms"], "cache_hit": False},
                "warnings": row["warnings"], "origin": "frozen_generation_only"}

    def stream(self, query, scope, document_ids=None, mode=None, history=None):
        # Agent receives completed tool output; do not label its replay as model TTFT.
        result = self.ask(query, scope, document_ids, mode, history)
        for citation in result["citations"]:
            yield {"type": "citation", "citation": citation}
        yield {"type": "answer_delta", "text": result["answer"], "replay": True}
        yield {"type": "done", **result}

    def retrieve(self, query, scope, document_ids=None, mode=None):
        from src.core.schemas import RetrievalHit, Chunk
        case = self.by_query.get(query, self.fallback_case)
        raw = self.inputs.by_id[case.case_id]
        result = []
        for hit in raw.get("hits", raw.get("retrieved", [])):
            chunk = Chunk.model_validate(public(hit["chunk"])).model_copy(update={"scope": SCOPE})
            if not document_ids or chunk.document_id in document_ids:
                result.append(RetrievalHit.model_validate(public(hit)).model_copy(update={"chunk": chunk}))
        return result[:self.settings.result_k]


def agent_stage(inputs, args):
    from src.agent.react_loop import ResearchAgent
    from src.generation.llm_client import LLMClient
    store = independent_store(inputs, args)
    factual = next(case for case in inputs.dev if case.category == "factual")
    examples = [("concept", "什么是自注意力？", []), ("calculator", "计算 `3.14*2.56`", ["calculator"]),
                ("time", "当前时间是什么？", ["current_time"]),
                ("query_keywords", "从问题中提取关键词：Swin Transformer uses shifted windows in vision tasks", ["extract_keywords"]),
                ("paper_fact", factual.question, ["knowledge_qa"])]
    records = []
    settings = inputs.settings
    for label, query, expected in examples:
        supplied = inputs.context(factual.case_id)
        fixed_case = factual.model_copy(update={"case_id": "route-" + label, "question": query, "split": "dev",
            "reference_answer": factual.reference_answer if label == "paper_fact" else "",
            "reviewed": bool(factual.reviewed) if label == "paper_fact" else False})
        fixed = generate(inputs, fixed_case, settings, context=supplied)
        records.append({**fixed, "condition": "fixed_rag", "expected_tools": expected, "invoked_tools": [],
                        "frozen_source_case_id": factual.case_id, "small_sample": True,
                        "context_source_checked": True, "reference_scope": "paper annotation" if label == "paper_fact" else "route-only exploratory query, no static answer grade"})
        rag = FrozenRAG(inputs, args, settings, factual)
        agent = ResearchAgent(settings, store, LLMClient(settings), rag)
        session = store.create_session(SCOPE, "isolated routing study")
        store.add_message(session["id"], SCOPE, "user", query)
        started = time.perf_counter()
        try:
            events = list(agent.stream(query, SCOPE, session["id"]))
        finally:
            agent.close()
        done = next((event for event in reversed(events) if event["type"] == "done"), {})
        invoked = [event["tool"] for event in events if event["type"] == "tool_end"]
        usage = done.get("metrics", {}).get("usage", done.get("metrics", {}))
        records.append({"case_id": "route-" + label, "condition": "agent", "question": query, "category": label,
            "split": "dev", "answer": done.get("answer", ""), "citations": done.get("citations", []), "events": events,
            "elapsed_ms": (time.perf_counter()-started)*1000, "prompt_tokens": usage.get("input_tokens") if not any(call["prompt_tokens"] is None for call in rag.calls) else None,
            "completion_tokens": usage.get("output_tokens") if not any(call["completion_tokens"] is None for call in rag.calls) else None,
            "success": bool(done.get("answer")) and not any(event["type"] == "error" for event in events),
            "expected_tools": expected, "invoked_tools": invoked, "routing_exact_match": set(expected) == set(invoked),
            "tool_call_count": len(invoked), "tool_error_count": sum(bool(event.get("error")) for event in events if event["type"] == "tool_end"),
            "audit": done.get("audit", {}), "metrics": done.get("metrics", {}), "origin": done.get("origin"),
            "frozen_source_case_id": factual.case_id, "small_sample": True, "source_checked": label == "paper_fact", "context_source_checked": True,
            "human_answer_review_pending": True, "human_review": {"reviewer": "", "correctness_1_to_5": None}})
        save(Path(args.output) / "agent" / "results.json", bundle(inputs, args, "small_sample_agent_routing", records,
            {"llm_backend": "ollama", "sample_count": 5, "retrieval": "fixed-real-context", "general_concept_origin": "general-model-knowledge"}, "running"))
        print(f"routing {label}: actual={invoked} expected={expected}", flush=True)
    result = bundle(inputs, args, "small_sample_agent_routing", records,
                    {"llm_backend": "ollama", "sample_count": 5, "retrieval": "fixed-real-context"}, small_sample=True)
    result["metrics"]["routing_exact_match"] = mean(row["routing_exact_match"] for row in records if row["condition"] == "agent")
    result["review_status"] = {"frozen_context_source_checked": True, "human_answer_review_pending": True,
                               "samples_are_route_exploration_not_five_paper_ground_truth_cases": True}
    save(Path(args.output) / "agent" / "results.json", result)
    return Path(args.output) / "agent" / "results.json"


def parallel_stage(inputs, args):
    from src.agent.executor import ToolExecutor
    from src.agent.registry import ToolCall, ToolContext
    from src.agent.tools.local import create_registry
    store = independent_store(inputs, args)
    real_text = "\n".join(inputs.context(inputs.dev[0].case_id)["citations"][i]["text"] for i in range(min(3, len(inputs.context(inputs.dev[0].case_id)["citations"]))))[:6000]
    calls = [ToolCall("calculator", {"expression": "sqrt(16)+2**5"}), ToolCall("extract_keywords", {"query": real_text, "count": 10})]
    records = []
    for workers in (1, 2):
        settings = replace(inputs.settings, max_tool_workers=workers)
        executor = ToolExecutor(create_registry(), settings)
        context = ToolContext(settings, store, None, None, SCOPE, "independent-local-tools")
        try:
            # Warm segmentation/module imports for both configurations; not included in measurements.
            executor.batch(context, [calls[1]])
            for repeat in range(10):
                started = time.perf_counter()
                outputs = []
                for offset in range(0, len(calls), workers):
                    outputs.extend(executor.batch(context, calls[offset:offset+workers]))
                records.append({"case_id": f"parallel-{repeat}", "condition": f"workers_{workers}", "split": "dev",
                    "elapsed_ms": (time.perf_counter()-started)*1000, "success": not any(row.get("error") for row in outputs),
                    "workers": workers, "outputs": outputs, "local_tools_only": True, "llm_called": False,
                    "prompt_tokens": 0, "completion_tokens": 0, "usage_estimated": False})
        finally:
            executor.close()
    result = bundle(inputs, args, "actual_local_tool_parallelism", records,
                    {"llm_backend": "none", "workers": [1, 2], "tools": [call.name for call in calls]},
                    formal_models=False, small_sample=True, measurement_scope="actual local independent tools; excludes LLM and retrieval")
    result["kind"] = "agent_parallel_tools"
    save(Path(args.output) / "parallel" / "results.json", result)
    return Path(args.output) / "parallel" / "results.json"


def cache_stage(inputs, args):
    from src.generation.cache import SemanticCache
    if not inputs.vectors:
        destination = Path(args.output) / "cache" / "not_executed.json"
        save(destination, {"status": "not_executed", "reason": "No genuine frozen query vectors; fake/hash vectors are forbidden"})
        return destination
    baseline = json.loads((Path(args.output) / "baseline" / "results.json").read_text(encoding="utf-8"))["records"]
    store = independent_store(inputs, args)
    cache = SemanticCache(inputs.settings, store, FrozenVectors(inputs))
    case_by_id = {case.case_id: case for case in inputs.cases}
    record = next((row for row in baseline if row.get("success") and cache.eligible(row["question"])
                   and not row.get("warnings") and not row["audit"].get("invalid_ids") and not row["audit"].get("needs_review")), None)
    if record is None:
        destination = Path(args.output) / "cache" / "not_executed.json"
        save(destination, {"status": "not_executed", "reason": "No baseline answer passed actual semantic cache admission policy"})
        return destination
    case = case_by_id[record["case_id"]]
    store.clear_cache(SCOPE)
    fingerprint_value = cache.fingerprint(SCOPE, None, args.mode, [])
    started = time.perf_counter()
    miss = cache.get(case.question, SCOPE, fingerprint_value)
    assert miss is None
    cold = generate(inputs, case, inputs.settings)
    cache.put(case.question, SCOPE, fingerprint_value, {"answer": cold["answer"], "citations": cold["citations"], "warnings": cold["warnings"], "audit": cold["audit"]})
    cold_elapsed = (time.perf_counter()-started)*1000
    rows = [{**cold, "condition": "cold_exact_query", "elapsed_ms": cold_elapsed, "cache_hit": False, "actual_model_calls": 1}]
    started = time.perf_counter()
    hot = cache.get(case.question, SCOPE, fingerprint_value)
    hot_elapsed = (time.perf_counter()-started)*1000
    rows.append({"case_id": case.case_id, "condition": "hot_exact_repeat", "split": "test", "question": case.question,
        "answer": hot.get("answer", "") if hot else "", "success": bool(hot), "cache_hit": bool(hot), "elapsed_ms": hot_elapsed,
        "actual_model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "vectors": "real frozen BGE query embedding reused",
        "answer_same_as_cold": bool(hot) and hot["answer"] == cold["answer"], "human_answer_review_pending": True})
    result = bundle(inputs, args, "exact_semantic_cache_cold_hot", rows,
        {"llm_backend": "ollama", "cache_threshold": inputs.settings.cache_threshold, "query_vector_source": "real frozen vectors",
         "cold_definition": "Empty isolated answer cache; model already resident after generation studies, not cold model startup",
         "answer_tokens": inputs.settings.answer_tokens, "answer_character_target": args.answer_character_target},
        small_sample=True, measurement_scope="cold generation+cache admission vs exact-repeat lookup; no new online embedding")
    save(Path(args.output) / "cache" / "results.json", result)
    return Path(args.output) / "cache" / "results.json"


def preparation(inputs, args):
    destination = Path(args.output) / "prepared_inputs.json"
    save(destination, {"created_at": stamp(), "status": "prepared_not_generated", "demo": False,
        "source_files": inputs.source_files, "test_count": len(inputs.test), "dev_count": sum(case.split == "dev" for case in inputs.cases),
        "selected_dev_cases": [{"case_id": case.case_id, "category": case.category} for case in inputs.dev],
        "context_hashes": {key: value["context_hash"] for key, value in inputs.contexts.items()},
        "dataset_hash": inputs.dataset_hash, "corpus_hash": inputs.corpus_hash,
        "genuine_query_vector_count": len(inputs.vectors), "measurement_scope": "frozen retrieval / generation-only",
        "generation_limits": {"answer_tokens": inputs.settings.answer_tokens, "answer_character_target": args.answer_character_target},
        "human_source_review_pending": True, "human_answer_scoring_pending": True})
    return destination


def summary_stage(inputs, args):
    results = []
    for stage in ("baseline", "parameters", "prompt", "agent", "parallel", "cache"):
        path = Path(args.output) / stage / "results.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append({"study": data["study"], "path": str(path), "sha256": digest(path), "status": data["status"], "metrics": data["metrics"]})
    save(Path(args.output) / "study_manifest.json", {"created_at": stamp(), "results": results,
         "scope": "", "human_scoring_pending": True, "source_files": inputs.source_files})
    lines = ["# 生成与Agent实验记录", "", "记录日期：" + datetime.now(timezone.utc).strftime("%Y-%m-%d") + "（UTC）。", "",
        "所有论文生成研究使用真实冻结检索结果与本地Qwen，不加载Embedding或Reranker。记录的延迟属于generation-only，不是在线端到端性能。", "",
        "60test基线答案可填写评分表：`reports/experiments/generation/baseline/human_scoring_60.csv`。评分人、正确性、完整性、引用质量和流畅性均留空，未伪造人工评分。", "",
        "开发集取四题型各1题，5个单因素配置共20问；旧Prompt额外4问，新Prompt复用相同context的4条开发集基线。小样本只用于规则现象与参数探索，不据此概括整体质量。", "",
        "| 研究 | 状态 | 实际记录数 | 成功执行率 | 平均耗时ms |", "|---|---|---:|---:|---:|"]
    for result in results:
        metric = result["metrics"]
        lines.append(f"| {result['study']} | {result['status']} | {metric['run_count']} | {metric['generation_protocol_success_rate']} | {metric['latency_mean_ms']} |")
    lines += ["", "出处与限制：", "", "- 原始结果、来源编号、实际Token和自动规则审校均保存在各研究results.json；自动规则不等于人工正确性评分。",
              "- Agent路由只有5个题型样本；固定RAG对照明确使用同一真实固定context，未重新在线检索。",
              "- 并行研究只测两个真实独立本地工具，不含LLM；线程调度可能比串行更慢，结果照实保留。",
              "- 缓存只测真实冻结query向量下的exactrepeat；没有向量或答案不满足缓存策略时明确不执行。",
              "- 案例来源已自动据原文核对，人工来源复核与答案评分仍待完成。公开JSON不包含真实工作区凭据。"]
    (ROOT / "docs" / "生成与Agent实验记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(args.output) / "study_manifest.json"


def make_figures(args):
    """Export only measured latency/rule flags; never make a quality score chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    directory = Path(args.output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for stage, title in (("parameters", "4题开发集单因素参数：真实生成耗时"),
                         ("prompt", "同一真实context的Prompt自动规则检查"),
                         ("agent", "5个样本：Agent与固定context RAG耗时"),
                         ("parallel", "真实独立本地工具：串行与并行耗时"),
                         ("cache", "真实缓存准入后的精确重复查询")):
        source = Path(args.output) / stage / "results.json"
        if not source.exists():
            continue
        result = json.loads(source.read_text(encoding="utf-8"))
        groups = {}
        for row in result["records"]:
            groups.setdefault(row.get("condition", "baseline"), []).append(row)
        labels = list(groups)
        if stage == "prompt":
            values = [sum(bool(row.get("audit", {}).get("needs_review")) or bool(row.get("audit", {}).get("invalid_ids")) for row in groups[label]) for label in labels]
            y_label = "触发引用/数字规则的回答数（不等于正确性）"
        else:
            values = [mean(row["elapsed_ms"] for row in groups[label]) for label in labels]
            y_label = "实测平均耗时 ms"
        figure, axis = plt.subplots(figsize=(8.6, 4.5), dpi=160)
        axis.bar(labels, values, color="#287c85")
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.tick_params(axis="x", labelrotation=18)
        figure.tight_layout()
        destination = directory / (stage + ".png")
        figure.savefig(destination)
        plt.close(figure)
        result.setdefault("artifacts", {})["measured_chart"] = str(destination.resolve())
        save(source, result)


def run_stages(args):
    inputs = FrozenInputs(args)
    if args.stage == "prepare":
        print(preparation(inputs, args)); return 0
    if args.stage not in {"parallel", "summary"}:
        available = free_memory_gib()
        if available is not None and available < args.min_free_gib:
            raise RuntimeError(f"Only {available:.2f}GiB RAM free; wait until serial retrieval model processes exit")
        inputs.llm_identity = model_identity(args)
    else:
        inputs.llm_identity = {}
    stages = ["prepare", "baseline", "parameters", "prompt", "agent", "parallel", "cache", "summary"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "prepare":
            path = preparation(inputs, args)
        elif stage in {"baseline", "parameters", "prompt"}:
            path = generation_stage(inputs, args, stage)
        elif stage == "agent":
            path = agent_stage(inputs, args)
        elif stage == "parallel":
            path = parallel_stage(inputs, args)
        elif stage == "cache":
            path = cache_stage(inputs, args)
        else:
            make_figures(args)
            path = summary_stage(inputs, args)
        if stage not in {"prepare", "parallel", "summary"}:
            after = model_identity(args)
            if after.get("digest") != inputs.llm_identity.get("digest"):
                raise RuntimeError("The actual local model digest changed during this study; measurements cannot be mixed")
        print(f"supporting_materials: {path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "baseline", "parameters", "prompt", "agent", "parallel", "cache", "summary", "all"])
    parser.add_argument("--frozen", default=str(ROOT / "reports/experiments/retrieval/frozen/hybrid.jsonl"))
    parser.add_argument("--mode", default="hybrid", choices=["vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--cases", default=str(ROOT / "data/runtime/benchmarks/retrieval-cache/cases.jsonl"))
    parser.add_argument("--chunks", default=str(ROOT / "data/runtime/benchmarks/retrieval-cache/standard_chunks.jsonl"))
    parser.add_argument("--snapshot", default=str(ROOT / "data/runtime/benchmarks/retrieval-cache/snapshot.json"))
    parser.add_argument("--query-vectors", default=str(ROOT / "data/runtime/benchmarks/retrieval-cache/standard_m3_queries.npz"))
    parser.add_argument("--config", default=str(ROOT / "config.standard.yaml"))
    parser.add_argument("--output", default=str(ROOT / "reports/experiments/generation"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="research-qwen:7b")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--answer-tokens", type=int, default=256, help="Explicit short-answer experiment cap; 0 preserves the application config")
    parser.add_argument("--answer-character-target", type=int, default=150, help="Soft length instruction shared by Prompt/parameter conditions; 0 disables it")
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)
    parsed = urlparse(args.ollama_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != 11435:
        raise ValueError("Only the independent project Ollama at localhost:11435 is permitted; do not use system11434")
    os.environ["NO_PROXY"] = "127.0.0.1,localhost," + os.getenv("NO_PROXY", "")
    with exclusive_study(args.output):
        return run_stages(args)


if __name__ == "__main__":
    raise SystemExit(main())
