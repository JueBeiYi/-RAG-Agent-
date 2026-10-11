"""Reindex this project's existing papers once after a tokenizer policy change."""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_CACHE"] = str(ROOT / "data/models/hub")
os.environ["HF_HUB_OFFLINE"] = "1"

from src.core.config import load_settings
from src.services.application import Application


def main():
    scope = json.loads((ROOT / "data/runtime/workspace-standard.json").read_text(encoding="utf-8-sig"))["workspace_token"]
    settings = replace(load_settings(ROOT / "config.standard.yaml"), data_dir=str(ROOT / "data/runtime/standard"),
                       embedding_batch_size=1, embedding_max_tokens=512, cpu_threads=8,
                       llm_model="research-qwen:7b", ollama_url="http://127.0.0.1:11435")
    application = Application(settings, recover_jobs=False)
    started = time.perf_counter()
    try:
        docs = application.store.list_documents(scope)
        fingerprint = application.retriever.embedder.fingerprint
        pending = [doc for doc in docs if doc.status != "ready" or doc.metadata.get("embedding_fingerprint") != fingerprint
                   or doc.metadata.get("chunker_implementation") != "recursive-character-proposals-v1"]
        jobs = {doc.document_id: application.retry(doc.document_id, scope, reindex=True)["job_id"] for doc in pending}
        completed = set()
        while len(completed) < len(jobs):
            for document_id, job_id in jobs.items():
                if document_id in completed:
                    continue
                job = application.store.get("jobs", job_id, scope)
                if job and job.get("status") == "error":
                    raise RuntimeError(f"Reindex failed for document {document_id}: {job.get('error')}")
                if job and job.get("status") == "done":
                    doc = application.store.get_document(document_id, scope)
                    doc.metadata["chunker_implementation"] = "recursive-character-proposals-v1"
                    application.store.put_document(doc)
                    completed.add(document_id)
                    print(f"Reindex {len(completed)}/{len(jobs)} complete; {doc.chunks} chunks", flush=True)
            if len(completed) < len(jobs):
                time.sleep(1)
        current = application.store.list_documents(scope)
        report = {"document_count": len(current), "ready_count": sum(doc.status == "ready" for doc in current),
                  "chunks": sum(doc.chunks for doc in current), "embedding_fingerprint": fingerprint,
                  "chunker_implementation": "recursive-character-proposals-v1", "reindexed_documents": len(jobs),
                  "elapsed_ms": round((time.perf_counter() - started)*1000, 3),
                  "index_version": application.store.index_version(scope),
                  "preserved_document_ids": sorted(doc.document_id for doc in current),
                  "preserved_history": True, "scope": "private workspace omitted"}
        target = ROOT / "reports/requirements/current-standard-index.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "preserved_document_ids"}), flush=True)
    finally:
        application.close()


if __name__ == "__main__":
    main()
