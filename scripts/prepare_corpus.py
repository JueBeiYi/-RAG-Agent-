"""Prepare verifiable open research corpus and map stable paper keys to server IDs.

No models are downloaded or loaded by this command. Download is an explicit
network action; validate operates entirely on local PDFs and dataset files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "corpus"
DATASET = ROOT / "reports" / "datasets"
PAPERS = [
    ("vit", "2010.11929", "ICLR", 2021),
    ("deit", "2012.12877", "ICML", 2021),
    ("swin", "2103.14030", "ICCV", 2021),
    ("pvt", "2102.12122", "ICCV", 2021),
    ("t2t", "2101.11986", "ICCV", 2021),
    ("cait", "2103.17239", "ICCV", 2021),
    ("cvt", "2103.15808", "ICCV", 2021),
    ("crossvit", "2103.14899", "ICCV", 2021),
    ("levit", "2104.01136", "ICCV", 2021),
    ("beit", "2106.08254", "ICLR", 2022),
]


class ArxivMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}
        self.license_url = ""
        self.in_license = False
        self.in_abstract = False
        self.abstract_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name", "").startswith("citation_"):
            self.meta.setdefault(attrs["name"], []).append(html.unescape(attrs.get("content", "")))
        if tag == "div" and "abs-license" in attrs.get("class", ""):
            self.in_license = True
        if tag == "a" and (self.in_license or attrs.get("id") == "license-link" or "license" in attrs.get("class", "")):
            self.license_url = attrs.get("href", "")
        if tag == "blockquote" and "abstract" in attrs.get("class", ""):
            self.in_abstract = True

    def handle_endtag(self, tag):
        if tag == "div":
            self.in_license = False
        if tag == "blockquote":
            self.in_abstract = False

    def handle_data(self, value):
        if self.in_abstract:
            self.abstract_parts.append(value)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def download_paper(item):
    key, arxiv_id, venue, publication_year = item
    page_url = "https://arxiv.org/abs/" + arxiv_id
    previous = {}
    manifest_path = CORPUS / "manifest.json"
    if manifest_path.exists():
        previous = next((p for p in json.loads(manifest_path.read_text(encoding="utf-8"))["papers"] if p["paper_key"] == key), {})
    with httpx.Client(follow_redirects=True, timeout=90, headers={"User-Agent": "CourseResearchCorpus/1.0 (educational source verification)"}) as client:
        response = client.get("https://arxiv.org/abs/" + previous.get("arxiv_version", arxiv_id))
        response.raise_for_status()
        parser = ArxivMetadata()
        parser.feed(response.text)
        revisions = re.findall(re.escape(arxiv_id) + r"v(\d+)", response.text)
        revision = max(map(int, revisions)) if revisions else 1
        version = previous.get("arxiv_version", arxiv_id + "v" + str(revision))
        revision = int(version.rsplit("v", 1)[1])
        pdf_url = "https://arxiv.org/pdf/" + version
        path = CORPUS / (key + "_" + version + ".pdf")
        downloaded_at = previous.get("downloaded_at", "")
        if not path.exists():
            response = client.get(pdf_url)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF-"):
                raise ValueError(f"{key}: 下载响应不是 PDF")
            # Validate the complete body before atomically publishing a file.
            with pymupdf.open(stream=response.content, filetype="pdf") as checked:
                if not len(checked) or not checked[0].get_text().strip():
                    raise ValueError(f"{key}: 下载PDF不可读取")
            temporary = path.with_suffix(".pdf.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            downloaded_at = stamp()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous.get("sha256") and digest != previous["sha256"]:
            raise ValueError(f"{key}: 本地文件哈希与固定manifest不一致，拒绝覆盖既有版本")
        with pymupdf.open(path) as pdf:
            pages = len(pdf)
            first_page = pdf[0].get_text()
            if not first_page.strip():
                raise ValueError(f"{key}: PDF 无可抽取文本")
        license_url = parser.license_url
        if not license_url:
            match = re.search(r'href="([^"]+)"[^>]*>\s*view license', response.text)
            license_url = html.unescape(match.group(1)) if match else ""
        if license_url.startswith("/"):
            license_url = "https://arxiv.org" + license_url
        result = {"paper_key": key, "arxiv_id": arxiv_id, "arxiv_version": version,
                  "title": parser.meta.get("citation_title", [""])[0],
                  "authors": parser.meta.get("citation_author", []),
                  "abstract": re.sub(r"^\s*Abstract:\s*", "", " ".join(parser.abstract_parts)).strip(),
                  "first_submission_date": parser.meta.get("citation_date", [""])[0],
                  "publication_year": publication_year, "venue": venue,
                  "source_url": page_url, "version_url": page_url + "v" + str(revision),
                  "pdf_url": pdf_url, "license_url": license_url,
                  "local_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                  "sha256": digest, "byte_count": path.stat().st_size,
                  "physical_pages": pages, "downloaded_at": downloaded_at or stamp(), "metadata_checked_at": stamp(),
                  "purpose": "Local course research and retrieval evaluation; retain author attribution and source licensing",
                  "metadata_review_method": "assistant_source_checked", "human_review_pending": True}
        print(f"{key}: {pages} pages; {license_url}; {result['title']}", flush=True)
        return result


def download(args):
    CORPUS.mkdir(parents=True, exist_ok=True)
    DATASET.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        papers = list(executor.map(download_paper, PAPERS))
    manifest = {"schema_version": 1, "topic": "Vision Transformers for image classification: architecture, data efficiency and training",
                "created_at": stamp(), "paper_count": len(papers), "papers": papers,
                "review_method": "assistant_source_checked", "human_review_pending": True}
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for paper in papers:
        with pymupdf.open(ROOT / paper["local_file"]) as pdf:
            pages = [{"page": n + 1, "text": page.get_text(sort=True)} for n, page in enumerate(pdf)]
        (DATASET / (paper["paper_key"] + "_pages.json")).write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", CORPUS / "manifest.json", flush=True)


def read_manifest():
    path = CORPUS / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("缺少corpus manifest，请先执行 download")
    return json.loads(path.read_text(encoding="utf-8"))


def normalise(text):
    # Same literal matching rule as src.evaluation.metrics.
    return re.sub(r"\s+", "", text).casefold()


def verify_files():
    manifest = read_manifest()
    for paper in manifest["papers"]:
        path = ROOT / paper["local_file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != paper["sha256"]:
            raise ValueError(f"{paper['paper_key']}: 原件缺失或sha256不匹配")
        with pymupdf.open(path) as pdf:
            if len(pdf) != paper["physical_pages"]:
                raise ValueError(f"{paper['paper_key']}: 物理页数不匹配")
            if any(not page.get_text().strip() for page in pdf):
                raise ValueError(f"{paper['paper_key']}: 存在空白或无文本页，需人工处理")
        if not paper.get("title") or not paper.get("authors") or not paper.get("license_url"):
            raise ValueError(f"{paper['paper_key']}: 缺少官方标题、作者或许可信息")
    return manifest


def source_cases(path=None):
    from src.evaluation.dataset import EvaluationCase
    path = Path(path or DATASET / "vision_transformer_80.jsonl")
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    cases = [EvaluationCase.model_validate(item) for item in raw]
    if len(cases) != 80 or len({case.case_id for case in cases}) != 80:
        raise ValueError("主评测集必须是80个唯一题号")
    return raw, cases


def validate(args):
    from collections import Counter
    from src.chunking.strategies import split_blocks
    from src.core.schemas import Document
    from src.data_loader.loaders import load_blocks
    from src.evaluation.metrics import map_evidence_to_chunks

    manifest = verify_files()
    papers = {paper["paper_key"]: paper for paper in manifest["papers"]}
    raw, cases = source_cases(args.cases)
    expected = {"factual": 30, "comparison": 20, "synthesis": 15, "reasoning": 15}
    if Counter(case.category for case in cases) != expected:
        raise ValueError("四类题分配不符合30/20/15/15")
    if Counter(case.split for case in cases) != {"dev": 20, "test": 60}:
        raise ValueError("dev/test不是20/60")
    quote_words = Counter()
    checked = []
    source_blocks = {key: load_blocks(ROOT / paper["local_file"]) for key, paper in papers.items()}
    for item, case in zip(raw, cases):
        if not case.reviewed or item.get("review_method") != "assistant_source_checked" or not item.get("human_review_pending"):
            raise ValueError(f"{case.case_id}: 必须如实保留AI原文核对/人核待完成标记")
        for number, source in enumerate(case.relevant_sources, 1):
            key = source.document_id
            if key not in papers or source.page is None:
                raise ValueError(f"{case.case_id}: 证据文档键或物理页无效")
            with pymupdf.open(ROOT / papers[key]["local_file"]) as pdf:
                if source.page > len(pdf):
                    raise ValueError(f"{case.case_id}: 物理页越界")
                text = pdf[source.page - 1].get_text(sort=True)
                if normalise(source.quote) not in normalise(text):
                    raise ValueError(f"{case.case_id}: 引用未出现在原始PDF的指定页")
            quote_words[key] += len(source.quote.split())
            block_matches = [b for b in source_blocks[key] if b.page == source.page and normalise(source.quote) in normalise(b.text)]
            if not block_matches:
                raise ValueError(f"{case.case_id}: 引用被当前pymupdf解析丢失")
            checked.append({"case_id": case.case_id, "evidence_number": number, "paper_key": key,
                            "physical_page": source.page, "quote": source.quote,
                            "paragraph_candidates": [b.paragraph for b in block_matches],
                            "bbox_candidates": [b.bbox for b in block_matches], "pdf_sha256": papers[key]["sha256"],
                            "exact_anchor_on_pdf_page": True, "exact_anchor_in_loaded_block": True,
                            "human_review_pending": True})
    if any(value > 25 for value in quote_words.values()):
        raise ValueError("某一论文的短证据摘录累计超过25英文词")
    matrix = []
    for strategy in ("fixed", "recursive", "semantic"):
        for size in (256, 512, 1024):
            chunks = []
            for key, paper in papers.items():
                document = Document(document_id=key, name=Path(paper["local_file"]).name, content_hash=paper["sha256"],
                                    format="pdf", status="ready", scope="corpus-validation")
                chunks.extend(split_blocks(document, source_blocks[key], strategy=strategy, size=size, overlap=64))
            missing = []
            for case in cases:
                mapping = map_evidence_to_chunks(case.relevant_sources, chunks)
                missing.extend({"case_id": case.case_id, "evidence_number": i + 1} for i, ids in enumerate(mapping) if not ids)
            if missing:
                raise ValueError(f"{strategy}/{size} 无法映射这些证据：{missing}")
            row = {"strategy": strategy, "chunk_size": size, "chunk_overlap": 64, "chunk_count": len(chunks),
                   "case_count": len(cases), "evidence_count": len(checked), "unmapped_evidence_count": len(missing)}
            matrix.append(row)
            print(f"Validated {strategy}/{size}: {len(chunks)} chunks, all evidence mapped", flush=True)
    result = {"checked_at": stamp(), "paper_count": len(papers), "total_physical_pages": sum(p["physical_pages"] for p in papers.values()),
              "case_count": len(cases), "evidence_count": len(checked), "short_quote_words": dict(quote_words),
              "all_hashes_verified": True, "all_original_physical_pages_verified": True,
              "all_loaded_evidence_verified": True, "chunking_matrix": matrix,
              "evidence_checks": checked, "review_method": "assistant_source_checked", "human_review_pending": True,
              "dataset_sha256": hashlib.sha256(Path(args.cases or DATASET / "vision_transformer_80.jsonl").read_bytes()).hexdigest(),
              "corpus_manifest_sha256": hashlib.sha256((CORPUS / "manifest.json").read_bytes()).hexdigest(),
              "not_an_answer_quality_score": True, "notes": ["此验证确认原件、定位锚点与分块映射，不冒充人工评分或正式模型效果。",
              "简短锚点可能匹配同页多个段落；候选段落与坐标全部保留，人工复核时需确认真正支持答案的段落。"]}
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "source_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Source validation complete: {len(papers)} papers, {len(cases)} cases, {len(checked)} evidence anchors", flush=True)


def validate_index(args):
    from src.core.storage import Storage
    from src.evaluation.metrics import map_evidence_to_chunks

    token = args.token or os.environ.get("RAG_WORKSPACE_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", token):
        raise ValueError("请用RAG_WORKSPACE_TOKEN指定现有知识库范围")
    directory = Path(args.data_dir).resolve()
    if not (directory / "state.sqlite3").is_file():
        raise FileNotFoundError("当前data-dir没有已存在的state.sqlite3；拒绝创建空库冒充验证")
    store = Storage(directory)
    path = args.cases or DATASET / "vision_transformer_80_mapped.jsonl"
    _, cases = source_cases(path)
    chunks = store.list_chunks(token)
    documents = {doc.document_id: doc for doc in store.list_documents(token)}
    manifest = verify_files()
    expected_hashes = {paper["sha256"] for paper in manifest["papers"]}
    rows = []
    for case in cases:
        for source in case.relevant_sources:
            doc = documents.get(source.document_id)
            if not doc or doc.status != "ready" or doc.content_hash not in expected_hashes:
                raise ValueError(f"{case.case_id}: 实际docID未就绪或不是该固定论文语料")
        mapped = map_evidence_to_chunks(case.relevant_sources, chunks)
        missing = [i + 1 for i, matches in enumerate(mapped) if not matches]
        if missing:
            raise ValueError(f"{case.case_id}: 当前持久化分块无法映射证据{missing}")
        rows.append({"case_id": case.case_id, "matching_chunk_ids": [sorted(matches) for matches in mapped]})
    result = {"checked_at": stamp(), "scope_fingerprint": hashlib.sha256(token.encode()).hexdigest()[:24],
              "index_version": store.index_version(token), "case_count": len(cases), "chunk_count": len(chunks),
              "referenced_document_count": len({s.document_id for case in cases for s in case.relevant_sources}),
              "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "all_actual_ids_ready": True, "all_actual_evidence_mapped": True,
              "review_method": "assistant_source_checked", "human_review_pending": True, "cases": rows}
    target = Path(args.output or DATASET / "index_validation.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated actual index: {len(cases)} cases, {len(chunks)} chunks; report {target}", flush=True)


def import_corpus(args):
    """Import existing exact PDFs; map keys to server IDs; never print the token."""
    manifest = verify_files()
    token = args.token or os.environ.get("RAG_WORKSPACE_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", token):
        raise ValueError("请用RAG_WORKSPACE_TOKEN提供与界面相同的workspace token（16-128字符）")
    headers = {"X-Workspace-Token": token}
    base = args.base_url.rstrip("/")
    mapping, last_status = {}, {}
    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        for paper in manifest["papers"]:
            path = ROOT / paper["local_file"]
            with path.open("rb") as stream:
                response = client.post("/documents", files=[("files", (path.name, stream, "application/pdf"))])
            response.raise_for_status()
            data = response.json()
            document = data["documents"][0]
            if document["content_hash"] != paper["sha256"]:
                raise ValueError("服务器返回了不同文件的文档记录")
            mapping[paper["paper_key"]] = document["document_id"]
            if document["status"] == "error":
                response = client.post(f"/documents/{document['document_id']}/retry")
                response.raise_for_status()
            print(f"Submitted {paper['paper_key']}: {document['document_id']}", flush=True)
        deadline = time.monotonic() + args.wait_seconds
        while True:
            response = client.get("/documents")
            response.raise_for_status()
            documents = {d["document_id"]: d for d in response.json()}
            pending = []
            for key, identifier in mapping.items():
                document = documents.get(identifier)
                if not document:
                    raise ValueError(f"{key}: 导入期间文档被删除")
                status = document["status"]
                if last_status.get(key) != status:
                    print(f"{key}: {status}", flush=True)
                    last_status[key] = status
                if status == "error":
                    raise RuntimeError(f"{key}: {document.get('error', '索引失败')}")
                if status != "ready":
                    pending.append(key)
            if not pending:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("等待索引超时；已提交任务保留在服务器，请检查Jobs后重跑map")
            time.sleep(2)
        for paper in manifest["papers"]:
            payload = {"title": paper["title"], "authors": paper["authors"], "year": paper["publication_year"],
                       "abstract": paper.get("abstract", ""), "doi": "10.48550/arXiv." + paper["arxiv_id"],
                       "aliases": [paper["paper_key"], paper["arxiv_id"],
                                   {"vit": "ViT", "deit": "DeiT", "swin": "Swin", "pvt": "PVT", "t2t": "T2T-ViT",
                                    "cait": "CaiT", "cvt": "CvT", "crossvit": "CrossViT", "levit": "LeViT", "beit": "BEiT"}[paper["paper_key"]]],
                       "source_url": paper["source_url"]}
            response = client.post(f"/documents/{mapping[paper['paper_key']]}/metadata", json=payload)
            response.raise_for_status()
    mapping_record = {"created_at": stamp(), "base_url": base, "document_ids": mapping,
                      "corpus_hashes": {p["paper_key"]: p["sha256"] for p in manifest["papers"]},
                      "note": "No workspace token is stored; IDs belong to the workspace used during import."}
    DATASET.mkdir(parents=True, exist_ok=True)
    mapping_target = Path(args.mapping_output or DATASET / "index-map.json")
    mapping_target.parent.mkdir(parents=True, exist_ok=True)
    mapping_target.write_text(json.dumps(mapping_record, ensure_ascii=False, indent=2), encoding="utf-8")
    map_cases(mapping, args.cases, args.output)


def map_cases(mapping, cases_path=None, output_path=None):
    raw, _ = source_cases(cases_path)
    for case in raw:
        for source in case["relevant_sources"]:
            key = source["document_id"]
            if key not in mapping:
                raise ValueError(f"缺少paper_key到实际document_id映射: {key}")
            source["document_id"] = mapping[key]
        case["annotation_note"] = "AI原文核对，尚待人工复核；document_id已映射实际知识库，原始出处保留在source_provenance。"
    path = Path(output_path or DATASET / "vision_transformer_80_mapped.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in raw) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        target = path.with_name(path.stem + "_" + split + path.suffix)
        target.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in raw if case["split"] == split) + "\n", encoding="utf-8")
    print(f"Mapped {len(raw)} cases to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Explicitly download pinned open PDFs and authoritative metadata")
    check = commands.add_parser("validate", help="Offline checksum, exact page anchors and 9 chunk configurations")
    check.add_argument("--cases", type=Path)
    load = commands.add_parser("import", help="Upload existing PDFs, apply authoritative metadata and map evaluation IDs")
    load.add_argument("--base-url", default="http://127.0.0.1:8002")
    load.add_argument("--token", default="", help="Prefer RAG_WORKSPACE_TOKEN environment variable to avoid shell history")
    load.add_argument("--wait-seconds", type=int, default=7200)
    load.add_argument("--cases", type=Path)
    load.add_argument("--output", type=Path)
    load.add_argument("--mapping-output", type=Path, help="Separate ID map for an independently imported workspace")
    remap = commands.add_parser("map", help="Map an existing JSON document-ID mapping without calling the server")
    remap.add_argument("--mapping", type=Path, default=DATASET / "index-map.json")
    remap.add_argument("--cases", type=Path)
    remap.add_argument("--output", type=Path)
    current = commands.add_parser("validate-index", help="Verify mapped evidence against the current local persistent index")
    current.add_argument("--data-dir", required=True, type=Path)
    current.add_argument("--token", default="", help="Prefer RAG_WORKSPACE_TOKEN environment variable")
    current.add_argument("--cases", type=Path)
    current.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "import":
        import_corpus(args)
    elif args.command == "map":
        record = json.loads(args.mapping.read_text(encoding="utf-8"))
        map_cases(record.get("document_ids", record), args.cases, args.output)
    elif args.command == "validate-index":
        validate_index(args)


if __name__ == "__main__":
    main()
