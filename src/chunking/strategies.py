from __future__ import annotations

import hashlib
import re
from bisect import bisect_left

from src.core.schemas import Chunk, Document, SourceBlock


# Lightweight, reproducible units: one Chinese character or one English word.
# The embedding adapter separately enforces its actual model tokenizer budget.
TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+(?:[.\-/][A-Za-z0-9_]+)*|[^\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _recursive_boundaries(text: str, size: int, overlap: int) -> list[int]:
    """Actual recursive-splitter proposals, located in the unchanged source.

    LangChain's add_start_index search subtracts overlap as characters even when
    length_function counts tokens. Validate its offsets and repair only location
    lookup; the splitter still determines all candidate recursive boundaries.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, length_function=count_tokens,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", " ", ""],
        keep_separator=True, add_start_index=True, strip_whitespace=False,
    )
    boundaries, previous_start = [], -1
    for piece in splitter.create_documents([text]):
        value = piece.page_content
        start = piece.metadata.get("start_index", -1)
        if start < previous_start or start < 0 or text[start:start + len(value)] != value:
            start = text.find(value, max(0, previous_start + 1))
            if start < 0:
                start = text.find(value, max(0, previous_start))
        if start < 0:
            raise ValueError("递归切分结果无法定位到原文，拒绝生成错误来源")
        previous_start = start
        boundaries.append(start + len(value))
    return sorted(set(boundaries))


def _ranges(text: str, strategy: str, size: int, overlap: int,
            atomic: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    matches = list(TOKEN_RE.finditer(text))
    if not matches:
        return []
    n = len(matches)
    starts = [m.start() for m in matches]
    protected = [span for span in (atomic or []) if count_tokens(text[span[0]:span[1]]) <= size]
    cuts = []
    if strategy == "recursive":
        cuts = _recursive_boundaries(text, size, overlap)
    elif strategy != "fixed":
        # Highest priority is paragraph, followed by sentence/row boundaries.
        boundary_re = r"\n\s*\n|[。！？!?](?:[\"'”’）)]*)\s*|(?<=[a-zA-Z0-9])[.](?=\s+[A-Z])\s*|\n"
        cuts = [m.end() for m in re.finditer(boundary_re, text)]
    result = []
    start_token = 0
    while start_token < n:
        end_token = min(start_token + size, n)
        start = 0 if start_token == 0 else matches[start_token].start()
        end = len(text) if end_token == n else matches[end_token].start()
        if strategy != "fixed" and end_token < n:
            lower = matches[min(n - 1, start_token + max(overlap + 1, size // 2))].start()
            options = [cut for cut in cuts if lower <= cut <= end]
            if options:
                if strategy == "recursive":
                    paragraphs = [cut for cut in options if text[max(0, cut - 3):cut].count("\n") >= 2]
                    end = (paragraphs or options)[-1]
                else:
                    end = options[-1]
                while end_token > start_token + 1 and matches[end_token - 1].start() >= end:
                    end_token -= 1
            for low, high in protected:
                if low < end < high:
                    # Move an entire short formula/table row into the next
                    # chunk, rather than cutting a number away from its unit.
                    if low > start:
                        end = low
                    elif high <= (len(text) if start_token + size >= n else matches[start_token + size].start()):
                        end = high
                    end_token = bisect_left(starts, end)
                    break
        result.append((start, end))
        if end >= len(text):
            break
        next_token = max(start_token + 1, end_token - overlap)
        if strategy != "fixed":
            for low, high in protected:
                atom_start, atom_end = bisect_left(starts, low), bisect_left(starts, high)
                if atom_start < next_token < atom_end:
                    next_token = atom_start if atom_start > start_token else atom_end
                # Reduce overlap if including it would stop the next whole
                # atomic unit fitting into the advertised size budget.
                if end == low and atom_end - next_token > size:
                    next_token = atom_start
        start_token = next_token
    return result


def split_blocks(document: Document, blocks: list[SourceBlock], strategy: str = "recursive",
                 size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split with fixed windows, actual RecursiveCharacterTextSplitter proposals,
    or sentence/row boundaries, preserving original source and atomic units.

    ``semantic`` is a reproducible sentence/paragraph boundary baseline, as
    required by the course; it does not pretend to use an embedding boundary
    model. Table rows/equations and cross-page source blocks retain their text
    and provenance even if a very long unit must span more than one chunk.
    Size uses ``count_tokens`` units; the embedding tokenizer is checked later.
    """
    if strategy not in {"fixed", "recursive", "semantic"}:
        raise ValueError("分块策略可选 fixed / recursive / semantic")
    if size < 8 or not 0 <= overlap < size:
        raise ValueError("size 至少为 8，overlap 必须在 [0,size) 内")
    text_parts, offsets, atomic = [], [], []
    position = 0
    for block in blocks:
        value = block.text.strip()
        if not value:
            continue
        if text_parts:
            text_parts.append("\n\n")
            position += 2
        offsets.append((position, position + len(value), block))
        if block.kind in {"formula", "code", "heading"}:
            atomic.append((position, position + len(value)))
        elif block.kind == "table":
            row_start = position
            for row in value.splitlines(keepends=True):
                atomic.append((row_start, row_start + len(row)))
                row_start += len(row)
        text_parts.append(value)
        position += len(value)
    full_text = "".join(text_parts)
    result = []
    for number, (start, end) in enumerate(_ranges(full_text, strategy, size, overlap, atomic)):
        chunk_text = full_text[start:end].strip()
        if not chunk_text:
            continue
        sources = []
        for low, high, source in offsets:
            if low < end and high > start:
                fragment = full_text[max(start, low):min(end, high)].strip()
                if fragment:
                    sources.append(source.model_copy(update={"text": fragment}))
        pages = [s.page for s in sources if s.page is not None]
        rectangles = [s.bbox for s in sources if s.bbox is not None]
        bbox = None
        if rectangles and pages and len(set(pages)) == 1:
            bbox = [min(r[0] for r in rectangles), min(r[1] for r in rectangles),
                    max(r[2] for r in rectangles), max(r[3] for r in rectangles)]
        identity = f"{document.scope}\0{document.document_id}\0{strategy}\0{size}\0{overlap}\0{number}\0{chunk_text}"
        result.append(Chunk(chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:32],
                            document_id=document.document_id, scope=document.scope, name=document.name,
                            text=chunk_text, page_start=min(pages) if pages else None,
                            page_end=max(pages) if pages else None,
                            paragraph=sources[0].paragraph if sources else None,
                            section=next((s.section for s in sources if s.section), ""),
                            bbox=bbox, sources=sources, strategy=strategy,
                            index_version=document.content_hash))
    return result
