from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Iterable

from src.core.schemas import SourceBlock


SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
PDF_PARSERS = {"pymupdf", "pdfplumber", "pypdf2"}


def _clean(text: str) -> str:
    """Keep equations, Unicode symbols and line breaks; remove only control noise."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _kind(text: str) -> str:
    if len(text) < 500 and re.search(r"[=∑∏∫√≤≥≈]|\\(?:frac|sum|int)\b", text):
        return "formula"
    return "text"


def _heading(text: str) -> bool:
    text = text.strip()
    if len(text) > 110 or len(text.splitlines()) > 2:
        return False
    return bool(re.match(
        r"^(?:\d+(?:\.\d+)*\s+[A-Z\u4e00-\u9fff]|[一二三四五六七八九十]+[、．.]|"
        r"(?:abstract|introduction|background|related work|method(?:s|ology)?|"
        r"experiments?|results?|discussion|conclusion(?:s)?|references|"
        r"摘要|引言|背景|方法|实验|结果|讨论|结论|参考文献)\b)", text, re.I))


def _markdown_table(rows: Iterable[Iterable]) -> str:
    rows = [[_clean(str(cell or "")).replace("\n", "<br>").replace("|", "\\|")
             for cell in row] for row in rows]
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    # Preserve the first row as extracted, rather than inventing column titles.
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _inside(rect: list[float], tables: list[list[float]]) -> bool:
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    return any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in tables)


def _reading_order(blocks: list[SourceBlock], width: float) -> list[SourceBlock]:
    """Read full-width bands then left/right columns; keep original bounding boxes.

    This is a layout heuristic for text-layer PDFs, not OCR or a universal layout
    model. Full-width titles/tables split the page into independent reading bands.
    """
    if not blocks or any(b.bbox is None for b in blocks):
        return blocks
    mid = width / 2
    left = [b for b in blocks if b.bbox[2] <= mid + width * .03]
    right = [b for b in blocks if b.bbox[0] >= mid - width * .03]
    # Avoid turning a single-column page's short headings into fake columns.
    overlap = sum(1 for a in left for b in right
                  if min(a.bbox[3], b.bbox[3]) >= max(a.bbox[1], b.bbox[1]))
    if len(left) < 2 or len(right) < 2 or overlap < 2:
        return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    full = sorted([b for b in blocks if b not in left and b not in right],
                  key=lambda b: (b.bbox[1], b.bbox[0]))
    pending = [b for b in blocks if b not in full]
    result: list[SourceBlock] = []
    for anchor in full:
        band = [b for b in pending if b.bbox[1] < anchor.bbox[1]]
        result.extend(sorted(band, key=lambda b: (b.bbox[0] >= mid, b.bbox[1], b.bbox[0])))
        pending = [b for b in pending if b not in band]
        result.append(anchor)
    result.extend(sorted(pending, key=lambda b: (b.bbox[0] >= mid, b.bbox[1], b.bbox[0])))
    return result


def _pymupdf(path: Path) -> list[SourceBlock]:
    import pymupdf as fitz

    blocks = []
    with fitz.open(path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF 已加密，请上传解除密码的副本")
        for page_num, page in enumerate(pdf, 1):
            local: list[SourceBlock] = []
            table_boxes = []
            # Keep a textual fallback on malformed tables: extraction exceptions
            # affect only table detection, never discard ordinary page text.
            try:
                tables = page.find_tables().tables
            except (ValueError, RuntimeError, AttributeError):
                tables = []
            for table in tables:
                try:
                    text = _markdown_table(table.extract())
                except (ValueError, TypeError, RuntimeError):
                    continue
                if text:
                    bbox = list(map(float, table.bbox))
                    table_boxes.append(bbox)
                    local.append(SourceBlock(text=text, page=page_num, bbox=bbox, kind="table"))
            content = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, sort=False)
            fonts = [s["size"] for b in content["blocks"] if b.get("type") == 0
                     for line in b.get("lines", []) for s in line.get("spans", []) if s.get("text", "").strip()]
            normal_size = median(fonts) if fonts else 11
            for raw in content["blocks"]:
                if raw.get("type") != 0:
                    continue
                # MuPDF sometimes joins same-height lines from different
                # columns into a single block. Separate these before ordering.
                raw_lines = raw.get("lines", [])
                mid, tolerance = page.rect.width / 2, page.rect.width * .03
                left_lines = [line for line in raw_lines if line["bbox"][2] <= mid + tolerance]
                right_lines = [line for line in raw_lines if line["bbox"][0] >= mid - tolerance]
                if left_lines and right_lines:
                    other = [line for line in raw_lines if line not in left_lines and line not in right_lines]
                    groups = [left_lines, right_lines] + ([other] if other else [])
                else:
                    groups = [raw_lines]
                for group in groups:
                    lines = []
                    for line in group:
                        spans = []
                        for span in line.get("spans", []):
                            value = span.get("text", "")
                            # Superscript spans are otherwise collapsed (e.g. x2).
                            if span.get("flags", 0) & 1 and value.strip():
                                value = "^{" + value + "}"
                            spans.append(value)
                        lines.append("".join(spans))
                    text = _clean("\n".join(lines))
                    rectangles = [line["bbox"] for line in group]
                    if not rectangles:
                        continue
                    bbox = [min(r[0] for r in rectangles), min(r[1] for r in rectangles),
                            max(r[2] for r in rectangles), max(r[3] for r in rectangles)]
                    if not text or _inside(bbox, table_boxes):
                        continue
                    sizes = [s["size"] for line in group for s in line.get("spans", [])]
                    kind = "heading" if (_heading(text) or (len(text) < 120 and sizes and max(sizes) > normal_size * 1.22)) else _kind(text)
                    local.append(SourceBlock(text=text, page=page_num, bbox=bbox, kind=kind))
            blocks.extend(_reading_order(local, page.rect.width))
    return blocks


def _pdfplumber(path: Path) -> list[SourceBlock]:
    import pdfplumber

    result = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.find_tables()
            table_boxes = [list(map(float, t.bbox)) for t in tables]
            local = [SourceBlock(text=_markdown_table(t.extract()), page=page_num,
                                 bbox=list(map(float, t.bbox)), kind="table") for t in tables]
            words = [w for w in page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
                     if not _inside([w["x0"], w["top"], w["x1"], w["bottom"]], table_boxes)]
            words.sort(key=lambda w: (round(w["top"] / 3), w["x0"]))
            rows = []
            for word in words:
                if rows and abs(rows[-1][0]["top"] - word["top"]) <= 3:
                    rows[-1].append(word)
                else:
                    rows.append([word])
            for row in rows:
                row.sort(key=lambda w: w["x0"])
                # A large horizontal gap splits same-height column text.
                groups: list[list] = [[]]
                for word in row:
                    if groups[-1] and word["x0"] - groups[-1][-1]["x1"] > max(24, page.width * .035):
                        groups.append([])
                    groups[-1].append(word)
                for group in groups:
                    text = _clean(" ".join(w["text"] for w in group))
                    if text:
                        bbox = [min(w["x0"] for w in group), min(w["top"] for w in group),
                                max(w["x1"] for w in group), max(w["bottom"] for w in group)]
                        local.append(SourceBlock(text=text, page=page_num, bbox=bbox,
                                                 kind="heading" if _heading(text) else _kind(text)))
            result.extend(_reading_order([b for b in local if b.text], page.width))
    return result


def _pypdf2(path: Path) -> list[SourceBlock]:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，请上传解除密码的副本")
    result = []
    for page_num, page in enumerate(reader.pages, 1):
        # PyPDF2's text ordering and table layout are deliberately left intact
        # for parser experiments. It does not supply reliable page rectangles.
        text = _clean(page.extract_text() or "")
        for paragraph in re.split(r"\n\s*\n", text):
            if paragraph.strip():
                result.append(SourceBlock(text=paragraph.strip(), page=page_num,
                                          kind="heading" if _heading(paragraph) else _kind(paragraph)))
    return result


def _docx(path: Path) -> list[SourceBlock]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn

    doc = Document(path)
    result = []
    section = ""
    for element in doc.element.body.iterchildren():
        if element.tag == qn("w:p"):
            paragraph = Paragraph(element, doc)
            text = paragraph.text
            # Office Math has separate m:t nodes, omitted by paragraph.text.
            formulas = element.xpath(".//m:t")
            if formulas:
                math_text = " ".join(node.text or "" for node in formulas)
                text += ("\n[Office Math] " + math_text)
            text = _clean(text)
            if not text:
                continue
            heading = paragraph.style is not None and (paragraph.style.name.lower().startswith("heading") or paragraph.style.name.startswith("标题"))
            if heading:
                section = text
            result.append(SourceBlock(text=text, section=section,
                                      kind="heading" if heading else ("formula" if formulas else _kind(text))))
        elif element.tag == qn("w:tbl"):
            table = Table(element, doc)
            text = _markdown_table([[cell.text for cell in row.cells] for row in table.rows])
            if text:
                result.append(SourceBlock(text=text, section=section, kind="table"))
    return result


def _text(path: Path) -> list[SourceBlock]:
    raw = path.read_bytes()
    decoded = None
    encodings = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    if decoded is None:
        raise ValueError("无法解码文本，请将文件另存为 UTF-8")
    markdown = path.suffix.lower() in {".md", ".markdown"}
    lines = decoded.replace("\r\n", "\n").splitlines()
    result, buffer = [], []
    section, fence = "", None

    def flush(kind="text"):
        nonlocal buffer
        text = _clean("\n".join(buffer))
        if text:
            result.append(SourceBlock(text=text, section=section, kind=kind if kind != "text" else _kind(text)))
        buffer = []

    for line in lines:
        stripped = line.strip()
        if markdown and re.match(r"^(```|~~~)", stripped):
            if fence:
                buffer.append(line)
                flush("code")
                fence = None
            else:
                flush()
                fence = stripped[:3]
                buffer.append(line)
            continue
        if fence:
            buffer.append(line)
        elif markdown and re.match(r"^#{1,6}\s+", stripped):
            flush()
            section = re.sub(r"^#{1,6}\s+", "", stripped).rstrip("# ")
            result.append(SourceBlock(text=stripped, section=section, kind="heading"))
        elif not stripped:
            flush("table" if markdown and buffer and sum("|" in row for row in buffer) >= 2 else "text")
        else:
            buffer.append(line)
    flush("code" if fence else ("table" if markdown and buffer and sum("|" in row for row in buffer) >= 2 else "text"))
    return result


def load_blocks(path: str | Path, pdf_parser: str = "pymupdf") -> list[SourceBlock]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持 {suffix or '无扩展名'}；支持 PDF/DOCX/TXT/Markdown")
    if suffix == ".pdf":
        parser = pdf_parser.lower().replace("-", "").replace("_", "")
        if parser not in PDF_PARSERS:
            raise ValueError(f"未知 PDF 解析器 {pdf_parser}，可选 {sorted(PDF_PARSERS)}")
        blocks = {"pymupdf": _pymupdf, "pdfplumber": _pdfplumber, "pypdf2": _pypdf2}[parser](path)
    elif suffix == ".docx":
        blocks = _docx(path)
    else:
        blocks = _text(path)
    if not blocks:
        raise ValueError("未提取到文本；扫描 PDF 需要先进行 OCR，当前解析器不会编造扫描页内容")
    section = ""
    previous_table = None
    for index, block in enumerate(blocks, 1):
        if block.kind == "heading":
            section = block.text.lstrip("# ").strip()
        if not block.section:
            block.section = section
        if block.kind == "table":
            header = block.text.splitlines()[0] if block.text else ""
            if (previous_table is not None and previous_table.page is not None and block.page is not None
                    and block.page == previous_table.page + 1
                    and previous_table.text.splitlines()[0] == header):
                # Keep each physical page's rows and box. A repeated extracted
                # header links continuations without inventing missing rows.
                block.section = previous_table.section or section
            previous_table = block
        block.paragraph = index
    return blocks


def extract_metadata(blocks: list[SourceBlock], name: str) -> dict:
    """Conservative extraction: absent author/date/DOI remain absent."""
    texts = [b.text for b in blocks if b.text.strip()]
    front = "\n".join(texts[:20])[:18000]
    title = Path(name).stem
    for text in texts[:5]:
        cleaned = text.lstrip("# ").strip()
        if 5 <= len(cleaned) <= 300 and not re.match(r"^(?:abstract|摘要|arxiv|doi|https?://)", cleaned, re.I):
            title = cleaned.replace("\n", " ")
            break
    doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", front, re.I)
    doi = doi_match.group(0).rstrip(".,;)]}") if doi_match else ""
    year_match = re.search(r"(?:published|publication|copyright|©|发表|出版|年份|year)\s*[:：]?[^\n]{0,35}?((?:19|20)\d{2})", front, re.I)
    if not year_match:
        year_match = re.search(r"\b((?:19|20)\d{2})\b", front[:3500])
    author_match = re.search(r"^(?:authors?|作者)\s*[:：]\s*(.+)$", front, re.I | re.M)
    authors = re.split(r"\s*(?:;|；|、|\band\b)\s*", author_match.group(1)) if author_match else []
    abstract = ""
    start = re.search(r"(?:^|\n)(?:#{1,6}\s*)?(?:abstract|摘要)\s*[:：.\-]?\s*", front, re.I)
    if start:
        rest = front[start.end():]
        stop = re.search(r"\n(?:#{1,6}\s*|\d+(?:\.\d+)*\s+)?(?:keywords?|关键词|introduction|引言|背景|1\s)\b", rest, re.I)
        abstract = rest[:stop.start() if stop else 2500].strip()[:5000]
    return {"title": title, "authors": [a.strip() for a in authors if a.strip()],
            "year": int(year_match.group(1)) if year_match else None,
            "abstract": abstract, "doi": doi}
