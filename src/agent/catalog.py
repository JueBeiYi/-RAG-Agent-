"""Scoped catalogue resolution; ambiguous paper names never pick the first row."""
from __future__ import annotations

import re
import unicodedata


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", unicodedata.normalize("NFKC", str(value)).casefold())


def scoped_documents(context):
    return [doc for doc in context.store.list_documents(context.scope)
            if not context.document_ids or field(doc, "document_id") in context.document_ids]


def paper_names(doc) -> list[str]:
    title = field(doc, "title", "") or ""
    filename = field(doc, "name", "") or ""
    stem = re.sub(r"\.(?:pdf|docx|txt|md|markdown)$", "", filename, flags=re.I)
    metadata = field(doc, "metadata", {}) or {}
    aliases = metadata.get("aliases", [])
    aliases = [aliases] if isinstance(aliases, str) else list(aliases) if isinstance(aliases, list) else []
    names = [str(field(doc, "document_id", "")), title, filename, stem]
    names.extend(str(metadata.get(key, "")) for key in ("short_title", "acronym"))
    names.extend(str(alias) for alias in aliases)
    prefix = re.split(r"[:：]", title, maxsplit=1)[0]
    if prefix and prefix != title and len(prefix) >= 6:
        names.append(prefix)
    names.extend(re.findall(r"[（(]([A-Za-z][A-Za-z0-9-]{1,15})[）)]", title))
    # Distinctive first tokens cover names like Swin/ResNet without accepting
    # generic beginnings such as A, The, Training, Attention or Transformer.
    first = re.match(r"([A-Za-z][A-Za-z0-9-]*)", title)
    generic = {"a", "an", "the", "on", "analyzing", "training", "attention", "transformer", "transformers",
               "learning", "deep", "towards", "rethinking", "image", "vision", "efficient", "paper"}
    if first and len(first.group(1)) >= 4 and first.group(1).casefold() not in generic:
        names.append(first.group(1))
    return list(dict.fromkeys(name for name in names if name.strip()))


def resolve_reference(reference: str, context) -> list[str]:
    """Resolve a single explicit ID/title/alias, returning every ambiguity."""
    docs = scoped_documents(context)
    reference_key = normalize(reference)
    if not reference_key:
        return []
    exact = [field(doc, "document_id") for doc in docs
             if any(normalize(name) == reference_key for name in paper_names(doc))]
    if exact:
        return list(dict.fromkeys(exact))
    if len(reference_key) < 4:
        return []
    partial = [field(doc, "document_id") for doc in docs
               if reference_key in normalize(field(doc, "title", "") or field(doc, "name", ""))]
    return list(dict.fromkeys(partial))


def resolve_mentions(message: str, context) -> list[str]:
    docs = scoped_documents(context)
    normalized_message = normalize(message)
    full_matches = []
    for doc in docs:
        identifiers = [field(doc, "document_id", ""), field(doc, "title", ""),
                       re.sub(r"\.(?:pdf|docx|txt|md|markdown)$", "", field(doc, "name", ""), flags=re.I)]
        if any(len(normalize(value)) >= 3 and normalize(value) in normalized_message for value in identifiers if value):
            full_matches.append(field(doc, "document_id"))
    selected = list(dict.fromkeys(full_matches))
    # Explicit book/quote names also support a shortened but unique title phrase.
    references = re.findall(r"《([^》]+)》|[\"“]([^\"”]+)[\"”]", message)
    for pair in references:
        matches = resolve_reference(next(value for value in pair if value), context)
        if len(matches) == 1:
            selected.extend(matches)
    names_to_ids = {}
    for doc in docs:
        for name in paper_names(doc):
            key = normalize(name)
            if len(key) >= 3:
                names_to_ids.setdefault(key, []).append((name, field(doc, "document_id")))
    for key, candidates in names_to_ids.items():
        ids = list(dict.fromkeys(document_id for _, document_id in candidates))
        if len(ids) != 1:
            continue
        name = candidates[0][0]
        # English aliases require word boundaries: ViT must not match "activity".
        present = (re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", message, re.I)
                   if re.fullmatch(r"[A-Za-z0-9_-]+", name) else key in normalized_message)
        if present:
            selected.append(ids[0])
    return list(dict.fromkeys(selected))
