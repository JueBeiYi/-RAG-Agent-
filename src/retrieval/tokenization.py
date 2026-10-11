from __future__ import annotations

import re


STOPWORDS = {"的", "了", "和", "与", "及", "是", "在", "为", "对", "中", "一", "一个", "这种", "这个",
             "请", "什么", "如何", "哪些", "the", "a", "an", "and", "or", "of", "to", "in", "is",
             "are", "on", "for", "with", "what", "how", "which", "paper", "论文"}
TERM_RE = re.compile(r"[A-Za-z0-9]+(?:[._/+\-][A-Za-z0-9]+)*|[\u3400-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Chinese words/bigrams plus preserved English identifiers and DOI terms."""
    import jieba

    terms = []
    for match in TERM_RE.finditer(text.casefold()):
        term = match.group(0)
        if re.match(r"[\u3400-\u9fff]", term):
            terms.extend(t for t in jieba.cut(term, HMM=False) if t not in STOPWORDS)
            # Unknown technical words still match before a domain dictionary is added.
            terms.extend(term[i:i + 2] for i in range(len(term) - 1)
                         if term[i:i + 2] not in STOPWORDS)
        elif term not in STOPWORDS:
            terms.append(term)
            if re.search(r"[._/+\-]", term):
                terms.extend(part for part in re.split(r"[._/+\-]", term)
                             if len(part) > 1 and part not in STOPWORDS)
    return terms


def lexical_coverage(query: str, document: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    doc_terms = set(tokenize(document))
    return len(query_terms & doc_terms) / len(query_terms)
