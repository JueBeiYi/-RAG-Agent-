from __future__ import annotations

import re

from .registry import ToolCall, ToolContext
from .catalog import resolve_mentions


def _field(obj, name, default=""):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def resolve_papers(message: str, context: ToolContext) -> list[str]:
    if context.document_ids:
        return list(context.document_ids)
    selected = resolve_mentions(message, context)
    if selected:
        return selected
    if re.search(r"这篇|该论文|这两篇|这几篇|刚才|上述|它|these papers|this paper", message, re.I):
        for item in reversed(context.store.list_messages(context.session_id, context.scope)):
            citations = item.get("citations", [])
            ids = list(dict.fromkeys(_field(citation, "document_id") for citation in citations if _field(citation, "document_id")))
            if ids:
                # Check ownership rather than accepting IDs from old text blindly.
                return [i for i in ids if context.store.get_document(i, context.scope) is not None]
    docs = context.store.list_documents(context.scope)
    return [_field(docs[0], "document_id")] if len(docs) == 1 else []


def calculator_expression(message: str) -> str | None:
    # Prefer a user-provided backtick expression; do not reinterpret prose numbers.
    code = re.search(r"`([^`]+)`", message)
    if code:
        expression = code.group(1).strip()
        if re.search(r"计算|算一下|算算|calculate", message, re.I):
            return expression
        if re.fullmatch(r"[\d\s.+\-*/%(),eE]+", expression) and re.search(r"[+*/%]|(?<=\d)-(?=\d)", expression):
            return expression
        if re.fullmatch(r"[A-Za-z\d_\s.+\-*/%(),]+", expression) and re.search(r"\b(?:sqrt|log|exp|sin|cos|tan|round|abs|min|max)\s*\(", expression):
            return expression
        return None
    text = re.sub(r"^(?:请|帮我|请帮我)?\s*(?:计算|算一下|算算|calculate)\s*[:：]?\s*", "", message, flags=re.I)
    text = re.sub(r"[=？?。\s]+$", "", text)
    if re.fullmatch(r"[\d\s.+\-*/%(),eE]+", text) and re.search(r"[+*/%]|(?<=\d)-(?=\d)", text):
        return text
    if re.fullmatch(r"[A-Za-z\d_\s.+\-*/%(),]+", text) and re.search(r"\b(?:sqrt|log|exp|sin|cos|tan|round|abs|min|max)\s*\(", text):
        return text
    return None


def use_model_knowledge(message: str, context: ToolContext) -> bool:
    """Route explicit general definitions without treating paper questions as facts.

    Selected papers, named library papers, citations and conversational references
    always keep the grounded route. Time-sensitive queries stay out of this path.
    """
    if context.document_ids or resolve_mentions(message, context):
        return False
    if re.search(r"论文|文献|文章|知识库|原文|这篇|该研究|刚才|上述|它|这两篇|作者|doi|\[S\d+\]|"
                 r"最新|目前|现在|今年|今天|当前|paper|article|document|library|latest|current|this|that|above", message, re.I):
        return False
    if re.search(r"关键词|关键字|摘要|对比|比较|计算|日期|时间|keywords?|summary|compare|calculate", message, re.I):
        return False
    return bool(re.search(r"^(?:请|请问|帮我|请解释|解释一下)?\s*(?:什么是.{1,80}|.{1,80}是什么|"
                          r"解释.{1,80}(?:概念|原理)|explain\s+.{1,80}|what\s+is\s+.{1,80}|define\s+.{1,80})[？?。.!\s]*$", message, re.I))


def route(message: str, context: ToolContext) -> list[ToolCall]:
    """Transparent deterministic routing also supplies the offline baseline.

    Independent explicit clauses are dispatched together; a calculation that
    depends on retrieved values is left to the model's next ReAct iteration.
    """
    ids = resolve_papers(message, context)
    calls = []
    clauses = re.split(r"[；;\n]|(?:，|,)\s*(?:然后|并且|同时|再)", message)
    for clause in clauses:
        expression = calculator_expression(clause.strip())
        if expression:
            calls.append(ToolCall("calculator", {"expression": expression}))
        if re.search(r"现在几点|当前时间|今天(?:是几号|日期)|日期|current time|what time|today.?s date", clause, re.I):
            calls.append(ToolCall("current_time", {}))
    if re.search(r"对比|比较|compare|comparison|区别|异同", message, re.I):
        if len(ids) >= 2:
            calls.append(ToolCall("compare_papers", {"document_ids": ids[:4], "question": message}))
        else:
            calls.append(ToolCall("paper_catalog_search", {"query": "", "limit": 20}))
    else:
        if re.search(r"摘要|总结|概括|summari[sz]e|summary", message, re.I):
            calls.append(ToolCall("summarize_paper", {"document_ids": ids} if ids else {}))
        if re.search(r"关键词|关键字|keywords?|key terms", message, re.I):
            from_question = bool(re.search(r"问题|文本|句子|keywords?\s*(?:from\s+(?:this\s+)?(?:question|text)|:)|关键词\s*[:：]", message, re.I))
            if from_question or not ids:
                query = re.split(r"[:：]", message, maxsplit=1)[-1].strip()
                calls.append(ToolCall("extract_keywords", {"query": query[:12000]}))
            else:
                calls.append(ToolCall("extract_keywords", {"document_ids": ids}))
        if re.search(r"作者|谁写|发表年份|哪一年|doi|元信息|metadata|author|publication year", message, re.I):
            calls.append(ToolCall("paper_metadata", {"document_ids": ids} if ids else {}))
        if re.search(r"论文(?:目录|列表)|有哪些论文|查找论文|检索论文|list papers|find papers|catalog", message, re.I):
            query = re.sub(r".*(?:查找论文|检索论文|find papers)\s*[:：]?\s*", "", message, flags=re.I)
            if query == message:
                query = ""
            calls.append(ToolCall("paper_catalog_search", {"query": query[:500]}))
    if not calls:
        args = {"query": message}
        if ids:
            args["document_ids"] = ids
        calls = [ToolCall("knowledge_qa", args)]
    deduplicated, seen = [], set()
    for call in calls:
        if call.fingerprint() not in seen:
            seen.add(call.fingerprint())
            deduplicated.append(call)
    return deduplicated[:8]
