from __future__ import annotations

import json


def citation_lines(citations: list[dict]) -> list[str]:
    lines = []
    for citation in citations:
        start, end = citation.get("page_start"), citation.get("page_end")
        location = f"物理页 {start}" + (f"–{end}" if end and end != start else "") if start else f"段落 {citation.get('paragraph') or '未标记'}"
        lines.append(f"- [{citation.get('citation_id', '?')}] {citation.get('name', '文档')} · {location}")
        lines.append("  " + citation.get("text", "").replace("\n", "\n  "))
    return lines


def export_session(session: dict, format: str) -> bytes:
    if format == "json":
        return json.dumps(session, ensure_ascii=False, indent=2).encode("utf-8")
    if format != "md":
        raise ValueError("会话导出仅支持 md/json")
    lines = ["# " + session.get("title", "科研对话"), "", "创建时间：" + session.get("created_at", ""), ""]
    for message in session.get("messages", []):
        lines += ["## " + ("用户" if message.get("role") == "user" else "科研助理"), "", message.get("content", ""), ""]
        if message.get("citations"):
            lines += ["### 证据", "", *citation_lines(message["citations"]), ""]
    return "\n".join(lines).encode("utf-8")


def export_run(run: dict, format: str) -> bytes:
    if format == "json":
        return json.dumps(run, ensure_ascii=False, indent=2).encode("utf-8")
    if format != "md":
        raise ValueError("运行导出仅支持 md/json")
    lines = ["# 问答执行记录", "", run.get("question", ""), "", "状态：" + run.get("status", ""), "",
             run.get("answer", ""), "", "## 证据", "", *citation_lines(run.get("citations", [])), "",
             "## 指标", "", "```json", json.dumps(run.get("metrics", {}), ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines).encode("utf-8")
