from __future__ import annotations

import json
import re
import queue

import streamlit as st

from .client import APIError
from .state import api, cached_get, documents, invalidate, notify, switch_workspace, title

CSS = """<style>
:root{--ink:#19374b;--teal:#287c85;--muted:#627887;--border:#dfe8ed}
.stApp{background:#f6f8fa;color:var(--ink)}
[data-testid="stSidebar"]{background:#eef3f5;border-right:1px solid var(--border)}
.block-container{max-width:1440px;padding:4.2rem 2.4rem 4rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.02em}
h1{font-size:2rem!important}h2{font-size:1.55rem!important}h3{font-size:1.2rem!important}
[data-testid="stChatMessage"]{background:white;border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem}
[data-testid="stMetric"]{background:white;border:1px solid var(--border);padding:.65rem .9rem;border-radius:10px}
[data-testid="stMetricValue"]{font-size:1.6rem}.stButton>button,.stDownloadButton>button{border-radius:8px}
[data-testid="stAppDeployButton"],.stAppDeployButton,[data-testid="stToolbar"] [data-testid="stDeployButton"],.stDeployButton{display:none}
[data-testid="stDialog"] [role="dialog"]{max-height:94vh}
[class*="st-key-citation_view_"] [data-testid="stBaseButton-tertiary"]{justify-content:flex-start!important;text-align:left!important;padding:.25rem .4rem;min-height:2.1rem}
[class*="st-key-citation_view_"] button [data-testid="stMarkdownContainer"]{text-align:left!important;width:100%!important;display:block!important}
[class*="st-key-citation_view_"] button p{text-align:left!important}
@media(max-width:1280px){.block-container{padding-left:1.5rem;padding-right:1.5rem}}
@media(max-width:700px){
 [data-testid="stHorizontalBlock"]:has([class*="st-key-citation_view_"]){flex-wrap:wrap}
 [data-testid="stHorizontalBlock"]:has([class*="st-key-citation_view_"])>[data-testid="stColumn"]{min-width:100%!important;flex:1 1 100%!important}
}
</style>"""

TOOL_NAMES = {"knowledge_qa": "知识库问答", "paper_summary": "论文摘要", "paper_metadata": "论文信息",
              "compare_papers": "论文对比", "paper_catalog_search": "论文目录检索", "keyword_extraction": "关键词提取",
              "extract_keywords": "关键词提取", "calculator": "数值计算", "current_time": "当前日期和时间",
              "summarize_paper": "论文摘要", "get_paper_metadata": "论文信息", "search_papers": "论文检索"}
ORIGINS = {"knowledge_rag": "知识库证据", "model_fallback": "普通模型补充回答", "demo_extractive": "原文抽取功能演示",
           "knowledge_base": "知识库原文", "document_metadata": "论文元信息", "document_analysis": "论文片段分析",
           "document_catalog": "论文目录", "system_clock": "系统日期和时间",
           "agent": "多工具协作", "agent_error": "工具执行含错误", "calculator": "计算器结果"}
STATUS = {"ready": "已就绪", "queued": "等待处理", "processing": "处理中", "running": "执行中", "done": "已完成",
          "error": "失败", "failed": "失败", "cancelled": "已中断", "cancelling": "等待停止"}


def heading(name, description):
    st.title(name)
    st.caption(description)


def friendly_error(exc):
    st.error(str(exc))
    if isinstance(exc, APIError) and exc.status_code is None:
        st.caption("本地启动：在项目目录运行 scripts/start.ps1；也可在侧栏恢复已保存的工作区配置。")


def download(label, path, filename, key, *, params=None, disabled=False, mime=None, width="content"):
    client = api()  # Capture the current scope; deferred callback runs in its own thread.
    notices = st.session_state.setdefault("download_notices", queue.Queue())
    def fetch():
        try:
            return client.bytes(path, params=params)
        except APIError as exc:
            notices.put(f"{label}未完成。{exc}")
            raise
    st.download_button(label, data=fetch, file_name=filename,
                       mime=mime, key=key, disabled=disabled, on_click="ignore", width=width)


@st.fragment(run_every="2s")
def download_feedback():
    notices = st.session_state.get("download_notices")
    errors = st.session_state.setdefault("download_errors", [])
    if notices is not None:
        while True:
            try:
                message = notices.get_nowait()
                if message not in errors:
                    errors.append(message)
            except queue.Empty:
                break
    for message in errors:
        st.error(message)
    if errors and st.button("清除下载提示", key="clear_download_errors"):
        st.session_state.download_errors = []
        st.rerun()


def usage(metrics, *, compact=False):
    if not metrics:
        return
    elapsed = metrics.get("end_to_end_ms", metrics.get("elapsed_ms", metrics.get("latency_ms")))
    token = metrics.get("usage", metrics)
    inputs = token.get("input_tokens", token.get("prompt_tokens"))
    outputs = token.get("output_tokens", token.get("completion_tokens"))
    values = [("耗时", f"{elapsed/1000:.2f}秒" if isinstance(elapsed, (int, float)) else "未记录"),
              ("输入Token", str(inputs) if inputs is not None else "未返回"),
              ("输出Token", str(outputs) if outputs is not None else "未返回")]
    if metrics.get("cache_hit") is not None:
        values.append(("缓存", "命中" if metrics["cache_hit"] else "未命中"))
    if compact:
        st.caption("　　".join(f"{label} {value}" for label, value in values))
    else:
        for column, (label, value) in zip(st.columns(len(values)), values):
            column.metric(label, value)
    if token.get("estimated"):
        st.caption("Token为估算值。")
    tools = metrics.get("tool_token_usage", {})
    if tools:
        with st.expander("按工具统计Token用量"):
            st.dataframe([{"工具": TOOL_NAMES.get(name, name), "调用次数": item.get("calls", 0),
                           "输入Token（已知部分）": item.get("input_tokens"), "输出Token（已知部分）": item.get("output_tokens"),
                           "未知用量调用": item.get("unknown_calls", 0), "估算": bool(item.get("estimated"))}
                          for name, item in tools.items()], hide_index=True, width="stretch")
            planner = metrics.get("planner_usage")
            if planner:
                st.caption(f"规划与综合单独统计：输入Token {planner.get('input_tokens', '未返回')}，输出Token {planner.get('output_tokens', '未返回')}。")


def _trace_tree(events):
    """Group public execution summaries by decision step and tool call."""
    steps, calls = {}, {}
    for event in events:
        kind = event.get("type")
        if kind in {"answer_delta", "citation", "usage", "done"}:
            continue
        step = event.get("step", 0)
        node = steps.setdefault(step, {"步骤": step, "Thought 执行状态摘要": [], "工具调用": []})
        if kind in {"tool_start", "tool_end"}:
            tool = event.get("tool", event.get("tool_name", ""))
            identifier = event.get("call_id") or f"{step}:{tool}"
            key = (step, identifier)
            if key not in calls:
                calls[key] = {"工具": TOOL_NAMES.get(tool, tool), "调用标识": identifier,
                              "Action 输入": {}, "Observation 结果": []}
                node["工具调用"].append(calls[key])
            if kind == "tool_start":
                calls[key]["Action 输入"] = event.get("arguments", {})
            else:
                calls[key]["Observation 结果"].append({
                    "状态": "失败" if event.get("error") else "完成", "尝试": event.get("attempt", 1),
                    "摘要": str(event.get("error") or event.get("summary", event.get("content", "")))[:350],
                    "耗时ms": event.get("elapsed_ms"), "Token": event.get("usage")})
        else:
            text = event.get("text", event.get("message", event.get("summary", "")))
            if text:
                node["Thought 执行状态摘要"].append(str(text)[:350])
    return {"决策步骤": list(steps.values())}


def trace(events, *, expanded=False, label="查看执行过程"):
    with st.expander(label, expanded=expanded):
        selected = [event for event in events if event.get("type") not in {"answer_delta", "citation", "usage", "done"}]
        if not selected:
            st.caption("此记录没有工具或阶段事件。")
        for event in selected:
            kind = event.get("type")
            tool = event.get("tool", event.get("tool_name", ""))
            if kind in {"tool_start", "tool_end"}:
                name = TOOL_NAMES.get(tool, "工具")
                action = "开始" if kind == "tool_start" else "失败" if event.get("error") else "完成"
                st.write(f"{action} · {name}")
                text = event.get("error") or event.get("summary", "")
                if text:
                    st.caption(str(text)[:350])
            else:
                text = event.get("text", event.get("message", event.get("summary", "")))
                if text:
                    st.write(str(text)[:350])
            if event.get("elapsed_ms") is not None:
                st.caption(f"{event['elapsed_ms']:.0f} ms")
            if kind == "tool_end" and event.get("usage") is not None:
                measured = event["usage"]
                if measured.get("unknown"):
                    st.caption("本次工具的模型用量未知。")
                else:
                    st.caption(f"本次工具Token：输入 {measured.get('input_tokens', '未返回')}，输出 {measured.get('output_tokens', '未返回')}" + ("（估算）" if measured.get("estimated") else ""))
        st.caption("按步骤展开状态摘要、Action输入与Observation结果。这里只展示公开执行摘要。")
        st.json(_trace_tree(selected), expanded=False)


def audit(audit_data, origin=""):
    if origin:
        st.caption("回答来源：" + ORIGINS.get(origin, "服务器返回的回答"))
    if not audit_data:
        return
    problems = (audit_data.get("invalid_ids", []) or audit_data.get("unsupported_numbers", [])
                or audit_data.get("warnings", []) or audit_data.get("needs_review", []))
    with st.expander("证据审校" + (" · 需要核对" if problems else " · 引用检查")):
        if problems:
            st.warning("引用或数值存在需要人工核对的项目。")
        else:
            st.caption("自动审校用于检查引用与证据对应关系，关键事实仍需结合原文判断。")
        if audit_data.get("page_reference_corrections"):
            st.caption("回答中的页码引用已按服务器保存的原文位置校正。")
        st.json(audit_data, expanded=False)


def preview_bytes(document_id, page, chunk_id=""):
    cache = st.session_state.preview_cache
    key = (document_id, int(page), chunk_id)
    if key not in cache:
        cache[key] = api().bytes(f"/documents/{document_id}/preview", params={"page": page, "chunk_id": chunk_id})
        while len(cache) > 16:
            cache.pop(next(iter(cache)))
    return cache[key]


def citation_location(citation):
    start, end = citation.get("page_start"), citation.get("page_end")
    if start:
        return f"物理页{start}" + (f"至{end}" if end and end != start else "")
    return f"段落{citation['paragraph']}" if citation.get("paragraph") else "原文片段"


def _dismiss_evidence():
    st.session_state.pop("pending_evidence_dialog", None)


def evidence_dialog(citation):
    st.session_state.pop("pending_delete_dialog", None)
    st.session_state.pop("pending_workspace_dialog", None)
    st.session_state.pending_evidence_dialog = citation
    st.rerun()


@st.dialog("证据与原文", width="large", position="right", on_dismiss=_dismiss_evidence)
def _evidence_dialog(citation):
    st.subheader(citation.get("name", "来源文档"))
    st.caption(f"[{citation.get('citation_id', '')}] · {citation_location(citation)}")
    document_id = citation.get("document_id")
    doc = next((item for item in documents() if item["document_id"] == document_id), None)
    if not doc:
        st.warning("原始文档已不在当前论文库中。历史证据片段仍可查看，原页与原件无法打开。")
        with st.container(height=260, border=True):
            st.code(citation.get("text", "没有保存证据片段。"), language=None, wrap_lines=True)
        return
    download("下载原始文档", f"/documents/{document_id}/file", doc["name"], "evidence_download")
    if doc.get("format") != "pdf":
        st.caption("引用片段与段落位置")
        with st.container(height=360, border=True):
            st.code(citation.get("text", "没有保存证据片段。"), language=None, wrap_lines=True)
        return
    page_tab, quote_tab = st.tabs(["原文页面", "引用片段"])
    with quote_tab:
        with st.container(height=260, border=True):
            st.code(citation.get("text", "没有保存证据片段。"), language=None, wrap_lines=True)
    pages = sorted({block.get("page") for block in citation.get("sources", []) if block.get("page")})
    start, end = citation.get("page_start"), citation.get("page_end")
    if not pages and start:
        pages = list(range(start, (end or start) + 1))
    with page_tab:
        if pages:
            page = st.selectbox("证据所在物理页", pages, key="evidence_page")
            try:
                with st.container(height=580, border=True):
                    st.image(preview_bytes(document_id, page, citation.get("chunk_id", "")), width="stretch")
                st.caption("有来源坐标时高亮证据。页码为PDF物理页码。")
            except APIError as exc:
                friendly_error(exc)
        else:
            st.caption("此证据没有保存可靠页码，可从论文库浏览原始PDF。")


def citations(citations, key):
    if not citations:
        return
    st.caption(f"来源证据 · {len(citations)}条")
    for start in range(0, len(citations), 2):
        for column, i in zip(st.columns(2), range(start, min(start+2, len(citations)))):
            citation = citations[i]
            name = citation.get("name", "来源文档")
            short_name = name if len(name) <= 24 else name[:23] + "…"
            number = citation.get("citation_id") or "未编号证据"
            label = f"[{number}] {short_name} · {citation_location(citation)}"
            with column:
                if st.button(label, key=f"citation_view_{key}_{i}", width="stretch", type="tertiary",
                             help=f"[{number}] {name} · {citation_location(citation)}"):
                    st.session_state.pop("evidence_page", None)
                    evidence_dialog(citation)


def _dismiss_delete():
    st.session_state.pop("pending_delete_dialog", None)


def delete_dialog(kind, record_id, label):
    st.session_state.pop("pending_evidence_dialog", None)
    st.session_state.pop("pending_workspace_dialog", None)
    st.session_state.pending_delete_dialog = {"kind": kind, "record_id": record_id, "label": label}
    st.rerun()


@st.dialog("确认删除", width="small", on_dismiss=_dismiss_delete)
def _delete_dialog(kind, record_id, label):
    st.write(f"将删除{label}。")
    st.caption("删除后该条记录不会继续出现在当前工作区。")
    c1, c2 = st.columns(2)
    if c1.button("取消", key="cancel_delete", width="stretch"):
        _dismiss_delete()
        st.rerun()
    if c2.button("确认删除", key="confirm_delete", type="primary", width="stretch"):
        try:
            api().delete(f"/{kind}/{record_id}")
            _dismiss_delete()
            invalidate()
            if kind == "documents":
                st.session_state.selected_document_ids = [value for value in st.session_state.selected_document_ids if value != record_id]
                st.session_state.pop("doc_detail_selector", None)
            elif kind == "sessions":
                if st.session_state.active_session_id == record_id:
                    st.session_state.active_session_id = None
                st.session_state.pop("session_selector", None)
                st.session_state.pop("compare_history", None)
                st.session_state.pop("comparison_session_id", None)
            elif kind == "experiments":
                st.session_state.pop("experiment_selection", None)
            elif kind == "runs":
                st.session_state.pop("run_selector", None)
            notify("已删除所选记录。")
            st.rerun()
        except APIError as exc:
            friendly_error(exc)


def _dismiss_workspace():
    st.session_state.pop("pending_workspace_dialog", None)


def workspace_dialog():
    st.session_state.pop("pending_evidence_dialog", None)
    st.session_state.pop("pending_delete_dialog", None)
    st.session_state.pending_workspace_dialog = True
    st.rerun()


@st.dialog("工作区设置", width="large", on_dismiss=_dismiss_workspace)
def _workspace_dialog():
    st.caption("保存工作区配置可以在刷新或换浏览器后恢复论文与会话。标识用于访问对应资料，请自行保管配置文件。")
    exported = json.dumps({"version": 1, "api_url": st.session_state.api_url,
                           "workspace_token": st.session_state.workspace_token}, ensure_ascii=False, indent=2)
    st.download_button("保存当前工作区配置", exported, file_name="科研助理工作区.json", mime="application/json",
                       key="workspace_export", on_click="ignore")
    uploaded = st.file_uploader("恢复已保存的工作区", type=["json"], key="workspace_import_file")
    if st.button("恢复配置文件", key="workspace_import", disabled=uploaded is None):
        try:
            value = json.loads(uploaded.getvalue().decode("utf-8-sig"))
            switch_workspace(value)
            notify("已恢复工作区。")
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error("配置文件无法使用。" + str(exc))
    st.divider()
    with st.form("workspace_connection"):
        base = st.text_input("服务地址", value=st.session_state.api_url, key="workspace_api_input")
        token = st.text_input("工作区标识", value=st.session_state.workspace_token, type="password", key="workspace_token_input")
        apply = st.form_submit_button("连接此工作区", key="workspace_apply", type="primary")
    if apply:
        try:
            switch_workspace({"api_url": base, "workspace_token": token})
            notify("工作区已切换。")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def render_dialogs():
    if st.session_state.get("pending_delete_dialog"):
        _delete_dialog(**st.session_state.pending_delete_dialog)
    elif st.session_state.get("pending_evidence_dialog"):
        _evidence_dialog(st.session_state.pending_evidence_dialog)
    elif st.session_state.get("pending_workspace_dialog"):
        _workspace_dialog()
