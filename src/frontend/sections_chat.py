from __future__ import annotations

import re
import queue
import threading

import streamlit as st

from .client import APIError
from .state import api, as_list, cached_get, documents, invalidate, notify, selected_ready_documents, title
from .ui import TOOL_NAMES, audit, citations, delete_dialog, download, friendly_error, heading, trace, usage

MODE_NAMES = {"vector": "纯向量检索", "hybrid": "混合检索", "hybrid_rerank": "混合检索与重排序"}


def _persist_docs():
    st.session_state.selected_document_ids = list(st.session_state.get("doc_selection", []))
    session_id = st.session_state.active_session_id
    if session_id:
        try:
            api().patch(f"/sessions/{session_id}", json={"document_ids": st.session_state.selected_document_ids})
            invalidate()
        except APIError as exc:
            notify(str(exc), "warning")


def _clear_docs():
    session_id = st.session_state.active_session_id
    if session_id:
        try:
            # Commit first. A stale enabled control must not change the local
            # range when the backend rejects an in-flight session with 409.
            api().patch(f"/sessions/{session_id}", json={"document_ids": []})
        except APIError as exc:
            notify(str(exc), "warning")
            return
    st.session_state.doc_selection = []
    st.session_state.selected_document_ids = []
    invalidate()


def _select_session():
    st.session_state.active_session_id = st.session_state.get("session_selector") or None


def _queue_question(question):
    if not question or not question.strip():
        return
    if not st.session_state.active_session_id:
        try:
            created = api().post("/sessions", json={"title": "新的科研对话"})
        except APIError as exc:
            notify(str(exc), "warning")
            return
        st.session_state.active_session_id = created["id"]
        st.session_state.session_selector = created["id"]
        st.session_state.session_search = ""
        invalidate()
    st.session_state.pending_chat_question = question.strip()


def _submit_question():
    _queue_question(st.session_state.get("chat_input", ""))


def _select_comparison_history(widget_key=None):
    key = widget_key or st.session_state.get("comparison_history_widget_key", "compare_history")
    if key != st.session_state.get("comparison_history_widget_key", "compare_history"):
        return  # Ignore a late event from the previous generation's control.
    st.session_state.comparison_session_id = st.session_state.get(key)


def _new_session():
    try:
        session = api().post("/sessions", json={"title": "新的科研对话"})
    except APIError as exc:
        notify(str(exc), "warning")
        return
    identifier = session["id"]
    # Callbacks run before widgets are registered. Assign the selection rather
    # than deleting it, so the browser cannot resend the previous option.
    st.session_state.active_session_id = identifier
    st.session_state.session_selector = identifier
    st.session_state.session_search = ""
    st.session_state.last_selected_session = identifier
    st.session_state.doc_selection = []
    st.session_state.selected_document_ids = []
    st.session_state.pop("last_failed_chat", None)
    invalidate()


def _prepare_comparison():
    docs = {doc["document_id"]: doc for doc in documents() if doc.get("status") == "ready"}
    first, second = st.session_state.get("compare_a"), st.session_state.get("compare_b")
    aspects = st.session_state.get("compare_aspects", [])
    if first not in docs or second not in docs or first == second or not aspects:
        notify("请选择两篇不同的已就绪论文和至少一个比较维度。", "warning")
        return
    name = f"论文对比 · {title(docs[first])[:24]}与{title(docs[second])[:24]}"
    try:
        created = api().post("/sessions", json={"title": name[:80]})
    except APIError as exc:
        notify(str(exc), "warning")
        return
    identifier = created["id"]
    st.session_state.comparison_session_id = identifier
    epoch = st.session_state.get("comparison_history_epoch", 0) + 1
    st.session_state.comparison_history_epoch = epoch
    st.session_state.comparison_history_widget_key = f"compare_history_{epoch}"
    st.session_state.compare_history = identifier
    st.session_state.selected_document_ids = [first, second]
    question = (f"请比较论文《{title(docs[first])}》与《{title(docs[second])}》。比较维度：{'、'.join(aspects)}。"
                "用表格组织，每项事实引用服务器提供的证据编号。先核对任务、数据集、划分、指标定义和实验条件，"
                "确认可比时才计算差异；原文未说明或不能比较时明确标记，不补造数字。"
                + st.session_state.get("compare_focus", ""))
    st.session_state.pending_comparison_request = {"session_id": identifier, "question": question,
                                                  "document_ids": [first, second]}
    invalidate()


def _cancel_request():
    request_id = st.session_state.get("active_chat_request_id")
    if not request_id:
        notify("正在建立连接，尚未收到可停止的执行标识。", "info")
        return
    try:
        response = api().post(f"/runs/{request_id}/cancel")
        notify(response.get("message", "停止请求已提交，当前模型或工具会在可停止的边界结束。"), "info")
    except APIError as exc:
        notify(str(exc), "warning")


def _network_events(client, session_id, payload, retry, events, abandoned):
    """Network reads stay off the UI thread so stop controls remain usable.

    The worker never calls Streamlit or writes session state. Abandoning a UI
    stream closes it after the current network read; model/tool cancellation is
    still bounded by the backend's real execution boundary.
    """
    try:
        stream = client.retry_last(session_id) if retry else client.chat(session_id, payload)
        try:
            for event in stream:
                if abandoned.is_set():
                    break
                events.put(("event", event))
        finally:
            stream.close()
    except Exception as exc:
        if not abandoned.is_set():
            events.put(("error", exc if isinstance(exc, APIError) else APIError("回答连接发生异常，请查看运行记录后重试。")))
    finally:
        events.put(("end", None))


def stream_answer(session_id, question, document_ids, mode, agent, *, retry=False):
    if not retry:
        with st.chat_message("user"):
            st.write(question)
    failed = False
    cancelled = False
    answer = ""
    events = []
    st.session_state.active_chat_request_id = None
    with st.chat_message("assistant"):
        status = st.status("正在连接科研助理…", expanded=False)
        st.button("请求停止", key="chat_cancel_request", on_click=_cancel_request,
                  help="提交停止请求后，当前模型或工具调用可能仍需释放。已完成的部分保留在运行记录中。")
        display = st.empty()
        heartbeat = st.empty()
        incoming = queue.Queue()
        abandoned = threading.Event()
        client = api()
        worker = threading.Thread(target=_network_events,
            args=(client, session_id, {"message": question, "document_ids": document_ids,
                                      "retrieval_mode": mode, "agent": agent}, retry, incoming, abandoned),
            daemon=True, name="research-ui-sse")
        worker.start()
        try:
            completed = False
            while True:
                try:
                    kind, event = incoming.get(timeout=.2)
                except queue.Empty:
                    # Enqueue a harmless UI update. Streamlit can then dispatch
                    # stop/rename/navigation events while a tool call is silent.
                    heartbeat.caption("正在等待当前模型或工具返回。停止请求在可释放的执行边界生效。")
                    continue
                if kind == "end":
                    break
                if kind == "error":
                    raise event
                heartbeat.empty()
                if event.get("request_id"):
                    st.session_state.active_chat_request_id = event["request_id"]
                kind = event.get("type")
                if kind == "answer_delta":
                    answer += event.get("text", event.get("delta", ""))
                    display.markdown(answer + " ▌")
                elif kind == "done":
                    answer = event.get("answer", answer)
                    display.markdown(answer)
                    completed = True
                    status.update(label="回答已保存，可打开来源证据核对", state="error" if failed else "complete")
                elif kind == "cancelled":
                    cancelled = True
                    answer = event.get("answer", answer)
                    display.markdown(answer or "此次执行已停止。")
                    status.update(label="已请求停止，已生成部分保留在运行记录", state="error")
                elif kind == "error":
                    failed = True
                    events.append(event)
                    status.update(label="本次回答未完成", state="error", expanded=True)
                    status.write(event.get("text", event.get("error", "回答失败")))
                elif kind in {"status", "tool_start", "tool_end", "warning", "decision"}:
                    events.append(event)
                    message = event.get("text", event.get("message", event.get("summary", "")))
                    if kind in {"tool_start", "tool_end"}:
                        tool = TOOL_NAMES.get(event.get("tool", event.get("tool_name", "")), "工具")
                        message = ("正在执行" if kind == "tool_start" else "已完成" if not event.get("error") else "执行失败") + " · " + tool
                    if message:
                        status.update(label=str(message)[:100], expanded=False)
                        status.write(str(message)[:250])
            if not completed and not cancelled:
                failed = True
                status.update(label="未收到完整回答，可重试上一次问题", state="error", expanded=True)
                display.markdown(answer or "没有返回完整答案。")
        except APIError as exc:
            failed = True
            status.update(label="连接中断", state="error", expanded=True)
            friendly_error(exc)
            display.markdown(answer or "服务尚未返回答案。")
        finally:
            abandoned.set()
            heartbeat.empty()
    st.session_state.active_chat_request_id = None
    if failed or cancelled:
        st.session_state.last_failed_chat = {"session_id": session_id, "question": question,
                                           "answer": answer, "events": events, "cancelled": cancelled}
    else:
        st.session_state.pop("last_failed_chat", None)
    invalidate()
    st.rerun()


@st.fragment(run_every="2s")
def session_execution_progress(session_id):
    current = api().get(f"/sessions/{session_id}")
    active = current.get("active_request")
    if not active:
        invalidate(); st.rerun()
    st.caption("停止请求正在等待当前调用释放。" if active.get("status") == "cancelling" else "当前请求仍在执行，完成后会自动更新此会话。")


def render_answer(message, key):
    st.markdown(message.get("content", message.get("answer", "")) or "此记录没有保存答案正文。")
    usage(message.get("metrics", {}), compact=True)
    audit(message.get("audit", {}), message.get("origin", ""))
    citations(message.get("citations", []), key)
    events = message.get("events", message.get("trace", []))
    request_id = message.get("request_id")
    if not events and request_id:
        try:
            record = cached_get(f"/runs/{request_id}", ttl=30)
            events = record.get("events", [])
        except APIError as exc:
            if exc.status_code != 404:
                friendly_error(exc)
    if events:
        trace(events)


def question_page():
    heading("科研问答", "提问论文中的方法、实验条件与结果，逐条核对原文证据。")
    all_sessions = sorted(as_list(api().get("/sessions"), "sessions"), key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
    toolbar = st.columns([1.2, 2.4, .9, .8])
    search = toolbar[0].text_input("搜索会话", placeholder="会话名称…", key="session_search")
    sessions = [session for session in all_sessions if not search or search.casefold() in session.get("title", "").casefold()]
    mapping = {session["id"]: session for session in sessions}
    active = st.session_state.active_session_id
    if active not in mapping and mapping:
        active = next(iter(mapping))
    if not mapping:
        active = None
    st.session_state.session_selector = (st.session_state.get("session_selector")
        if st.session_state.get("session_selector") in mapping else active or "")
    session_id = toolbar[1].selectbox("当前会话", list(mapping) or [""],
        index=list(mapping).index(active) if active in mapping else 0,
        format_func=lambda value: f"{mapping[value].get('title', '未命名会话')} · {value[:8]}" if value else "尚无匹配会话",
        key="session_selector", disabled=not mapping, on_change=_select_session)
    toolbar[2].button("新建会话", key="new_session", width="stretch", on_click=_new_session)
    st.session_state.active_session_id = session_id or None
    session = api().get(f"/sessions/{session_id}") if session_id else {"messages": [], "document_ids": []}
    if session_id and st.session_state.get("last_selected_session") != session_id:
        # Restore a session's scope when deliberately switching sessions. Adding
        # papers on another page remains intact when returning to the same one.
        if (st.session_state.get("last_selected_session") is not None
                or not st.session_state.selected_document_ids):
            st.session_state.selected_document_ids = session.get("document_ids", [])
        st.session_state.last_selected_session = session_id
        st.session_state.doc_selection = list(st.session_state.selected_document_ids)
    with toolbar[3].popover("管理", disabled=not session_id, width="stretch"):
        with st.form("rename_session_form"):
            renamed = st.text_input("会话名称", value=session.get("title", ""), key=f"rename_title_{session_id}", max_chars=80)
            # Public stable key for automated and keyboard workflows.
            submitted = st.form_submit_button("保存名称", key="rename_session")
        if submitted:
            if not renamed.strip():
                st.error("会话名称不能留空。")
            else:
                api().patch(f"/sessions/{session_id}", json={"title": renamed.strip()})
                invalidate(); notify("会话名称已保存。"); st.rerun()
        if session_id:
            download("导出Markdown", f"/sessions/{session_id}/export", "科研对话.md", "session_export_md", params={"format": "md"})
            download("导出JSON", f"/sessions/{session_id}/export", "科研对话.json", "session_export_json", params={"format": "json"})
            if st.button("删除此会话", key="delete_session", disabled=bool(session.get("active_request"))):
                delete_dialog("sessions", session_id, "此会话及所有消息")
    docs = documents()
    ready = {doc["document_id"]: doc for doc in docs if doc.get("status") == "ready"}
    selected = selected_ready_documents(docs)
    if "doc_selection" not in st.session_state:
        st.session_state.doc_selection = selected
    else:
        current_selection = st.session_state.doc_selection
        valid_selection = [value for value in current_selection if value in ready]
        if valid_selection != current_selection:
            st.session_state.doc_selection = valid_selection
    with st.expander("论文范围与回答设置", expanded=not session.get("messages")):
        columns = st.columns([3, 1.6, 1.5])
        columns[0].multiselect("所选论文", list(ready), format_func=lambda value: f"{title(ready[value])} · {value[:8]}",
                              key="doc_selection", on_change=_persist_docs,
                              disabled=bool(session.get("active_request")), placeholder="留空检索全部已就绪论文")
        mode = columns[1].selectbox("检索方式", list(MODE_NAMES), index=2, format_func=MODE_NAMES.get, key="chat_mode")
        agent = columns[2].toggle("启用多工具协作", value=True, key="chat_agent_enabled")
        st.button("清空论文限定", key="clear_document_selection", on_click=_clear_docs,
                  disabled=not selected or bool(session.get("active_request")))
        st.caption(f"当前限定{len(st.session_state.selected_document_ids)}篇论文。" if st.session_state.selected_document_ids else "当前范围为全部已就绪论文。")
    if not ready:
        st.info("论文库尚未准备好。可以先到论文库导入资料；无知识库证据时，回答会明确说明来源与限制。")
    active_run = session.get("active_request")
    if active_run:
        st.session_state.active_chat_request_id = active_run["id"]
        st.warning("此会话有请求正在执行。当前模型或工具调用结束后，结果会保存在运行记录中。")
        st.button("请求停止当前执行", key="chat_cancel_existing", on_click=_cancel_request)
        if st.button("刷新执行状态", key="chat_refresh_running"):
            invalidate(); st.rerun()
        session_execution_progress(session_id)
    messages = session.get("messages", [])
    if not messages:
        st.markdown("### 从一个具体问题开始")
        examples = ["概括所选论文的研究问题与方法，并引用证据。", "论文使用哪些数据集和评测指标？", "梳理论文中的主要局限与可继续研究的问题。"]
        for i, sample in enumerate(examples):
            st.button(sample, key=f"chat_example_{i}", width="stretch", on_click=_queue_question,
                      args=(sample,), disabled=bool(active_run))
    for i, message in enumerate(messages):
        if message.get("role") not in {"user", "assistant"}:
            continue
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message.get("content", ""))
            else:
                render_answer(message, message.get("request_id") or f"{session_id}_{i}")
    failed = st.session_state.get("last_failed_chat")
    unfinished = bool(messages and messages[-1].get("role") == "user")
    if session_id and not active_run and ((failed and failed.get("session_id") == session_id) or unfinished):
        st.warning("上一次回答未完整结束。可以重试上一次问题，也可在运行记录中查看已执行部分。")
        if failed and failed.get("answer"):
            with st.expander("已收到的部分答案"):
                st.markdown(failed["answer"])
        if st.button("重试上一次问题", key="retry_last", type="primary"):
            question = next((message["content"] for message in reversed(messages) if message["role"] == "user"), failed.get("question", "") if failed else "")
            stream_answer(session_id, question, selected, mode, agent, retry=True)
    st.chat_input("输入问题，例如论文的方法、数据集或实验结果…", key="chat_input",
                  disabled=bool(active_run), on_submit=_submit_question)
    pending = st.session_state.pop("pending_chat_question", None)
    if pending and session_id:
        if active_run:
            st.warning("此会话仍在执行，问题尚未发送。请在完成或停止后重新提交。")
        else:
            stream_answer(session_id, pending, st.session_state.selected_document_ids, mode, agent)


def compare_page():
    heading("论文对比", "对齐研究方法、数据集与实验条件，保留每一项结论的原文证据。")
    docs = {doc["document_id"]: doc for doc in documents() if doc.get("status") == "ready"}
    if len(docs) < 2:
        st.info("需要两篇不同的已就绪论文才能提交对比。历史结果仍可在下方查看。")
    options = list(docs)
    first_candidates = [value for value in st.session_state.selected_document_ids if value in docs]
    default_a = options.index(first_candidates[0]) if first_candidates else 0
    default_b = options.index(first_candidates[1]) if len(first_candidates) > 1 else min(1, len(options)-1) if options else 0
    for key, index in (("compare_a", default_a), ("compare_b", default_b)):
        st.session_state[key] = (st.session_state.get(key) if st.session_state.get(key) in docs
                                 else options[index] if options else "")
    columns = st.columns(2)
    first = columns[0].selectbox("论文A", options or [""], index=None, format_func=lambda value: f"{title(docs[value])} · {value[:8]}" if value else "尚无已就绪论文", key="compare_a", disabled=not options)
    second = columns[1].selectbox("论文B", options or [""], index=None, format_func=lambda value: f"{title(docs[value])} · {value[:8]}" if value else "尚无已就绪论文", key="compare_b", disabled=not options)
    aspects = st.multiselect("比较维度", ["研究问题", "方法与创新", "数据集", "实验条件", "性能指标", "局限与未来方向"],
                            default=["方法与创新", "数据集", "实验条件", "性能指标"], key="compare_aspects")
    extra = st.text_input("补充关注点", key="compare_focus", placeholder="例如训练成本、消融实验、泛化条件")
    valid = len(docs) >= 2 and first != second and bool(aspects)
    if first and first == second:
        st.caption("请选择两篇不同论文。")
    if not aspects:
        st.caption("至少选择一个比较维度。")
    st.button("生成证据对比", key="compare_generate", type="primary", disabled=not valid, on_click=_prepare_comparison)
    pending_comparison = st.session_state.pop("pending_comparison_request", None)
    if pending_comparison:
        stream_answer(pending_comparison["session_id"], pending_comparison["question"],
                      pending_comparison["document_ids"], "hybrid", True)
    st.divider()
    sessions = sorted(as_list(api().get("/sessions"), "sessions"),
                      key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
    comparisons = [session for session in sessions if session.get("title", "").startswith("论文对比") or len(session.get("document_ids", [])) >= 2]
    if not comparisons:
        st.info("尚无已保存的多论文对比。提交后答案与来源会永久保存在当前工作区。")
        return
    mapping = {session["id"]: session for session in comparisons}
    active = st.session_state.get("comparison_session_id")
    if active not in mapping:
        active = next(iter(mapping))
    history_key = st.session_state.get("comparison_history_widget_key", "compare_history")
    st.session_state.comparison_history_widget_key = history_key
    # The canonical non-widget SID drives this fresh epoch. A valid old widget
    # value is never consulted after a long stream or an interrupted rerun.
    st.session_state[history_key] = active
    identifier = st.selectbox("已保存的论文对比与多论文对话", list(mapping), index=None,
                              format_func=lambda value: f"{mapping[value].get('title', '未命名结果')} · {value[:8]}",
                              key=history_key, on_change=_select_comparison_history, args=(history_key,))
    st.session_state.comparison_session_id = identifier
    session = api().get(f"/sessions/{identifier}")
    actions = st.columns([1, 1, 1])
    with actions[0]:
        download("下载对比Markdown", f"/sessions/{identifier}/export", "论文对比.md", "compare_export_md", params={"format": "md"}, width="stretch")
    with actions[1]:
        download("下载完整JSON", f"/sessions/{identifier}/export", "论文对比.json", "compare_export_json", params={"format": "json"}, width="stretch")
    if actions[2].button("删除此结果", key="compare_delete", disabled=bool(session.get("active_request")), width="stretch"):
        delete_dialog("sessions", identifier, "这份对比结果")
    answers = [message for message in session.get("messages", []) if message.get("role") == "assistant"]
    if not answers:
        st.warning("这份对比尚未生成完整答案。请查看运行记录或重新提交。")
    for i, message in enumerate(answers):
        if not message.get("citations"):
            st.warning("此结果缺少可核对的论文证据，不应作为事实比较结论。")
        render_answer(message, message.get("request_id") or f"compare_{identifier}_{i}")
