from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from .client import APIError
from .state import api, as_list, cached_get, documents, invalidate, notify, selected_ready_documents, title
from .ui import STATUS, delete_dialog, download, friendly_error, heading, preview_bytes

STAGES = {"queued": "等待开始", "parsing": "解析原文", "chunking": "组织片段", "embedding": "建立向量索引",
          "committing": "保存索引", "ready": "已就绪", "failed": "导入失败", "interrupted": "任务中断"}


def _sync_pending_imports(jobs):
    pending = st.session_state.pending_upload_jobs
    discovered = [job.get("job_id", job.get("id")) for job in jobs if job.get("status") in {"queued", "running"}]
    discovered = [identifier for identifier in discovered if identifier and identifier not in pending]
    if discovered and not pending:
        st.session_state.completed_upload_jobs = []
    pending.extend(dict.fromkeys(discovered))


@st.fragment(run_every="2s")
def import_progress():
    jobs = as_list(api().get("/jobs"), "jobs")
    # Restored browser sessions did not submit these jobs locally. Track active
    # server jobs too so their completion still refreshes the document list.
    _sync_pending_imports(jobs)
    pending = set(st.session_state.pending_upload_jobs)
    active = [job for job in jobs if job.get("status") in {"queued", "running"} or job.get("job_id", job.get("id")) in pending]
    if active:
        st.markdown("#### 导入进度")
        for job in active:
            job_id = job.get("job_id", job.get("id"))
            st.write(f"{job.get('name', '文档')} · {STAGES.get(job.get('stage'), STATUS.get(job.get('status'), '处理中'))}")
            amount = float(job.get("progress", 0))
            st.progress(max(0., min(1., amount / 100 if amount > 1 else amount)))
            if job.get("error"):
                st.error(job["error"])
            if job.get("status") in {"done", "error", "failed"} and job_id in pending:
                st.session_state.completed_upload_jobs.append(job)
                st.session_state.pending_upload_jobs.remove(job_id)
        if pending and not st.session_state.pending_upload_jobs:
            invalidate()
            failed = sum(job.get("status") != "done" for job in st.session_state.completed_upload_jobs)
            notify("导入任务已结束，论文列表已自动更新。" + (f"有{failed}个失败任务，可查看原因并重试。" if failed else ""), "warning" if failed else "success")
            st.rerun()


def _change_page(key, delta, maximum):
    value = int(st.session_state.get(key, 1)) + delta
    st.session_state[key] = max(1, min(value, maximum)) if maximum else max(1, value)


def _change_chunk_page(document_id, delta):
    offsets = st.session_state.setdefault("chunk_offsets", {})
    offsets[document_id] = max(0, offsets.get(document_id, 0) + delta)


def _metadata(doc):
    identifier = doc["document_id"]
    metadata = doc.get("metadata", {})
    if doc.get("status") in {"processing", "queued"}:
        st.info("原文正在处理，完成后可以核对与编辑元信息。")
        return
    st.caption("核对真实标题、作者、年份与来源。修改后目录检索和论文比较会使用新的元信息。")
    with st.form(f"metadata_form_{identifier}"):
        name = st.text_input("论文标题", value=doc.get("title", ""), key=f"meta_title_{identifier}", max_chars=500)
        cols = st.columns([3, 1])
        authors = cols[0].text_area("作者（每行一位，或用逗号分隔）", value="\n".join(doc.get("authors", [])), key=f"meta_authors_{identifier}", height=100)
        year = cols[1].text_input("发表年份", value=str(doc.get("year") or ""), key=f"meta_year_{identifier}", placeholder="未知可留空")
        doi = st.text_input("DOI", value=doc.get("doi", ""), key=f"meta_doi_{identifier}")
        source = st.text_input("论文来源网址", value=metadata.get("source_url", metadata.get("metadata_source", "")) if str(metadata.get("metadata_source", "")).startswith("http") else metadata.get("source_url", ""), key=f"meta_source_{identifier}")
        aliases = st.text_input("检索别名（逗号分隔）", value="，".join(metadata.get("aliases", [])), key=f"meta_aliases_{identifier}")
        abstract = st.text_area("摘要", value=doc.get("abstract", ""), height=150, key=f"meta_abstract_{identifier}")
        submitted = st.form_submit_button("保存元信息", type="primary", key=f"meta_save_{identifier}")
    if submitted:
        try:
            parsed_year = int(year) if year.strip() else None
            if parsed_year is not None and not 1800 <= parsed_year <= 2100:
                raise ValueError("发表年份请填写1800至2100之间的整数，未知时留空。")
            api().post(f"/documents/{identifier}/metadata", json={"title": name.strip(),
                "authors": [part.strip() for part in re.split(r"[\n,，;；]", authors) if part.strip()],
                "year": parsed_year, "doi": doi.strip(), "source_url": source.strip(),
                "aliases": [part.strip() for part in re.split(r"[,，;；]", aliases) if part.strip()], "abstract": abstract.strip()})
            invalidate(); notify("论文元信息已保存。")
            st.rerun()
        except (APIError, ValueError) as exc:
            friendly_error(exc)


def _preview(doc):
    identifier = doc["document_id"]
    if doc.get("format") != "pdf":
        st.info("此格式没有稳定物理页码，可以在“检索片段”查看段落证据，或下载原件阅读。")
        return
    maximum = doc.get("metadata", {}).get("page_count")
    maximum = int(maximum) if isinstance(maximum, (int, float)) and maximum > 0 else None
    key = f"doc_page_{identifier}"
    if maximum and st.session_state.get(key, 1) > maximum:
        st.session_state[key] = maximum
    st.session_state.setdefault(key, 1)
    cols = st.columns([1, 2, 1, 2])
    cols[0].button("上一页", key=f"doc_prev_{identifier}", disabled=st.session_state[key] <= 1,
                   on_click=_change_page, args=(key, -1, maximum), width="stretch")
    page = cols[1].number_input("物理页码", min_value=1, max_value=maximum, step=1, key=key, label_visibility="collapsed")
    cols[2].button("下一页", key=f"doc_next_{identifier}", disabled=bool(maximum and page >= maximum),
                   on_click=_change_page, args=(key, 1, maximum), width="stretch")
    if cols[3].button("刷新原页", key=f"doc_preview_{identifier}", width="stretch"):
        st.session_state.preview_cache.pop((identifier, int(page), ""), None)
    st.caption(f"物理页 {page}" + (f" / {maximum}" if maximum else "，页数暂未记录") + "。PDF印刷页码可能与物理页码不同。")
    try:
        with st.container(height=650, border=True):
            st.image(preview_bytes(identifier, page), width="stretch")
    except APIError as exc:
        friendly_error(exc)


def _chunks(doc):
    identifier = doc["document_id"]
    offsets = st.session_state.setdefault("chunk_offsets", {})
    offset = offsets.get(identifier, 0)
    result = cached_get(f"/documents/{identifier}/chunks", params={"offset": offset, "limit": 20})
    chunks = result.get("items", result.get("chunks", [])) if isinstance(result, dict) else as_list(result)
    total = result.get("total", doc.get("chunks", 0)) if isinstance(result, dict) else doc.get("chunks", len(chunks))
    if not chunks:
        st.info("这篇文档还没有可查看的检索片段。请等待索引完成，失败时重试导入。")
        return
    mapping = {chunk["chunk_id"]: chunk for chunk in chunks}
    key = f"doc_chunk_{identifier}"
    st.session_state[key] = st.session_state.get(key) if st.session_state.get(key) in mapping else next(iter(mapping))
    selected = st.selectbox("选择片段", list(mapping), format_func=lambda value:
        f"片段{offset + list(mapping).index(value) + 1} · " + re.sub(r"\s+", " ", mapping[value].get("text", ""))[:65], key=key)
    chunk = mapping[selected]
    page = chunk.get("page_start")
    location = f"物理页{page}" + (f"至{chunk['page_end']}" if chunk.get("page_end") != page and chunk.get("page_end") else "") if page else f"段落{chunk.get('paragraph') or '未标注'}"
    st.caption(f"{location}　{chunk.get('section', '')}　分块策略 {chunk.get('strategy', '')}")
    st.code(chunk.get("text", ""), language=None, wrap_lines=True)
    if page and st.button("定位此片段原页", key=f"doc_chunk_preview_{identifier}"):
        from .ui import evidence_dialog
        st.session_state.pop("evidence_page", None)
        evidence_dialog({"citation_id": "片段", "document_id": identifier, "chunk_id": selected,
                         "name": doc["name"], "text": chunk["text"], "page_start": page,
                         "page_end": chunk.get("page_end"), "sources": chunk.get("sources", [])})
    previous, marker, following = st.columns([1, 2, 1])
    previous.button("上一组片段", key=f"doc_chunks_prev_{identifier}", disabled=offset == 0,
                    on_click=_change_chunk_page, args=(identifier, -20))
    marker.caption(f"显示 {offset+1}至{offset+len(chunks)} / {total} 个片段")
    following.button("下一组片段", key=f"doc_chunks_next_{identifier}", disabled=offset + len(chunks) >= total,
                     on_click=_change_chunk_page, args=(identifier, 20))


def library_page():
    heading("论文库", "导入资料，核对论文信息，浏览原文与检索片段。")
    docs = documents()
    with st.expander("上传研究资料", expanded=not docs):
        generation = st.session_state.get("upload_generation", 0)
        uploads = st.file_uploader("选择文件", type=["pdf", "docx", "txt", "md", "markdown"],
                                   accept_multiple_files=True, key=f"library_upload_{generation}")
        config = cached_get("/config", ttl=15)
        st.caption(f"支持PDF、Word、TXT、Markdown。单文件上限{config.get('max_upload_mb', 50)}MB，每批最多20份、总计200MB。扫描PDF须先OCR。")
        if not uploads:
            st.caption("选择至少一个文件后即可提交导入。")
        if st.button("导入所选资料", key="upload_documents", type="primary", disabled=not uploads):
            try:
                if len(uploads) > 20:
                    raise ValueError("每批最多20份，请分批上传。")
                response = api().post("/documents", files=[("files", (upload.name, upload.getvalue(), upload.type or "application/octet-stream")) for upload in uploads])
                st.session_state.pending_upload_jobs.extend(job.get("job_id", job.get("id")) for job in response.get("jobs", []))
                st.session_state.completed_upload_jobs = []
                duplicates = [doc.get("name", "文档") for doc in response.get("documents", []) if doc.get("duplicate")]
                st.session_state.last_upload_summary = {"new": len(response.get("jobs", [])), "duplicates": duplicates}
                st.session_state.upload_generation = generation + 1
                invalidate(); notify(f"已提交{len(response.get('jobs', []))}份资料。" + (f"{len(duplicates)}份内容重复，已复用已有文档。" if duplicates else ""))
                st.rerun()
            except (APIError, ValueError) as exc:
                friendly_error(exc)
    summary = st.session_state.last_upload_summary
    if summary and summary.get("duplicates"):
        st.info("已存在的重复资料：" + "、".join(summary["duplicates"]))
    import_progress()
    ready = [doc for doc in docs if doc.get("status") == "ready"]
    stats = st.columns(3)
    stats[0].metric("全部文档", len(docs)); stats[1].metric("已可检索", len(ready)); stats[2].metric("检索片段", sum(doc.get("chunks", 0) for doc in docs))
    cols = st.columns([3, 1.4, .8])
    query = cols[0].text_input("查找论文", key="document_search", placeholder="标题、作者、年份或DOI")
    status = cols[1].selectbox("文档状态", ["全部", "已就绪", "处理中", "失败"], key="document_status_filter")
    if cols[2].button("刷新", key="library_refresh", width="stretch"):
        invalidate(); st.rerun()
    allowed = {"已就绪": {"ready"}, "处理中": {"queued", "processing"}, "失败": {"error", "failed"}}
    filtered = [doc for doc in docs if (status == "全部" or doc.get("status") in allowed[status]) and
                (not query.strip() or query.casefold().strip() in " ".join(str(doc.get(field, "")) for field in ("name", "title", "authors", "year", "doi")).casefold())]
    if not docs:
        st.info("论文库还没有资料。展开“上传研究资料”导入第一篇论文。")
        return
    if not filtered:
        st.info("当前筛选没有匹配文档。可以清空搜索词或切换文档状态。")
        return
    st.dataframe(pd.DataFrame([{"论文": title(doc), "作者": "、".join(doc.get("authors", [])) or "待核对", "年份": doc.get("year"),
                  "状态": STATUS.get(doc.get("status"), "未知"), "片段": doc.get("chunks", 0)} for doc in filtered]),
                 hide_index=True, width="stretch", column_config={"论文": st.column_config.TextColumn(width="large")})
    mapping = {doc["document_id"]: doc for doc in filtered}
    st.session_state.doc_detail_selector = (st.session_state.get("doc_detail_selector")
        if st.session_state.get("doc_detail_selector") in mapping else next(iter(mapping)))
    identifier = st.selectbox("查看文档详情", list(mapping), key="doc_detail_selector", format_func=lambda value: f"{title(mapping[value])} · {value[:8]}")
    doc = cached_get(f"/documents/{identifier}")
    st.subheader(title(doc))
    st.caption(f"原文件：{doc['name']}　文档状态：{STATUS.get(doc.get('status'), '未知')}")
    actions = st.columns([1.2, 1.2, 1.2, 1])
    with actions[0]:
        download("下载原件", f"/documents/{identifier}/file", doc["name"], f"doc_download_{identifier}", width="stretch")
    if doc.get("status") == "ready" and actions[1].button("加入问答范围", key=f"doc_select_{identifier}", width="stretch"):
        if identifier not in st.session_state.selected_document_ids:
            st.session_state.selected_document_ids.append(identifier)
        st.toast("已加入问答范围，切换页面仍会保留。")
    if doc.get("status") in {"failed", "error"}:
        if actions[2].button("重试导入", key=f"doc_retry_{identifier}", width="stretch"):
            job = api().post(f"/documents/{identifier}/retry")
            if not st.session_state.pending_upload_jobs:
                st.session_state.completed_upload_jobs = []
            st.session_state.pending_upload_jobs.append(job.get("job_id", job.get("id")))
            invalidate(); notify("重试任务已提交，完成后自动更新列表。"); st.rerun()
    elif actions[2].button("重建索引", key=f"doc_reindex_{identifier}", disabled=doc.get("status") != "ready", width="stretch"):
        job = api().post(f"/documents/{identifier}/reindex")
        if not st.session_state.pending_upload_jobs:
            st.session_state.completed_upload_jobs = []
        st.session_state.pending_upload_jobs.append(job.get("job_id", job.get("id")))
        invalidate(); notify("按当前配置重建索引，处理阶段将在上方显示。"); st.rerun()
    if actions[3].button("删除文档", key=f"doc_delete_{identifier}", disabled=doc.get("status") in {"queued", "processing"}, width="stretch"):
        delete_dialog("documents", identifier, "这篇文档及索引")
    if doc.get("error"):
        st.error(doc["error"])
    preview, metadata, chunks = st.tabs(["原文预览", "论文信息", "检索片段"])
    with preview:
        _preview(doc)
    with metadata:
        _metadata(doc)
    with chunks:
        _chunks(doc)
