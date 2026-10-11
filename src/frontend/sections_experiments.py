from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from src.evaluation.dataset import EvaluationCase
from .client import APIError
from .sections_chat import MODE_NAMES
from .state import api, as_list, cached_get, documents, invalidate, notify
from .ui import STATUS, audit, citations, delete_dialog, download, evidence_dialog, friendly_error, heading, trace, usage

CATEGORIES = {"factual": "事实查询", "comparison": "论文比较", "synthesis": "综合归纳", "reasoning": "推理与计算"}


def display_time(value):
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return date.astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value or "未记录")


def validate_dataset(raw, split, ready_ids):
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("评测集超过5MB，请拆分或去除无关字段。")
    seen, selected = set(), []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"第{number}行格式有误，请核对JSON字段和题型。{str(exc)[:200]}") from exc
        if case.case_id in seen:
            raise ValueError(f"题目编号重复：{case.case_id}")
        seen.add(case.case_id)
        if case.split != split:
            continue
        if not case.reviewed:
            raise ValueError(f"第{number}行 {case.case_id} 来源尚未核对 reviewed=true，不能提交评测。")
        if not case.reference_answer.strip() or not case.relevant_sources:
            raise ValueError(f"第{number}行缺少参考答案或原文证据。")
        if any(source.document_id not in ready_ids for source in case.relevant_sources):
            raise ValueError(f"第{number}行的来源论文不在当前已就绪论文库。请核对工作区和文档编号。")
        selected.append(case)
    if not selected:
        raise ValueError("所选数据划分没有题目。请检查split字段和当前选择。")
    return selected


def _sync_pending_experiments(records):
    pending = st.session_state.pending_experiments
    discovered = [record["id"] for record in records if record.get("status") in {"queued", "running"} and record["id"] not in pending]
    pending.extend(dict.fromkeys(discovered))


@st.fragment(run_every="3s")
def experiment_progress():
    _sync_pending_experiments(as_list(api().get("/experiments"), "experiments"))
    finished = False
    for identifier in list(st.session_state.pending_experiments):
        record = api().get(f"/experiments/{identifier}")
        status = record.get("status")
        st.write(f"{MODE_NAMES.get(record.get('mode'), '检索实验')} · {STATUS.get(status, status)} · {record.get('case_count', '未记录')}题")
        if status == "error":
            st.error(record.get("error", "实验未完成，请检查标注证据和模型状态。"))
        if status in {"done", "error"}:
            st.session_state.pending_experiments.remove(identifier)
            finished = True
    if finished:
        invalidate(); st.rerun()


def _comparison(results):
    if len(results) < 2:
        return
    identities = {(r.get("dataset_hash"), r.get("corpus_hash"), tuple(sorted(r.get("split", []) if isinstance(r.get("split"), list) else [str(r.get("split"))])),
                   r.get("kind", "retrieval"), bool(r.get("demo"))) for r in results}
    if len(identities) != 1:
        st.warning("所选实验的语料、题集、数据划分或数据性质不同，不能直接比较，也不会绘制方法优劣图。可分别打开详情核对。")
        return
    rows = [{"实验": {"vector": "vector", "hybrid": "hybrid", "hybrid_rerank": "rerank"}.get(r.get("mode"), "experiment") + f" {i+1}",
             "Hit@5": r.get("metrics", {}).get("hit_at_5"), "MRR": r.get("metrics", {}).get("mrr"),
             "Recall@5": r.get("metrics", {}).get("recall_at_5")} for i, r in enumerate(results)]
    if any(row[key] is None for row in rows for key in ("Hit@5", "MRR", "Recall@5")):
        st.info("所选结果缺少完整检索指标，不能绘制对比图。")
        return
    st.markdown("#### 同一评测集的检索对比")
    st.bar_chart(pd.DataFrame(rows).set_index("实验"), color=["#287c85", "#4e71a8", "#78997e"],
                 stack=False, sort=False, y_label="指标分数（0至1）", height=320)
    st.caption("vector为纯向量，hybrid为混合检索，rerank为混合与重排。三项指标各自范围为0至1，分组展示，不相加。")
    if results[0].get("demo"):
        st.warning("以上为功能演示结果，不能作为真实模型性能结论。")


def _single_case(identifier, result):
    records = result.get("records", [])
    if not records:
        st.info("这份实验没有逐题记录。")
        return
    only_failed = st.toggle("只看失败或前五条未命中的题目", key=f"experiment_failures_{identifier}")
    filtered = [r for r in records if not only_failed or r.get("error") or r.get("hit_at_5", 1) < 1]
    if not filtered:
        st.success("没有匹配当前筛选条件的失败题目。")
        return
    st.dataframe(pd.DataFrame([{"题目编号": r["case_id"], "问题": r["question"], "题型": CATEGORIES.get(r.get("category"), r.get("category")),
                   "Hit@5": r.get("hit_at_5"), "MRR": r.get("mrr"), "Recall@5": r.get("recall_at_5"),
                   "用时ms": r.get("elapsed_ms"), "错误": r.get("error", "")} for r in filtered]), hide_index=True, width="stretch",
                 column_config={"问题": st.column_config.TextColumn(width="large")})
    mapping = {r["case_id"]: r for r in filtered}
    st.session_state.experiment_case_selector = (st.session_state.get("experiment_case_selector")
        if st.session_state.get("experiment_case_selector") in mapping else next(iter(mapping)))
    case_id = st.selectbox("单题排查", list(mapping), format_func=lambda value: f"{value} · {mapping[value]['question'][:55]}", key="experiment_case_selector")
    row = mapping[case_id]
    st.write(row["question"])
    if row.get("error"):
        st.error(row["error"])
    st.caption(f"参考答案：{row.get('reference_answer') or '未保存'}")
    hits = row.get("retrieved", [])
    if not hits:
        st.info("此次检索没有返回片段。可核查文档状态、问题用词、模型服务和标注原文。")
    for rank, hit in enumerate(hits[:5], 1):
        chunk = hit.get("chunk", {})
        location = f"物理页{chunk['page_start']}" if chunk.get("page_start") else f"段落{chunk.get('paragraph') or '未记录'}"
        with st.expander(f"第{rank}条 · {chunk.get('name', '来源文档')} · {location}", expanded=rank == 1):
            with st.container(height=280, border=True):
                st.code(chunk.get("text", ""), language=None, wrap_lines=True)
            scores = {"向量": hit.get("vector_score"), "关键词": hit.get("bm25_score"), "融合": hit.get("rrf_score"), "重排": hit.get("rerank_score")}
            st.caption("　".join(f"{name} {score:.4f}" for name, score in scores.items() if isinstance(score, (int, float))))
            if st.button("打开此检索证据", key=f"case_source_{identifier}_{case_id}_{rank}"):
                st.session_state.pop("evidence_page", None)
                evidence_dialog({**chunk, "citation_id": f"检索结果{rank}", "name": chunk.get("name", "来源文档")})
    with st.expander("标注原文与相关位置"):
        names = {d["document_id"]: d.get("title") or d["name"] for d in documents()}
        for source in row.get("relevant_sources", []):
            st.write(names.get(source.get("document_id"), "已删除或未找到的来源论文"))
            st.caption(f"物理页{source['page']}" if source.get("page") else "段落来源")
            st.write(source.get("quote", ""))
        st.json(row.get("matching_chunk_ids", []), expanded=False)


def _experiments():
    with st.expander("提交新评测", expanded=True):
        input_mode = st.radio("评测集输入方式", ["上传文件", "粘贴 JSONL"], horizontal=True, key="experiment_input_mode")
        if input_mode == "上传文件":
            uploaded = st.file_uploader("评测JSONL文件", type=["jsonl"], key="experiment_file")
            raw, name = (uploaded.getvalue(), uploaded.name) if uploaded else (b"", "cases.jsonl")
        else:
            raw = st.text_area("每行一道题的JSON对象", height=170, key="experiment_dataset_text", placeholder="粘贴已核对的JSONL，不填写模拟分数。").encode("utf-8")
            name = "pasted-cases.jsonl"
        columns = st.columns(2)
        split = columns[0].selectbox("数据划分", ["dev", "test"], format_func=lambda value: "开发集 用于调参" if value == "dev" else "保留测试集 用于最终验证", key="experiment_split")
        mode = columns[1].selectbox("检索模式", list(MODE_NAMES), index=2, format_func=MODE_NAMES.get, key="experiment_mode")
        st.caption("题目须reviewed=true，并有参考答案和原文证据。先用开发集调参，再固定配置验证测试集。课程需50题以上和四类题，单题文件适合调试。")
        columns = st.columns(2)
        single = columns[0].button("运行所选模式", key="experiment_run", disabled=not raw.strip(), width="stretch")
        all_modes = columns[1].button("依次运行三档对比", key="experiment_run_all", type="primary", disabled=not raw.strip(), width="stretch")
        if not raw.strip():
            st.caption("上传文件或粘贴题目后才可提交。")
        if single or all_modes:
            try:
                cases = validate_dataset(raw, split, {d["document_id"] for d in documents() if d.get("status") == "ready"})
                modes = ["vector", "hybrid", "hybrid_rerank"] if all_modes else [mode]
                for chosen in modes:
                    record = api().post("/experiments", files={"file": (name, raw, "application/jsonl")}, data={"mode": chosen, "split": split})
                    st.session_state.pending_experiments.append(record["id"])
                invalidate(); notify(f"已提交{len(modes)}个实验，每个{len(cases)}题。状态由实际任务更新。"); st.rerun()
            except (ValueError, UnicodeError, APIError) as exc:
                friendly_error(exc)
    experiment_progress()
    columns = st.columns([2, 1, .7])
    mode_filter = columns[0].selectbox("筛选实验模式", ["全部", *MODE_NAMES], format_func=lambda value: MODE_NAMES.get(value, value), key="experiment_mode_filter")
    state_filter = columns[1].selectbox("筛选状态", ["全部", "已完成", "执行中", "失败"], key="experiment_status_filter")
    if columns[2].button("刷新", key="experiments_refresh", width="stretch"):
        invalidate(); st.rerun()
    experiments = as_list(api().get("/experiments"), "experiments")
    allowed = {"已完成": {"done"}, "执行中": {"queued", "running"}, "失败": {"error"}}
    filtered = [r for r in experiments if (mode_filter == "全部" or r.get("mode") == mode_filter) and (state_filter == "全部" or r.get("status") in allowed[state_filter])]
    if not filtered:
        st.info("没有匹配的实验记录。可提交已核对的题目集，或调整筛选条件。")
        return
    st.dataframe(pd.DataFrame([{"时间": display_time(r.get("created_at")), "模式": MODE_NAMES.get(r.get("mode"), "检索实验"),
                  "状态": STATUS.get(r.get("status"), r.get("status")), "题数": r.get("case_count"), "错误": r.get("error", "")} for r in filtered]), hide_index=True, width="stretch")
    completed = {r["id"]: r for r in filtered if r.get("status") == "done"}
    st.session_state.setdefault("experiment_selection", [])
    current_selection = st.session_state.experiment_selection
    valid_selection = [value for value in current_selection if value in completed]
    if valid_selection != current_selection:
        st.session_state.experiment_selection = valid_selection
    selected = st.multiselect("选择同一语料与题目集的实验进行对比", list(completed), key="experiment_selection", max_selections=6,
                             format_func=lambda value: f"{display_time(completed[value].get('created_at'))} · {MODE_NAMES.get(completed[value].get('mode'), '实验')} · {value[:8]}")
    _comparison([cached_get(f"/experiments/{identifier}", ttl=120).get("result", {}) for identifier in selected])
    mapping = {r["id"]: r for r in filtered}
    st.session_state.experiment_detail = (st.session_state.get("experiment_detail")
        if st.session_state.get("experiment_detail") in mapping else next(iter(mapping)))
    identifier = st.selectbox("实验详情", list(mapping), key="experiment_detail", format_func=lambda value:
        f"{display_time(mapping[value].get('created_at'))} · {MODE_NAMES.get(mapping[value].get('mode'), '实验')} · {STATUS.get(mapping[value].get('status'), '')} · {value[:8]}")
    record = cached_get(f"/experiments/{identifier}", ttl=120 if mapping[identifier].get("status") == "done" else 1)
    result = record.get("result", {}); finished = record.get("status") == "done"
    columns = st.columns(4)
    for column, fmt in zip(columns[:3], ("json", "csv", "xlsx")):
        with column:
            download(f"下载{fmt.upper()}", f"/experiments/{identifier}/export", f"实验结果.{fmt}", f"experiment_export_{fmt}", params={"format": fmt}, disabled=not finished, width="stretch")
    if columns[3].button("删除实验", key="delete_experiment", disabled=record.get("status") in {"queued", "running"}, width="stretch"):
        delete_dialog("experiments", identifier, "这份实验记录")
    if not finished:
        st.info("只有成功完成的实验才能导出结果。")
        if record.get("error"):
            st.error(record["error"])
        return
    if result.get("demo"):
        st.warning("功能演示实验：这些分数用于检查流程，不代表真实模型质量。")
    metrics = result.get("metrics", {})
    for column, field, label in zip(st.columns(4), ["hit_at_5", "mrr", "recall_at_5", "latency_p95_ms"], ["Hit@5", "MRR", "Recall@5", "P95耗时ms"]):
        value = metrics.get(field)
        column.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "未测量")
    if metrics.get("error_count"):
        st.warning(f"有{metrics['error_count']}题检索异常，质量指标按零分计入。")
    st.markdown("#### 逐题结果与排错")
    _single_case(identifier, result)
    with st.expander("配置与复现信息"):
        st.caption("MRR按完整返回排名统计，Hit@5与Recall@5看前五条。不同阶段分数不处于同一尺度。")
        st.json({field: result.get(field) for field in ("dataset_hash", "corpus_hash", "config_hash", "split", "index_version", "models", "config", "matching_rule")}, expanded=False)


def _open_session(session_id):
    try:
        api().get(f"/sessions/{session_id}")
    except APIError as exc:
        notify("对应会话已删除，运行记录与已保存的证据仍可在此查看。" if exc.status_code == 404 else str(exc), "warning")
        return
    st.session_state.active_session_id = session_id
    st.session_state.session_selector = session_id
    st.session_state.session_search = ""
    st.session_state.nav_page = "科研问答"


@st.fragment(run_every="3s")
def live_run_progress(identifier):
    record = api().get(f"/runs/{identifier}")
    if record.get("status") not in {"running", "cancelling"}:
        invalidate(); st.rerun()
    st.info("正在执行。停止请求在当前调用释放或后续事件边界生效。")


def _runs():
    columns = st.columns([2.5, 1.2, .7])
    query = columns[0].text_input("查找运行记录", placeholder="按问题文字搜索", key="run_search")
    state_filter = columns[1].selectbox("运行状态", ["全部", "已完成", "执行中", "失败", "已中断"], key="run_status_filter")
    if columns[2].button("刷新", key="runs_refresh", width="stretch"):
        invalidate(); st.rerun()
    records = as_list(api().get("/runs"), "runs")
    allowed = {"已完成": {"done"}, "执行中": {"running", "cancelling"}, "失败": {"error"}, "已中断": {"cancelled"}}
    filtered = [r for r in records if (not query or query.casefold() in r.get("question", "").casefold()) and
                (state_filter == "全部" or r.get("status") in allowed[state_filter])]
    if not filtered:
        st.info("没有匹配的运行记录。问答后可在这里查看执行过程、导出或排查失败。")
        return
    mapping = {r.get("id", r.get("request_id")): r for r in filtered}
    st.session_state.run_selector = (st.session_state.get("run_selector")
        if st.session_state.get("run_selector") in mapping else next(iter(mapping)))
    identifier = st.selectbox("运行记录", list(mapping), key="run_selector", format_func=lambda value:
        f"{display_time(mapping[value].get('created_at'))} · {mapping[value].get('question', '')[:55]} · {STATUS.get(mapping[value].get('status'), '')} · {value[:8]}")
    record = cached_get(f"/runs/{identifier}", ttl=1 if mapping[identifier].get("status") in {"running", "cancelling"} else 60)
    live = record.get("status") in {"running", "cancelling"}
    columns = st.columns(4)
    for column, fmt in zip(columns[:2], ("md", "json")):
        with column:
            download("下载Markdown" if fmt == "md" else "下载JSON", f"/runs/{identifier}/export", f"运行记录.{fmt}", f"run_export_{fmt}", params={"format": fmt}, disabled=live, width="stretch")
    columns[2].button("打开对应会话", key="run_open_session", disabled=not record.get("session_id"), on_click=_open_session, args=(record.get("session_id"),), width="stretch")
    if columns[3].button("删除记录", key="delete_run", disabled=live, width="stretch"):
        delete_dialog("runs", identifier, "这条运行记录")
    st.write(record.get("question", ""))
    st.caption(f"历史执行状态：{STATUS.get(record.get('status'), record.get('status'))}。回放读取已保存的数据，不会重新调用模型。")
    if live:
        if st.button("请求停止当前执行", key="run_cancel"):
            st.info(api().post(f"/runs/{identifier}/cancel").get("message", "停止请求已提交。")); invalidate()
        live_run_progress(identifier)
        return
    if record.get("error"):
        st.warning(record["error"])
    st.markdown(record.get("answer") or "这条记录没有保存答案，详情见执行过程。")
    usage(record.get("metrics", {})); audit(record.get("audit", {}), record.get("origin", ""))
    citations(record.get("citations", []), f"run_{identifier}")
    trace(record.get("events", []), expanded=True, label="执行轨迹回放")
    with st.expander("运行标识与配置"):
        st.json({field: record.get(field) for field in ("request_id", "created_at", "index_version", "configuration")}, expanded=False)


def experiments_page():
    heading("实验与运行记录", "比较真实检索结果，排查单题证据，回放每一次工具执行。")
    experiments, runs = st.tabs(["检索实验", "运行回放"])
    with experiments:
        _experiments()
    with runs:
        _runs()
