from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.frontend.client import APIError
from src.frontend.sections_chat import compare_page, question_page
from src.frontend.sections_experiments import experiments_page
from src.frontend.sections_library import library_page
from src.frontend.state import PAGES, cached_get, flash, initialise, invalidate
from src.frontend.ui import CSS, download_feedback, friendly_error, render_dialogs, workspace_dialog

st.set_page_config(page_title="研知 智能科研助理", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
initialise()

with st.sidebar:
    st.markdown("## 研知")
    st.caption("智能科研助理")
    st.radio("工作台", PAGES, key="nav_page", label_visibility="collapsed")
    st.divider()
    if st.button("工作区设置与恢复", key="workspace_settings", width="stretch"):
        workspace_dialog()
    if st.button("刷新连接", key="refresh_connection", width="stretch"):
        invalidate(); st.rerun()
    st.caption("配置文件可以恢复论文和会话。工作区标识保存在本地配置中，不写入浏览器地址。")

connected = False
health = config = {}
try:
    health = cached_get("/health", ttl=8)
    config = cached_get("/config", ttl=30)
    if not isinstance(health, dict) or not isinstance(config, dict) or "alive" not in health:
        raise APIError("地址不是兼容的科研助理后端，请检查服务端口和版本。")
    connected = True
except APIError as exc:
    with st.sidebar:
        st.error("服务未连接")
        st.caption(str(exc))

if connected:
    mode = "功能演示模式" if config.get("demo") else "标准模型模式"
    with st.sidebar:
        st.success("API 已连接")
        st.caption(mode)
    status, info = st.columns([1.15, 3])
    if health.get("ready"):
        status.success("模型服务已就绪")
    else:
        status.warning("模型或索引尚未就绪")
    if config.get("demo"):
        info.info("功能演示：哈希向量与原文抽取。结果用于检查流程，不作为真实模型质量评测。")
    else:
        info.caption(f"标准模式　生成模型 {config.get('llm_model', '未记录')}　检索与生成状态以服务健康检查为准。")
    with st.expander("查看服务健康与模型状态"):
        for name, label in [("llm", "生成模型"), ("retrieval", "检索模型"), ("storage", "数据存储")]:
            component = health.get(name, {})
            st.write(f"{label}：" + ("已就绪" if component.get("ready") else "尚未就绪"))
            if component.get("error") or component.get("message"):
                st.caption(component.get("error", component.get("message", "")))
        st.json(health, expanded=False)
    flash()
    download_feedback()
    try:
        {"科研问答": question_page, "论文库": library_page, "论文对比": compare_page,
         "实验与运行记录": experiments_page}[st.session_state.nav_page]()
    except APIError as exc:
        friendly_error(exc)
        st.caption("修复连接后点击侧栏“刷新连接”。当前工作区与已保存记录不会因此丢失。")
else:
    st.title("连接科研助理工作区")
    st.error("无法连接应用服务。")
    st.write("在项目目录运行 scripts/start.ps1，或从侧栏打开“工作区设置与恢复”导入已有配置。")
    st.caption("服务重新启动后点击“刷新连接”，即可继续当前工作区。")

render_dialogs()
