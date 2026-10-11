from __future__ import annotations

import json
import os
import re
import time
import uuid
from urllib.parse import urlparse

import streamlit as st

from .client import APIClient

PAGES = ["科研问答", "论文库", "论文对比", "实验与运行记录"]


def initialise():
    st.session_state.setdefault("workspace_token", os.getenv("RAG_WORKSPACE_TOKEN") or str(uuid.uuid4()))
    st.session_state.setdefault("api_url", os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    st.session_state.setdefault("nav_page", PAGES[0])
    identity = (st.session_state.api_url, st.session_state.workspace_token)
    if st.session_state.get("scope_identity") != identity:
        for key in list(st.session_state):
            if key not in {"workspace_token", "api_url", "nav_page"}:
                del st.session_state[key]
        st.session_state.scope_identity = identity
    for key, value in {"selected_document_ids": [], "active_session_id": None, "api_cache": {},
                       "preview_cache": {}, "pending_upload_jobs": [], "pending_experiments": [],
                       "completed_upload_jobs": [], "last_upload_summary": None}.items():
        st.session_state.setdefault(key, value)


def api() -> APIClient:
    return APIClient(st.session_state.api_url, st.session_state.workspace_token)


def cached_get(path: str, *, ttl=2.0, params=None):
    key = (path, json.dumps(params or {}, sort_keys=True))
    cache = st.session_state.api_cache
    now = time.monotonic()
    if key in cache and now - cache[key][0] < ttl:
        return cache[key][1]
    value = api().get(path, params=params)
    cache[key] = (now, value)
    return value


def invalidate():
    st.session_state.api_cache = {}
    st.session_state.preview_cache = {}


def as_list(value, key="items"):
    return value if isinstance(value, list) else value.get(key, []) if isinstance(value, dict) else []


def documents():
    return as_list(cached_get("/documents"), "documents")


def title(document):
    return document.get("title") or document.get("name") or "未命名文档"


def selected_ready_documents(docs):
    ready = {doc["document_id"] for doc in docs if doc.get("status") == "ready"}
    selection = [value for value in st.session_state.selected_document_ids if value in ready]
    st.session_state.selected_document_ids = selection
    return selection


def choose_page(page):
    st.session_state.nav_page = page


def validate_workspace(value: dict):
    if not isinstance(value, dict):
        raise ValueError("工作区配置应为包含服务地址和标识的JSON对象。")
    base = str(value.get("api_url", "")).rstrip("/")
    token = str(value.get("workspace_token", ""))
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("服务地址应为完整HTTP地址，例如 http://127.0.0.1:8000。")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", token):
        raise ValueError("工作区标识须为16至128位字母、数字、下划线或横线。")
    return base, token


def switch_workspace(value: dict):
    base, token = validate_workspace(value)
    st.session_state.api_url, st.session_state.workspace_token = base, token
    for key in list(st.session_state):
        if key not in {"workspace_token", "api_url", "nav_page"}:
            del st.session_state[key]
    initialise()


def notify(message: str, kind="success"):
    st.session_state.flash = {"message": message, "kind": kind}


def flash():
    value = st.session_state.pop("flash", None)
    if value:
        getattr(st, value["kind"])(value["message"])
