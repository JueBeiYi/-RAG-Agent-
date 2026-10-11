from __future__ import annotations

import json
from typing import Iterator

import httpx


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class APIClient:
    def __init__(self, base_url: str, workspace_token: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-Workspace-Token": workspace_token}
        self.timeout = timeout

    def request(self, method: str, path: str, **kwargs):
        try:
            with httpx.Client(timeout=min(self.timeout, 60.0), trust_env=False) as client:
                response = client.request(method, self.base_url + path, headers=self.headers, **kwargs)
            self._check(response)
            return response
        except httpx.HTTPError as exc:
            raise APIError("无法连接应用服务。请确认后端已启动，或在工作区设置中检查服务地址。") from exc

    def _check(self, response: httpx.Response):
        if response.is_success:
            return
        try:
            payload = response.json()
            detail = payload.get("detail", response.text) if isinstance(payload, dict) else payload
        except ValueError:
            detail = response.text[:400]
        if isinstance(detail, list):
            detail = "；".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
        prefix = {401: "工作区凭据无效，请恢复或重新连接工作区。", 404: "这条记录已不存在，请刷新列表。",
                  409: "当前操作暂时无法执行。", 413: "上传内容超过服务允许的大小。", 422: "填写内容不符合要求。"}.get(response.status_code, "服务未能完成此操作。")
        raise APIError(prefix + (" " + str(detail)[:700] if detail else ""), response.status_code, detail)

    def get(self, path: str, **kwargs):
        return self._json(self.request("GET", path, **kwargs))

    def post(self, path: str, **kwargs):
        return self._json(self.request("POST", path, **kwargs))

    def patch(self, path: str, **kwargs):
        return self._json(self.request("PATCH", path, **kwargs))

    def _json(self, response):
        try:
            data = response.json()
        except ValueError as exc:
            raise APIError("服务没有返回有效数据。请检查地址是否指向FastAPI后端，而不是网页界面的端口。") from exc
        if not isinstance(data, (dict, list)):
            raise APIError("服务返回的数据格式不符合应用接口，请检查后端地址与版本。")
        return data

    def _decode_event(self, raw):
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                raise ValueError("invalid event")
            return value
        except ValueError as exc:
            raise APIError("回答事件格式异常，连接已结束。请查看运行记录中的已完成部分，检查服务版本后重试。") from exc

    def delete(self, path: str):
        return self.request("DELETE", path)

    def bytes(self, path: str, **kwargs) -> bytes:
        return self.request("GET", path, **kwargs).content

    def chat(self, session_id: str, payload: dict) -> Iterator[dict]:
        yield from self._stream(f"/sessions/{session_id}/chat", payload)

    def retry_last(self, session_id: str) -> Iterator[dict]:
        yield from self._stream(f"/sessions/{session_id}/retry-last")

    def _stream(self, path: str, payload: dict | None = None) -> Iterator[dict]:
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10.0), trust_env=False) as client:
                with client.stream("POST", self.base_url + path,
                                   headers=self.headers, json=payload) as response:
                    if not response.is_success:
                        response.read()
                        self._check(response)
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise APIError("服务未返回回答事件流，请检查后端地址与接口版本。")
                    data_lines: list[str] = []
                    for line in response.iter_lines():
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                        elif not line and data_lines:
                            yield self._decode_event("\n".join(data_lines))
                            data_lines.clear()
                    if data_lines:
                        yield self._decode_event("\n".join(data_lines))
        except httpx.HTTPError as exc:
            raise APIError(f"回答连接中断：{type(exc).__name__}。部分回答可能已保存，请查看会话记录。") from exc
