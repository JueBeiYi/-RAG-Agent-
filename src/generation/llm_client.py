from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from collections.abc import Iterator

import httpx
from src.core.config import Settings
from src.generation.context import estimate_tokens


class ModelUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gate = threading.BoundedSemaphore(1)

    @contextmanager
    def slot(self):
        if not self.gate.acquire(timeout=self.settings.tool_timeout_seconds):
            raise ModelUnavailable("模型正在处理其他请求，请稍后重试")
        try:
            yield
        finally:
            self.gate.release()

    def _url(self) -> str:
        if self.settings.llm_backend == "cloud":
            if self.settings.offline or not self.settings.cloud_url:
                raise ModelUnavailable("云端服务未启用；离线模式不允许发送文献")
            return self.settings.cloud_url.rstrip("/") + "/chat/completions"
        return self.settings.ollama_url.rstrip("/") + "/api/chat"

    def _body(self, messages: list[dict], stream: bool, tools=None) -> dict:
        messages = json.loads(json.dumps(messages, ensure_ascii=False))
        for message in messages:
            if message.get("role") == "tool" and self.settings.llm_backend == "ollama":
                message["tool_name"] = message.get("name", "")
                message.pop("tool_call_id", None)
            for call in message.get("tool_calls", []) or []:
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                if self.settings.llm_backend == "ollama" and isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments)
                if self.settings.llm_backend == "cloud" and not isinstance(arguments, str):
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        payload = {"model": self.settings.llm_model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if self.settings.llm_backend == "cloud":
            payload.update(temperature=self.settings.temperature, top_p=self.settings.top_p,
                           max_tokens=self.settings.answer_tokens)
            if stream:
                payload["stream_options"] = {"include_usage": True}
        else:
            payload["options"] = {"temperature": self.settings.temperature, "top_p": self.settings.top_p,
                                  "top_k": self.settings.top_k_sampling, "num_ctx": self.settings.context_tokens,
                                  "num_predict": self.settings.answer_tokens}
        return payload

    def _headers(self):
        return {"Authorization": "Bearer " + self.settings.cloud_api_key} if self.settings.llm_backend == "cloud" else {}

    def _demo(self, messages: list[dict]) -> str:
        text = messages[-1].get("content", "")
        matches = re.findall(r"\[(S\d+)\] ([^\n]+)\n(.*?)(?=\n\n\[S\d+\] |\n\n用户问题：|$)", text, re.S)
        if matches:
            return "**功能检查模式 · 原文摘录**\n\n" + "\n\n".join(f"{body.strip()[:450]} [{sid}]" for sid, _, body in matches[:4]) + "\n\n此模式未使用大语言模型，不能用于正式答案质量评测。"
        return "功能检查模式未连接大语言模型。请在标准模式准备 Ollama 后提问；当前可检查文档、检索、计算和执行流程。"

    def complete_json(self, messages: list[dict], *, schema=None) -> dict:
        """Use the provider's JSON grammar for structured local tools."""
        return self.complete(messages, json_mode=schema or True)

    def complete(self, messages: list[dict], tools=None, *, json_mode=False) -> dict:
        if self.settings.llm_backend == "extractive":
            content = self._demo(messages)
            return {"content": content, "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": False}}
        try:
            with self.slot(), httpx.Client(timeout=self.settings.tool_timeout_seconds) as client:
                payload = self._body(messages, False, tools)
                if json_mode:
                    if self.settings.llm_backend == "cloud":
                        payload["response_format"] = {"type": "json_object"}
                    else:
                        payload["format"] = json_mode if isinstance(json_mode, dict) else "json"
                response = client.post(self._url(), json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
            cloud = self.settings.llm_backend == "cloud"
            message = data["choices"][0]["message"] if cloud else data["message"]
            calls = []
            for call in message.get("tool_calls", []) or []:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"id": call.get("id"), "name": fn.get("name"), "arguments": args})
            raw = data.get("usage", {})
            usage = {"input_tokens": raw.get("prompt_tokens", 0) if cloud else data.get("prompt_eval_count", 0),
                     "output_tokens": raw.get("completion_tokens", 0) if cloud else data.get("eval_count", 0),
                     "estimated": cloud and not bool(raw)}
            if usage["estimated"]:
                usage.update(input_tokens=sum(estimate_tokens(m.get("content", "")) for m in messages),
                             output_tokens=estimate_tokens(message.get("content") or ""))
            return {"content": message.get("content") or "", "tool_calls": calls, "usage": usage}
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ModelUnavailable(f"模型调用失败，请检查模型是否已下载、服务地址与健康状态 ({type(exc).__name__})") from exc

    def stream(self, messages: list[dict]) -> Iterator[dict]:
        if self.settings.llm_backend == "extractive":
            answer = self._demo(messages)
            for i in range(0, len(answer), 32):
                yield {"type": "answer_delta", "text": answer[i:i+32]}
            yield {"type": "usage", "input_tokens": 0, "output_tokens": 0, "estimated": False}
            return
        cloud, got_usage, output, terminal = self.settings.llm_backend == "cloud", False, "", False
        try:
            with self.slot(), httpx.Client(timeout=self.settings.tool_timeout_seconds) as client:
                with client.stream("POST", self._url(), json=self._body(messages, True), headers=self._headers()) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if cloud:
                            if not line.startswith("data:"):
                                continue
                            line = line[5:].strip()
                            if line == "[DONE]":
                                terminal = True
                                break
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise ModelUnavailable("模型服务返回错误，请检查服务日志后重试")
                        reason = ((data.get("choices") or [{}])[0].get("finish_reason")) if cloud else data.get("done_reason")
                        if (cloud and reason is not None) or (not cloud and data.get("done")):
                            terminal = True
                        if reason == "length":
                            yield {"type": "status", "warning": True, "text": "模型达到输出长度上限，回答可能截断，请继续追问或调整预算"}
                        text = ((data.get("choices") or [{}])[0].get("delta", {}).get("content") or "") if cloud else data.get("message", {}).get("content", "")
                        if text:
                            output += text
                            yield {"type": "answer_delta", "text": text}
                        usage = data.get("usage") if cloud else ({"prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0)} if data.get("done") else None)
                        if usage:
                            got_usage = True
                            yield {"type": "usage", "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0), "estimated": False}
            if not terminal:
                raise ModelUnavailable("模型响应意外中断，未收到完整结束标记；请重试")
            if not output.strip():
                raise ModelUnavailable("模型返回了空答案，请检查服务或调整生成参数后重试")
            if not got_usage:
                yield {"type": "usage", "input_tokens": sum(estimate_tokens(m.get("content", "")) for m in messages), "output_tokens": estimate_tokens(output), "estimated": True}
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailable(f"模型流式调用失败，请检查服务并重试 ({type(exc).__name__})") from exc

    def health(self) -> dict:
        if self.settings.llm_backend == "extractive":
            return {"ready": True, "backend": "extractive", "demo": True, "message": "无大模型功能检查模式"}
        if self.settings.llm_backend == "cloud":
            return {"ready": bool(self.settings.cloud_url and self.settings.cloud_api_key and not self.settings.offline), "backend": "cloud", "message": "配置就绪，未发送文献进行联网探测"}
        try:
            r = httpx.get(self.settings.ollama_url.rstrip("/") + "/api/tags", timeout=2)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            return {"ready": self.settings.llm_model in names, "backend": "ollama", "models": names,
                    "message": "模型已准备" if self.settings.llm_model in names else "指定模型尚未下载"}
        except httpx.HTTPError:
            return {"ready": False, "backend": "ollama", "message": "Ollama 未启动或地址不可达"}
