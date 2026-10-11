from __future__ import annotations

import concurrent.futures
import threading
import time

import httpx

from .registry import ToolCall, ToolContext, ToolRegistry


class ToolExecutor:
    """Bounded read-only dispatch with honest timeout accounting.

    A Python thread cannot forcibly cancel a running call. Timed-out workers
    remain occupied until completion. No replacement worker and no speculative
    retry is started, so repeated requests cannot create unbounded threads.
    Network/model adapters must impose their own transport timeout.
    """

    def __init__(self, registry: ToolRegistry, settings):
        self.registry = registry
        self.timeout = max(0.01, float(settings.tool_timeout_seconds))
        self.workers = max(1, min(int(settings.max_tool_workers), 8))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="research-tool")
        self.slots = threading.BoundedSemaphore(self.workers)
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _error(call: ToolCall, message: str, start: float, *, pending: bool = False,
               completed: bool = False, retryable: bool = False) -> dict:
        return {"tool": call.name, "content": "", "data": {"worker_pending": pending, "task_completed": completed},
                "citations": [], "error": message, "retryable": retryable,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)}

    @staticmethod
    def _temporary_error(exc: Exception) -> bool:
        # Only completed local errors or explicit transient HTTP responses retry.
        # A transport read timeout may leave server work running: do not replay it.
        for _ in range(4):
            if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
                return True
            response = getattr(exc, "response", None)
            if response is not None and getattr(response, "status_code", None) in {429, 502, 503, 504}:
                return True
            cause = exc.__cause__
            if not isinstance(cause, Exception):
                break
            exc = cause
        return False

    def _invoke(self, context: ToolContext, call: ToolCall, start: float) -> dict:
        try:
            spec = self.registry.validate(call)
            result = spec.handler(context, call.arguments)
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json")
            if not isinstance(result, dict):
                raise TypeError("工具返回值必须为对象")
            result = {"tool": call.name, "content": "", "data": {}, "citations": [],
                      "error": "", "retryable": False, **result}
            result["data"] = {**result.get("data", {}), "task_completed": True}
            result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result
        except Exception as exc:
            return self._error(call, f"{type(exc).__name__}: {exc}", start, completed=True,
                               retryable=self._temporary_error(exc))

    def _invoke_releasing(self, context: ToolContext, call: ToolCall, start: float) -> dict:
        try:
            return self._invoke(context, call, start)
        finally:
            # Release before Future.result returns so a legitimate retry has capacity.
            self.slots.release()

    def batch(self, context: ToolContext, calls: list[ToolCall], timeout_seconds: float | None = None) -> list[dict]:
        start = time.perf_counter()
        submitted: list[tuple[ToolCall, concurrent.futures.Future | None, dict | None]] = []
        for call in calls:
            try:
                self.registry.validate(call)
            except Exception as exc:
                submitted.append((call, None, self._error(call, str(exc), start)))
                continue
            # Do not queue behind unknown-duration work. Saturation is visible.
            if self._closed or not self.slots.acquire(blocking=False):
                submitted.append((call, None, self._error(call, "工具执行资源已占用，请稍后重试", start)))
                continue
            try:
                future = self.pool.submit(self._invoke_releasing, context, call, start)
                future.add_done_callback(lambda f: self.slots.release() if f.cancelled() else None)
                submitted.append((call, future, None))
            except Exception as exc:
                self.slots.release()
                submitted.append((call, None, self._error(call, str(exc), start)))
        results = []
        deadline = start + (self.timeout if timeout_seconds is None else max(0, min(self.timeout, timeout_seconds)))
        for call, future, error in submitted:
            if error is not None:
                results.append(error)
                continue
            try:
                results.append(future.result(timeout=max(0, deadline - time.perf_counter())))
            except concurrent.futures.TimeoutError:
                cancelled = future.cancel()
                results.append(self._error(call, "工具超过响应时间上限；运行中的任务仍占用受限工作线程", start, pending=not cancelled))
        return results
