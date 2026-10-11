"""Backward-compatible entry point for the maintained UI workflow checks.

The original --api argument is still supported:
    python scripts/smoke_ui.py --api http://127.0.0.1:8003

Omitting --api now starts an isolated ephemeral-port demo API instead of
assuming a demo service at port 8000. Existing services and research data are
not stopped or reused. --report is also forwarded to check_ui_workflows.py.
There is one shared test implementation, not a second label-based smoke suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_ui_workflows import main as check_workflows


def main(argv: list[str] | None = None) -> int:
    return check_workflows(argv, description=(
        "旧 smoke_ui.py 的兼容入口，执行完整UI工作流。保留 --api 参数连接专用本机demo API；"
        "省略时自动启动独立随机端口服务，不使用或停止已有8000服务。支持 --report 保存结果。"))


if __name__ == "__main__":
    raise SystemExit(main())
