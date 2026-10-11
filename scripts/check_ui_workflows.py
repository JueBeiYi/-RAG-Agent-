"""Run reproducible UI workflows against a private demo backend and save evidence.

Usage: python scripts/check_ui_workflows.py
       python scripts/check_ui_workflows.py --api http://127.0.0.1:8003
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Results:
    def __init__(self):
        self.items = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when in {"setup", "teardown"} and (report.failed or report.skipped)):
            self.items.append({"test": report.nodeid, "phase": report.when, "outcome": report.outcome,
                               "duration_seconds": round(report.duration, 3),
                               "failure": str(report.longrepr)[:10000] if report.failed else ""})


def main(argv: list[str] | None = None, *, description: str | None = None):
    parser = argparse.ArgumentParser(description=description or "真实隔离 demo API + Streamlit AppTest 全流程操作检查")
    parser.add_argument("--api", default="", help="专用本机 demo API，例如 http://127.0.0.1:8003；留空则临时启动独立服务")
    parser.add_argument("--report", default=str(ROOT / "reports" / "ui" / "workflow-tests.json"))
    args = parser.parse_args(argv)
    if args.api:
        os.environ["RAG_UI_TEST_API"] = args.api
    else:
        os.environ.pop("RAG_UI_TEST_API", None)
    testing_root = (ROOT / "data" / "runtime" / "ui-tests").resolve()
    testing_root.mkdir(parents=True, exist_ok=True)
    basetemp = (testing_root / ("pytest-" + uuid.uuid4().hex)).resolve()
    if not basetemp.is_relative_to(testing_root) or basetemp.exists():
        raise RuntimeError("Test temporary directory must be a new child of data/runtime/ui-tests")
    import pytest
    results = Results()
    code = pytest.main([str(ROOT / "tests" / "test_frontend_workflows.py"), "-q", "--tb=short", "--show-capture=no", "--basetemp", str(basetemp)], plugins=[results])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "exit_code": int(code),
              "engine": "Streamlit AppTest with real HTTP backend", "backend": args.api or "dedicated ephemeral local demo API",
              "scope_policy": "Only unpredictable workspaces created by these tests are modified/deleted",
              "coverage": ["sessions", "document metadata/preview/reindex/retry", "chat/evidence/scope isolation",
                           "comparison persistence", "three-mode experiments", "exports", "run replay/deletion", "offline recovery"],
              "limits": "OS file-picker and download-manager interactions are not emulated by AppTest; actual upload/export HTTP payloads are checked. Cancellation uses a bounded controlled-stream pause only in the automatically owned test API. Pixel layout requires browser inspection.",
              "results": results.items}
    destination = Path(args.report).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"UI workflow report: {destination}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
