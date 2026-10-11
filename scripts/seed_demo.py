"""Import transparently synthetic notes for a no-model smoke demonstration."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.frontend.client import APIClient

NOTES = {
    "DEMO_研究报告A.md": """# 演示研究报告 A：检索基线
这份资料是人工编写的虚构演示文本，不是发表论文，所有数字只用于功能验证，不能写入课程真实实验结论。

## 方法
方法 A 使用关键词检索。演示数据集 DemoSet 包含 100 条人工记录，固定拆分为 80 条训练与 20 条测试。
## 演示结果
在相同的 DemoSet 测试记录和准确率定义下，方法 A 的演示准确率为 70%，参数量为 10 万。
## 局限
仅在小型人工资料上演示，没有真实科研评测，也没有泛化证据。
""",
    "DEMO_研究报告B.md": """# 演示研究报告 B：混合检索
这份资料是人工编写的虚构演示文本，不是发表论文，所有数字只用于功能验证，不能写入课程真实实验结论。

## 方法
方法 B 使用关键词与向量检索后进行 RRF 融合。演示数据集 DemoSet 包含 100 条人工记录，固定拆分为 80 条训练与 20 条测试。
## 演示结果
在相同的 DemoSet 测试记录和准确率定义下，方法 B 的演示准确率为 80%，参数量为 12 万。
对照 A 的 70%，绝对差异为 10 个百分点，相对变化为 (80-70)/70×100%。
## 局限
结果完全来自虚构演示，没有证明任何真实模型优劣。此资料只用于来源引用、对比和计算器的界面检查。
""",
    "DEMO_阅读指南.md": """# 演示资料阅读指南
这些 DEMO 资料用于验证上传、解析、分块、三档检索、引用、会话与工具轨迹的连通性。
标准模式需要真实 Embedding、重排模型与 Ollama。DEMO 模式使用哈希向量和抽取式回答，不执行真实大模型推理。
正式评测须另选一个研究方向，收集 10–20 篇真实论文并人工核对至少 50 个问答对。
PDF 引用显示物理页码，Markdown、Word、TXT 展示段落位置，不能编造页码。
""",
}


def main():
    parser = argparse.ArgumentParser(description="导入明确标注的虚构演示资料")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--scope", default=None)
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    scope = args.scope or str(uuid.uuid4())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in NOTES.items():
        (output / name).write_text(text, encoding="utf-8")
    client = APIClient(args.api, scope)
    result = client.post("/documents", files=[("files", (name, text.encode("utf-8"), "text/markdown"))
                                             for name, text in NOTES.items()])
    deadline = time.monotonic() + 180
    pending = [job["job_id"] for job in result.get("jobs", [])]
    while pending and time.monotonic() < deadline:
        for jid in list(pending):
            job = client.get(f"/jobs/{jid}")
            if job["status"] == "done":
                pending.remove(jid)
            elif job["status"] == "error":
                raise SystemExit(f"演示资料导入失败：{job.get('error')}")
        if pending:
            time.sleep(1)
    if pending:
        raise SystemExit("演示资料仍在处理，请在论文库查看真实进度。")
    (output / "workspace.json").write_text(json.dumps({"workspace_token": scope, "demo": True}, indent=2), encoding="utf-8")
    print(f"演示资料已导入。前端侧栏填入此工作区标识：{scope}")
    print("所有资料为虚构 DEMO；不能作为真实论文评测或课程性能结论。")


if __name__ == "__main__":
    main()
