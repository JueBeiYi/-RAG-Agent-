"""Assemble measured online samples, actual bad cases and delivery evidence.

This does not invent human scores or turn functional samples into a formal test.
Heavy experiments are read from their completed exports rather than rerun.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def write(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ids = ["b68e34a7df634d4992b3127b6ff336f1", "51617c72a0634eb3ad65e7b40e2baced",
           "2c6dfa1536cd44b09afb0f8df77ca5fd", "287d1a873c5d49928482b5931832de27",
           "54ffc841003946d89bd8a60ef731b060", "90009b0b7fac4c38811df8489ac73e65",
           "8cf5f63d5209411d9a312f4a7ea79718", "a83112deab0c41d78594cd6ee8b0c99f",
           "7837919cdc5f4ce591b358f1fa8b3ba6", "e3012095efc349f89eb964d11ecb5b48"]
    records = []
    for identifier in ids:
        path = ROOT / f"data/runtime/standard/logs/{identifier}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8-sig"))
        metrics = run.get("metrics", {})
        records.append({"request_id": identifier, "created_at": run.get("created_at"),
                        "question": run.get("question"), "status": run.get("status"),
                        "elapsed_ms": run.get("elapsed_ms"),
                        "input_tokens": metrics.get("input_tokens"), "output_tokens": metrics.get("output_tokens"),
                        "tool_calls": metrics.get("tool_calls"), "tool_successes": metrics.get("tool_successes"),
                        "agent_steps": metrics.get("agent_steps"), "origin": run.get("origin"),
                        "comparison_method": metrics.get("comparison_method"), "warnings": metrics.get("warnings", [])})
    latencies = [r["elapsed_ms"] for r in records if isinstance(r.get("elapsed_ms"), (int, float))]
    online = {"experiment_id": "online-functional-history-20261010", "kind": "online_functional",
              "study": "online_functional", "demo": False, "formal_models": True,
              "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "split": ["functional"],
              "delivery_scope": "线上真实功能样本 含问题和配置差异 不是固定测试集",
              "config": {"llm_backend": "ollama", "llm_model": "research-qwen:7b", "embedding_model": "BAAI/bge-m3"},
              "metrics": {"sample_count": len(records), "latency_mean_ms": mean(latencies) if latencies else None,
                          "input_tokens_total": sum(r.get("input_tokens") or 0 for r in records),
                          "output_tokens_total": sum(r.get("output_tokens") or 0 for r in records),
                          "average_agent_steps": mean(r["agent_steps"] for r in records if r.get("agent_steps") is not None) if records else None},
              "records": records, "review_status": {"human_answer_review_pending": True},
              "notes": ["耗时为真实在线请求的检索、模型、工具和传输总时间。",
                        "包含迭代过程的不完整对比，问题和配置不同，不作配置优劣的因果比较。",
                        "与冻结检索的生成阶段实验分开，不相加为独立测量。人工答案质量未评分。"]}
    write("reports/experiments/online-functionality/results.json", online)
    cases = [
        {"case_id": "BC-RETRIEVAL-01", "type": "检索失败", "observed_error": "实际4题验收中，ViT架构事实题前五条没有覆盖全部标注证据，其他论文的相似架构段落干扰排名。",
         "root_cause": "中文问题与英文实体的语义/关键词差异，以及相近架构文本；短锚点相关性标注还需人工核对。",
         "improvement": "保留BM25与RRF，完成真实分块、模型、RRF/TopK对比；用dev选取配置，test只记录冻结结果。",
         "before_result": "reports/ui/standard-acceptance/hybrid.json",
         "after_result": "待同一dev题集RRF/TopK研究完成后填写，不将4题和60题分数直接比较。",
         "improvement_status": "实现和实验进行中", "evidence": ["reports/ui/standard-acceptance/summary.json"]},
        {"case_id": "BC-GENERATION-01", "type": "生成错误", "observed_error": "真实模型返回不完整JSON或在定性对比中添加无关性能数字，不能形成完整中文表。",
         "root_cause": "只有自然语言格式提示，未限制行数和结构；条件列与性能列混淆。",
         "improvement": "原生JSON schema限制指定行数和列，逐列验证来源和数值，最多一次有界修复，失败明确降级。",
         "before_result": "请求90009b0b7fac4c38811df8489ac73e65为evidence_excerpts",
         "after_result": "请求e3012095efc349f89eb964d11ecb5b48为llm_grounded；中文表结构可用，语义仍需评分。",
         "improvement_status": "已实现 有真实线上前后样例", "evidence": ["reports/ui/browser-standard-20261010.json", "reports/testing/agent-generation-final-20261010.xml"],
         "limitation": "线上前后关注点有变化，仅为诊断样例；受控Prompt实验另外记录。"},
        {"case_id": "BC-CITATION-01", "type": "引用错误", "observed_error": "Swin早期回答把S1写为模型推断的第4页，实际S1来自物理页2；合并引用格式也曾被误判为无来源。",
         "root_cause": "模型自行生成页码；解析器只接受单个来源编号。",
         "improvement": "页码由服务器原文元数据校正；合并来源展开后逐一核对论文归属，不放宽未知引用与数字规则。",
         "before_result": "请求b68e34a7df634d4992b3127b6ff336f1的来源页与模型页码不一致。",
         "after_result": "新版原文抽屉默认展示真实物理页；已知错误页码的回归会校正，未知来源仍保留待核对。",
         "improvement_status": "已实现 定向回归与真实浏览器原页通过", "evidence": ["reports/testing/ui-final-critical-20261010.xml", "docs/screenshots/新版原文证据.jpg"]},
        {"case_id": "BC-AGENT-01", "type": "Agent决策错误", "observed_error": "已明确选定两篇论文时，规划模型仍编造DOI作为文档ID，首次工具失败后目录恢复增加额外步骤。",
         "root_cause": "把本应由服务器确认的范围再次交给规划模型生成。",
         "improvement": "显式两篇论文比较固定服务器范围，保留一次实际模型证据生成；一般概念、计算和问题关键词走对应短路由。",
         "before_result": "请求54ffc841003946d89bd8a60ef731b060含错误文档ID、3次工具调用。",
         "after_result": "请求e3012095efc349f89eb964d11ecb5b48使用两个真实ID、1次工具调用完成结构化表。",
         "improvement_status": "已实现 定向回归通过 路由参数实验待归档", "evidence": ["reports/ui/browser-standard-20261010.json", "reports/testing/requirement-gaps-20261010.xml"]},
        {"case_id": "BC-WINDOW-01", "type": "检索失败", "observed_error": "较长WordPiece文档窗口decode后再次encode会膨胀，导致合法分块的Embedding或重排输入超限。",
         "root_cause": "文本窗口还原不保证token-ID边界保持，原实现为避免丢尾而拒绝请求。",
         "improvement": "改为原token-ID完整分窗，特殊token预算精确，尾部全部保留；短文本与原编码核对，索引策略指纹更新。",
         "before_result": "真实m3e分块实验触发decode/re-encode长度Guard。",
         "after_result": "待完整窗口helper与同一真实模型实验结果归档。",
         "improvement_status": "修复与定向验证进行中", "evidence": ["src/retrieval/token_windows.py"]},
    ]
    write("reports/experiments/bad-cases/actual-bad-cases.json", cases)
    implementation = read("reports/requirements/implementation-audit.json")
    evidence = ["reports/requirements/implementation-audit.json", "reports/ui/browser-standard-20261010.json",
                "reports/testing/requirement-gaps-20261010.xml", "reports/testing/recursive-requirement-20261010.xml",
                "reports/testing/per-tool-usage-20261010.xml"]
    manifest = {"generated_at": online["created_at"], "audit": "reports/requirements/implementation-audit.json",
                "modules": [{"name": name, "status": "implemented", "verified": True, "evidence": evidence}
                            for name in ["文档处理与检索代码", "RAG生成代码", "Agent决策代码", "系统集成与界面代码"]] +
                           [{"name": "评测与项目交付", "status": "partial", "verified": False,
                             "evidence": ["reports/datasets/dataset_manifest.json", "reports/experiments/retrieval/parsers/results.json",
                                          "reports/experiments/bad-cases/actual-bad-cases.json"],
                             "pending": ["正式实验与最终材料正在完成", "人工评分按用户选择留待填写"]}],
                "notes": ["代码实现、执行验证、人工评分分别判定。旧359块不是新递归切分构建的索引。",
                          "真实学生协作记录和实际提交日未由编码助手推定。", "可选联网搜索未选择。"]}
    write("reports/requirements/delivery-implementation-manifest.json", manifest)
    print(json.dumps({"online_samples": len(records), "bad_cases": len(cases), "implementation_items": len(implementation["requirements"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
