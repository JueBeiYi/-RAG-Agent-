"""Refresh delivery inputs from completed measured exports without model calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "artifacts/delivery-final/delivery-config.json"
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    known = {entry["path"] for entry in config["results"]}
    additions = [
        ("reports/experiments/retrieval/chunking/bge-zh/recursive-512/results.json", "BGE-large与M3e同递归342块对比", "embedding", "同20dev与同342块 不同真实模型"),
        ("reports/experiments/retrieval/encoding/bge-zh/results.json", "BGE-large真实编码速度", "embedding", "20查询独立计时 文档编码分开"),
        ("reports/experiments/retrieval/m3/test/hybrid_rerank/results.json", "M3测试集混合加重排", "retrieval", "同60test 旧359索引 完整原token-ID精排"),
        ("reports/experiments/generation/baseline/results.json", "60题短答案生成基线", "generation", "真实冻结检索输入 生成阶段 含待人工评分"),
        ("reports/experiments/generation/parameters/results.json", "五组生成参数单因素对比", "generation", "同4dev×5配置 共20次模型回答"),
        ("reports/experiments/generation/prompt/results.json", "旧与新Prompt规则对比", "prompt", "同4dev上下文 规则检查不当人工质量"),
        ("reports/experiments/generation/agent/results.json", "Agent与固定RAG路由样例", "routing", "小规模固定问题组 不推广为全题准确率"),
        ("reports/experiments/generation/parallel/results.json", "本地工具串行与并行", "parallel", "独立本地工具实际耗时 不冒充LLM并发加速"),
        ("reports/experiments/generation/cache/results.json", "真实查询缓存冷暖", "cache", "真实模型冷请求和精确重复缓存请求"),
    ]
    for relative, label, study, scope in additions:
        source = ROOT / relative
        if relative not in known and source.exists():
            result = json.loads(source.read_text(encoding="utf-8-sig"))
            if result.get("status") == "running":
                continue
            config["results"].append({"path": relative, "label": label, "study": study, "scope": scope})
    config["project"]["submission_date"] = "待填写"
    config["project"]["report_date"] = "2026-10-11"
    config["implementation_manifest"] = "reports/requirements/delivery-implementation-manifest.json"
    config["bad_cases"] = "reports/experiments/bad-cases/actual-bad-cases.json"
    config["bad_case_findings"] = ["保留所有实际失败类型。结构与引用规则通过不等于语义正确。",
                                   "同20开发题RRF60与20的Hit@5均0.40、Recall@5均0.35，MRR从0.3148到0.3247。该改动小幅改善排序，未改善漏召回。"]
    config["prompt_findings"] = ["采用相同4题上下文做单因素参数与旧/新Prompt对比。自动指标只评格式、引用与数字规则。",
                                 "60题生成基线使用256个输出token上限与100至150字提示，属于短答案生成研究；不冒充生产768上限的完整系统成绩。"]
    config["evaluation_note"] = "人工质量评分按用户选择待填写；题集证据待学生确认；Docker容器未在本机实跑。所有模型/解析数据均来自实际执行，旧359实验快照与新生产索引分开。"
    config["verification_files"] = list(dict.fromkeys(config.get("verification_files", []) + [
        "reports/requirements/implementation-audit.json", "reports/requirements/course-audit.json",
        "reports/testing/review-final-20261011.xml", "reports/requirements/git-history.txt"]))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Delivery input groups: {len(config['results'])}")


if __name__ == "__main__":
    main()
