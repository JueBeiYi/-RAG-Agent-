"""Generate 80 unreviewed annotation prompts, never fabricate answers/evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.frontend.client import APIClient

QUESTIONS = {
    "factual": ["主要研究问题是什么？", "提出的方法名称与核心结构是什么？", "训练数据来自哪些数据集？", "测试集划分和样本数是多少？", "主要评测指标如何定义？", "报告的主要实验结果是什么？", "基线方法有哪些？", "训练轮数和优化器是什么？", "硬件与推理成本如何描述？", "关键超参数如何选择？", "论文年份和发表信息是什么？", "模型输入和输出分别是什么？", "使用了哪些数据增强？", "预训练模型和训练策略是什么？", "消融实验移除了哪些模块？", "最有效的模块对应什么证据？", "是否开源代码和数据？", "数据预处理有哪些步骤？", "实验重复次数与误差报告如何？", "论文声明的主要局限是什么？"],
    "comparison": ["两篇论文的任务定义是否一致？", "两篇论文的核心方法有哪些异同？", "两篇论文使用的数据集与划分有何不同？", "两篇论文的指标定义是否可比？", "在相同条件下哪一种方法表现更好？", "两篇论文的训练资源和时间如何比较？", "两篇论文的参数量和复杂度如何比较？", "两篇论文的实验基线设置有何差异？", "两篇论文如何处理数据不足？", "两篇论文的消融结论是否一致？", "两篇论文的预训练方式有何差异？", "两篇论文的推理流程有何差异？", "两篇论文对噪声与分布变化如何处理？", "两篇论文的泛化实验如何比较？", "两篇论文报告的失败场景有何异同？", "两篇论文的复现材料是否完整？", "两篇论文的损失函数有什么不同？", "两篇论文的数据增强策略如何比较？", "两篇论文的实际适用场景有何不同？", "两篇论文的局限性如何比较？"],
    "synthesis": ["综合这些论文，研究问题的发展趋势是什么？", "这些论文可以归纳为哪些方法类型？", "综合方法与实验，关键设计原则是什么？", "这些论文共同使用哪些评测数据集？", "综合消融结果，哪些组件较为关键？", "这些论文共同暴露哪些实验不足？", "如何整理这组论文的相关工作脉络？", "综合论文，常见复现困难是什么？", "综合论文，数据与模型规模关系如何？", "这些论文的性能与资源取舍是什么？", "哪些结论得到多篇论文的支持？", "哪些结论在论文间出现冲突？", "如何用证据组织该方向的综述提纲？", "综合论文，哪些指标最需要同时报告？", "这组论文研究了哪些应用场景？", "这些方法共同依赖哪些前提假设？", "哪些结果只在特定条件下成立？", "综合失败案例，优先改进什么？", "这组论文的开放问题有哪些？", "如何给新研究者制定阅读顺序？"],
    "reasoning": ["根据实验表格，指定指标的绝对提升是多少？", "条件一致时，相对提升百分比是多少？", "参数量与性能变化是否支持作者结论？", "消融证据能否独立证明模块有效？", "训练数据差异是否影响方法优劣比较？", "指标下降能否由实验条件变化解释？", "如何检验论文的因果解释是否充分？", "是否有可能存在数据泄露或划分混淆？", "统计波动是否足以影响主要结论？", "在资源受限条件下应如何选择方法？", "若换一个数据分布，哪些假设需要重验？", "从误差案例能推出哪些验证实验？", "如何设计实验区分两个可能解释？", "作者结论是否超出提供的证据范围？", "缺少哪些信息会阻止数值比较？", "方法对具体假设失效时可能怎样？", "在性能相同时成本差异如何计算？", "该结果是否有合适的对照实验？", "根据实际资源，怎样制定复现方案？", "哪些后续实验最能验证论文提出的假说？"],
}


def main():
    parser = argparse.ArgumentParser(description="生成 80 题人工标注模板（20 dev + 60 test）")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--scope", help="用于从API读取当前真实文档目录")
    parser.add_argument("--documents", help="可选已导出的文档目录 JSON，替代API")
    parser.add_argument("--output", default="data/evaluation/annotation-template.jsonl")
    args = parser.parse_args()
    if args.documents:
        docs = json.loads(Path(args.documents).read_text(encoding="utf-8-sig"))
    elif args.scope:
        docs = APIClient(args.api, args.scope).get("/documents")
    else:
        docs = []
    docs = [doc for doc in docs if doc.get("status", "ready") == "ready"][:20]
    records = []
    for category, questions in QUESTIONS.items():
        for i, question in enumerate(questions):
            selected = [docs[i % len(docs)]] if docs else []
            if category in {"comparison", "reasoning"} and len(docs) > 1:
                selected.append(docs[(i + 1) % len(docs)])
            titles = [doc.get("title") or doc.get("name") or doc["document_id"] for doc in selected]
            prefix = f"针对《{'》与《'.join(titles)}》：" if titles else "针对已选定的真实论文语料："
            records.append({"case_id": f"{category[:3]}-{i+1:02d}", "question": prefix + question,
                            "category": category, "split": "dev" if i < 5 else "test",
                            "reference_answer": "", "relevant_sources": [], "expected_tools": [],
                            "reviewed": False, "candidate_document_ids": [doc["document_id"] for doc in selected],
                            "annotation_hint": "改为具体、可回答的问题；填写参考答案、真实文档ID、物理页码/段落和原文摘录；人工复核后才将 reviewed 设为 true。"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    print(f"80题待标注模板：{output.resolve()}（20 dev / 60 test，全部 reviewed=false）")
    print("这是问题草案，尚非可提交评测集。真实课程评测应选一个研究方向，收集10–20篇论文并人工核对50题以上。")


if __name__ == "__main__":
    main()
