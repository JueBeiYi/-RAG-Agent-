from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RelevantSource(BaseModel):
    """Stable source evidence, deliberately independent of chunk identifiers."""

    document_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1)

    @field_validator("quote")
    @classmethod
    def quote_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("证据摘录不能只含空白字符")
        return value.strip()


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: Literal["factual", "comparison", "synthesis", "reasoning"]
    split: Literal["dev", "test"] = "test"
    reference_answer: str = ""
    relevant_sources: list[RelevantSource] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    reviewed: bool = False
    review_method: Literal["human", "assistant_source_checked", "unspecified"] = "unspecified"
    human_review_pending: bool = True
    inference_basis: str = ""


def load_cases(path: str | Path, *, split: str | None = None,
               require_reviewed: bool = True) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"评测文件第 {number} 行格式错误：{exc}") from exc
        if case.case_id in seen:
            raise ValueError(f"重复题目编号：{case.case_id}")
        seen.add(case.case_id)
        if split is None or case.split == split:
            if require_reviewed and not case.reviewed:
                raise ValueError(f"{case.case_id} 尚未来源核对 reviewed=true，不能进入正式评测")
            if require_reviewed and (not case.relevant_sources or not case.reference_answer.strip()):
                raise ValueError(f"{case.case_id} 缺少参考答案或来源证据")
            cases.append(case)
    if not cases:
        raise ValueError("所选 split 没有可评测题目")
    return cases
