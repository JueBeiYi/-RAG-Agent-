from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    settings: Any
    store: Any
    llm: Any
    rag: Any
    scope: str
    session_id: str
    document_ids: list[str] = field(default_factory=list)
    retrieval_mode: str | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""

    def fingerprint(self) -> str:
        return self.name + ":" + json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[[ToolContext, dict], dict]
    # Parallel calls may share a worker only when neither mutates shared data.
    read_only: bool = True

    def as_function(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"重复工具名: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict]:
        return [tool.as_function() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ValueError(f"工具不存在: {name}")
        return self._tools[name]

    def validate(self, call: ToolCall) -> ToolSpec:
        spec = self.get(call.name)
        if not isinstance(call.arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        _validate(call.arguments, spec.parameters, "arguments")
        return spec


def _validate(value: Any, schema: dict, path: str) -> None:
    kind = schema.get("type")
    valid = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
    }
    if kind in valid and not valid[kind](value):
        raise ValueError(f"{path} 必须为 {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} 不在可选值中")
    if kind == "object":
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} 含未知参数: {sorted(set(value) - set(properties))}")
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path} 缺少 {key}")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], path + "." + key)
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
            raise ValueError(f"{path} 元素数量超出限制")
        for i, item in enumerate(value):
            _validate(item, schema.get("items", {}), f"{path}[{i}]")
    if kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 20000):
        raise ValueError(f"{path} 长度超出限制")
    if kind in {"number", "integer"}:
        if not schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf")):
            raise ValueError(f"{path} 数值超出限制")


def object_schema(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "maxLength": 12000}
DOC_ID = {"type": "string", "minLength": 1, "maxLength": 128}
DOC_IDS = {"type": "array", "items": DOC_ID, "maxItems": 8}
