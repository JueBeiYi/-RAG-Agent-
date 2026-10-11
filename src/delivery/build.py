from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .inputs import deck_payload, digest, load_inputs


def default_runtime() -> dict:
    home = Path.home()
    base = home / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    skills = home / ".codex/plugins/cache/openai-primary-runtime"
    return {"python": str(base / "python/python.exe"), "node": str(base / "node/bin/node.exe"),
            "node_modules": str(base / "node/node_modules"), "override_bin": str(base / "bin/override"),
            "fallback_bin": str(base / "bin/fallback"),
            "document_skill": str(skills / "documents/26.1009.11318/skills/documents"),
            "presentation_skill": str(skills / "presentations/26.1009.11318/skills/presentations")}


def _run(args: list[str], cwd: Path, *, env=None) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(f"命令失败 {Path(args[0]).name}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result


def _mark(runtime: dict, skill: str, count: int, format: str, operation: str = "create"):
    directory = Path(runtime[skill])
    _run([runtime["node"], str(directory / "container_tools/mark_artifact_operation_started.mjs"),
          "--operation-kind", operation, "--expected-output-count", str(count), "--output-format", format], directory)


def render_documents(artifacts: dict, runtime: dict, build_dir: Path) -> dict:
    """Use the skill renderer, never the user's installed desktop LibreOffice."""
    environment = os.environ.copy()
    receipts = {}
    renderer = Path(__file__).with_name("render_with_explicit_tools.py")
    soffice = Path(runtime.get("soffice", str(Path(runtime.get("override_bin", "")) / "soffice.exe")))
    poppler = Path(runtime.get("poppler_dir", str(Path(runtime["python"]).parent.parent / "native/poppler/Library/bin")))
    for key, value in artifacts.items():
        path = Path(value)
        if path.suffix != ".docx":
            continue
        if not soffice.is_file():
            receipts[key] = {"status": "renderer_unavailable_or_failed", "visual_review": "not_completed",
                             "note": "未配置可用的隔离LibreOffice，不使用用户桌面Office，不声明已渲染。"}
            continue
        target = build_dir / (key + "-render")
        result = subprocess.run([runtime["python"], str(renderer), runtime["document_skill"], str(soffice), str(poppler),
                                 str(path), "--output_dir", str(target), "--emit_pdf"],
                                env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        log = build_dir / (key + "-render.log")
        log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        pages = sorted(target.glob("page-*.png"), key=lambda page: int(page.stem.split("-")[-1]))
        receipts[key] = {"status": "rendered_needs_visual_review" if result.returncode == 0 and pages else "renderer_unavailable_or_failed",
                         "page_count": len(pages), "renderer": str(soffice),
                         "page_images": [str(page) for page in pages], "log": str(log),
                         "visual_review": "not_completed",
                         "note": "保留可编辑DOCX。另行生成同内容PDF不能代替DOCX的实际渲染与视觉核对。" if not pages else "须逐页检查后才能标记视觉验收。"}
    return receipts


def build_delivery(config_path: str | Path, *, documents_only: bool = False, render: bool = True) -> dict:
    inputs = load_inputs(config_path)
    runtime = {**default_runtime(), **inputs.config.get("runtime", {})}
    required = ["python", "node", "document_skill"] + ([] if documents_only else ["node_modules", "presentation_skill"])
    for key in required:
        if not Path(runtime[key]).exists():
            raise RuntimeError(f"缺少authoring运行时 {key}：{runtime[key]}。请用Codex workspace dependency loader提供的路径配置。")
    # DOCX authoring runs under the authoritative bundled interpreter too.
    if Path(sys.executable).resolve() != Path(runtime["python"]).resolve():
        raise RuntimeError("请使用config.runtime.python对应的bundled Python运行本生成器；应用.venv只用于业务服务。")
    inputs.output.mkdir(parents=True, exist_ok=True)
    build_dir = inputs.output.parent / (".delivery-build-" + uuid.uuid4().hex[:10])
    build_dir.mkdir(parents=True)
    template = Path(inputs.config.get("course_template", inputs.root / "项目交付模板.docx"))
    if not template.is_absolute():
        template = inputs.root / template
    contract = f"""# 课程交付模板执行契约

参考路径：{template.resolve()}
参考SHA256：{digest(template)}
参考渲染：artifacts/.delivery-reference-render，共14页，已逐页检查。第4页空白，封面例日期单独跨页，均为待替换占位流。
源：单节A4纵向，页宽7560310、高10692130 EMU，上下914400、左右1143000 EMU。
封面：考核表、学期、学院、课程、考核内容、成绩和教师签名。用户未知字段待填写，原例日期不当作截止日期。
报告结构：项目概述、团队分工、模块交付清单、实现记录、技术设计、总结展望、附录。
正文：参考仿宋14pt、1.5倍行距；标题黑体黑色；复用源模板styles和sectPr。
可改槽：模板所有指导性正文与示例状态均为内容占位，须整体替换真实实现与实验，不继承示例已完成标记。填入后清理占位空白页和错误跨页，保持七章节顺序。
保持槽：参考文件字节保持不变，复制模板的页尺寸、边距、样式与未变包对象。
界面：3至5真实运行截图，来源文件与SHA256清单保存。
渲染工具：仅使用配置指定的隔离LibreOffice与bundled Poppler绝对路径，不修改PATH、不调用用户桌面Office。视觉核对状态另行记录。
"""
    (build_dir / "artifact.md").write_text(contract, encoding="utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "generated_needs_review",
                "project_title": inputs.title, "pending_human_fields": inputs.pending_fields,
                "source_files": inputs.source_files, "formal_result_count": len(inputs.formal_results),
                "demo_result_count": len(inputs.results) - len(inputs.formal_results),
                "course_evaluation_complete": False,
                "course_evaluation_note": inputs.config.get("evaluation_note", "人工质量评分尚未完成；生成文件不代表课程验收全部完成。"),
                "build_dir": str(build_dir), "artifacts": {}, "rendering": {}}
    architecture = retrieval_chart = None
    if not documents_only:
        final = inputs.output / "演示PPT.pptx"
        if final.exists():
            final = inputs.output / ("演示PPT-" + uuid.uuid4().hex[:8] + ".pptx")
        job = {"runtime": runtime, "payload": deck_payload(inputs), "workspace_dir": str(inputs.root),
               "build_dir": str(build_dir), "final_path": str(final)}
        payload_path = build_dir / "presentation-input.json"
        payload_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        _mark(runtime, "presentation_skill", 1, "pptx")
        node_environment = os.environ.copy()
        node_environment["RUNTIME_NODE_MODULES"] = runtime["node_modules"]
        node_environment["RUNTIME_NODE"] = runtime["node"]
        node_environment["RUNTIME_PYTHON"] = runtime["python"]
        result = _run([runtime["node"], str(Path(__file__).with_name("presentation.mjs")), str(payload_path)], inputs.root, env=node_environment)
        (build_dir / "presentation-authoring.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        pptx = json.loads((build_dir / "pptx-result.json").read_text(encoding="utf-8"))
        architecture = Path(pptx["architecture"])
        retrieval_chart = Path(pptx["retrieval_chart"]) if pptx.get("retrieval_chart") else None
        manifest["artifacts"]["presentation"] = pptx["final_path"]
        manifest["rendering"]["presentation"] = {"status": "rendered_needs_visual_review", "slide_count": pptx["slide_count"],
                                                   "slide_images": pptx["previews"], "visual_review": "not_completed",
                                                   "validation_receipt": str(build_dir / "pptx-validation.json")}
    _mark(runtime, "document_skill", 8, "docx", "edit" if any(inputs.output.glob("*.docx")) else "create")
    from .documents import build_documents
    documents = build_documents(inputs, architecture, retrieval_chart)
    manifest["artifacts"].update({key: value for key, value in documents.items() if key != "template_sha256"})
    if digest(template) != documents["template_sha256"]:
        raise RuntimeError("参考模板SHA256意外改变")
    manifest["reference_template"] = {"path": str(template), "sha256": documents["template_sha256"], "unchanged": True}
    if render:
        manifest["rendering"].update(render_documents(documents, runtime, build_dir))
    else:
        manifest["rendering"].update({key: {"status": "not_rendered", "visual_review": "not_completed"}
                                     for key in documents if key != "template_sha256"})
    manifest["artifact_sha256"] = {key: digest(Path(path)) for key, path in manifest["artifacts"].items()}
    destination = inputs.output / "交付状态.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
