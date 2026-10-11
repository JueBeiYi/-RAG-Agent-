"""Package only source, measured results and final artifacts; exclude caches."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts/course-submission"
    output.mkdir(parents=True, exist_ok=True)
    history = output / "仓库历史.bundle"
    subprocess.run(["git", "bundle", "create", str(history), "--all"], cwd=ROOT, check=True)
    roots = ["src", "scripts", "tests", "docs", "reports", "docker", ".github", ".streamlit", "artifacts/delivery-final"]
    singles = ["README.md", ".gitignore", ".dockerignore", ".env.example", "pyproject.toml", "requirements.txt",
               "requirements-models.txt", "requirements-tested.txt", "config.yaml", "config.standard.yaml",
               "南京农业大学课程实践.docx", "项目交付模板.docx", "data/corpus/manifest.json"]
    files = [ROOT / name for name in singles if (ROOT / name).is_file()]
    for name in roots:
        files.extend(path for path in (ROOT / name).rglob("*") if path.is_file() and
                     "__pycache__" not in path.parts and path.suffix not in {".pyc", ".log", ".tmp"})
    archive = output / "智能科研助理课程交付包.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(set(files)):
            package.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
        package.write(history, history.name)
        package.writestr("恢复Git历史.txt", "真实历史包含原始提交和Codex实现提交，未伪造成员协作历史。\n"
                         "git clone -b codex/course-practice-completion 仓库历史.bundle restored-repository\n")
    receipt = {"archive": str(archive), "file_count": len(set(files)) + 2, "bytes": archive.stat().st_size,
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "excluded": [".venv", "data/models", "data/tools", "data/runtime", ".env", ".idea", "临时渲染工作目录"],
               "human_scoring": "按用户选择提供空白人工评分字段，未代填"}
    (output / "提交包清单.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
