"""Package only source, measured results and final artifacts; exclude caches."""
from __future__ import annotations
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = ("src", "scripts", "tests", "docs", "reports", "docker", ".github", ".streamlit",
                 "artifacts/delivery-final", "artifacts/course-report", "artifacts/academic-figures")
SINGLE_FILES = ("README.md", ".gitignore", ".dockerignore", ".env.example", "pyproject.toml", "requirements.txt",
                "requirements-models.txt", "requirements-tested.txt", "config.yaml", "config.standard.yaml",
                "data/corpus/manifest.json")
EXCLUDED_DIRECTORIES = {"node_modules", ".runtime", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
EXCLUDED_QA = {"reports/presentation/qa", "artifacts/academic-figures/workbook-previews", "docs/report-drafts"}


def _is_link_or_reparse(path: Path) -> bool:
    """Reject file links and Windows junctions before walking their targets."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) &
                                             getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _has_link_component(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if _is_link_or_reparse(current):
            return True
    return False


def submission_files(root: Path) -> list[Path]:
    files = [root / name for name in SINGLE_FILES
             if (root / name).exists() and not _has_link_component(root, root / name) and (root / name).is_file()]
    for name in PACKAGE_ROOTS:
        location = root / name
        if not location.exists() or _has_link_component(root, location) or not location.is_dir():
            continue
        for current, directories, filenames in os.walk(location, topdown=True, followlinks=False):
            current = Path(current)
            retained = []
            for directory in directories:
                path = current / directory
                relative = path.relative_to(root).as_posix()
                temporary_qa = relative.startswith("reports/testing/latex-")
                if (directory not in EXCLUDED_DIRECTORIES and relative not in EXCLUDED_QA
                        and not temporary_qa and not _is_link_or_reparse(path)):
                    retained.append(directory)
            directories[:] = retained
            for filename in filenames:
                path = current / filename
                if (path.suffix not in {".pyc", ".log", ".tmp"} and filename not in {"raw-results.json", ".env"}
                        and path.relative_to(root).as_posix() != ".streamlit/secrets.toml"
                        and not _is_link_or_reparse(path) and path.is_file()):
                    files.append(path)
    return sorted(set(files))


def main():
    output = ROOT / "artifacts/packages/course-submission"
    output.mkdir(parents=True, exist_ok=True)
    history = output / "仓库历史.bundle"
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "bundle", "create", str(history), branch], cwd=ROOT, check=True)
    files = submission_files(ROOT)
    archive = output / "智能科研助理课程交付包.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(set(files)):
            package.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
        package.write(history, history.name)
        package.writestr("恢复Git历史.txt", "真实历史包含原始提交和Codex实现提交，未伪造成员协作历史。\n"
                         "git clone -b codex/course-practice-completion 仓库历史.bundle restored-repository\n")
    receipt = {"archive": str(archive), "file_count": len(set(files)) + 2, "bytes": archive.stat().st_size,
               "git_history_scope": branch,
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "excluded": [".venv", "data/models", "data/tools", "data/runtime", ".env", ".idea",
                            "node_modules", ".runtime", "__pycache__", "raw-results.json", ".streamlit/secrets.toml",
                            "临时渲染工作目录", "符号链接和junction"],
               "human_scoring": "人工评分状态见评分说明；AI辅助分数不计入人工完成数"}
    (output / "提交包清单.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
