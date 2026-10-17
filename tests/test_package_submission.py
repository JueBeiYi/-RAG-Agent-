from pathlib import Path

import pytest

from scripts import package_submission


def _file(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")
    return path


def test_package_keeps_reorganized_deliverables_and_original_experiments(tmp_path):
    included = {
        "docs/course/南京农业大学课程实践.docx",
        "docs/course/项目交付模板.docx",
        "docs/references/method.pdf",
        "artifacts/course-report/main.tex",
        "artifacts/course-report/figures/fig01.pdf",
        "artifacts/academic-figures/tables/retrieval.csv",
        "reports/experiments/retrieval/vector/results.json",
        "reports/experiments/retrieval/vector/query-results.jsonl",
    }
    for relative in included:
        _file(tmp_path, relative)
    assert {p.relative_to(tmp_path).as_posix() for p in package_submission.submission_files(tmp_path)} == included


def test_package_prunes_dependencies_qa_and_generated_aggregates(tmp_path):
    excluded = [
        "src/node_modules/dependency/index.js", "reports/experiments/.runtime/index.db",
        "scripts/__pycache__/script.pyc", "reports/presentation/qa/page.png",
        "reports/testing/latex-report-20261015/page.png",
        "artifacts/academic-figures/workbook-previews/sheet.png",
        "reports/presentation/source-data/raw-results.json",
        "artifacts/academic-figures/data/raw-results.json", "docs/report-drafts/draft.md",
        "reports/experiments/run.log", "reports/experiments/run.tmp", ".streamlit/secrets.toml",
        "docs/.env",
    ]
    for relative in excluded:
        _file(tmp_path, relative)
    keep = _file(tmp_path, "reports/experiments/results.json")
    assert package_submission.submission_files(tmp_path) == [keep]


def test_package_prunes_reparse_directory_without_entering_it(tmp_path, monkeypatch):
    _file(tmp_path, "src/junction/foreign.txt")
    keep = _file(tmp_path, "src/real.py")
    original = package_submission._is_link_or_reparse
    monkeypatch.setattr(package_submission, "_is_link_or_reparse",
                        lambda path: path.name == "junction" or original(path))
    assert package_submission.submission_files(tmp_path) == [keep]


def test_package_never_follows_directory_or_file_symlinks(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "foreign.txt").write_text("outside package roots", encoding="utf-8")
    (tmp_path / "src").mkdir()
    try:
        (tmp_path / "src/link").symlink_to(external, target_is_directory=True)
        (tmp_path / "src/link.txt").symlink_to(external / "foreign.txt")
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable: {error}")
    assert package_submission.submission_files(tmp_path) == []
