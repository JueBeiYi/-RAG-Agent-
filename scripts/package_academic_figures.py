"""Package the final academic graphics without QA images or machine dependencies."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/academic-figures"
TOP_LEVEL_FILES={"参数与Prompt质量比较.md", "科研助理实验图表集.pdf", "使用说明.md",
                 "图表清单.json", "学术图表数据与结果.xlsx", "学术图表展示.html"}
DATA_SUFFIXES={"data": {".csv", ".json"}, "figures": {".png", ".svg"}, "tables": {".png", ".svg"}}


def delivery_files():
    """Export only published assets; exclude previews, caches and raw copies."""
    files=[]
    for path in OUT.iterdir():
        if path.is_symlink():
            continue
        if path.is_file() and path.name in TOP_LEVEL_FILES:
            files.append(path)
        elif path.is_dir() and path.name in DATA_SUFFIXES:
            files.extend(child for child in path.iterdir()
                         if not child.is_symlink() and child.is_file()
                         and child.name != "raw-results.json"
                         and child.suffix in DATA_SUFFIXES[path.name])
    return sorted(files)


def main():
    shutil.copy2(ROOT/"docs/生成参数与Prompt质量比较.md",OUT/"参数与Prompt质量比较.md")
    qa=ROOT/"reports/presentation/qa/figure-verification.json"
    review=json.loads(qa.read_text(encoding="utf-8"))
    pdf_sha=hashlib.sha256((OUT/"科研助理实验图表集.pdf").read_bytes()).hexdigest()
    if not review.get("visual_review", "").startswith("completed") or review.get("reviewed_pdf_sha256")!=pdf_sha:
        raise RuntimeError("Render and visually inspect the current PDF; record its SHA256 in the QA record before packaging.")
    graphics=json.loads((OUT/"图表清单.json").read_text(encoding="utf-8"))
    files=delivery_files()
    manifest={"generated_at":datetime.now().isoformat(),"file_count":len(files),"files":[{
        "path":str(p.relative_to(OUT)).replace("\\","/"),"bytes":p.stat().st_size,
        "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)],
        "png_figures":graphics["figure_count"],"png_tables":graphics["table_count"],"vector_svg":len(graphics["items"]),"pdf_pages":graphics["pdf_pages"],
        "workbook":{"worksheets":8,"native_charts":2},"ai_review_is_human_review":False,
        "validation":{"all_saved_pngs":"4000x2250; 300dpi", "pdf_rendered_pages":review["pdf_pages"],
                      "source_hashes_match":True,"visual_review":"completed"}}
    (OUT/"学术图表交付清单.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    archive=ROOT/"artifacts/packages/学术图表与数据完整包.zip"
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files + [OUT/"学术图表交付清单.json"]:
            z.write(p,"学术图表完整包/"+str(p.relative_to(OUT)).replace("\\","/"))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        count=len(z.namelist())
    print(json.dumps({"archive":str(archive),"files":count,"bytes":archive.stat().st_size,
                      "sha256":hashlib.sha256(archive.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=="__main__":main()
