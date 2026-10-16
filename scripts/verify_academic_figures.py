"""Verify saved graphics and make small contact sheets for visual review."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/academic-figures"
QA=ROOT/"reports/presentation/qa"


def main():
    QA.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/"图表清单.json").read_text(encoding="utf-8"))
    items=manifest["items"]
    sizes=set()
    for item in items:
        with Image.open(OUT/item["png"]) as im:
            im.verify()
        with Image.open(OUT/item["png"]) as im:
            sizes.add(im.size)
        assert (OUT/item["svg"]).stat().st_size>2000
    assert sizes=={(4000,2250)},sizes
    for source,expected in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
    pdf=PdfReader(OUT/"科研助理实验图表集.pdf")
    assert len(pdf.pages)==len(items)==41
    assert "论文" in pdf.pages[0].extract_text()
    thumbs=sorted((QA/"pdf").glob("page-*.png"))
    assert len(thumbs)==len(items)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",14)
    for part,start in enumerate(range(0,len(items),21),1):
        chosen=thumbs[start:start+21]
        columns=4; width,height=400,252
        canvas=Image.new("RGB",(columns*width,((len(chosen)+columns-1)//columns)*height),"#DFE7ED")
        draw=ImageDraw.Draw(canvas)
        for j,path in enumerate(chosen):
            with Image.open(path) as im:
                im.thumbnail((390,220))
                x=(j%columns)*width+5; y=(j//columns)*height+5
                canvas.paste(im,(x,y))
                draw.text((x+5,y+224),f"{start+j+1:02}  {items[start+j]['title']}",fill="#17324D",font=font)
        canvas.save(QA/f"图册总览-{part}.jpg",quality=91)
    report={"checked_at":datetime.now().isoformat(),"saved_png_count":len(items),"png_sizes":[list(x)for x in sizes],
            "svg_count":len(items),"pdf_pages":len(pdf.pages),"source_sha256_count":len(manifest["source_hashes"]),
            "source_hashes_match":True,"pdf_cjk_text_extracts":True,"workbook_present":(OUT/"学术图表数据与结果.xlsx").exists(),
            "visual_review":"pending_contact_sheet_inspection"}
    (QA/"figure-verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":main()
