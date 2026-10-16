"""Read-only export checks; native number cells and editable chart bindings."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parents[2]
output = root / "artifacts/academic-figures/学术图表数据与结果.xlsx"
ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "c": "http://schemas.openxmlformats.org/drawingml/2006/chart", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = [s.attrib["name"] for s in workbook.findall("s:sheets/s:sheet", ns)]
    assert len(sheets) == 8
    numeric = formulas = 0
    errors = []
    for name in archive.namelist():
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
            sheet = ET.fromstring(archive.read(name))
            for cell in sheet.findall("s:sheetData/s:row/s:c", ns):
                numeric += cell.get("t") == "n" and cell.find("s:v", ns) is not None
                formulas += cell.find("s:f", ns) is not None
                if cell.get("t") == "e":
                    errors.append({"sheet": name, "cell": cell.get("r"), "error": cell.findtext("s:v", namespaces=ns)})
    charts = [n for n in archive.namelist() if "/charts/chart" in n and n.endswith(".xml")]
    chart_records = []
    for name in charts:
        chart = ET.fromstring(archive.read(name))
        bindings = [f.text for f in chart.findall(".//c:f", ns)]
        colors = [c.get("val") for c in chart.findall(".//a:srgbClr", ns)]
        assert bindings, name
        chart_records.append({"part": name, "cell_references": bindings, "colors": colors})
    assert len(charts) == 2
    assert numeric > 1000
    assert formulas == 23
    assert not errors
    result = {"file": str(output), "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "sheets": sheets, "native_numeric_cells": numeric, "formula_cells": formulas, "formula_errors": errors, "native_charts": chart_records}
(root / "reports/presentation/qa/workbook/export-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in ["native_charts", "sheets"]}, ensure_ascii=True))
