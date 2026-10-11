"""Export reviewer forms or import actual human scores; never auto-fill scores."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.review import export_review_sheet, import_reviews

parser = argparse.ArgumentParser()
parser.add_argument("results")
parser.add_argument("--export")
parser.add_argument("--import-csv")
parser.add_argument("--output")
args = parser.parse_args()
result = json.loads(Path(args.results).read_text(encoding="utf-8-sig"))
if args.export:
    export_review_sheet(result, Path(args.export))
if args.import_csv:
    if not args.output:
        parser.error("导入评分后需要 --output 新结果路径，不覆盖原始记录")
    updated = import_reviews(result, Path(args.import_csv))
    Path(args.output).write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
