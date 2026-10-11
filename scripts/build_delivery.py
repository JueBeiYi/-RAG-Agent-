"""Generate actual editable DOCX/PPTX deliverables from measured project data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.delivery.build import build_delivery


def main():
    parser = argparse.ArgumentParser(description="生成课程报告 评测报告 技术设计 用户手册 实验附件与实际PPTX")
    parser.add_argument("config", help="按docs/交付材料说明.md准备的JSON配置")
    parser.add_argument("--documents-only", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="仅生成源文件，状态明确保留未视觉验证")
    args = parser.parse_args()
    result = build_delivery(args.config, documents_only=args.documents_only, render=not args.no_render)
    print(json.dumps({"artifacts": result["artifacts"], "pending_human_fields": result["pending_human_fields"],
                      "rendering": {key: value["status"] for key, value in result["rendering"].items()}}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
