"""Remove known local workspace credentials from public experiment artifacts.

The live workspace files and model caches remain private and are never edited.
Output counts identify files, never the credential values themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".yaml", ".yml", ".xml", ".py", ".ps1", ".mjs"}


def private_values() -> set[str]:
    values = set()
    private = ROOT / "data/runtime"
    for path in [*private.glob("workspace-*.json"), private / "course-workspace.json"]:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8-sig")).get("workspace_token")
            if isinstance(value, str) and len(value) >= 16:
                values.add(value)
    return values


def public_files():
    for directory in ("reports", "docs", "artifacts/delivery-final"):
        location = ROOT / directory
        if location.exists():
            yield from (path for path in location.rglob("*") if path.is_file())


def sanitize(redact: bool = False) -> dict:
    secrets = private_values()
    changed, remaining = [], []
    for path in public_files():
        if path.suffix.lower() in TEXT_SUFFIXES:
            original = path.read_bytes()
            content = original
            for value in secrets:
                replacement = "redacted-workspace-" + hashlib.sha256(value.encode()).hexdigest()[:16]
                content = content.replace(value.encode(), replacement.encode())
            if content != original:
                if redact:
                    path.write_bytes(content)
                    changed.append(str(path.relative_to(ROOT)))
                else:
                    remaining.append(str(path.relative_to(ROOT)))
        elif path.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
            with ZipFile(path) as archive:
                if any(value.encode() in archive.read(name) for name in archive.namelist()
                       if name.endswith(".xml") for value in secrets):
                    remaining.append(str(path.relative_to(ROOT)))
    result = {"known_credentials_checked": len(secrets), "changed_files": changed,
              "remaining_files": remaining, "passed": not remaining,
              "policy": "Only known private values are checked; live runtime and model data are excluded."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redact", action="store_true")
    parser.add_argument("--report", default="reports/requirements/delivery-privacy-check.json")
    args = parser.parse_args()
    result = sanitize(args.redact)
    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
