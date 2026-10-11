"""Invoke the packaged DOCX renderer with explicit, isolated native tools.

No PATH mutation and no fallback to a user's installed desktop Office.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import partial
from pathlib import Path


def main():
    skill, soffice, poppler, *arguments = sys.argv[1:]
    renderer = Path(skill) / "render_docx.py"
    executable = Path(soffice).resolve()
    poppler_dir = Path(poppler).resolve()
    for required in (renderer, executable, poppler_dir / "pdftoppm.exe", poppler_dir / "pdfinfo.exe"):
        if not required.is_file():
            raise FileNotFoundError(f"Explicit render dependency is missing: {required}")
    spec = importlib.util.spec_from_file_location("course_packaged_docx_renderer", renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._resolve_soffice = lambda: str(executable)
    module.convert_from_path = partial(module.convert_from_path, poppler_path=str(poppler_dir))
    module.pdfinfo_from_path = partial(module.pdfinfo_from_path, poppler_path=str(poppler_dir))
    sys.argv = [str(renderer), *arguments]
    module.main()


if __name__ == "__main__":
    main()
