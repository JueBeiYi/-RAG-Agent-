from __future__ import annotations

import json
import threading
from pathlib import Path


class RequestLogger:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, request_id: str, record: dict):
        # Per-request files avoid interleaved writes and cross-workspace UI leakage.
        with self.lock:
            path = self.directory / f"{request_id}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
