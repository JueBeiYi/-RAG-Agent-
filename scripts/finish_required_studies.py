"""Resume only unfinished required study stages, using serial model processes."""
from __future__ import annotations
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def active(pid):
    if not pid:
        return False
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def run(script, *args):
    print(f"Required stage: {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, default=0)
    args = parser.parse_args()
    while active(args.wait_pid):
        time.sleep(10)
    bge = ROOT / "reports/experiments/retrieval/chunking/bge-zh/recursive-512/results.json"
    if not bge.exists():
        run("benchmark_retrieval.py", "bge-zh")
    if not (ROOT / "reports/experiments/retrieval/m3/test/hybrid_rerank/results.json").exists():
        run("benchmark_retrieval.py", "rerank")
    run("benchmark_retrieval.py", "summary")
    # New window fingerprints must never relabel the old persisted vectors.
    run("rebuild_standard_index.py")
    # The preceding process exits here, releasing its large embedding model.
    run("run_course_generation.py", "all", "--answer-tokens", "256", "--min-free-gib", "0.25")
    run("sanitize_delivery.py", "--redact")
    print("Required model studies complete. Final artifact build remains.", flush=True)


if __name__ == "__main__":
    main()
