"""Prepare the official portable Windows CLI inside this project, with SHA256.

No registry changes, permanent PATH changes, or system service installation.
This is an explicit network operation. Large binaries and models stay ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="下载官方Ollama Windows便携版并核对SHA256")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--directory", default=str(ROOT / "data" / "tools"))
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("此脚本面向Windows；其他系统请使用Ollama官方安装方式")
    endpoint = "https://api.github.com/repos/ollama/ollama/releases/" + ("latest" if args.version == "latest" else "tags/" + args.version)
    folder = Path(args.directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        response = client.get(endpoint)
        response.raise_for_status()
        release = response.json()
        asset = next(a for a in release["assets"] if a["name"] == "ollama-windows-amd64.zip")
        expected = (asset.get("digest") or "").removeprefix("sha256:")
        if len(expected) != 64:
            raise RuntimeError("官方release未提供可信SHA256，拒绝运行未校验的程序")
        archive = folder / (release["tag_name"] + "-" + asset["name"])
        destination = folder / ("ollama-" + release["tag_name"])
        if not archive.exists() or sha256(archive) != expected:
            partial = archive.with_suffix(".zip.partial")
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    with client.stream("GET", asset["browser_download_url"], headers=headers) as downloaded:
                        downloaded.raise_for_status()
                        append = downloaded.status_code == 206 and offset > 0
                        if not append:
                            offset = 0
                        last_logged = offset
                        with partial.open("ab" if append else "wb") as target:
                            for block in downloaded.iter_bytes(8 * 1024 * 1024):
                                target.write(block)
                                offset += len(block)
                                if offset - last_logged >= 128 * 1024 * 1024:
                                    print(f"官方文件下载 {offset / 1024**2:.0f}/{asset['size'] / 1024**2:.0f} MiB", flush=True)
                                    last_logged = offset
                    break
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            if partial.stat().st_size != asset["size"] or sha256(partial) != expected:
                raise RuntimeError("官方文件大小或SHA256不一致；保留partial供诊断，未执行")
            partial.replace(archive)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError("压缩包包含越界路径")
            package.extractall(destination)
        executable = destination / "ollama.exe"
        if not executable.is_file():
            raise RuntimeError("官方压缩包中没有预期CLI")
        manifest = {"version": release["tag_name"], "release_url": release["html_url"],
                    "download_url": asset["browser_download_url"], "sha256": expected,
                    "executable": str(executable), "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (folder / "ollama-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已验证官方便携版：{executable}", flush=True)


if __name__ == "__main__":
    main()
