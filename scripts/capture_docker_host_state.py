"""Capture real Windows/Docker startup diagnostics without starting application containers."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/deployment/docker/host-20261015"
DOCKER="C:/Program Files/Docker/Docker/resources/bin/docker.exe"


def decode(raw):
    if not raw:return ""
    encoding="utf-16-le" if raw.count(b"\0")>len(raw)//8 else "utf-8"
    return raw.decode(encoding,errors="replace").replace("\ufeff","")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    commands=[("wsl-version",["wsl.exe","--version"],30),
              ("wsl-status",["wsl.exe","--status"],30),
              ("docker-client",[DOCKER,"--version"],20),
              ("compose-version",[DOCKER,"compose","version"],20),
              ("desktop-start",[DOCKER,"desktop","start","--timeout","45"],55),
              ("docker-version",[DOCKER,"version"],30),
              ("docker-info",[DOCKER,"info","--format","{{.ServerVersion}}"],30)]
    rows=[]
    for name,argv,timeout in commands:
        row={"name":name,"argv":argv,"started_at":datetime.now(timezone.utc).isoformat()}
        try:
            p=subprocess.run(argv,capture_output=True,timeout=timeout,check=False)
            row.update(exit_code=p.returncode,stdout=decode(p.stdout),stderr=decode(p.stderr))
        except subprocess.TimeoutExpired as exc:
            row.update(exit_code=None,timed_out=True,stdout=decode(exc.stdout),stderr=decode(exc.stderr))
        except OSError as exc:row.update(exit_code=None,error=str(exc))
        rows.append(row)
        (OUT/"runtime-probe.json").write_text(json.dumps({"status":"actual_probe_recorded_not_container_acceptance","commands":rows},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"{name}: exit={row['exit_code']}",flush=True)


if __name__=="__main__":main()
