"""Summarize measured local runtime evidence and package only the public records."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"reports/deployment/local-20261015"


def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))


def local_time(s):return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")


def main():
    r=read(EVIDENCE/"runtime-record.json"); a=read(EVIDENCE/"acceptance/acceptance.json")
    assert r["status"]==a["status"]=="passed" and r["failed_check_count"]==0
    m=r["generation_metrics"]; job=read(EVIDENCE/"acceptance/http-indexing-job.json")["response"]
    run=read(EVIDENCE/"acceptance/http-persisted-run.json")["response"]
    lines=["# 本地127.0.0.1实际运行记录","",f"结论：**标准模型模式实际运行与核心流程验收通过**。本轮共{r['unique_check_count']}类检查，失败{r['failed_check_count']}类。轮询在内的通过记录共{r['poll_inclusive_passed_count']}条，不将重复轮询冒称独立功能测试。", "",
        f"- 实际流程验收：{local_time(a['started_at'])} 至 {local_time(a['finished_at'])}。",
        f"- 最新进程、端口与就绪状态采集：{local_time(r['captured_at'])}。",
        f"- 前端：{r['ui_url']}；API：{r['api_url']}；本项目Ollama：{r['model_url']}。",
        "- API、前端和项目模型服务均仅监听回环地址；用户原有11434服务未改动。",
        "- 按用户最新要求采用本地运行；不继续Docker，本记录不声明容器验收或服务重启测试通过。", "",
        "## 启动命令","","```powershell",r["start_command"],"```","",
        r"服务当前保持运行；停止应用可使用`.\scripts\stop.ps1`。启动前应确认8000和8501未被占用。", "",
        "## 实际模型与组件","","| 项目 | 实测配置 |","| --- | --- |",
        f"| 大模型 | {r['actual_model']}；真实本地Ollama，demo=false |",
        f"| Embedding | {r['embedding_model']} / {r['embedding_backend']} |",
        f"| 重排 | {r['reranker_model']} |",f"| 向量存储 | {r['vector_backend']} |",
        f"| 后端状态 | alive={r['api_alive']}，ready={r['api_ready']} |","",
        "## 本轮实际操作","","| 步骤 | 实际结果 |","| --- | --- |",
        f"| 论文导入 | {r['document']['filename']}，{r['document']['bytes']:,}字节 |",
        f"| 解析与索引 | done / ready，{job['chunks']}个真实片段；使用独立随机工作区 |",
        f"| 检索 | hybrid_rerank，候选{m['retrieval_stages']['candidates']}条，返回{m['retrieved_count']}条 |",
        "| 流式回答 | 实际SSE answer_delta与done；origin=knowledge_rag |",
        "| 引用追溯 | 使用有效来源编号；引用对应已持久化片段，PDF物理页预览返回PNG |",
        "| 原文件下载 | 下载SHA-256与上传论文一致 |",
        "| 记录保存 | 会话回答与run保存；会话JSON导出成功 |","",
        "## 本次问答资源记录","","| 指标 | 实测值 |","| --- | --- |",
        f"| 端到端问答耗时 | {m['end_to_end_ms']/1000:.2f}秒 |",
        f"| 检索与重排耗时 | {m['retrieval_ms']/1000:.2f}秒 |",
        f"| 首个Token | {m['first_token_ms']/1000:.2f}秒 |",
        f"| 输入 / 输出Token | {m['input_tokens']} / {m['output_tokens']}（实际用量，非估算） |",
        f"| 缓存命中 | {m['cache_hit']} |",f"| 请求ID | {run.get('request_id',run.get('id','见原始run记录'))} |","",
        "这些数值只描述本次真实请求，不是并发负载测试、稳定均时或完整题集成绩。模型回答的自动证据审校仍有"
        f"{r['quality_review_warning_count']}条待核查提示；运行通过不等于回答全部事实已获语义认证，原始回答与提示完整保留。", "",
        "## 实际证据文件","",
        "- `runtime-record.json`：本地运行结论、组件、端口、实际用量与边界。",
        "- `processes-and-listeners.json`：真实监听地址、PID、可执行文件和启动参数。",
        "- `start-transcript.txt`及API/UI/Ollama日志：启动与执行现场。",
        "- `health.json`、`ollama-model-ps.json`：采集时的后端、检索和模型状态。",
        "- `acceptance/acceptance.json`、`acceptance/验收记录.md`：逐项检查与时间。",
        "- `acceptance/http-*.json`、`rag-events.json`：HTTP响应、原回答、SSE与引用审校。",
        "- `acceptance/citation-page-preview.png`：本次返回的论文物理页。",
        "- `本地界面运行截图.jpg`：本轮本地页面与模型就绪状态；不将页面中既有历史会话冒称本次生成。", "",
        "证据中workspace token与认证字段脱敏；独立验收工作区数据留存供核对。", ""]
    (EVIDENCE/"运行记录.md").write_text("\n".join(lines),encoding="utf-8")
    proof={"created_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"status":"passed","files":[]}
    known=[]
    for p in (ROOT/"data/runtime").glob("*workspace*.json"):
        try:
            v=read(p)
            for key in ("workspace_token","scope"):
                if isinstance(v.get(key),str) and len(v[key])>=12:known.append(v[key])
        except (ValueError,OSError):pass
    for p in sorted(EVIDENCE.rglob("*")):
        if not p.is_file() or p.name=="证据清单.json":continue
        raw=p.read_bytes()
        if p.suffix in (".json",".md",".txt",".log"):
            assert all(secret.encode() not in raw for secret in known),"Private workspace value found in evidence; inspect locally before packaging."
        proof["files"].append({"path":str(p.relative_to(EVIDENCE)).replace("\\","/"),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    proof["known_private_workspace_values_found"]=0
    (EVIDENCE/"证据清单.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding="utf-8")
    target=ROOT/"artifacts/packages/本地127运行验收记录.zip"
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(EVIDENCE.rglob("*")):
            if p.is_file():z.write(p,"本地127运行验收记录/"+str(p.relative_to(EVIDENCE)).replace("\\","/"))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(json.dumps({"status":r["status"],"unique_checks":r["unique_check_count"],"failures":r["failed_check_count"],"files":len(proof["files"])+1,"archive":str(target),"bytes":target.stat().st_size},ensure_ascii=False))


if __name__=="__main__":main()
