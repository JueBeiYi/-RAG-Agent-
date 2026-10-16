# Docker实际部署验收

本文件说明可复现的验收方法。**部署结果以实际生成的 `acceptance.json` 和 `验收记录.md` 为准；脚本存在本身不代表已通过。**

使用单独的 Compose 项目和端口，避免影响已有本机服务。真实模型验收使用 `standard`；`demo` 仅证明上传、控制流和存储，不可冒充大模型部署验收。

## 执行

先按实际运行环境准备模型、Compose覆盖文件并启动容器，再执行：

```powershell
python scripts/verify_docker_deployment.py --project rag-acceptance-20261015 --compose docker/docker-compose.yml --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --profile standard --evidence-dir reports/docker/20261015-standard --restart
```

如果使用覆盖文件，重复传入 `--compose`，顺序与启动命令一致。`--docker` 可指定 Docker CLI 完整路径。`--document` 可指定真实论文 PDF，`--question` 可指定论文相关问题；省略文档时会明确生成并标注“部署验收夹具”TXT，其数字不属于论文实验结果。

## 实测范围与证据

- Docker客户端/服务端、Engine信息、Compose版本及解析配置、镜像。
- 容器Compose项目归属、进程状态、API健康检查、持久卷挂载。
- 后端存活及模型/检索就绪，前端Streamlit健康端点。
- 独立随机工作区中的真实上传、解析、分块、索引任务完成。
- 实际混合检索与重排、SSE逐步回答、模型后端、非演示且非降级的回答来源。
- 引用编号对应已保存片段，原件下载SHA-256一致；PDF物理页预览实际返回PNG。
- 会话、回答、调用记录与导出。
- 仅重启指定验收项目的API与前端，再验证文档、所有片段、会话、调用记录和原文件的持久性。
- 无论成功失败均保留实际命令返回码、脱敏输出、HTTP响应、SSE事件和容器日志；失败以非零退出码结束。

工作区token只保留在进程内存，报告仅保存其SHA-256标识。API密钥、密码和身份认证字段会脱敏，不保存请求认证头。容器inspect只获取指定字段，不保存包含密钥的Config.Env。

## 判定边界

只有 `standard` 全部检查通过且完成 `--restart`，总状态才为 `passed`。未验证重启为 `partial_restart_not_tested`，demo通过为 `demo_control_flow_passed`。该验收证明容器实际部署与核心流程，不替代完整题集质量评分或浏览器视觉验收。

实际记录位置与运行结论将在真实执行后补入本页；不得在执行前预写“部署通过”。
