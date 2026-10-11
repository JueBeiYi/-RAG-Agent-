import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

// The caller resolves these authoritative bundled paths; no global packages.
const config = JSON.parse(await fs.readFile(process.argv[2], "utf8"));
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(config.runtime.node_modules, "@oai/artifact-tool/dist/artifact_tool.mjs")).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(config.runtime.presentation_skill, "container_tools/artifact_tool_utils.mjs")).href);
const payload = config.payload;
const FONT = "Microsoft YaHei";
const INK = "#17354A", MUTED = "#607484", TEAL = "#287C85";
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
presentation.theme.defaultFont = FONT;
const chartOwners = [], tableOwners = [];
// Chart workbooks retain four displayed decimals; unrounded metrics remain in source JSON.
const chartValue = (value) => typeof value === "number" ? Number(value.toFixed(4)) : 0;
await fs.mkdir(config.build_dir, { recursive: true });

function text(slide, value, x, y, width, height, size = 27, bold = false, color = INK) {
  const shape = slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width, height },
    fill: "none", line: { fill: "none", width: 0 } });
  shape.text = String(value);
  shape.text.style = { typeface: FONT, fontSize: size, bold, color, alignment: "left" };
  return shape;
}

function slide(title, subtitle = "", sources = []) {
  const s = presentation.slides.add();
  s.background.fill = "#FFFFFF";
  text(s, title, 72, 48, 1120, 76, 43, true);
  if (subtitle) text(s, subtitle, 74, 123, 1120, 58, 23, false, MUTED);
  text(s, "南京农业大学课程实践", 74, 673, 860, 26, 16, false, MUTED);
  text(s, String(presentation.slides.items.length).padStart(2, "0"), 1140, 673, 66, 26, 16, false, MUTED);
  s.speakerNotes.text = sources.map(source => "来源 " + source).join("\n");
  return s;
}

function lines(s, values, y = 220, size = 29, spacing = 74) {
  values.forEach((value, index) => text(s, value, 82, y + index * spacing, 1110, spacing - 10, size));
}

function table(s, headers, rows, widths) {
  const values = [headers, ...rows];
  const t = s.tables.add({ rows: values.length, columns: headers.length, left: 75, top: 203,
    width: 1130, height: values.length * 58, values, columnWidths: widths });
  t.borders.assign({ style: "solid", fill: "#D9E3E8", width: 1 });
  for (let r = 0; r < values.length; r++) for (let c = 0; c < headers.length; c++) {
    const cell = t.getCell(r, c);
    cell.fill = r === 0 ? "#203C50" : r % 2 ? "#FFFFFF" : "#F0F5F7";
    cell.text.style = { typeface: FONT, fontSize: 23, color: r === 0 ? "#FFFFFF" : INK, bold: r === 0 };
  }
  tableOwners.push(presentation.slides.items.length);
  return t;
}

const cover = slide("智能科研助理");
text(cover, "基于 RAG 与 Agent 的\n论文知识库问答系统", 76, 205, 1090, 172, 56, true);
text(cover, payload.project.course || "课程名称待填写", 82, 418, 1050, 50, 28, false, TEAL);
const people = (payload.project.members || []).map(member => member.name || "姓名待填写").join("  ");
text(cover, people || "团队信息待填写", 82, 490, 1050, 42, 27);
text(cover, payload.project.deadline ? `提交截止 ${payload.project.deadline}` : "提交日期待填写", 82, 562, 1050, 38, 23, false, MUTED);

const scope = slide("科研阅读与证据核对", "通用产品支持不同领域 课程评测固定具体研究方向");
lines(scope, ["资料导入和论文知识库管理", "带引用问答与 PDF 原页证据定位", "论文比较与多工具执行轨迹", "真实实验记录和可复现导出"]);

const architecture = slide("系统架构", "展示 服务 调度 检索与持久化", ["src/api", "src/services", "src/agent", "src/retrieval"]);
const layers = ["Streamlit 展示层   论文库 问答 对比 实验", "FastAPI 服务层   REST 接口 SSE 事件", "Agent 与 RAG   多工具调度 会话记忆 引用", "检索层   Embedding BM25 RRF 重排序", "持久化   原件 SQLite 向量索引 日志与实验"];
layers.forEach((label, i) => {
  const y = 196 + i * 83;
  const node = architecture.shapes.add({ geometry: "rect", position: { left: 196, top: y, width: 890, height: 62 },
    fill: i % 2 ? "#F0F5F7" : "#EAF3F4", line: { fill: "#A5BEC5", width: 1.2 } });
  node.text = label;
  node.text.style = { typeface: FONT, fontSize: 26, color: INK, alignment: "center", bold: false };
  if (i < layers.length - 1) architecture.shapes.add({ geometry: "downArrow", position: { left: 626, top: y + 65, width: 30, height: 15 }, fill: TEAL, line: { fill: "none", width: 0 } });
});
await fs.writeFile(path.join(config.build_dir, "architecture.png"), new Uint8Array(await (await presentation.export({ slide: architecture, format: "png", scale: 1.5 })).arrayBuffer()));

const moduleSlide = slide("模块实现与验证", "状态来自明确的实现验证清单", payload.source_files.filter(s => s.kind === "implementation").map(s => s.path));
const modules = payload.modules.slice(0, 6).map(row => [row[0], row[1]]);
table(moduleSlide, ["模块", "已记录状态"], modules, [590, 540]);

const retrieval = slide("检索实验", "Hit@5 MRR Recall@5 来自同一语料与题目集合", payload.retrieval.map(r => r.source_path));
retrieval.speakerNotes.text += "\n图表数据保留四位小数，来源JSON保留未舍入指标。人工题集共识与答案质量评分未完成。";
if (payload.retrieval.length) {
  const chart = retrieval.charts.add("bar", { position: { left: 95, top: 207, width: 1090, height: 390 },
    categories: payload.retrieval.map(r => ({ vector: "纯向量", hybrid: "混合检索", hybrid_rerank: "混合与重排" }[r.mode] || r.mode)),
    series: [
      { name: "Hit@5", values: payload.retrieval.map(r => chartValue(r.metrics.hit_at_5)), fill: TEAL },
      { name: "MRR", values: payload.retrieval.map(r => chartValue(r.metrics.mrr)), fill: "#4E71A8" },
      { name: "Recall@5", values: payload.retrieval.map(r => chartValue(r.metrics.recall_at_5)), fill: "#759478" },
    ], hasLegend: true, legend: { position: "bottom", textStyle: { fontSize: 22 } }, barOptions: { direction: "column", grouping: "clustered" },
    xAxis: { textStyle: { fontSize: 22 } },
    yAxis: { min: 0, max: 1, majorUnit: 0.2, numberFormatCode: "0.0", textStyle: { fontSize: 20 } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000", textStyle: { fontSize: 20 } },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  chartOwners.push(presentation.slides.items.length);
  text(retrieval, `每个实验题数 ${payload.retrieval.map(r => r.metrics.case_count).join("  ")}`, 83, 612, 1100, 35, 21, false, MUTED);
} else {
  lines(retrieval, ["尚未提供可对比的真实模型检索结果", "功能演示分数不进入正式检索图表", "固定语料与人工标注题目后再记录差异"], 245, 29, 90);
}
const retrievalImage = payload.retrieval.length ? path.join(config.build_dir, "retrieval-results.png") : null;
if (retrievalImage) await fs.writeFile(retrievalImage, new Uint8Array(await (await presentation.export({ slide: retrieval, format: "png", scale: 1.5 })).arrayBuffer()));

const parameter = slide("分块策略实验", "完整矩阵与原始记录见独立分块报告");
const studies = payload.experiments.filter(r => !r.demo && r.study === "chunking").slice(0, 6);
if (studies.length) table(parameter, ["实验", "分块策略与长度", "Hit@5"], studies.map(r => [r.label,
  `${r.config.chunk_strategy || "未记录"} ${r.config.chunk_size || ""}`,
  typeof r.metrics.hit_at_5 === "number" ? r.metrics.hit_at_5.toFixed(4) : "未测量"]), [390, 480, 260]);
else lines(parameter, ["三种分块策略与 256 512 1024 长度", "相同语料与题集 独立映射原文证据", "尚未提供真实结果的实验保留为待执行"], 235, 28, 98);

const promptRuns = payload.experiments.filter(r => !r.demo && ["prompt", "context", "generation"].includes(r.study)).slice(0, 6);
const promptSlide = slide("生成参数与证据约束", "参数实验保留真实答案 人工质量评分单独填写", promptRuns.map(r => r.source_path));
if (promptRuns.length) table(promptSlide, ["实验", "T  p  k", "平均用时ms"], promptRuns.map(r => [r.label,
  `${r.config.temperature ?? "未记"}  ${r.config.top_p ?? "未记"}  ${r.config.top_k_sampling ?? "未记"}`,
  typeof r.metrics.latency_mean_ms === "number" ? r.metrics.latency_mean_ms.toFixed(1) : "未测量"]), [450, 320, 360]);
else lines(promptSlide, ["来源编号由服务分配 回答后核对", "事实依据不足时说明证据限制", "当前尚未提供真实生成参数测量"], 235, 28, 98);

const agent = slide("Agent 与回答质量", "执行成功与证据覆盖分别统计 答案质量仍需人工评分", payload.generation.map(r => r.source_path));
if (payload.generation.length) {
  const r = payload.generation[0];
  const pairs = [["回答完成率", r.metrics.answer_success_rate], ["工具成功率", r.metrics.tool_success_rate], ["路由覆盖", r.metrics.routing_recall], ["证据覆盖", r.metrics.citation_evidence_recall]].filter(pair => typeof pair[1] === "number");
  table(agent, ["指标", "实际测量"], pairs.map(([name, value]) => [name, value.toFixed(4)]), [660, 470]);
  text(agent, `平均耗时 ${Number(r.metrics.latency_mean_ms || 0).toFixed(1)} ms`, 83, 535, 1110, 40, 25);
} else lines(agent, ["工具路由 并行 缓存与会话隔离均可独立评测", "正确性 完整性 引用质量 流畅性由人工评分", "当前尚未提供真实生成评测结果"], 232, 28, 98);

if (payload.online.length) {
  const online = slide("真实线上功能样本", "问题与配置有变化 此表用于记录实际运行", payload.online.map(r => r.source_path));
  const m = payload.online[0].metrics;
  table(online, ["指标", "记录值"], [["请求样本数", String(m.sample_count)], ["平均端到端耗时", `${(m.latency_mean_ms / 1000).toFixed(2)} 秒`],
    ["输入Token总计", String(m.input_tokens_total)], ["输出Token总计", String(m.output_tokens_total)], ["平均Agent步骤", String(m.average_agent_steps)]], [680, 450]);
  text(online, "包含历史降级样本 统一任务基准与人工语义评分另列", 83, 606, 1100, 41, 23, false, MUTED);
}

if (payload.validation_summary.length) {
  const validation = slide("代码与运行验证", "自动验证证明对应行为 人工论文质量单独验收", payload.validation_summary.map(r => r.source));
  table(validation, ["验证项目", "实际结果"], payload.validation_summary.slice(0, 6).map(r => [r.name, r.result]), [680, 450]);
}

for (const shot of payload.screenshots) {
  const s = slide(shot.title || "系统运行界面", shot.description || "真实运行截图", [shot.path]);
  const bytes = await fs.readFile(shot.path);
  s.images.add({ blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: path.extname(shot.path).toLowerCase() === ".jpg" ? "image/jpeg" : "image/png",
    alt: shot.title || "系统运行截图", fit: "contain", position: { left: 76, top: 194, width: 1125, height: 446 } });
}
if (!payload.screenshots.length) {
  const s = slide("界面演示材料", "真实截图接口已准备");
  lines(s, ["交付前插入 3 至 5 张实际运行截图", "论文库 科研问答 证据定位 论文对比 实验", "设计稿和模拟截图不用于证明功能已经运行"], 235, 28, 98);
}

const cases = slide("失败案例与实际改进", "检索 生成 引用 Agent 四类记录");
if (payload.bad_cases.length) {
  const shortStatus = r => {
    const status = r.improvement_status || "";
    if (/进行中|待归档|待完成|待复测/.test(status)) return /已实现/.test(status) ? "代码已实现 复测待完成" : "修复与验证进行中";
    if (/线上/.test(status)) return "代码已修复 线上样例有记录";
    if (/已实现/.test(status)) return "代码已修复 验证有记录";
    return status || (r.after_result ? "已有重跑记录 待复核" : "尚未完成重跑");
  };
  cases.speakerNotes.text += "\n完整状态、前后请求和限制见独立Bad Case报告。";
  table(cases, ["案例", "类型", "改进状态"], payload.bad_cases.slice(0, 5).map(r => [r.case_id || "案例", r.type || "待分类", shortStatus(r)]), [400, 250, 480]);
} else lines(cases, ["当前尚未提供已分析的 Bad Case 台账", "每项记录原错误 根因 具体改动与重跑结果", "固定测试集的前后数据用于检验实际改进"], 230, 28, 98);

const finish = slide("交付与待完成事项", "源码 可编辑报告 演示文稿 原始实验与复现记录");
lines(finish, [`真实测量记录 ${payload.formal_result_count} 份`, `功能演示实验 ${payload.demo_result_count} 份`,
  payload.pending_fields.length ? `待填写信息 ${payload.pending_fields.length} 项` : "团队与提交信息已经提供"], 216, 27, 76);
text(finish, payload.evaluation_note, 82, 475, 1110, 146, 24, false, MUTED);

const candidate = path.join(config.build_dir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const previews = [];
for (const [index, s] of presentation.slides.items.entries()) {
  const preview = path.join(config.build_dir, `slide-${String(index + 1).padStart(2, "0")}.png`);
  await fs.writeFile(preview, new Uint8Array(await (await presentation.export({ slide: s, format: "png", scale: 1 })).arrayBuffer()));
  previews.push(preview);
}
const receipt = await finalizePresentation({ workspaceDir: config.workspace_dir, candidatePath: candidate,
  finalPath: config.final_path, pythonExecutable: config.runtime.python,
  integrityValidatorPath: path.join(config.runtime.presentation_skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(config.runtime.presentation_skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit", ...tableOwners.flatMap(i => ["--require-native-table-slide", String(i)])],
  requiredNativeTableOwnerSlides: tableOwners, requiredNativeChartOwnerSlides: chartOwners,
  fontPolicy: { basis: "design", families: [FONT] }, verifyArtifactToolImport: true,
  materializeLiteralChartWorkbooks: true, receiptPath: path.join(config.build_dir, "pptx-validation.json") });
await fs.writeFile(path.join(config.build_dir, "pptx-result.json"), JSON.stringify({ final_path: config.final_path,
  slide_count: presentation.slides.items.length, previews, architecture: path.join(config.build_dir, "architecture.png"), retrieval_chart: retrievalImage, receipt }, null, 2));
console.log(JSON.stringify({ final_path: config.final_path, slide_count: presentation.slides.items.length }));
