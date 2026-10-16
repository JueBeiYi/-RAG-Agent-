import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from './academic-workbook-support/runtime.mjs';

// Reproducible workbook authoring with the bundled artifact-tool runtime.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'artifacts/academic-figures');
const qaOut = path.join(root, 'reports/presentation/qa/workbook');
const dataPath = path.join(root, 'reports/presentation/source-data/experiments.json');
const source = JSON.parse(await fs.readFile(dataPath, 'utf8'));
// Quality judgements are additive records, not replacements for human-score columns.
const qualityDir='reports/experiments/generation/quality-comparison';
const quality=JSON.parse(await fs.readFile(path.join(root,qualityDir,'ai_quality_summary.json'),'utf8'));
const qualityRows=JSON.parse(await fs.readFile(path.join(root,qualityDir,'ai_quality_rows.json'),'utf8'));
source.tables.quality_conditions ??= {
  title:'生成参数与Prompt的AI质量评价', columns:Object.keys(quality.conditions[0]),rows:quality.conditions,
  notes:['每条件4题、单次生成；正确性/完整性/引用准确性/流畅性均为1–5分。','三维均分为描述性等权汇总，不是课程官方总分。','28个条件记录含17个独立答案评分；相同输出复用既有评审，当前Prompt复用参数baseline。','历史359片段冻结检索生成研究，输出上限256 token；未进行显著性推断。'],
  source_paths:[`${qualityDir}/ai_quality_summary.json`],
};
source.tables.quality_records ??= {
  title:'参数与Prompt逐题AI评审',columns:Object.keys(qualityRows[0]),rows:qualityRows,
  notes:['复用记录通过 score_reused_from/review_id 追踪。原始模型回答与AI评分依据一并保留。'],
  source_paths:[`${qualityDir}/ai_quality_rows.json`],
};
if(!source.tables.quality_conditions.notes?.length) source.tables.quality_conditions.notes=['AI辅助评分，量表1–5分；每个条件4题、每题单次生成。','28个条件记录含17个独立答案评分；相同输出复用评审，当前Prompt复用参数baseline。','三维均分为描述性等权汇总，不是课程官方总分；小样本不作显著性与普适优劣推断。'];
const C = { navy: '#142D4E', teal: '#087F8C', orange: '#D88C3A', ink: '#24374D', muted: '#607488', pale: '#EDF4F7', line: '#CCD8E1', white: '#FFFFFF' };
const font = 'Arial';
const wb = Workbook.create();
const definitions = [
  ['总览', []],
  ['检索与优化', ['retrieval_test','retrieval_dev','retrieval_category','rrf_grid','retrieval_records','rrf_records']],
  ['文档与索引', ['parsers','chunking','embeddings','vectorstores','parser_records','embedding_batches']],
  ['生成与Prompt', ['quality_conditions','prompt_ai_quality','generation_conditions','baseline_ai_quality','baseline_ai_distribution','baseline_ai_category']],
  ['Agent与运行', ['agent','parallel','cache','online','online_records']],
  ['逐题结果', ['baseline_ai_scores','generation_records','quality_records']],
  ['配置与数据', ['corpus','dataset','experiment_config','bad_cases','acceptance_summary','acceptance']],
  ['来源与方法', []],
];
const byName = Object.fromEntries(definitions.map(([name]) => [name, wb.worksheets.add(name)]));
const placed = new Set();
const refs = {};
const previews = [];
const audit = { source: path.relative(root, dataPath), tables: [], previews: [], sheets: [] };

const labels = {
  mode:'检索方式', split:'数据划分', category:'问题类型', condition:'实验条件', condition_id:'条件编号', study:'研究', name:'名称',
  hit_at_5:'Hit@5', mrr:'MRR', recall_at_5:'Recall@5', hit_at_k:'Hit@K', recall_at_k:'Recall@K',
  hit_5:'Hit@5', recall_5:'Recall@5', top_k:'Top-K', rrf_k:'RRF常数', result_k:'返回数量',
  count:'数量', case_count:'问题数', run_count:'回答数', query_count:'查询数', sample_count:'样本数',
  latency_ms:'延迟 (ms)', latency_s:'延迟 (s)', elapsed_s:'耗时 (s)', elapsed_seconds:'耗时 (s)', mean_latency_ms:'平均延迟 (ms)',
  mean_latency_seconds:'平均延迟 (s)', total_tokens:'总Token数', prompt_tokens:'输入Token数', completion_tokens:'输出Token数',
  input_tokens:'输入Token数', output_tokens:'输出Token数', temperature:'Temperature', top_p:'Top-p', top_k_sampling:'Top-k采样',
  parser:'PDF解析器', chunk_strategy:'分块策略', chunk_size:'块大小 (token)', chunk_overlap:'重叠 (token)', chunk_count:'片段数',
  embedding_model:'向量模型', reranker_model:'重排模型', backend:'后端', vector_backend:'向量存储', model:'模型',
  correctness_1_to_5:'正确性 (1–5)', completeness_1_to_5:'完整性 (1–5)', citation_quality_1_to_5:'引用准确性 (1–5)', fluency_1_to_5:'流畅性 (1–5)',
  correctness:'正确性', completeness:'完整性', citation_quality:'引用准确性', fluency:'流畅性',
  correctness_mean:'AI正确性均分', completeness_mean:'AI完整性均分', citation_quality_mean:'AI引用准确性均分', fluency_mean:'AI流畅性均分', completion_tokens_mean:'平均输出Token数', quality_three_dimension_mean:'质量三维均分',
  ai_correctness_mean:'AI正确性均分', ai_completeness_mean:'AI完整性均分', ai_citation_quality_mean:'AI引用准确性均分', ai_fluency_mean:'AI流畅性均分',
  question:'问题', answer:'模型回答', reference_answer:'参考答案', question_id:'问题ID', case_id:'问题ID', row_id:'记录ID',
  reviewer:'评分者', grading_method:'评分方法', review_notes:'评分说明', notes:'说明', evidence:'依据', page_evidence:'页码依据',
  status:'状态', source:'来源', source_file:'源文件', source_path:'源路径', sha256:'SHA-256', file:'文件',
  scope:'实验范围', index_version:'索引版本', config:'配置', config_json:'配置JSON', models_json:'模型JSON',
  dimension:'评价维度', n:'样本数', mean:'均值', std:'标准差', min:'最小值', max:'最大值', scale_min:'量表下界', scale_max:'量表上界', grading_date:'评分日期', score:'评分', proportion:'占比',
  paper_key:'论文编号', title:'论文标题', arxiv_version:'arXiv版本', publication_year:'年份', venue:'会议/期刊', physical_pages:'物理页数', byte_count:'文件大小 (B)', pdf_url:'论文链接', local_file:'语料文件',
  total:'总计', dev:'开发集', test:'测试集', paper_count:'论文数', parse_ms:'解析耗时 (ms)', anchor_retention:'锚点保留率', matched_anchors:'保留锚点数', total_anchors:'锚点总数', table_blocks:'表格块数', formula_blocks:'公式块数', pages_with_text:'含文字页数', bbox_blocks:'含位置块数',
  document_id:'文档ID', content_hash:'内容校验值', block_count:'内容块数', text_characters:'字符数', blocks_with_bbox:'含位置块数', source_anchor_count:'源锚点数', source_anchors_retained:'保留锚点数',
  successful_count:'成功数', error_count:'错误数', latency_mean_ms:'平均耗时 (ms)', latency_p50_ms:'P50耗时 (ms)', latency_p95_ms:'P95耗时 (ms)', error:'错误信息', evidence_count:'依据数量', recalled_evidence_count:'召回依据数', first_relevant_rank:'首个相关排名', top1_score:'Top-1相关分', elapsed_ms:'耗时 (ms)', vector_db_ms:'向量检索 (ms)', bm25_ms:'BM25 (ms)', fusion_ms:'融合 (ms)', rerank_ms:'重排 (ms)', amortized_query_encoding_ms:'均摊查询编码 (ms)', latency_scope:'耗时测量范围',
  variant:'实验变体', strategy:'分块策略', split_ms:'分块耗时 (ms)', chunks_hash:'片段校验值', implementation_sha256:'实现校验值', cpu_threads:'CPU线程数', model_batch_size:'批次大小', window_token_cap:'窗口token上限', query_encoding_total_ms:'查询编码总计 (ms)', query_encoding_amortized_ms:'查询编码均摊 (ms)', model_load_ms:'模型加载 (ms)', unique_document_text_count:'去重文本数', document_encoding_total_ms:'文档编码 (ms)', document_texts_per_second:'文本吞吐量 (条/s)', artifact_revision:'模型版本', embedding_fingerprint:'向量指纹', batch_index:'批次序号', text_count:'文本数量', window_count:'窗口数量', measured_at:'测量时间',
  dimensions:'向量维度', build_and_persist_ms:'建库与持久化 (ms)', restart_first_query_ms:'重启首查询 (ms)', delete_ms:'删除 (ms)', restore_ms:'恢复 (ms)', delete_probe_passed:'删除探针结果', metric_k:'指标K', hit:'Hit@K', recall:'Recall@K', mrr_at_20:'MRR@20', additional_model_calls:'额外模型调用', model_calls:'模型调用次数',
  generation_protocol_success_rate:'生成协议成功率', prompt_tokens_total:'输入Token总数', completion_tokens_total:'输出Token总数', token_usage_known_count:'Token已知记录数', invalid_citation_run_count:'引用规则异常次数', numeric_rule_review_run_count:'数字规则提醒次数', answers_with_source_markers:'含来源标记回答数', output_truncation_count:'截断次数', human_correctness_mean:'真人正确性均分', human_completeness_mean:'真人完整性均分', human_citation_quality_mean:'真人引用准确性均分', human_scoring_completed_count:'真人评分条数', measurement_scope:'测量范围', routing_exact_match:'工具路由准确率',
  first_token_ms:'首Token (ms)', success:'执行成功', truncated:'输出截断', expected_tools:'预期工具', invoked_tools:'实际工具', cache_hit:'缓存命中', actual_model_calls:'实际模型调用', workers:'并行工作数', local_tools_only:'仅本地工具', llm_called:'调用大模型', relevant_sources:'参考来源', citations:'回答引用', source_checked:'来源已核对', review_method:'审阅方式', human_review_pending:'待真人复核', bad_case_type:'问题类型', root_cause:'根因', grading_evidence:'评分依据',
  input_tokens_total:'输入Token总数', output_tokens_total:'输出Token总数', average_agent_steps:'平均Agent步数', request_id:'请求ID', created_at:'记录时间', tool_calls:'工具调用数', tool_successes:'工具成功数', agent_steps:'Agent步数', origin:'请求来源', comparison_method:'比较方式', warnings:'警告', type:'类型', observed_error:'错误表现', improvement:'改进方案', before_result:'改进前', after_result:'改进后', improvement_status:'改进状态', limitation:'解释边界',
  id:'编号', source_paragraph_id:'课程段落', module:'课程模块', requirement:'要求', source_text:'课程原文', mandatory:'必做', implementation_status:'实现状态', execution_status:'执行状态', evidence_paths:'依据路径', interpretation:'验收说明', pending:'待完成项', requirement_count:'条款数', config_hash:'配置校验值', dataset_hash:'题集校验值', corpus_hash:'语料校验值', models:'模型配置', item:'项目', path:'源文件', bytes:'大小 (B)',
};
const label = key => labels[key] ?? key;
function col(n) { let s=''; for(let v=n+1;v>0;v=Math.floor((v-1)/26)) s=String.fromCharCode(65+(v-1)%26)+s; return s; }
function cellValue(v) { if(v===undefined||v===null) return null; if(typeof v==='object') return JSON.stringify(v); return v; }
function base(sheet, endRow=100, endCol=14) {
  sheet.showGridLines=false;
  const r=sheet.getRange(`A1:${col(endCol-1)}${Math.max(endRow,10)}`);
  r.format.font={name:font,size:10,color:C.ink};
  r.format.rowHeight=21;
  r.format.verticalAlignment='center';
  r.format.columnWidth=15;
  sheet.getRange(`A1:A${Math.max(endRow,10)}`).format.columnWidth=24;
  sheet.tabColor=C.navy;
}
function title(sheet,text,sub='') {
  sheet.getRange('A2').values=[[text]];
  sheet.getRange('A2').format.font={name:font,size:16,bold:true,color:C.navy};
  sheet.getRange('A2').format.rowHeight=27;
  sheet.getRange('A3:N3').format.borders={bottom:{style:'thin',color:C.teal}};
  if(sub) { sheet.getRange('A4').values=[[sub]]; sheet.getRange('A4').format.font={name:font,size:10,color:C.muted,italic:true}; }
}
function makeTable(sheet,id,t,start) {
  const scoreOrder=id==='quality_records'?['study','condition','case_id','correctness','completeness','citation_quality','fluency','grading_method','notes']:['case_id','category','correctness_1_to_5','completeness_1_to_5','citation_quality_1_to_5','fluency_1_to_5','grading_method','review_notes'];
  const columns=['baseline_ai_scores','quality_records'].includes(id)?[...scoreOrder.filter(k=>t.columns.includes(k)),...t.columns.filter(k=>!scoreOrder.includes(k))]:t.columns;
  const n=columns.length;
  if(!n) return start;
  sheet.getRange(`A${start}`).values=[[t.title || id]];
  sheet.getRange(`A${start}`).format.font={name:font,size:12,bold:true,color:C.navy};
  const h=start+2, end=h+t.rows.length;
  const header=sheet.getRange(`A${h}:${col(n-1)}${h}`);
  header.values=[columns.map(label)];
  header.format={fill:C.navy,font:{name:font,size:10,bold:true,color:C.white},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:34};
  if(t.rows.length) {
    const grid=sheet.getRange(`A${h+1}:${col(n-1)}${end}`);
    grid.values=t.rows.map(row=>columns.map(k=>cellValue(row[k])));
    grid.format.rowHeight=id.includes('scores')||id==='quality_records'?54:24;
    for(let i=0;i<columns.length;i++) {
      const k=columns[i], r=sheet.getRange(`${col(i)}${h+1}:${col(i)}${end}`);
      const present=t.rows.map(x=>x[k]).filter(v=>v!==null&&v!==undefined);
      const numeric=present.length>0&&present.every(v=>typeof v==='number');
      if(numeric) {
        r.format.horizontalAlignment='right';
        if(k==='publication_year') r.setNumberFormat('0');
        else if(/hit|recall|mrr|rate|accuracy|fraction|ratio|success|coverage|retention/.test(k)&&present.every(v=>v>=0&&v<=1)) r.setNumberFormat('0.0%');
        else if(/mean|latency|elapsed|seconds|score|correctness|completeness|quality|fluency|speed|median|p95|temperature|top_p|_ms$|_s$/.test(k)) r.setNumberFormat('0.00');
        else r.setNumberFormat(present.every(Number.isInteger)?'#,##0':'0.0000');
      } else {
        r.format.horizontalAlignment='left';
        if(/answer|question|notes|evidence|description|reason|root_cause|improvement|config|json|interpretation|^title$|^path$|scope/.test(k)) { r.format.wrapText=true; r.format.verticalAlignment='top'; }
      }
    }
    for(let y=h+1;y<=end;y++) if((y-h)%2===0) sheet.getRange(`A${y}:${col(n-1)}${y}`).format.fill=C.pale;
    sheet.getRange(`A${end}:${col(n-1)}${end}`).format.borders={bottom:{style:'thin',color:C.line}};
    if(id==='corpus') grid.format.rowHeight=58;
  }
  // Context is part of the scientific record, and is kept close to each result.
  let r=end+2;
  const notes=[...(t.notes??[]),...(t.source_paths??[]).map(x=>`数据源：${x}`)];
  for(const note of notes) {
    sheet.getRange(`A${r}`).values=[[String(note)]];
    sheet.getRange(`A${r}:${col(Math.min(n-1,9))}${r}`).format={font:{name:font,size:10,color:C.muted},wrapText:false,rowHeight:23,verticalAlignment:'center'};
    r++;
  }
  refs[id]={sheet:sheet.name,headerRow:h,dataStart:h+1,dataEnd:end,columns};
  audit.tables.push({id,sheet:sheet.name,rows:t.rows.length,columns:n,range:`A${h}:${col(n-1)}${end}`});
  placed.add(id);
  return r+3;
}
function tableData(id) { return source.tables?.[id]; }
function formulaRef(id,row,key) {
  const ref=refs[id]; if(!ref) return null;
  const ci=ref.columns.indexOf(key); if(ci<0) return null;
  return `='${ref.sheet}'!${col(ci)}${ref.dataStart+row}`;
}
function chart(sheet,range,titleText,from,to,numberFormat='0.0%') {
  const c=sheet.charts.add('bar',sheet.getRange(range));
  c.setPosition(from,to); c.title=titleText;
  c.titleTextStyle.fontSize=14; c.titleTextStyle.typeface=font;
  c.legend={position:'top',textStyle:{typeface:font,fontSize:11}};
  c.xAxis={axisType:'textAxis',textStyle:{typeface:font,fontSize:11}};
  c.yAxis={numberFormatCode:numberFormat,numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
  c.series.items.forEach((s,i)=>s.fill=[C.teal,C.orange,C.navy][i%3]);
  return c;
}

// Source tables remain intact. Sections share sheets only where they serve the same reader.
for(const [name,ids] of definitions.slice(1,-1)) {
  const sheet=byName[name];
  const tables=ids.filter(id=>tableData(id));
  const anticipated=tables.reduce((n,id)=>n+tableData(id).rows.length+15,6);
  const ncols=Math.max(14,...tables.map(id=>tableData(id).columns.length));
  base(sheet,anticipated,ncols);
  title(sheet,name,'真实实验记录；AI评分保留原始方法标识。不同索引版本与数据划分分别列示。');
  let row=6;
  for(const id of tables) row=makeTable(sheet,id,tableData(id),row);
  for(let i=0;i<ncols;i++) {
    const keys=tables.map(id=>tableData(id).columns[i]).filter(Boolean);
    let width=16;
    if(keys.some(k=>/question$|answer$|notes|evidence|config|json|description|interpretation|reason|root_cause|source_path|^file$/.test(k))) width=42;
    else if(keys.some(k=>/name|model|condition|^source$|strategy|backend|experiment|study/.test(k))) width=27;
    sheet.getRange(`${col(i)}1:${col(i)}${row}`).format.columnWidth=width;
  }
  sheet.freezePanes.freezeRows(8); sheet.freezePanes.freezeColumns(2);
  previews.push({sheetName:name,range:`A1:${col(Math.min(ncols-1,7))}${Math.min(row,24)}`,file:`workbook-${name}.png`});
  audit.sheets.push({name,lastRow:row,columnCount:ncols});
}

const trace=byName['来源与方法'];
const unknown=Object.keys(source.tables??{}).filter(id=>!placed.has(id));
base(trace,250,12); title(trace,'来源与方法','统计总体、实验范围及来源校验用于追踪图表数值。');
let tr=6;
for(const id of unknown) tr=makeTable(trace,id,source.tables[id],tr);
const scopes=source.experimental_scopes;
const scopeRows=Array.isArray(scopes)?scopes.map((v,i)=>typeof v==='object'?v:{item:String(i+1),scope:v}):Object.entries(scopes??{}).map(([item,v])=>({item,scope:typeof v==='object'?JSON.stringify(v):v}));
if(scopeRows.length) tr=makeTable(trace,'__scopes',{title:'实验范围',columns:[...new Set(scopeRows.flatMap(Object.keys))],rows:scopeRows,notes:[],source_paths:[]},tr);
const hashes=source.source_hashes;
const hashRows=Array.isArray(hashes)?hashes.map(x=>typeof x==='object'?x:{file:String(x)}):Object.entries(hashes??{}).map(([file,v])=>({file,sha256:typeof v==='object'?JSON.stringify(v):v}));
if(hashRows.length) tr=makeTable(trace,'__hashes',{title:'源文件校验',columns:Object.keys(hashRows[0]),rows:hashRows,notes:['原始实验文件与汇总表共同保留；历史359片段结果不外推为当前342片段版本成绩。','本工作簿全部质量评分属于AI辅助评价，不表示真人独立评分。'],source_paths:[]},tr);
trace.getRange(`A1:A${tr}`).format.columnWidth=40;
trace.getRange(`B1:B${tr}`).format.columnWidth=100;
trace.freezePanes.freezeRows(8);
previews.push({sheetName:'来源与方法',range:'A1:D20',file:'workbook-来源与方法.png'});
audit.sheets.push({name:'来源与方法',lastRow:tr,columnCount:12});

const main=byName['总览']; base(main,44,14); title(main,'智能科研助理  实验结果','课程实践学术结果工作簿。详细数据、配置与来源保留在后续工作表。');
main.getRange('A1:A44').format.columnWidth=25;
main.getRange('B1:B44').format.columnWidth=20;
main.getRange('C1:E44').format.columnWidth=14;
main.getRange('F1:F44').format.columnWidth=3;
main.getRange('G1:N44').format.columnWidth=12;
main.getRange('A6:E6').values=[['检索模式','Hit@5','MRR','Recall@5','测试题数']];
main.getRange('A6:E6').format={fill:C.navy,font:{name:font,size:10,color:C.white,bold:true},rowHeight:30,horizontalAlignment:'center'};
const rt=tableData('retrieval_test');
if(rt) {
  const guess=(keys,patterns)=>keys.find(k=>patterns.some(p=>p.test(k)));
  const modeKey=guess(rt.columns,[/^mode$/,/retrieval_mode/]);
  const hKey=guess(rt.columns,[/^hit_at_5$/,/^hit_at_k$/,/^hit_5$/]);
  const mKey=guess(rt.columns,[/^mrr$/]);
  const rKey=guess(rt.columns,[/^recall_at_5$/,/^recall_at_k$/,/^recall_5$/]);
  const nKey=guess(rt.columns,[/^case_count$/,/^query_count$/,/^count$/,/^n$/]);
  const keys=[modeKey,hKey,mKey,rKey,nKey];
  for(let i=0;i<rt.rows.length;i++) main.getRange(`A${7+i}:E${7+i}`).formulas=[keys.map(k=>k?formulaRef('retrieval_test',i,k):'="n.a."')];
  main.getRange(`B7:D${6+rt.rows.length}`).setNumberFormat('0.0%');
  main.getRange(`E7:E${6+rt.rows.length}`).setNumberFormat('#,##0');
  chart(main,`A6:D${6+rt.rows.length}`,'三档检索效果  历史359片段','G6','N21');
}
main.getRange('A12:E13').merge();
main.getRange('A12').values=[['正式检索测试使用历史359片段索引与60题测试集；当前生产索引为342片段，分开报告。']];
main.getRange('A12:E13').format={wrapText:true,font:{name:font,size:10,color:C.muted},rowHeight:24};
main.getRange('A16:C16').values=[['基线答案质量','AI均分 (1–5)','评分题数']];
main.getRange('A16:C16').format={fill:C.navy,font:{name:font,size:10,color:C.white,bold:true},rowHeight:30,horizontalAlignment:'center'};
const q=tableData('baseline_ai_quality');
if(q) {
  for(let i=0;i<q.rows.length;i++) {
    const row=q.rows[i]; const labelKey=q.columns.find(k=>/metric|dimension/.test(k));
    const meanKey=q.columns.find(k=>/mean|score/.test(k)&&typeof row[k]==='number');
    const countKey=q.columns.find(k=>/count|^n$/.test(k));
    main.getRange(`A${17+i}`).values=[[label(row[labelKey]??q.columns[0])]];
    main.getRange(`B${17+i}`).formulas=[[meanKey?formulaRef('baseline_ai_quality',i,meanKey):'="n.a."']];
    main.getRange(`C${17+i}`).formulas=[[countKey?formulaRef('baseline_ai_quality',i,countKey):'=60']];
  }
  main.getRange(`B17:B${16+q.rows.length}`).setNumberFormat('0.00');
  chart(main,`A16:B${16+q.rows.length}`,'基线答案质量  AI辅助评分','G24','N39','0.0');
}
main.getRange('A24:E26').merge();
main.getRange('A24').values=[['质量比较依据实际生成回答与参考依据逐条评价。基线60题采用256输出token上限；生成参数实验每个条件4题，Prompt实验每组4题。']];
main.getRange('A24:E26').format={wrapText:true,font:{name:font,size:10,color:C.muted},rowHeight:24};
main.getRange('A29:E32').merge();
main.getRange('A29').values=[['使用方法：在结果页编辑原生数值；总览及原生图表通过单元格引用更新。AI评分明细保留问题、模型回答、参考答案与评分依据。统计比较仅适用于记录的样本、条件和版本。']];
main.getRange('A29:E32').format={wrapText:true,font:{name:font,size:10,color:C.ink},rowHeight:24};
previews.unshift({sheetName:'总览',range:'A1:N40',file:'workbook-总览.png'});
audit.sheets.unshift({name:'总览',lastRow:40,columnCount:14});

await fs.mkdir(out,{recursive:true});
await fs.mkdir(qaOut,{recursive:true});
wb.recalculate();
const checks=await wb.inspect({kind:'region',sheetId:'总览',range:'A6:E21',maxChars:4000,tableMaxRows:16,tableMaxCols:5});
audit.overview_inspect=checks.ndjson;
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});
audit.error_scan=errors.ndjson;
console.log(checks.ndjson);
console.log(errors.ndjson);
for(const view of previews) {
  try {
    const png=await wb.render({sheetName:view.sheetName,range:view.range,scale:1.35,format:'png'});
    await fs.writeFile(path.join(qaOut,view.file),new Uint8Array(await png.arrayBuffer()));
    audit.previews.push(view);
    console.log(`Rendered ${view.sheetName}`);
  } catch(error) { audit.previews.push({...view,error:String(error)}); throw error; }
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(out,'学术图表数据与结果.xlsx'));
await fs.rename(path.join(out,'学术图表数据与结果.xlsx.inspect.ndjson'),path.join(qaOut,'xlsx.inspect.ndjson')).catch(error=>{if(error.code!=='ENOENT') throw error;});
await fs.writeFile(path.join(qaOut,'verification.json'),JSON.stringify(audit,null,2));
console.log(JSON.stringify({output:path.join(out,'学术图表数据与结果.xlsx'),sheets:audit.sheets.length,tables:audit.tables.length,previews:audit.previews.length}));
