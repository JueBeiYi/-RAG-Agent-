"""Rebuild the source-linked module 1-4 audit without rerunning model experiments.

This is an evidence inventory, not a grading rubric or automatic course sign-off.
It reads the exact DOCX paragraphs and existing test records; workspace credentials
and runtime requests are deliberately not copied into the audit.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "南京农业大学课程实践.docx"
OUTPUT = ROOT / "reports/requirements/implementation-audit.json"
BASELINE = "reports/testing/full-suite-20261010.xml"
TARGETED = "reports/testing/requirement-gaps-20261010.xml"
BROWSER = "reports/ui/browser-standard-20261010.json"
RETRIEVAL_SAMPLE = "reports/ui/standard-acceptance/summary.json"
IMPORT = "reports/datasets/standard/import_validation_summary.json"
WORKFLOW = "reports/ui/workflow-tests-20261010.json"


def main():
    with zipfile.ZipFile(SOURCE) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", ns))
                  for p in document.findall(".//w:p", ns)]
    tests = {}
    for relative in (BASELINE, TARGETED, "reports/testing/recursive-requirement-20261010.xml",
                     "reports/testing/per-tool-usage-20261010.xml", "reports/testing/agent-generation-final-20261010.xml"):
        path = ROOT / relative
        if not path.exists():
            continue
        for test in ET.parse(path).getroot().iter("testcase"):
            name = test.get("name", "").split("[")[0]
            if not any(child.tag in {"failure", "error", "skipped"} for child in test):
                tests.setdefault(name, set()).add(relative)

    entries = []

    def add(p, module, suffix, summary, code, test_names=(), *, note="", pending=(),
            real=(), implementation="implemented", optional=False, artifact=False):
        references = []
        for relative, symbol in code:
            path = ROOT / relative
            lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
            line = next((i for i, value in enumerate(lines, 1) if symbol in value), None)
            references.append({"path": relative, "line": line, "symbol": symbol,
                               "exists": path.exists() and line is not None,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None})
        evidence = []
        for name in test_names:
            evidence.extend({"path": path, "kind": "automated_test", "test": name,
                             "passed_in_record": True} for path in sorted(tests.get(name, ())))
        for relative in real:
            if (ROOT / relative).exists():
                evidence.append({"path": relative, "kind": "real_functional_record",
                                 "claim_scope": "functional_not_formal_quality"})
        if optional:
            executed = "not_applicable_unselected_optional"
        elif artifact:
            executed = "artifact_pending_external_delivery_task"
        elif any(e["kind"] == "real_functional_record" for e in evidence):
            executed = "real_functional_executed"
        elif any(e["path"] in {TARGETED, "reports/testing/recursive-requirement-20261010.xml", "reports/testing/per-tool-usage-20261010.xml"} for e in evidence):
            executed = "targeted_automated_executed"
        elif evidence:
            executed = "previous_automated_executed"
        else:
            executed = "code_reviewed_execution_pending"
        entries.append({
            "id": f"p{p}.{suffix}", "source_paragraph_id": p, "module": module,
            "source_text": paragraphs[p], "requirement": summary,
            "mandatory": not optional, "implementation_status": implementation,
            "execution_status": executed,
            "human_review_status": "pending" if pending else "not_required_for_code_presence",
            "code": references, "evidence": evidence, "interpretation": note,
            "pending": list(pending),
        })

    loader = ("src/data_loader/loaders.py", "def load_blocks")
    chunk = ("src/chunking/strategies.py", "def split_blocks")
    index = ("src/retrieval/engine.py", "def index")
    search = ("src/retrieval/engine.py", "def search")
    rag = ("src/generation/rag_pipeline.py", "def stream")
    agent = ("src/agent/react_loop.py", "def stream")
    tools = ("src/agent/tools/local.py", "def create_registry")
    memory = ("src/agent/memory.py", "def load")
    matrix = ("src/evaluation/cli.py", "PRESETS =")
    experiment_pending = ("Complete formal experiments on fixed source-checked cases; assistant source checking is not student/teacher human review.",)
    add(124, 1, "pdf_parsers", "PDF loader: PyPDF2/pdfplumber/PyMuPDF", [loader, matrix],
        ["test_three_pdf_parsers_extract_real_page_numbers"], pending=experiment_pending,
        note="All three parsers exist; selecting the best parser requires results, not character counts alone.")
    add(124, 1, "other_formats", "Word/TXT/Markdown loaders", [loader],
        ["test_docx_interleaved_paragraphs_tables_and_no_invented_pages", "test_markdown_sections_table_formula_and_provenance", "test_text_chinese_encoding_and_empty_file"])
    add(124, 1, "batch_import", "Batch import, progress and failed-job retry", [("src/api/main.py", "async def upload"),
        ("src/services/application.py", "def _ingest"), ("src/services/application.py", "def retry"),
        ("src/frontend/sections_library.py", "def library_page")], ["test_upload_concurrency_deduplicates_and_profile_change_requires_reindex"], real=[IMPORT, WORKFLOW])
    add(126, 1, "three_strategies", "At least three splitting strategies", [chunk], ["test_chunking_keeps_tail_and_real_multpage_provenance"])
    add(127, 1, "fixed_sizes", "Fixed sizes 256/512/1024 and comparison", [chunk, matrix],
        ["test_chunking_keeps_tail_and_real_multpage_provenance"], pending=experiment_pending,
        note="Sizes count reproducible lightweight Chinese-character/English-word/punctuation units, not provider tokens.")
    add(128, 1, "recursive", "Recursive character/separator splitting", [chunk, ("src/chunking/strategies.py", "def _recursive_boundaries")],
        ["test_chunking_keeps_tail_and_real_multpage_provenance", "test_recursive_splitter_candidates_preserve_all_source_positions_and_budget", "test_recursive_proposals_do_not_split_short_table_rows_or_formula"],
        note="Gap repaired: actual RecursiveCharacterTextSplitter proposes recursive boundaries using count_tokens/add_start_index/strip_whitespace=False. Offsets are validated against unchanged text; source/page/atomic-row protections remain.")
    add(129, 1, "semantic_boundaries", "Sentence/paragraph boundary splitting", [chunk], ["test_chunking_keeps_tail_and_real_multpage_provenance"],
        note="The course describes sentence/paragraph boundaries. This implementation is not embedding-based semantic clustering.")
    add(130, 1, "chunk_recall_study", "Same-document recall comparison", [matrix, ("src/evaluation/metrics.py", "def map_evidence_to_chunks")],
        ["test_ground_truth_survives_changed_chunk_ids_and_boundaries"], pending=experiment_pending)
    add(131, 1, "academic_layout", "Double columns, cross-page tables and formula preservation", [loader, chunk],
        ["test_pymupdf_two_columns_read_left_before_right", "test_pymupdf_ruled_tables_keep_rows_and_cross_page_provenance", "test_markdown_sections_table_formula_and_provenance"],
        pending=["Human-check complex real-paper tables and formulas against original physical pages; OCR is intentionally outside the text-layer parser."], real=[BROWSER])
    add(132, 1, "chunk_report", "Chunking strategy comparison report", [matrix], pending=experiment_pending, artifact=True,
        implementation="artifact_generation_supported")
    add(134, 1, "embedding_comparison", "BGE-large-zh and M3E-base quality/speed comparison", [("src/retrieval/embeddings.py", "class Embedder"), matrix],
        ["test_embedding_fingerprint_separates_models", "test_real_embedding_window_policy_does_not_drop_tail"], pending=experiment_pending)
    add(135, 1, "vector_stores", "Vector store selection and encapsulation", [("src/retrieval/vectorstores.py", "class ChromaVectors"), ("src/retrieval/vectorstores.py", "class FaissVectors")],
        ["test_optional_real_vector_adapter_persistence_filter_update_delete"], note="Chroma and FAISS adapters are implemented. The course does not require two simultaneous production stores.")
    add(136, 1, "incremental", "Batch embedding, index construction and incremental updates", [index],
        ["test_incremental_index_reuses_vectors_and_replacement_removes_old_rows", "test_atomic_commit_publishes_version_and_invalidates"], real=[IMPORT])
    add(137, 1, "vector_topk", "Similarity Top-K retrieval", [search], ["test_search_modes_persistence_and_scope_isolation"], real=[RETRIEVAL_SAMPLE])
    add(139, 1, "bm25", "rank_bm25 keyword search", [("src/retrieval/engine.py", "def _bm25")], ["test_search_modes_persistence_and_scope_isolation"], real=[RETRIEVAL_SAMPLE])
    add(140, 1, "rrf", "Handwritten RRF", [("src/retrieval/rrf.py", "def reciprocal_rank_fusion")], ["test_rrf_combines_independent_ranks_and_deduplicates"])
    add(141, 1, "rerank", "Real reranker adapter", [("src/retrieval/reranking.py", "class Reranker"), search],
        ["test_rerank_failure_returns_hybrid_results_and_visible_warning"], real=[RETRIEVAL_SAMPLE])
    add(142, 1, "three_tier_eval", "Vector/hybrid/hybrid-rerank Hit@5/MRR comparison", [matrix, ("src/evaluation/runner.py", "def run_retrieval")],
        ["test_runner_real_ranking_and_demo_label"], real=[RETRIEVAL_SAMPLE], pending=experiment_pending,
        note="Existing 4-case standard-mode results prove execution only; they do not satisfy a formal 50+ case quality claim.")
    add(144, 1, "retrieval_code", "Complete hybrid+rerank module code", [search], ["test_search_modes_persistence_and_scope_isolation"])
    add(145, 1, "retrieval_benchmark", "Three-tier benchmark data", [("src/evaluation/runner.py", "def export_result")], real=[RETRIEVAL_SAMPLE], pending=experiment_pending)
    add(146, 1, "chunk_report_output", "Chunking experiment report output", [matrix], implementation="artifact_generation_supported", artifact=True, pending=experiment_pending)

    add(151, 2, "rag_prompt", "Role/context/question/output RAG prompt", [("src/generation/context.py", "SYSTEM_PROMPT"), rag],
        ["test_full_upload_chat_cache_delete_and_scope"], real=[BROWSER])
    add(152, 2, "context_budget", "Ranked context assembly and dynamic truncation", [("src/generation/context.py", "def build_context"), rag],
        ["test_cross_page_context_and_budget_preserve_true_location", "test_agent_preserves_budget_cropped_cross_page_sources_in_true_rag_stream"])
    add(153, 2, "source_locations", "Document-name and physical-page citations", [("src/generation/context.py", "def normalise_page_references"),
        ("src/frontend/ui.py", "def citations")], ["test_page_claims_use_server_location_and_keep_unknown_sources_visible"], real=[BROWSER],
        note="For Word/TXT/Markdown, real paragraph locations replace unavailable physical pages; page numbers are not fabricated.")
    add(154, 2, "sampling_parameters", "Temperature/Top-p/Top-k comparisons", [("src/generation/llm_client.py", "def _body"), matrix],
        ["test_structured_completion_requests_provider_json_grammar"], pending=experiment_pending,
        note="Ollama receives all three parameters; cloud-compatible backend does not claim unsupported Top-k sampling.")
    add(156, 2, "streaming", "SSE/model streaming and Streamlit output", [("src/generation/llm_client.py", "def stream"), rag,
        ("src/agent/react_loop.py", "def _stream_knowledge"), ("src/frontend/sections_chat.py", "def question_page")],
        ["test_agent_streams_rag_tokens_and_preserves_history_without_resaving"], real=[BROWSER])
    add(157, 2, "stream_citations", "Streaming answer with inline reference positions", [rag, ("src/agent/react_loop.py", "def _stream_knowledge")],
        ["test_agent_streams_rag_tokens_and_preserves_history_without_resaving"], real=[BROWSER],
        note="Source table is sent before answer tokens; inline reference marks appear during the answer. Split transport references are held until complete.")
    add(159, 2, "semantic_cache", "Embedding-similarity answer cache", [("src/generation/cache.py", "class SemanticCache")],
        ["test_semantic_cache_blocks_negative_intent_and_revision", "test_optional_cache_failure_does_not_drop_completed_answer"],
        pending=["Measure real-model cache cold/hot hit behavior and latency if making a performance claim."])
    add(161, 2, "empty_retrieval", "Explicit ungrounded-model fallback for empty retrieval", [rag], ["test_agent_calculator_and_empty_knowledge_base"])
    add(162, 2, "low_relevance", "Low-relevance warning and inspectable sources", [rag, ("src/frontend/ui.py", "def citations")],
        ["test_search_modes_persistence_and_scope_isolation"], pending=["Calibrate minimum_relevance on dev data; backend scores are not answer-correctness probabilities."])
    add(163, 2, "model_failure", "Friendly provider failure and retry advice", [("src/generation/llm_client.py", "class ModelUnavailable"), agent],
        ["test_provider_partial_eof_and_errors_fail", "test_llm_failure_returns_tools_and_discloses_fallback"])
    add(165, 2, "request_log", "Questions, retrieved Top-K, answers, latency and Token logs", [("src/services/application.py", "def chat"),
        ("src/observability/logger.py", "class RequestLogger")], ["test_explicit_empty_selection_clears_previous_scope_and_log_failure_unlocks"],
        note="Complete events include the RAG done event and ranked hits; JSON logs are supplementary to scoped SQLite records.")
    add(166, 2, "top1_distribution", "Top-1 score records and distribution output", [rag, ("src/evaluation/runner.py", "def export_result")],
        ["test_search_timing_and_warnings_are_thread_local"], pending=experiment_pending,
        note="Histogram labels disclose backend-specific scores; they are not calibrated confidence probabilities.")
    add(168, 2, "rag_code", "RAG module including streaming and source attribution", [rag], ["test_full_upload_chat_cache_delete_and_scope"], real=[BROWSER])
    add(169, 2, "cache_degradation_output", "Implemented cache and fallback mechanisms", [rag, ("src/generation/cache.py", "class SemanticCache")],
        ["test_semantic_cache_blocks_negative_intent_and_revision", "test_provider_partial_eof_and_errors_fail"])
    add(170, 2, "prompt_experiment_record", "Prompt design documentation and parameter experiments", [("docs/技术设计文档.md", "生成与证据"), matrix],
        pending=experiment_pending, artifact=True, implementation="artifact_generation_supported")

    add(175, 3, "react", "Handwritten ReAct orchestration", [agent], ["test_native_tool_call_response_protocol_and_bounded_steps"],
        note="No high-level agent framework replaces the loop. About 200 lines in background prose is not a line-count constraint.")
    add(176, 3, "decision", "Short observable next-action decision", [agent], ["test_native_and_json_tool_call_parsing"],
        note="Public task decision summaries are recorded; private chain-of-thought is neither requested nor fabricated.")
    add(177, 3, "action", "Tool selection and execution", [agent, ("src/agent/executor.py", "class ToolExecutor")], ["test_registry_has_eight_callable_tools"])
    add(178, 3, "observation", "Tool observations determine continuation/termination", [("src/agent/react_loop.py", "def _append_observations"), agent], ["test_native_tool_call_response_protocol_and_bounded_steps"])
    add(179, 3, "termination", "Maximum iterations, completion, repeat and timeout termination", [agent],
        ["test_agent_stops_repeated_model_action", "test_native_tool_call_response_protocol_and_bounded_steps"])
    add(180, 3, "agent_prompt", "Agent role/tools/output schema prompt", [("src/agent/react_loop.py", "SYSTEM_PROMPT"), tools], ["test_native_and_json_tool_call_parsing"])
    add(182, 3, "rag_tool", "RAG engine wrapped as the core knowledge tool", [("src/agent/tools/local.py", "def knowledge_qa")],
        ["test_registry_has_eight_callable_tools", "test_fast_rag_records_actual_tokens_per_tool_call"], real=[BROWSER])
    add(183, 3, "metadata_tool", "Title/authors/year/abstract/DOI metadata tool", [("src/data_loader/loaders.py", "def extract_metadata"),
        ("src/agent/tools/local.py", "def paper_metadata"), ("src/api/main.py", "def update_metadata")],
        ["test_metadata_does_not_invent_missing_fields", "test_metadata_does_not_fill_unknown_doi"], real=[IMPORT],
        pending=["Unlabelled PDF author lines may be unrecognised; current official corpus metadata was source-checked/applied. Human spot-check automatic fields."])
    add(184, 3, "comparison_tool", "Two-paper methods/dataset/results comparison", [("src/agent/tools/local.py", "def compare_papers")],
        ["test_standard_comparison_produces_requested_chinese_rows_real_sources_and_usage"], real=[BROWSER],
        pending=["Human-check comparison semantics and experimental comparability; source/numeric checks do not prove every statement correct."])
    add(185, 3, "keywords_tool", "Keywords from either user question or document", [("src/agent/tools/local.py", "def extract_keywords"),
        ("src/agent/router.py", "def route")], ["test_keywords_accept_question_without_documents_or_fake_citations", "test_registry_has_eight_callable_tools"],
        note="Gap repaired: query input requires no uploaded document and produces no fabricated paper citation.")
    add(186, 3, "summary_tool", "Structured background/method/result/conclusion summary", [("src/agent/tools/local.py", "def summarize_paper")],
        ["test_standard_summary_uses_llm_four_required_sections_real_sources_and_usage", "test_standard_summary_can_mark_missing_conclusion_without_fabrication"],
        pending=["Human-check generated summary meaning on actual papers. Evidence excerpts are explicitly a fallback, not equivalent to successful model synthesis."])
    add(187, 3, "time_tool", "Real current time tool", [("src/agent/tools/local.py", "def current_time")], ["test_current_time_uses_real_clock_and_timezone"])
    add(188, 3, "optional_web_search", "Optional DuckDuckGo/academic search tool", [("src/core/config.py", "network_search_enabled")],
        implementation="optional_not_selected", optional=True,
        note="No web-search handler is registered in this implementation. Offline mode and the eight local tools do not require it; an unused config flag is not an implemented search feature.")
    add(190, 3, "tool_routing", "Fast question-type routing", [("src/agent/router.py", "def route"), ("src/agent/router.py", "def use_model_knowledge"), agent],
        ["test_router_resolves_paper_titles_and_independent_tasks", "test_general_definition_route_keeps_selected_papers_and_references_grounded", "test_general_definition_streams_model_without_retrieval_and_labels_origin"],
        note="Gap repaired: explicit general definitions (e.g. deep learning) can call the model directly. Selected/named papers and referential/latest questions stay grounded. Demo does not pretend to perform real general-knowledge inference.")
    add(191, 3, "parallel_tools", "Independent tool execution in parallel", [("src/agent/executor.py", "def batch"), agent],
        ["test_executor_dispatches_independent_tasks_concurrently"], pending=["Real-model parallel speedup needs independent multi-tool tasks and measured sequential/parallel runs."])
    add(193, 3, "timeout_recovery", "Timeout retry or skip with bounded workers", [("src/agent/executor.py", "def batch"), ("src/agent/react_loop.py", "def _rag_stream_with_retry")],
        ["test_timeout_preserves_worker_limit_and_error_is_visible", "test_agent_never_retries_task_still_running_after_timeout", "test_true_rag_stream_retries_completed_transient_errors_before_answer"])
    add(194, 3, "exception_alternative", "Error logs and alternative tools", [("src/agent/react_loop.py", "def _recovery_calls")],
        ["test_compare_failure_recovers_each_scoped_paper_and_merges_evidence", "test_metadata_failure_uses_catalog_without_crossing_selection"])
    add(195, 3, "loop_guard", "Forced stop and user warning for repeated actions", [agent], ["test_agent_stops_repeated_model_action"])
    add(197, 3, "session_isolation", "User/session isolation", [memory, ("src/core/storage.py", "def get_memory")],
        ["test_memory_compression_budget_and_scope_isolation", "test_session_memory_isolation"])
    add(198, 3, "memory_window", "Token-bounded recent window", [memory], ["test_memory_compression_budget_and_scope_isolation"],
        note="Conservative token estimates are labelled; provider Token usage is separately measured.")
    add(199, 3, "memory_compression", "Archived-dialogue summary compression", [memory], ["test_memory_compression_budget_and_scope_isolation"],
        note="Extractive archived excerpts compress history and preserve originals. They are background, not original-paper evidence or a claimed abstractive LLM summary.")
    add(201, 3, "react_code_output", "ReAct code output", [agent], ["test_native_tool_call_response_protocol_and_bounded_steps"])
    add(202, 3, "eight_tools_output", "Eight registered callable tools", [tools], ["test_registry_has_eight_callable_tools"],
        note="The source lists six required tools plus optional web search but requests eight outputs. Required six + calculator (background example) + local paper catalog = eight working offline tools. Catalog is an explicit compatibility choice, not a source-prescribed tool.")
    add(203, 3, "decision_optimisation_output", "Routing/parallel/recovery code output", [agent, ("src/agent/executor.py", "class ToolExecutor")],
        ["test_executor_dispatches_independent_tasks_concurrently", "test_agent_stops_repeated_model_action"])
    add(204, 3, "memory_output", "Conversation memory module output", [memory], ["test_memory_compression_budget_and_scope_isolation"])

    add(209, 4, "core_rag_tool", "RAG as core Agent tool", [("src/agent/tools/local.py", "def knowledge_qa"), agent], ["test_registry_has_eight_callable_tools"], real=[BROWSER])
    add(210, 4, "rag_web_routing", "RAG/search routing when optional networking is chosen", [("src/agent/router.py", "def route"), ("src/core/config.py", "network_search_enabled")],
        implementation="implemented_offline_optional_search_unselected", optional=True,
        note="Mandatory local routing exists (p190); RAG-versus-web tuning is conditional on choosing the optional p188 tool. No phantom web feature is counted.")
    add(211, 4, "context_integration", "Agent uses isolated conversation context", [agent, memory], ["test_agent_streams_rag_tokens_and_preserves_history_without_resaving"])
    add(212, 4, "unified_data", "Unified Message/Context/Chunk/Event structures", [("src/core/schemas.py", "class Chunk"),
        ("src/agent/registry.py", "class ToolContext"), ("src/core/schemas.py", "def event")], ["test_native_tool_call_response_protocol_and_bounded_steps"])
    add(215, 4, "tokens_per_tool", "Per-call and per-tool Token metrics shown in frontend", [("src/agent/react_loop.py", "def _record_tool_usage"),
        ("src/frontend/ui.py", "def usage"), ("src/frontend/ui.py", "def trace")], ["test_fast_rag_records_actual_tokens_per_tool_call",
        "test_true_rag_stream_retries_completed_transient_errors_before_answer", "test_agent_accounts_for_summary_llm_tokens_in_addition_to_planning"],
        note="Gap repaired: tool_end.usage and done.metrics.tool_token_usage include known amounts, estimates and unknown-call counts. Planning/synthesis usage is separate; non-LLM local tools use zero LLM Token. New frontend presentation awaits its owner's targeted UI check.")
    add(216, 4, "retrieval_metrics", "Retrieval latency and labelled Hit@5 display", [rag, ("src/frontend/ui.py", "def usage"),
        ("src/frontend/sections_experiments.py", "def experiments_page")], ["test_search_timing_and_warnings_are_thread_local"], real=[RETRIEVAL_SAMPLE, BROWSER],
        note="Latency/relevance are available per request. True hit rate requires labelled experiment cases; ordinary live requests do not invent correctness labels.")
    add(217, 4, "decision_trace", "Decision/Action/Observation trace", [agent, ("src/frontend/ui.py", "def trace")],
        ["test_native_tool_call_response_protocol_and_bounded_steps"], real=[WORKFLOW, BROWSER],
        note="Shows genuine task stages/actions/observations and parallel calls as an expandable event trajectory; private reasoning is not fabricated.")
    add(218, 4, "tool_metrics", "Tool success rate and execution time", [agent, ("src/agent/executor.py", "class ToolExecutor"), ("src/frontend/ui.py", "def trace")],
        ["test_timeout_preserves_worker_limit_and_error_is_visible"], real=[BROWSER])
    add(219, 4, "health", "LLM/vector/store health endpoint", [("src/services/application.py", "def health"), ("src/api/main.py", "def health")],
        ["test_full_upload_chat_cache_delete_and_scope"], real=[BROWSER, IMPORT])
    add(222, 4, "document_panel", "Upload/progress/list/delete document management", [("src/frontend/sections_library.py", "def library_page"),
        ("src/api/main.py", "async def upload")], real=[WORKFLOW, BROWSER],
        note="Current product uses a dedicated paper-library page plus navigation/sidebar rather than placing all document controls on the chat's left edge.")
    add(223, 4, "chat_panel", "Streaming Markdown chat and references", [("src/frontend/sections_chat.py", "def question_page"), ("src/frontend/ui.py", "def citations")], real=[WORKFLOW, BROWSER])
    add(224, 4, "process_panel", "Inspectable Agent execution-process panel", [("src/frontend/ui.py", "def trace")], real=[WORKFLOW, BROWSER],
        note="Expandable real-event trajectory replaces a literal tree diagram; no synthetic Thought trace is used.")
    add(225, 4, "history", "Create/switch/delete historical sessions", [("src/frontend/sections_chat.py", "def question_page"), ("src/api/main.py", "def delete_session")],
        ["test_new_session_explicitly_updates_browser_value_and_stays_selected"], real=[WORKFLOW, BROWSER])
    add(226, 4, "knowledge_management", "Uploaded documents and index state", [("src/frontend/sections_library.py", "def library_page")], real=[WORKFLOW, BROWSER, IMPORT])
    add(228, 4, "end_to_end", "Upload/index/chat/decision/answer integration", [("src/services/application.py", "class Application"),
        ("src/api/main.py", "def create_app")], ["test_full_upload_chat_cache_delete_and_scope"], real=[WORKFLOW, IMPORT, BROWSER])
    add(229, 4, "concurrency", "Concurrent/multi-user scenario as applicable", [("src/services/application.py", "def _session_lock"),
        ("src/core/storage.py", "class Storage")], ["test_upload_concurrency_deduplicates_and_profile_change_requires_reindex"],
        note="Scoped concurrent-import and session-lock tests exist; these are not throughput/load-test claims.")
    add(230, 4, "edge_cases", "Empty library, large documents and long dialogue", [("src/api/main.py", "async def upload"), memory,
        ("src/retrieval/embeddings.py", "def encode")], ["test_agent_calculator_and_empty_knowledge_base", "test_memory_compression_budget_and_scope_isolation", "test_real_embedding_window_policy_does_not_drop_tail"],
        pending=["Actual 50MB-scale document stress and long real-model conversation quality remain separate from limit guards/unit tests."])
    add(232, 4, "web_app_output", "Interactive application output", [("src/frontend/app.py", "st.set_page_config")], real=[WORKFLOW, BROWSER])
    add(233, 4, "integration_output", "Integrated module code output", [("src/services/application.py", "class Application")], ["test_full_upload_chat_cache_delete_and_scope"], real=[BROWSER])
    add(234, 4, "test_records_output", "End-to-end test records", [("docs/界面功能验收.md", "#")], real=[WORKFLOW, BROWSER])
    add(68, 3, "background_direct_model_example", "General-knowledge direct-model route background example", [("src/agent/router.py", "def use_model_knowledge"),
        ("src/agent/react_loop.py", "def _stream_model_knowledge")], ["test_general_definition_streams_model_without_retrieval_and_labels_origin"],
        note="Functional routing example only. Background 80% Token savings is not a measured project result or required grade threshold.")

    # Every concrete requirement paragraph in modules 1-4 is accounted for.
    concrete = [124,126,127,128,129,130,131,132,134,135,136,137,139,140,141,142,144,145,146,
                151,152,153,154,156,157,159,161,162,163,165,166,168,169,170,175,176,177,178,179,180,
                182,183,184,185,186,187,188,190,191,193,194,195,197,198,199,201,202,203,204,
                209,210,211,212,215,216,217,218,219,222,223,224,225,226,228,229,230,232,233,234]
    missing = sorted(set(concrete) - {entry["source_paragraph_id"] for entry in entries})
    unresolved_references = [f"{entry['id']}:{ref['path']}:{ref['symbol']}" for entry in entries
                             for ref in entry["code"] if not ref["exists"]]
    report = {
        "schema_version": "1.0", "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source": {"path": SOURCE.name, "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   "paragraph_id_convention": "Zero-based index of all w:p elements in word/document.xml, including blank/table paragraphs.",
                   "source_paragraph_count": len(paragraphs)},
        "scope": {"modules": [1,2,3,4], "excluded": ["module5 formal evaluation execution", "DOCX/PPT delivery", "student identity/signature", "student Git history"]},
        "policy": ["Implemented code, executed evidence and human review are independent statuses.",
                   "Historical passing tests are explicitly prior records; this audit does not rerun the full suite.",
                   "4-case real-model samples are functional evidence, not a 50+ case performance conclusion.",
                   "65%/85% and 80% savings in background prose are not project results or scoring requirements.",
                   "Optional web search is not marked as a missing mandatory feature.",
                   "No workspace token, API key or private runtime scope is included."],
        "gap_fixes": ["question-input keywords", "explicit general-concept model-only route", "per-call/per-tool Token accounting",
                      "actual RecursiveCharacterTextSplitter proposals with source/atomic protections"],
        "validation": {"concrete_paragraphs": len(concrete), "missing_source_paragraph_ids": missing,
                       "unresolved_code_references": unresolved_references,
                       "new_targeted_test_record": TARGETED},
        "summary": {"requirement_items": len(entries), "implementation_status": dict(Counter(e["implementation_status"] for e in entries)),
                    "execution_status": dict(Counter(e["execution_status"] for e in entries)),
                    "human_review_pending_items": sum(e["human_review_status"] == "pending" for e in entries)},
        "requirements": sorted(entries, key=lambda entry: (entry["source_paragraph_id"], entry["id"])),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "items": len(entries), "missing_paragraphs": missing,
                      "unresolved_code_references": unresolved_references}, ensure_ascii=False))


if __name__ == "__main__":
    main()
