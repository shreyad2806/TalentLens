# Repository Cleanup Audit Report

Generated from a static scan of 155 Python files plus the import graph.  
**No files have been modified or deleted.** This report is for approval before any cleanup.

## Executive Summary

- **One entry point:** `app.py`.
- **One legacy chunking package (`src/chunking/`) duplicates the active `src/chunks/` package.**
- **One legacy BM25 package (`src/retrieval/bm25/`) duplicates the active `src/retrieval/sparse/` package.**
- **Several root-level diagnostic / benchmark scripts should move into `scripts/` or `benchmarks/`.**
- **A number of dead utility files (`src/analytics.py`, `src/llm.py`, `src/tools.py`) and small one-off scripts are unused.**
- **Many `print()` calls and `BOOTSTRAP-TRACE` markers are scattered through production files; they should be replaced by logging.**

---

## Methodology

1. AST scan of every `.py` file (excluding `.venv`, `test_torch`, `__pycache__`, `.git`).
2. Built reverse import map (`imported by`).
3. Detected duplicate class / function names.
4. Flagged unused imports, unused top-level definitions, and debug markers (`print`, `BOOTSTRAP-TRACE`, `DEBUG TRACE`, etc.).
5. Manually reviewed key boundary files (`app.py`, `factory.py`, `vector_store/adapters/*.py`, `src/pipeline.py`, etc.).

---

========================

## KEEP

### Production Core (must keep)

These files are actively imported by `app.py`, `src/bootstrap/composition_root.py`, or other production core paths.

- `app.py`
- `requirements.txt`
- `.env.example`
- `README.md`
- `src/config.py`
- `src/debug_logger.py`
- `src/bootstrap/__init__.py`
- `src/bootstrap/bootstrap_service.py`
- `src/bootstrap/composition_root.py`
- `src/bootstrap/csv_ingestion.py`
- `src/bootstrap/resume_loader.py`
- `src/bootstrap/startup_report.py`
- `src/bootstrap/startup_validator.py`
- `src/chunks/__init__.py`
- `src/chunks/schema.py`
- `src/chunks/factory.py`
- `src/chunks/service.py`
- `src/chunks/validator.py`
- `src/embeddings/__init__.py`
- `src/embeddings/cache.py`
- `src/embeddings/embedding_service.py`
- `src/embeddings/model_loader.py`
- `src/embeddings/schema.py`
- `src/embeddings/vectorizer.py`
- `src/indexing/__init__.py`
- `src/indexing/indexing_service.py`
- `src/indexing/pipeline.py`
- `src/indexing/resume_ingestor.py`
- `src/resume_parser/__init__.py`
- `src/resume_parser/extractor.py`
- `src/resume_parser/metadata_parser.py`
- `src/resume_parser/parser_service.py`
- `src/resume_parser/schema.py`
- `src/retrieval/dense/__init__.py`
- `src/retrieval/dense/cache.py`
- `src/retrieval/dense/candidate_aggregator.py`
- `src/retrieval/dense/dense_retrieval_service.py`
- `src/retrieval/dense/query_embedder.py`
- `src/retrieval/dense/schema.py`
- `src/retrieval/dense/score_normalizer.py`
- `src/retrieval/dense/validator.py`
- `src/retrieval/hybrid/__init__.py`
- `src/retrieval/hybrid/fusion_service.py`
- `src/retrieval/hybrid/hybrid_retrieval_service.py`
- `src/retrieval/hybrid/schema.py`
- `src/retrieval/hybrid/validator.py`
- `src/retrieval/metadata/__init__.py`
- `src/retrieval/metadata/filter_engine.py`
- `src/retrieval/metadata/filter_parser.py`
- `src/retrieval/metadata/metadata_service.py`
- `src/retrieval/metadata/schema.py`
- `src/retrieval/metadata/schema_alignment.py`
- `src/retrieval/metadata/validator.py`
- `src/retrieval/reranker/__init__.py`
- `src/retrieval/reranker/batch_processor.py`
- `src/retrieval/reranker/cache.py`
- `src/retrieval/reranker/model_loader.py`
- `src/retrieval/reranker/reranker_service.py`
- `src/retrieval/reranker/schema.py`
- `src/retrieval/reranker/scorer.py`
- `src/retrieval/reranker/validator.py`
- `src/retrieval/sparse/__init__.py`
- `src/retrieval/sparse/bm25_index.py`
- `src/retrieval/sparse/cache.py`
- `src/retrieval/sparse/index_builder.py`
- `src/retrieval/sparse/schema.py`
- `src/retrieval/sparse/search_service.py`
- `src/retrieval/sparse/sparse_retrieval_service.py`
- `src/retrieval/sparse/tokenizer.py`
- `src/retrieval/sparse/validator.py`
- `src/vector_store/__init__.py`
- `src/vector_store/config.py`
- `src/vector_store/factory.py`
- `src/vector_store/interface.py`
- `src/vector_store/schema.py`
- `src/vector_store/service.py`
- `src/vector_store/validator.py`
- `src/vector_store/adapters/__init__.py`
- `src/vector_store/adapters/memory.py`
- `src/vector_store/adapters/pinecone_adapter.py`
- `src/vector_store/adapters/qdrant_adapter.py`
- `src/vector_store/qdrant/__init__.py`
- `src/vector_store/qdrant/collection_manager.py`
- `src/vector_store/qdrant/health_check.py`
- `src/vector_store/qdrant/qdrant_adapter.py`
- `src/vector_store/qdrant/schema.py`

### Production Support (can keep, but should be cleaned / consolidated)

- `src/embed.py` — `app.py` imports `load_embedding_model` from here. It overlaps with `src.embeddings` and should be consolidated, but removing it now would break `app.py`.
- `src/pipeline.py` — imported by `tests/test_pipeline_integration.py`; contains `IngestionPipeline` which processes single resumes. It is similar in purpose to `src/indexing/pipeline.py` and may be consolidated.
- `src/pinecone_client.py` — only imported by `scripts/upsert_resumes.py`.
- `scripts/build_index.py`
- `scripts/preload_models.py`
- `scripts/upsert_resumes.py`
- `tests/*.py` — keep if actively useful; several reference legacy modules and need review.
- `benchmarks/*.py` — keep as benchmarks; many duplicate `print_success` / `print_failure` helpers.

========================

## MOVE

These files are not part of the production runtime but are useful for development, diagnostics, or benchmarking. They should be relocated into canonical folders.

| File | Destination | Reason |
|------|-------------|--------|
| `analyze_chunking.py` | `scripts/` | Diagnostic / ad-hoc analysis script. |
| `benchmark_embeddings.py` | `benchmarks/` | Benchmark script, belongs with other benchmarks. |
| `diagnostics.py` | `scripts/` | One-shot diagnostic runner. |
| `preload_model.py` | `scripts/` | Overlaps with `scripts/preload_models.py`; should be merged. |
| `test_startup_repair.py` | `scripts/` or `tests/` | Repair / one-off test script. |
| `BOOTSTRAP_EXECUTION_AUDIT.md` | `docs/` | Audit artifact, not source. |
| `METADATA_AUDIT_REPORT.md` | `docs/` | Audit artifact, not source. |

========================

## DELETE

### Duplicate legacy packages

| File / Package | Reason |
|----------------|--------|
| `src/chunking/` entire package | Legacy duplicate of `src/chunks/`. Only `tests/test_chunking.py` and `tests/test_pipeline.py` still import from it. |
| `src/retrieval/bm25/` entire package | Legacy duplicate of `src/retrieval/sparse/`. Only `tests/test_bm25.py` and `scripts/build_index.py` still import from it. |
| `src/vector_store/adapters/pinecone.py` | Old `PineconeVectorStore`; not imported anywhere. `factory.py` uses `pinecone_adapter.py`. |

### Dead utility files

| File | Reason |
|------|--------|
| `src/analytics.py` | `load_data`, `extract_locations`, `category_distribution` not referenced anywhere. |
| `src/llm.py` | No importers, no `__main__`; not part of current architecture. |
| `src/tools.py` | `extract_experience`, `extract_location`, `extract_skills`, `CategoryEnum`, `classify_category`, `compute_candidate_score` not referenced anywhere. Duplicates `src/analytics.py` functionality. |
| `src/query_pipeline.py` | `retrieve` / `answer` not imported anywhere; `src.retrieval.hybrid` already provides the search API. |

### One-off / temporary scripts

| File | Reason |
|------|--------|
| `scripts/ast_check.py` | One-off AST utility. |
| `scripts/indent_inspect.py` | One-off formatting utility. |
| `scripts/print_lines.py` | 128-byte helper, no production use. |
| `test_startup_repair.py` | One-off repair script. |
| `repo_audit.py` | Temporary audit tool used to produce this report. |
| `audit_output.json` | Temporary audit artifact. |
| `repo_files.csv` | Temporary audit artifact. |
| `repo_file_summary.txt` | Temporary audit artifact. |

### Tests to review

| File | Reason |
|------|--------|
| `tests/test_parser.py` | 486-byte stub; not actively useful. |

========================

## DUPLICATES

### Whole-package duplicates

| Old | New / Active | Resolution |
|-----|--------------|------------|
| `src/chunking/` | `src/chunks/` | Delete `src/chunking/`; migrate `tests/test_chunking.py` and `tests/test_pipeline.py` to `src.chunks`. |
| `src/retrieval/bm25/` | `src/retrieval/sparse/` | Delete `src/retrieval/bm25/`; migrate `tests/test_bm25.py` and `scripts/build_index.py` to `src.retrieval.sparse`. |

### Adapter duplicates

| Old | Active | Resolution |
|-----|--------|------------|
| `src/vector_store/adapters/pinecone.py` (`PineconeVectorStore`) | `src/vector_store/adapters/pinecone_adapter.py` (`PineconeAdapter`) | Delete old `pinecone.py`. |
| `src/vector_store/adapters/qdrant_adapter.py` | `src/vector_store/qdrant/qdrant_adapter.py` | The `adapters` wrapper delegates to `qdrant/` implementation. Keep both for now or merge; `adapters/qdrant_adapter.py` is the one `factory.py` imports. |

### Functional duplicates

| Functionality | Locations | Resolution |
|---------------|-----------|------------|
| Embedding model loading | `src/embed.load_embedding_model` and `src.embeddings.model_loader` | `app.py` uses `src.embed.load_embedding_model`. Refactor `app.py` to use `src.embeddings.model_loader.get_model_loader()` / `EmbeddingService`, then remove `src/embed.py`. |
| Resume text extraction utilities | `src/tools.py` and `src/analytics.py` | Both define `extract_experience`, `extract_location`, `extract_skills`. Keep one in `src/services/resume_analysis.py` or `src/utils/` after confirming no consumers. |
| Pipeline | `src/pipeline.py` (`IngestionPipeline`) and `src/indexing/pipeline.py` (`IndexingPipeline`) | Single-resume vs batch indexing. Consider merging into `src/indexing/pipeline.py` or keep both with clearer names. |
| `print_success`, `print_failure`, `print_info`, `print_header`, `print_warning` | Repeated in every `tests/*.py` and `benchmarks/*.py` | Extract into `tests/utils.py` and `benchmarks/utils.py`. |

### Class-name collisions

These are not necessarily wrong, but indicate overlapping domains and should be reviewed.

| Name | Files |
|------|-------|
| `ValidationError` | `src/vector_store/validator.py`, `src/retrieval/dense/validator.py`, `src/retrieval/hybrid/validator.py`, `src/retrieval/metadata/validator.py`, `src/retrieval/reranker/validator.py`, `src/retrieval/sparse/validator.py` |
| `ChunkService` | `src/chunking/chunk_service.py`, `src/chunks/service.py` |
| `ChunkValidator` | `src/chunking/chunk_validator.py`, `src/chunks/validator.py` |
| `ChunkMetadata` | `src/chunking/schema.py`, `src/chunks/schema.py` |
| `Chunk` | `src/chunking/schema.py`, `src/chunks/schema.py` |
| `BM25Index` | `src/retrieval/bm25/bm25_index.py`, `src/retrieval/sparse/bm25_index.py` |
| `IndexBuilder` | `src/retrieval/bm25/index_builder.py`, `src/retrieval/sparse/index_builder.py` |
| `BM25Document` | `src/retrieval/bm25/schema.py`, `src/retrieval/sparse/schema.py` |
| `SearchResult` | `src/retrieval/bm25/cache.py`, `src/vector_store/qdrant/schema.py` |
| `QueryCache` | `src/retrieval/dense/cache.py`, `src/retrieval/sparse/cache.py` |
| `RetrievalMetrics` | `src/retrieval/dense/schema.py`, `src/retrieval/sparse/schema.py` |
| `QdrantAdapter` | `src/vector_store/adapters/qdrant_adapter.py`, `src/vector_store/qdrant/qdrant_adapter.py` |
| `ModelLoader` | `src/embeddings/model_loader.py`, `src/retrieval/reranker/model_loader.py` |

========================

## DEAD CODE

Functions/classes defined but never referenced outside their own module (excluding tests). These are safe deletion candidates after a quick manual sanity check.

| Function / Class | File |
|------------------|------|
| `load_data` | `src/analytics.py` |
| `extract_locations` | `src/analytics.py` |
| `category_distribution` | `src/analytics.py` |
| `embed_text` | `src/embed.py` |
| `get_or_create_resume_embeddings` | `src/embed.py` |
| `get_cached_index` | `src/pinecone_client.py` |
| `retrieve` | `src/query_pipeline.py` |
| `answer` | `src/query_pipeline.py` |
| `print_timing_summary` | `src/pipeline.py` |
| `print_stats_summary` | `src/pipeline.py` |
| `PineconeVectorStore` (class) | `src/vector_store/adapters/pinecone.py` |
| `CategoryEnum`, `classify_category`, `compute_candidate_score`, `extract_experience`, `extract_location`, `extract_skills` | `src/tools.py` |
| `cached_query` | `src/retrieval/dense/cache.py` |
| `RetrievalMetrics` (class) | `src/retrieval/dense/schema.py` |
| `RerankMetrics` (class) | `src/retrieval/reranker/schema.py` |
| `MatchOperator` (class) | `src/vector_store/qdrant/schema.py` |

**Note:** Some names are re-exported by `__init__.py` files and may be part of the public API even if not referenced internally. Treat the `__init__.py` re-exports as intentional unless you explicitly want to tighten the public API.

========================

## UNUSED IMPORTS

These are imports that the AST scan did not see being used. Many are type-hint only or only used in forward references; confirm before removing.

| File | Unused imports |
|------|----------------|
| `app.py` | `load_dotenv`, `html` |
| `src/bootstrap/bootstrap_service.py` | `LoadResult`, `log_error` |
| `src/bootstrap/composition_root.py` | `annotations` |
| `src/bootstrap/csv_ingestion.py` | `pickle`, `ResumeDocument` |
| `src/embeddings/embedding_service.py` | `EMBEDDING_MODEL` |
| `src/indexing/pipeline.py` | `IngestionResult` |
| `src/pipeline.py` | `ResumeDocument`, `Chunk`, `EmbeddingRecord` |
| `src/retrieval/bm25/bm25_index.py` | `pickle` |
| `src/retrieval/dense/dense_retrieval_service.py` | `RetrievalMetrics`, `ValidationError` |
| `src/retrieval/hybrid/fusion_service.py` | `ReciprocalRankFusion` |
| `src/retrieval/hybrid/hybrid_retrieval_service.py` | `FusionMetrics`, `log_error` |
| `src/retrieval/metadata/filter_engine.py` | `FilterLogic` |
| `src/retrieval/reranker/cache.py` | `timedelta` |
| `src/retrieval/reranker/reranker_service.py` | `RerankMetrics` |
| `src/retrieval/reranker/validator.py` | `Enum` |
| `src/retrieval/sparse/bm25_index.py` | `math` |
| `src/retrieval/sparse/cache.py` | `wraps` |
| `src/retrieval/sparse/index_builder.py` | `BM25IndexStats` |
| `src/retrieval/sparse/sparse_retrieval_service.py` | `RetrievalMetrics`, `ValidationError` |
| `src/retrieval/sparse/tokenizer.py` | `defaultdict` |
| `src/retrieval/sparse/validator.py` | `BM25IndexStats` |
| `src/vector_store/adapters/pinecone.py` | `ServerlessSpec` |
| `src/vector_store/qdrant/collection_manager.py` | `CollectionParams`, `HnswConfig` |
| `src/vector_store/qdrant/qdrant_adapter.py` | `Filter`, `SearchRequest`, `VectorParams`, `Distance`, `Batch` |
| `src/vector_store/service.py` | `VectorStoreFactory` |

**Note:** `__init__.py` files intentionally re-export symbols and are not listed here.

========================

## UNUSED FUNCTIONS

Same list as `DEAD CODE` (above), filtered to functions defined at module level and never called/used anywhere in the scanned code. The most important cleanup targets are:

- `src/analytics.py` — entire module is dead.
- `src/llm.py` — entire module is dead.
- `src/tools.py` — entire module is dead and duplicates `src/analytics.py`.
- `src/query_pipeline.py` — `retrieve` and `answer` are dead.
- `src/embed.py` — `embed_text` and `get_or_create_resume_embeddings` are dead (`load_embedding_model` is used by `app.py`).
- `src/pipeline.py` — `print_timing_summary` and `print_stats_summary` are dead.

========================

## DEBUGGING / TRACING TO REMOVE

Production and test files contain raw `print()` calls and trace markers that should be replaced with `logging` or `src.debug_logger`.

### Files with the most debug markers

| File | Marker Count |
|------|--------------|
| `src/bootstrap/bootstrap_service.py` | 147 |
| `src/indexing/indexing_service.py` | 32 |
| `src/indexing/pipeline.py` | 68 |
| `src/vector_store/qdrant/health_check.py` | 10 |
| `src/retrieval/hybrid/fusion_service.py` | 19 |
| `src/retrieval/hybrid/hybrid_retrieval_service.py` | 6 |
| `src/bootstrap/startup_report.py` | 70 |
| `src/bootstrap/startup_validator.py` | 18 |
| `src/bootstrap/csv_ingestion.py` | 55 |
| `app.py` | 38 |

### Trace markers found

- `BOOTSTRAP-TRACE` in `app.py` and several `src/bootstrap/*.py` files.
- `print_success`, `print_failure`, `print_info`, `print_header`, `print_warning` duplicated across every test and benchmark.

### Recommendation

1. Replace all `print()` in `src/` and `app.py` with `logging.getLogger(__name__)` or `src.debug_logger`.
2. Remove `BOOTSTRAP-TRACE`, `META TRACE`, `DEBUG TRACE`, `TEMP LOG`, `TODO DEBUG` comments / strings.
3. Move the test/benchmark `print_*` helpers into shared `tests/utils.py` and `benchmarks/utils.py`.

========================

## RECOMMENDED STRUCTURE

Produced without changing import behavior; this is the target layout after approved cleanup.

```
TalentLens/
├── app.py                           # Streamlit entry point (keep)
├── requirements.txt
├── .env.example
├── README.md
├── docs/                            # Audit/markdown artifacts
│   ├── BOOTSTRAP_EXECUTION_AUDIT.md
│   └── METADATA_AUDIT_REPORT.md
├── data/                            # Indexes / cache (already exists)
├── Resume/                          # Sample resumes (already exists)
├── models/                          # Cached models (already exists)
│
├── src/                             # Production code
│   ├── config.py
│   ├── debug_logger.py              # keep; convert prints to logging
│   │
│   ├── bootstrap/                   # startup / composition root
│   │   ├── bootstrap_service.py
│   │   ├── composition_root.py
│   │   ├── csv_ingestion.py
│   │   ├── resume_loader.py
│   │   ├── startup_report.py
│   │   └── startup_validator.py
│   │
│   ├── ingestion/                   # parsing + chunking + embedding
│   │   ├── resume_parser/
│   │   ├── chunks/
│   │   ├── embeddings/
│   │   └── pipeline.py              # single-resume ingestion pipeline
│   │
│   ├── indexing/                    # batch indexing
│   │   ├── indexing_service.py
│   │   ├── pipeline.py
│   │   └── resume_ingestor.py
│   │
│   ├── retrieval/                   # search layers
│   │   ├── dense/
│   │   ├── hybrid/
│   │   ├── sparse/                  # active BM25 implementation
│   │   ├── reranker/
│   │   └── metadata/
│   │
│   └── vector_store/                # vector store abstraction
│       ├── config.py
│       ├── service.py
│       ├── factory.py
│       ├── interface.py
│       ├── schema.py
│       ├── validator.py
│       ├── adapters/
│       │   ├── memory.py
│       │   ├── pinecone_adapter.py
│       │   └── qdrant_adapter.py    # thin wrapper -> qdrant/
│       └── qdrant/
│           ├── qdrant_adapter.py
│           ├── collection_manager.py
│           ├── health_check.py
│           └── schema.py
│
├── scripts/                         # one-shot / support scripts
│   ├── analyze_chunking.py
│   ├── build_index.py
│   ├── diagnostics.py
│   ├── preload_models.py
│   ├── test_startup_repair.py
│   └── upsert_resumes.py
│
├── benchmarks/                      # performance tests
│   ├── benchmark_embeddings.py
│   ├── bm25_benchmark.py
│   ├── dense_retrieval_benchmark.py
│   ├── hybrid_benchmark.py
│   ├── metadata_filter_benchmark.py
│   ├── qdrant_benchmark.py
│   ├── reranker_benchmark.py
│   ├── vector_store_benchmark.py
│   └── utils.py                     # shared benchmark print helpers
│
└── tests/                           # unit/integration tests
    ├── utils.py                     # shared test print helpers
    ├── test_bootstrap.py
    ├── test_chunk_objects.py
    ├── test_csv_ingestion.py
    ├── test_dense_retrieval.py
    ├── test_embeddings.py
    ├── test_hybrid_retrieval.py
    ├── test_indexing_pipeline.py
    ├── test_memory_vector_store.py
    ├── test_metadata_filtering.py
    ├── test_model_loading.py
    ├── test_pinecone_adapter.py
    ├── test_pinecone_vector_store.py
    ├── test_qdrant_adapter.py
    ├── test_reranker.py
    ├── test_schema_alignment.py
    ├── test_sparse_retrieval.py
    ├── test_vector_adapters.py
    ├── test_vector_store.py
    ├── test_vector_store_abstraction.py
    └── test_vector_store_factory.py
```

### What the restructure does

1. **Removes the two legacy packages**: `src/chunking/` and `src/retrieval/bm25/`.
2. **Consolidates ingestion**: parsing, chunking, and embedding under `src/ingestion/`.
3. **Keeps indexing separate**: `src/indexing/` for batch workflows.
4. **Keeps retrieval/search layers separate**: `src/retrieval/`.
5. **Keeps vector store abstraction**: `src/vector_store/`.
6. **Collects scripts/benchmarks/tests** into their own top-level folders with shared utility modules.

========================

## Next Steps (pending your approval)

1. Approve this report.
2. I will delete / move files in the order:
   1. Temporary audit artifacts (`repo_audit.py`, `audit_output.json`, etc.).
   2. Dead utility files (`src/analytics.py`, `src/llm.py`, `src/tools.py`, `src/query_pipeline.py`).
   3. Legacy duplicate packages (`src/chunking/`, `src/retrieval/bm25/`, `src/vector_store/adapters/pinecone.py`).
   4. Relocate root scripts to `scripts/` / `benchmarks/` / `docs/`.
   5. Replace `print()` / trace markers with `logging` (this is a cleanup-only change, no behavior change).
3. After the structural moves, update any stale imports (`tests/test_chunking.py`, `tests/test_pipeline.py`, `tests/test_bm25.py`, `scripts/build_index.py`) so the test suite still imports the canonical modules.

---

*Report generated by static analysis. Confirm any public-API re-exports in `__init__.py` before deleting symbols flagged as "unused".*
