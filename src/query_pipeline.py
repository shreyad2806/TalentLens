import time
from typing import Any

from .bootstrap.composition_root import create_retrieval_bundle
from .context_builder import ContextBuilder
from .llm import AnswerGenerator
from .retrieval.dense.query_embedder import QueryEmbedder
from .retrieval.dense.schema import DenseSearchResult
from .retrieval.hybrid.fusion_service import FusionService, FusionStrategy
from .retrieval.sparse.schema import SparseSearchResult


def _dense_result_to_dict(result: DenseSearchResult) -> dict[str, Any]:
    """Convert DenseSearchResult to the dict format expected by FusionService."""
    return {
        "chunk_id": result.chunk_id,
        "candidate_name": result.candidate_name,
        "resume_id": result.resume_id,
        "section": result.section,
        "score": result.score,
        "matched_text": result.matched_text,
        "offset": result.offset,
        "metadata": result.metadata,
        "rank": result.rank,
    }


def _sparse_result_to_dict(result: SparseSearchResult) -> dict[str, Any]:
    """Convert SparseSearchResult to the dict format expected by FusionService."""
    return {
        "chunk_id": result.chunk_id,
        "candidate_name": result.candidate_name,
        "resume_id": result.resume_id,
        "section": result.section,
        "bm25_score": result.bm25_score,
        "matched_text": result.matched_text,
        "offset": result.offset,
        "metadata": result.metadata,
        "matched_terms": result.matched_terms,
        "rank": result.rank,
    }


def run(user_query: str, top_k: int = 5) -> dict[str, Any]:
    """
    End-to-end recruiter QA pipeline.

    Trace:
        Question -> Query Embedding -> Dense Retrieval -> BM25 -> Fusion
        -> Context Builder -> LLM -> Response

    Returns a dictionary with the final answer, cited context, telemetry,
    and a per-stage trace.
    """
    if not user_query or not user_query.strip():
        return {
            "answer": "Not found in indexed resumes.",
            "context": "",
            "entries": [],
            "trace": [{"step": "Input Validation", "error": "Empty query"}],
        }

    trace: list[dict[str, Any]] = []
    t0_total = time.perf_counter()

    bundle = create_retrieval_bundle()

    # Stage 1: Query Embedding
    t0 = time.perf_counter()
    query_embedder = QueryEmbedder(embedding_service=bundle.embedding_service)
    query_vector = query_embedder.embed_query(user_query)
    embed_time = time.perf_counter() - t0
    trace.append({
        "step": "Query Embedding",
        "duration_ms": embed_time * 1000,
        "vector_dimension": len(query_vector),
    })

    # Stage 2: Dense Retrieval
    t0 = time.perf_counter()
    dense_results = bundle.dense_service.search(user_query, top_k=top_k * 2)
    dense_time = time.perf_counter() - t0
    trace.append({
        "step": "Dense Retrieval",
        "duration_ms": dense_time * 1000,
        "results_count": len(dense_results),
    })

    # Stage 3: BM25 (Sparse)
    t0 = time.perf_counter()
    sparse_results = bundle.sparse_service.search(user_query, top_k=top_k * 2)
    sparse_time = time.perf_counter() - t0
    trace.append({
        "step": "BM25",
        "duration_ms": sparse_time * 1000,
        "results_count": len(sparse_results),
    })

    # Stage 4: Fusion
    t0 = time.perf_counter()
    fusion_service = FusionService(strategy_name=FusionStrategy.RRF)
    dense_dicts = [_dense_result_to_dict(r) for r in dense_results]
    sparse_dicts = [_sparse_result_to_dict(r) for r in sparse_results]
    fused_results, _ = fusion_service.fuse_results(dense_dicts, sparse_dicts, user_query)
    fusion_time = time.perf_counter() - t0
    trace.append({
        "step": "Fusion",
        "duration_ms": fusion_time * 1000,
        "results_count": len(fused_results),
    })

    # Stage 5: Context Builder
    t0 = time.perf_counter()
    context = ContextBuilder().build_context(fused_results, top_k=top_k)
    context_time = time.perf_counter() - t0
    trace.append({
        "step": "Context Builder",
        "duration_ms": context_time * 1000,
        "retrieved_chunks": context["retrieved_chunks"],
        "context_length": context["context_length"],
        "top_k": context["top_k"],
    })

    # Stage 6: LLM
    t0 = time.perf_counter()
    llm_result = AnswerGenerator().generate(user_query, context)
    llm_time = time.perf_counter() - t0
    trace.append({
        "step": "LLM",
        "duration_ms": llm_time * 1000,
        "prompt_tokens": llm_result["prompt_tokens"],
        "completion_tokens": llm_result["completion_tokens"],
    })

    # Stage 7: Response
    total_time = time.perf_counter() - t0_total
    trace.append({
        "step": "Response",
        "duration_ms": total_time * 1000,
        "response_time_ms": total_time * 1000,
    })

    return {
        "answer": llm_result["answer"],
        "context": context["context"],
        "entries": context["entries"],
        "prompt": llm_result["prompt"],
        "prompt_tokens": llm_result["prompt_tokens"],
        "completion_tokens": llm_result["completion_tokens"],
        "response_time_ms": total_time * 1000,
        "trace": trace,
    }




def answer(user_query: str, retrieved: dict, top_k: int = 5) -> dict:
    """
    Generate a cited, context-only answer from retrieved documents.

    This function is the legacy UI entry point.  It builds a context payload
    from the retrieved docs and delegates to AnswerGenerator so the final
    answer includes candidate citations and hallucination guardrails.

    Args:
        user_query: The search query
        retrieved: Dictionary with "docs" key containing retrieved documents
        top_k: Maximum number of chunks to include in the answer context

    Returns:
        Dictionary with "answer", "context", "trace" and telemetry keys
    """
    docs = retrieved.get("docs", [])[:top_k]

    entries = []
    for i, doc in enumerate(docs, start=1):
        candidate_name = doc.get("candidate_name") or doc.get("id") or "Unknown"
        resume_id = doc.get("id") or "N/A"
        chunk_id = doc.get("chunk_id") or f"chunk-{i}"
        section = doc.get("section") or "unknown"
        matched_text = doc.get("text") or ""
        score = doc.get("score") or 0.0
        offset = doc.get("offset") or 0

        entries.append({
            "candidate_name": candidate_name,
            "resume_id": resume_id,
            "chunk_id": chunk_id,
            "section": section,
            "matched_text": matched_text,
            "score": score,
            "offset": offset,
            "retrieval_source": "retrieved",
        })

    context_lines = [
        f"[{i}] Candidate: {e['candidate_name']} | Resume ID: {e['resume_id']} | "
        f"Section: {e['section']}\n{e['matched_text'][:400]}"
        for i, e in enumerate(entries, start=1)
    ]
    context_payload = {
        "context": "\n\n".join(context_lines),
        "entries": entries,
    }

    llm_result = AnswerGenerator().generate(user_query, context_payload)
    return {
        "answer": llm_result["answer"],
        "context": context_payload["context"],
        "trace": {
            "step": "Answer generation",
            "tool": "AnswerGenerator",
            "input_docs_count": len(docs),
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "response_time_ms": llm_result["response_time_ms"],
        },
    }


