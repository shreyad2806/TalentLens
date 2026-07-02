import time
from typing import Dict, List, Tuple

from .llm import generate_answer_with_trace
from .bootstrap.composition_root import create_retrieval_bundle


def retrieve(user_query: str, top_k: int = 5) -> Dict:
    """
    Backward-compatible retrieve function using HybridRetrievalService.
    
    This function is a thin wrapper around the new HybridRetrievalService
    to maintain backward compatibility with existing code that imports
    from src.query_pipeline.
    
    Args:
        user_query: The search query
        top_k: Number of results to return (default: 5)
        
    Returns:
        Dictionary with format: {"category": str, "docs": List[Dict], "trace": List[Dict]}
    """
    print(f"------------------------------------")
    print(f"STAGE: query_pipeline.retrieve()")
    print(f"Input: query='{user_query}', top_k={top_k}")
    print(f"------------------------------------")
    
    trace: List[Dict] = []
    
    t0 = time.time()
    
    # Get retrieval bundle (creates services if needed)
    bundle = create_retrieval_bundle()
    
    # Use hybrid retrieval service
    results = bundle.hybrid_service.search(query=user_query, top_k=top_k)
    
    print(f"HybridRetrievalService returned {len(results)} results")
    if results:
        print(f"Example result: resume_id='{results[0].resume_id}', candidate_name='{results[0].candidate_name}', rrf_score={results[0].rrf_score}")
    
    t1 = time.time()
    
    # Convert HybridSearchResult to legacy format
    docs = []
    for result in results:
        doc = {
            "id": result.resume_id,
            "text": result.metadata.get("text", ""),
            "resume": result.metadata.get("text", ""),
            "score": result.rrf_score,
            "section": result.section,
            "candidate_name": result.candidate_name,
            "chunk_id": result.chunk_id,
            "metadata": result.metadata
        }
        docs.append(doc)
    
    print(f"Converted to legacy format: {len(docs)} docs")
    if docs:
        print(f"Example doc: id='{docs[0]['id']}', candidate_name='{docs[0]['candidate_name']}', score={docs[0]['score']}")
    
    # Build trace
    trace.append({
        "step": "Hybrid retrieval",
        "tool": "HybridRetrievalService",
        "model": "RRF Fusion",
        "duration_ms": int((t1 - t0) * 1000),
        "results_count": len(docs)
    })
    
    # Default category (could be extracted from metadata if needed)
    category = "GENERAL"
    
    print(f"Output: {len(docs)} docs, category='{category}'")
    print(f"------------------------------------")
    
    return {"category": category, "docs": docs, "trace": trace}


def answer(user_query: str, retrieved: Dict) -> Dict:
    """
    Generate answer from retrieved documents.
    
    Args:
        user_query: The search query
        retrieved: Dictionary with "docs" key containing retrieved documents
        
    Returns:
        Dictionary with "answer" and "trace" keys
    """
    docs = retrieved.get("docs", [])[:5]
    docs_with_ids: List[Tuple[str, str]] = [(d["id"], d["text"]) for d in docs]
    ans_text, ans_trace = generate_answer_with_trace(user_query, docs_with_ids)
    return {"answer": ans_text, "trace": ans_trace}


