"""Diagnostic script for indexing / retrieval verification."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.composition_root import create_retrieval_bundle


def step1(bundle):
    """Print first 20 chunks from BM25 and Qdrant."""
    print("\n" + "=" * 70)
    print("STEP 1: First 20 indexed chunks")
    print("=" * 70)

    bm25_docs = list(bundle.bm25_index.document_store.values())[:20]
    print("\nBM25 (first 20):")
    for i, doc in enumerate(bm25_docs, 1):
        print(f"  {i:<3} resume_id={doc.resume_id:<12} chunk_id={doc.chunk_id:<40} section={doc.section:<20} candidate_name={doc.candidate_name!r}")

    print("\nQdrant (first 20 by chunk_id from BM25):")
    missing = 0
    for i, doc in enumerate(bm25_docs, 1):
        v = bundle.vector_store_service.fetch(str(doc.chunk_id))
        if v is None:
            missing += 1
            print(f"  {i:<3} MISSING chunk_id={doc.chunk_id}")
        else:
            print(f"  {i:<3} resume_id={v.resume_id:<12} chunk_id={v.chunk_id:<40} section={v.section:<20} candidate_name={v.resume_metadata.candidate_name!r}")
    print(f"\nQdrant missing for first 20 BM25 chunk_ids: {missing}")


def step2(bundle):
    """Compare embedded text for first 5 chunks."""
    print("\n" + "=" * 70)
    print("STEP 2: Embedded text comparison (first 5 chunks)")
    print("=" * 70)

    for i, doc in enumerate(list(bundle.bm25_index.document_store.values())[:5], 1):
        v = bundle.vector_store_service.fetch(str(doc.chunk_id))
        print(f"\n  {i}. chunk_id={doc.chunk_id}")
        print(f"     BM25 text   : {doc.text[:200]!r}")
        if v and hasattr(v, "text"):
            print(f"     Qdrant text : {v.text[:200]!r}")
        else:
            print(f"     Qdrant text : <NOT STORED>")


def step3(bundle):
    """Compare dense vs sparse retrieval for 'Banking'."""
    print("\n" + "=" * 70)
    print("STEP 3: Dense vs Sparse retrieval for query='Banking'")
    print("=" * 70)

    dense = bundle.dense_service.search(query="Banking", top_k=20)
    sparse = bundle.sparse_service.search(query="Banking", top_k=20)

    print("\nDense top 20:")
    for r in dense:
        print(f"  resume_id={r.resume_id:<12} chunk_id={r.chunk_id:<40} score={r.score:.4f}")

    print("\nSparse top 20:")
    for r in sparse:
        print(f"  resume_id={r.resume_id:<12} chunk_id={r.chunk_id:<40} score={r.bm25_score:.4f}")


def step6(bundle):
    """Compute overlap for several queries."""
    queries = ["Banking", "Java", "Python", "Finance", "Software Engineer"]
    print("\n" + "=" * 70)
    print("STEP 6: Retrieval overlap")
    print("=" * 70)
    for q in queries:
        dense = bundle.dense_service.search(query=q, top_k=20)
        sparse = bundle.sparse_service.search(query=q, top_k=20)

        dense_resumes = {r.resume_id for r in dense}
        sparse_resumes = {r.resume_id for r in sparse}
        dense_chunks = {r.chunk_id for r in dense}
        sparse_chunks = {r.chunk_id for r in sparse}

        resume_overlap = len(dense_resumes & sparse_resumes)
        chunk_overlap = len(dense_chunks & sparse_chunks)

        print(f"\nQuery: {q}")
        print(f"  Dense resumes:  {len(dense_resumes):<3}  Dense chunks:  {len(dense_chunks):<3}")
        print(f"  Sparse resumes: {len(sparse_resumes):<3}  Sparse chunks: {len(sparse_chunks):<3}")
        print(f"  Resume overlap: {resume_overlap:<3}  Chunk overlap: {chunk_overlap:<3}")


if __name__ == "__main__":
    bundle = create_retrieval_bundle()
    step1(bundle)
    step2(bundle)
    step3(bundle)
    step6(bundle)
