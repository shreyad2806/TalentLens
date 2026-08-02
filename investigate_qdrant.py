"""Direct Qdrant/BM25 forensic investigation."""
from __future__ import annotations

import os
import sys
import json
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.composition_root import create_retrieval_bundle


def qdrant_client_from_bundle(bundle):
    """Return the underlying Qdrant client from the bundle."""
    return bundle.vector_store_service.vector_store._adapter.client


def collect_all_points(client, collection, batch=1000):
    """Scroll all points; return list of (id, payload) dicts."""
    all_points = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload if hasattr(p, "payload") else {}
            all_points.append({
                "id": str(p.id),
                "payload": payload,
                "resume_id": str(payload.get("resume_id") or ""),
                "chunk_id": str(payload.get("chunk_id") or ""),
            })
        if next_offset is None or (isinstance(next_offset, int) and next_offset == 0):
            break
    return all_points


def hash_payload(payload):
    """Stable hash for payload comparison."""
    import hashlib
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()


def step_duplicates(points):
    by_chunk = defaultdict(list)
    by_resume = defaultdict(list)
    for p in points:
        by_chunk[p["chunk_id"]].append(p)
        by_resume[p["resume_id"]].append(p)

    dup_chunk_ids = {cid: len(items) for cid, items in by_chunk.items() if len(items) > 1}
    dup_resume_ids = {rid: len(items) for rid, items in by_resume.items() if len(items) > 1}

    print("\n--- Qdrant duplicate analysis ---")
    print(f"Total points: {len(points)}")
    print(f"Unique chunk_ids: {len(by_chunk)}")
    print(f"Unique resume_ids: {len(by_resume)}")
    extra_chunks = sum(c - 1 for c in dup_chunk_ids.values())
    extra_resumes = sum(c - 1 for c in dup_resume_ids.values())
    print(f"Duplicate chunk_ids: {len(dup_chunk_ids)} (extra copies: {extra_chunks})")
    print(f"Duplicate resume_ids: {len(dup_resume_ids)} (extra copies: {extra_resumes})")

    print("\n--- Duplicate chunk_id examples (first 20) ---")
    for i, (cid, count) in enumerate(sorted(dup_chunk_ids.items(), key=lambda x: -x[1])[:20]):
        for p in by_chunk[cid][:3]:
            phash = hash_payload(p["payload"])
            print(f"  {i+1}. chunk_id={cid} qdrant_id={p['id']} resume_id={p['resume_id']} payload_hash={phash}")

    print("\n--- Resume IDs with most point copies (first 10) ---")
    for i, (rid, count) in enumerate(sorted(dup_resume_ids.items(), key=lambda x: -x[1])[:10]):
        print(f"  {i+1}. resume_id={rid} point_count={count}")


def step_raw_search(client, collection, bundle, query="banking", top_k=20):
    print(f"\n--- Raw Qdrant search for '{query}' (top {top_k}) ---")
    qvec = bundle.dense_service.query_embedder.embed_query(query)
    response = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=top_k,
        with_payload=True,
    )
    results = response.points if hasattr(response, "points") else []
    for i, r in enumerate(results, 1):
        payload = r.payload if hasattr(r, "payload") else {}
        print(f"  {i}. qdrant_id={r.id} resume_id={payload.get('resume_id')} chunk_id={payload.get('chunk_id')} score={r.score:.4f} has_text={bool(payload.get('text'))}")


def step_overlap(bundle):
    print("\n--- Dense vs Sparse overlap for 'banking' (top 20) ---")
    dense = bundle.dense_service.search(query="banking", top_k=20)
    sparse = bundle.sparse_service.search(query="banking", top_k=20)

    print("\nDense top 20:")
    for i, r in enumerate(dense, 1):
        print(f"  {i}. resume_id={r.resume_id} chunk_id={r.chunk_id} score={r.score:.4f}")

    print("\nSparse top 20:")
    for i, r in enumerate(sparse, 1):
        print(f"  {i}. resume_id={r.resume_id} chunk_id={r.chunk_id} score={r.bm25_score:.4f}")

    d_res = {r.resume_id for r in dense}
    s_res = {r.resume_id for r in sparse}
    d_chunk = {r.chunk_id for r in dense}
    s_chunk = {r.chunk_id for r in sparse}
    print(f"\nResume overlap: {len(d_res & s_res)} / dense={len(d_res)} sparse={len(s_res)}")
    print(f"Chunk overlap: {len(d_chunk & s_chunk)} / dense={len(d_chunk)} sparse={len(s_chunk)}")


def step_bm25_qdrant_alignment(bundle):
    print("\n--- BM25 vs Qdrant chunk/resume ID alignment (first 20) ---")
    bm25_docs = list(bundle.bm25_index.document_store.values())[:20]
    found = 0
    missing = 0
    for i, doc in enumerate(bm25_docs, 1):
        v = bundle.vector_store_service.fetch(str(doc.chunk_id))
        if v:
            found += 1
            print(f"  {i}. OK chunk_id={doc.chunk_id} resume_id={doc.resume_id} candidate_name={doc.candidate_name!r}")
        else:
            missing += 1
            print(f"  {i}. MISSING chunk_id={doc.chunk_id} resume_id={doc.resume_id}")
    print(f"\nFound: {found} Missing: {missing}")


def main():
    bundle = create_retrieval_bundle()
    client = qdrant_client_from_bundle(bundle)
    collection = bundle.vector_store_service.vector_store._adapter.collection_name
    actual = bundle.vector_store_service.count()
    print(f"Qdrant collection '{collection}': actual={actual}")
    print(f"BM25 documents: {bundle.bm25_index.total_documents}")

    points = collect_all_points(client, collection)
    step_duplicates(points)
    step_raw_search(client, collection, bundle, query="banking", top_k=20)
    step_overlap(bundle)
    step_bm25_qdrant_alignment(bundle)


if __name__ == "__main__":
    main()
