import time
from typing import Dict, List, Tuple

from .embed import embed_text
from .llm import generate_answer_with_trace
from .pinecone_client import get_cached_index
from .tools import classify_category, extract_location, compute_candidate_score
from .config import EMBEDDING_MODEL, EMBEDDING_DIM, PINECONE_INDEX


def retrieve(user_query: str, top_k: int = 5) -> Dict:
    trace: List[Dict] = []

    t0 = time.time()
    category = classify_category(user_query)
    t1 = time.time()
    trace.append(
        {
            "step": "Category classification",
            "tool": "Keyword-based Classifier",
            "model": "Rule-based",
            "chosen_category": category,
            "duration_ms": int((t1 - t0) * 1000),
        }
    )

    t2 = time.time()
    # embed the query (cached)
    query_vec = embed_text(user_query)
    t3 = time.time()
    trace.append(
        {
            "step": "Query embedding",
            "tool": "SentenceTransformers",
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIM,
            "duration_ms": int((t3 - t2) * 1000),
        }
    )

    # Use cached Pinecone index connection
    index = get_cached_index()
    pinecone_filter = {"category": category}
    t4 = time.time()
    result = index.query(
        vector=query_vec,
        top_k=top_k * 2,  # Get more candidates to re-rank with location logic
        include_metadata=True,
        filter=pinecone_filter,
    )
    t5 = time.time()
    trace.append(
        {
            "step": "Vector search",
            "tool": "Pinecone",
            "index": PINECONE_INDEX,
            "metric": "cosine",
            "top_k": top_k * 2,
            "filter": pinecone_filter,
            "duration_ms": int((t5 - t4) * 1000),
        }
    )

    matches = result.get("matches", [])
    docs = []

    # Fast heuristic explanation generator to avoid expensive LLM calls per match.
    def explain_match_fast(query: str, resume_text: str, score: float, candidate_location: str) -> Dict:
        q = query.lower()
        r = resume_text.lower()

        reasons = []
        # Skills / keywords from query
        for token in q.split():
            token = token.strip(".,()[]")
            if len(token) < 3:
                continue
            if token in r and token not in reasons:
                reasons.append(token)

        # Years of experience (simple patterns)
        import re

        yrs = []
        for mm in re.finditer(r"(\d+)\+?\s*(?:years|yrs)\b", resume_text, flags=re.I):
            yrs.append(mm.group(0))
        if yrs:
            reasons.append(f"Experience: {yrs[0]}")

        # ✅ FIXED LOCATION LOGIC
        query_location = extract_location(query)
        if query_location != "unknown" and candidate_location != "unknown":
            if query_location in candidate_location or candidate_location in query_location:
                reasons.append(f"Location match: {candidate_location}")
            else:
                reasons.append(f"Location mismatch: {candidate_location} (looking for {query_location})")

        # Assemble short explanation
        if reasons:
            bullets = [f"✔ {r_item}" for r_item in reasons]
            explanation = "\n".join(bullets)
        else:
            snippet = (resume_text[:200] + "...") if len(resume_text) > 200 else resume_text
            explanation = f"No direct keyword matches found. Snippet: {snippet}"

        return {"score": score, "explanation": explanation, "reasons": reasons}

    # Process candidates with location-aware scoring
    scored_candidates = []
    for m in matches:
        doc_id = m.get("id")
        meta = m.get("metadata", {}) or {}
        text = meta.get("text", "")
        raw_score = m.get("score") or 0.0
        try:
            score = round(float(raw_score) * 100, 2)
        except Exception:
            score = 0.0

        # Extract location and compute enhanced score
        candidate_location = extract_location(text)
        enhanced_score = compute_candidate_score({"text": text, "location": candidate_location}, user_query)
        
        explain = explain_match_fast(user_query, text, score, candidate_location)

        scored_candidates.append({
            "id": doc_id,
            "text": text,
            "resume": text,
            "score": enhanced_score,  # Use enhanced score with location logic
            "original_score": score,   # Keep original semantic score for reference
            "explain": explain,
            "category": meta.get("category"),
            "location": candidate_location,
        })

    # Sort by enhanced score (location-aware) and take top_k
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    docs = scored_candidates[:top_k]

    trace.append(
        {
            "step": "Location-aware re-ranking",
            "tool": "Custom Scoring",
            "candidates_processed": len(scored_candidates),
            "final_count": len(docs),
            "location_query": extract_location(user_query),
        }
    )

    return {"category": category, "docs": docs, "trace": trace}


def answer(user_query: str, retrieved: Dict) -> Dict:
    docs = retrieved.get("docs", [])[:5]
    docs_with_ids: List[Tuple[str, str]] = [(d["id"], d["text"]) for d in docs]
    ans_text, ans_trace = generate_answer_with_trace(user_query, docs_with_ids)
    return {"answer": ans_text, "trace": ans_trace}


