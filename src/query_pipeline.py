import time
from typing import Dict, List, Tuple

from .embed import embed_text
from .llm import generate_answer_with_trace
from .pinecone_client import get_cached_index
from .tools import classify_category
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
        top_k=top_k,
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
            "top_k": top_k,
            "filter": pinecone_filter,
            "duration_ms": int((t5 - t4) * 1000),
        }
    )

    matches = result.get("matches", [])
    docs = []

    # Fast heuristic explanation generator to avoid expensive LLM calls per match.
    def explain_match_fast(query: str, resume_text: str, score: float) -> Dict:
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

        # Location - look for capitalized tokens from query
        locs = []
        for w in query.split():
            if w and w[0].isupper() and w.lower() in r:
                locs.append(w)
        if locs:
            reasons.append(f"Location: {', '.join(locs)}")

        # Assemble short explanation
        if reasons:
            bullets = [f"✔ {r_item}" for r_item in reasons]
            explanation = "\n".join(bullets)
        else:
            snippet = (resume_text[:200] + "...") if len(resume_text) > 200 else resume_text
            explanation = f"No direct keyword matches found. Snippet: {snippet}"

        return {"score": score, "explanation": explanation, "reasons": reasons}

    for m in matches:
        doc_id = m.get("id")
        meta = m.get("metadata", {}) or {}
        text = meta.get("text", "")
        raw_score = m.get("score") or 0.0
        try:
            score = round(float(raw_score) * 100, 2)
        except Exception:
            score = 0.0

        explain = explain_match_fast(user_query, text, score)

        docs.append({
            "id": doc_id,
            "text": text,
            "resume": text,
            "score": score,
            "explain": explain,
            "category": meta.get("category"),
        })

    trace.append(
        {
            "step": "Fetch top 5",
            "tool": "Pinecone",
            "result_count": len(docs),
            "ids": [d["id"] for d in docs[:5]],
        }
    )

    return {"category": category, "docs": docs, "trace": trace}


def answer(user_query: str, retrieved: Dict) -> Dict:
    docs = retrieved.get("docs", [])[:5]
    docs_with_ids: List[Tuple[str, str]] = [(d["id"], d["text"]) for d in docs]
    ans_text, ans_trace = generate_answer_with_trace(user_query, docs_with_ids)
    return {"answer": ans_text, "trace": ans_trace}


