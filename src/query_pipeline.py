import time
from typing import Dict, List, Tuple

from .embed import embed_text
from .llm import generate_answer_with_trace
from .pinecone_client import get_index
from .tools import classify_category
from .config import EMBEDDING_MODEL, EMBEDDING_DIM, PINECONE_INDEX
from .llm import get_generator


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
    index = get_index()
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
    # Helper: generate explanation for a match
    def explain_match(query: str, resume_text: str, score: float) -> Dict:
        """
        Generate explanation for why this resume matches the query.
        """
        prompt = f"""
You are a recruiter assistant.

Explain why the following candidate resume matches the query.

Query:
{query}

Resume:
{resume_text}

Match score: {score}%

Return:
- Match Score
- Key reasons (skills, experience, location if present)
- Short bullet explanation
"""

        try:
            generator = get_generator()
            # Use the generator directly to produce a short explanation
            resp = generator(prompt, max_new_tokens=120, do_sample=False, temperature=0.0)
            explanation = resp[0].get("generated_text", "")
        except Exception as e:
            explanation = f"Error generating explanation: {str(e)}"

        return {"score": score, "explanation": explanation}

    for m in matches:
        doc_id = m.get("id")
        meta = m.get("metadata", {}) or {}
        text = meta.get("text", "")
        raw_score = m.get("score") or 0.0
        try:
            score = round(float(raw_score) * 100, 2)
        except Exception:
            score = 0.0

        explain = explain_match(user_query, text, score)

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


