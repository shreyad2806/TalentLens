from typing import Dict, List, Tuple
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct",
    max_new_tokens=300
)

def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    # docs_with_ids: list of (id, text)
    docs_formatted = "\n\n".join([f"ID: {doc_id}\n{doc_text}" for doc_id, doc_text in docs_with_ids])
    prompt = (
        f"User Query:\n{user_query}\n\nContext Documents (top 5):\n{docs_formatted}\n\n"
        "Answer the user's question using the context above. Be concise, accurate, and reference document IDs inline when citing (e.g., [ID: 123])."
    )
    result = generator(prompt)
    return result[0]["generated_text"]

def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    answer_text = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "LLM answer generation",
        "tool": "HuggingFace Transformers",
        "model": "mistralai/Mistral-7B-Instruct",
        "input_docs_count": len(docs_with_ids),
    }
    return answer_text, trace


