from typing import Dict, List, Tuple


def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    """Fast, lightweight answer generator to avoid heavy LLM latency.

    This returns a concise placeholder summary referencing the top document IDs.
    For production, replace with a small local model or a tuned lightweight generator.
    """
    ids = [doc_id for doc_id, _ in docs_with_ids]
    if not ids:
        return "No supporting documents found for this query."
    # Simple, fast summary
    return f"Top candidates based on your query: {', '.join(ids)}. Use 'View Resume' for details."


def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    answer_text = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "Fast answer generation",
        "tool": "Rule-based fast summarizer",
        "model": "fast-summary-v1",
        "input_docs_count": len(docs_with_ids),
    }
    return answer_text, trace


