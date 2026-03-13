from typing import Dict, List, Tuple
import os
import warnings

# Suppress warnings to avoid threading issues
warnings.filterwarnings("ignore")

# Load model lazily to avoid threading issues
_generator = None

def get_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline
        _generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            max_new_tokens=150,
            device=-1  # Force CPU to avoid GPU threading issues
        )
    return _generator

def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    """
    Generate an answer using DialoGPT-small model.
    """
    generator = get_generator()
    docs_formatted = "\n\n".join([f"ID: {doc_id}\n{doc_text}" for doc_id, doc_text in docs_with_ids])
    prompt = (
        f"User Query:\n{user_query}\n\nContext Documents (top 5):\n{docs_formatted}\n\n"
        "Answer the user's question using the context above. Be concise and reference document IDs like [ID: 123]."
    )
    try:
        result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating answer: {str(e)}. Please try again."

def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    answer_text = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "LLM answer generation",
        "tool": "HuggingFace Transformers",
        "model": "microsoft/DialoGPT-small",
        "input_docs_count": len(docs_with_ids),
    }
    return answer_text, trace


