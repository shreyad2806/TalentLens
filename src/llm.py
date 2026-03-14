from typing import Dict, List, Tuple
import streamlit as st
import warnings

# Suppress warnings to avoid threading issues
warnings.filterwarnings("ignore")


@st.cache_resource
def load_llm():
    from transformers import pipeline
    # Use a smaller chat-capable model; load once per Streamlit server
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1,
    )


def get_generator():
    return load_llm()


def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    """
    Generate an answer using the cached HF pipeline.
    """
    generator = get_generator()
    docs_formatted = "\n\n".join([f"ID: {doc_id}\n{doc_text}" for doc_id, doc_text in docs_with_ids])
    prompt = (
        f"User Query:\n{user_query}\n\nContext Documents (top 5):\n{docs_formatted}\n\n"
        "Answer the user's question using the context above. Be concise and reference document IDs like [ID: 123]."
    )
    try:
        result = generator(prompt, max_new_tokens=150, do_sample=False, temperature=0.0)
        return result[0].get("generated_text", "")
    except Exception as e:
        return f"Error generating answer: {str(e)}. Please try again."


def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    answer_text = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "LLM answer generation",
        "tool": "HuggingFace Transformers",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "input_docs_count": len(docs_with_ids),
    }
    return answer_text, trace


