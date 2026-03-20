import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List


@st.cache_resource
def load_embedding_model():
    """Load and cache the SentenceTransformer model once per Streamlit session."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def _clean_text(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    return str(text)[:max_len]


@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts and cache the result. Texts should be pre-cleaned.

    Returns list of embedding vectors as native Python lists to be JSON-serializable.
    """
    model = load_embedding_model()
    cleaned = [_clean_text(t) for t in texts]
    embs = model.encode(cleaned)
    # Convert numpy arrays to lists for easier downstream use/storage
    return [e.tolist() for e in embs]


@st.cache_data(show_spinner=False)
def embed_text(text: str) -> List[float]:
    """Embed single text and cache the embedding for repeated queries."""
    model = load_embedding_model()
    cleaned = _clean_text(text)
    emb = model.encode(cleaned)
    return emb.tolist()


@st.cache_data(show_spinner=False)
def load_and_embed_resumes(resume_texts: List[str]) -> List[List[float]]:
    """Convenience to embed many resumes once and cache the embeddings."""
    return embed_texts(resume_texts)


