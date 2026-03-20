import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List
import os
import json
from pathlib import Path
import numpy as np


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


def _emb_paths(base_dir: str = "Resume") -> tuple[str, str]:
    emb_path = os.path.join(base_dir, "embeddings.npy")
    ids_path = os.path.join(base_dir, "emb_ids.json")
    return emb_path, ids_path


def persist_resume_embeddings(df, base_dir: str = "Resume") -> tuple[list[list[float]], list[str]]:
    """Compute embeddings for resumes in `df` and persist to disk.

    Returns (embeddings, ids).
    """
    emb_path, ids_path = _emb_paths(base_dir)
    texts = []
    ids = []
    # Expecting a column named 'Resume' or use first text column
    text_col = "Resume" if "Resume" in df.columns else next((c for c in df.columns if df[c].dtype == object), None)
    if not text_col:
        return [], []

    for i, row in df.iterrows():
        texts.append(_clean_text(row[text_col]))
        ids.append(str(row.get("id") or row.get("ID") or f"row_{i}"))

    embs = load_and_embed_resumes(texts)

    # Ensure base dir exists
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    np.save(emb_path, np.array(embs, dtype=object))
    with open(ids_path, "w", encoding="utf8") as fh:
        json.dump(ids, fh)

    return embs, ids


def load_persisted_resume_embeddings(base_dir: str = "Resume") -> tuple[list[list[float]], list[str]]:
    emb_path, ids_path = _emb_paths(base_dir)
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        return [], []

    embs = np.load(emb_path, allow_pickle=True)
    with open(ids_path, "r", encoding="utf8") as fh:
        ids = json.load(fh)

    # Convert numpy arrays to lists
    embs_list = [e.tolist() if hasattr(e, "tolist") else e for e in embs]
    return embs_list, ids


def get_or_create_resume_embeddings(df, base_dir: str = "Resume") -> tuple[list[list[float]], list[str]]:
    """Load persisted resume embeddings if present, otherwise compute and persist them."""
    embs, ids = load_persisted_resume_embeddings(base_dir)
    if embs and ids:
        return embs, ids
    return persist_resume_embeddings(df, base_dir)


