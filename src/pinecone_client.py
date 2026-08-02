import time

from pinecone import Pinecone, ServerlessSpec

from .config import (
    EMBEDDING_DIM,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_REGION,
)

# Optional streamlit import for Streamlit-specific caching
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_pc() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(index_name: str | None = None, dimension: int = EMBEDDING_DIM) -> None:
    index_name = index_name or PINECONE_INDEX
    pc = get_pc()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    # Wait briefly for index to be ready
    for _ in range(30):
        desc = pc.describe_index(index_name)
        if desc.get("status", {}).get("ready"):
            break
        time.sleep(2)


def get_index(index_name: str | None = None):
    index_name = index_name or PINECONE_INDEX
    pc = get_pc()
    return pc.Index(index_name)


def get_cached_index(index_name: str | None = None):
    if STREAMLIT_AVAILABLE:
        @st.cache_resource
        def _get_cached():
            return get_index(index_name)
        return _get_cached()
    else:
        return get_index(index_name)


