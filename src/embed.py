import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]):
    model = load_embedding_model()
    return [model.encode(t).tolist() for t in texts]


def embed_text(text: str):
    model = load_embedding_model()
    return model.encode(text).tolist()


