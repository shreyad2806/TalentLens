from sentence_transformers import SentenceTransformer
import numpy as np

# Load model lazily to avoid threading issues
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: list[str]):
    model = get_model()
    return [model.encode(t).tolist() for t in texts]

def embed_text(text: str):
    model = get_model()
    return model.encode(text).tolist()


