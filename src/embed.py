from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]):
    return [model.encode(t).tolist() for t in texts]

def embed_text(text: str):
    return model.encode(text).tolist()


