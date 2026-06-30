from sentence_transformers import SentenceTransformer
from app.api.core.config import settings
import numpy as np

# Load model once at startup — cached in memory
_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.
    Returns numpy array of shape (len(texts), embedding_dim)
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # normalise for cosine similarity
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns numpy array of shape (embedding_dim,)
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding