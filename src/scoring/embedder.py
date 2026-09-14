"""
Wraps the SentenceTransformer model used for both:
  - resume <-> job description similarity scoring
  - features for the resume category classifier

Loading the model is slow (a few seconds) so we load it once per process
and reuse it, rather than reloading on every request.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    """
    Returns a cached SentenceTransformer instance, loading it on first call.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str):
    """
    Encodes a single piece of text into a 384-dim embedding vector.
    """
    model = get_model()
    return model.encode([text])[0]


def embed_texts(texts: list[str]):
    """
    Encodes a list of texts into embedding vectors in one batch call.
    More efficient than calling embed_text() in a loop.
    """
    model = get_model()
    return model.encode(texts)