"""Sentence Transformer embedding service."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_product_text(name: str, description: str) -> str:
    """Create the canonical text used for product embeddings."""
    return f"{name.strip()}\n{description.strip()}"


def get_embedding(text: str) -> list[float]:
    """Generate a dense vector embedding for the provided text."""
    vector = get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()
