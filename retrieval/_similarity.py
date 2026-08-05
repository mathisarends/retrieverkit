import math

from retrieval.types import Embedding


def normalize(embedding: Embedding) -> Embedding:
    magnitude = math.sqrt(sum(value * value for value in embedding))
    if magnitude == 0:
        raise ValueError("Embedding vectors must not be empty or zero")
    return [value / magnitude for value in embedding]


def cosine(left: Embedding, right: Embedding) -> float:
    """Cosine similarity of two vectors that are already normalized."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
