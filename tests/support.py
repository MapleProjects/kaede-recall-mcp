"""Offline test helpers.

The integration tests used to load all-MiniLM, which downloads weights the
first time it runs. A test suite must not depend on the network, so these
tests use a deterministic bag-of-words embedding: same text, same vector,
and cosine similarity that tracks word overlap.
"""

from __future__ import annotations

import hashlib
import re

from kaede_recall_mcp.embeddings.base import EmbeddingProvider


class DeterministicEmbedding(EmbeddingProvider):
    """Feature-hashed bag of words, 384 dims, no model and no network."""

    def __init__(self, size: int = 384) -> None:
        self.size = size

    def _vec(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return [1.0] + [0.0] * (self.size - 1)
        return [value / norm for value in vector]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._vec(document) for document in documents]

    async def embed_query(self, query: str) -> list[float]:
        return self._vec(query)

    def get_vector_name(self) -> str:
        return "dense"

    def get_vector_size(self) -> int:
        return self.size