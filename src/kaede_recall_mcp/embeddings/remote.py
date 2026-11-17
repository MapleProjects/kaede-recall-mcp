"""Embedding provider backed by the memory OS daemon.

A stdio MCP process used to load its own copy of the ONNX model, which costs
around 878 MB of resident memory per client. The daemon already holds the model,
so the client asks 127.0.0.1:7451 for vectors and stays near 35 MB.

If the daemon is unreachable at startup, the provider falls back to a local
model so the server keeps working; the fallback only loads on demand.
"""

from __future__ import annotations

import logging
import os

import httpx

from kaede_recall_mcp.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = os.environ.get("MEMORY_OS_EMBED_URL", "http://127.0.0.1:7451")
PROBE_TIMEOUT = float(os.environ.get("MEMORY_OS_PROBE_TIMEOUT", "1.5"))
REQUEST_TIMEOUT = float(os.environ.get("MEMORY_OS_EMBED_TIMEOUT", "60"))


class ServiceEmbeddingProvider(EmbeddingProvider):
    """Vectors from the daemon, with a lazy local fallback."""

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._local: EmbeddingProvider | None = None
        self._vector_size: int | None = None
        self._vector_name: str | None = None
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            with httpx.Client(timeout=PROBE_TIMEOUT) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                data = response.json()
            size = int(data.get("vector_size") or 0)
            if size:
                self._vector_size = size
            name = data.get("vector_name")
            if name:
                self._vector_name = str(name)
            return True
        except Exception as exc:
            logger.info("embedding service not reachable (%s)", exc)
            return False

    # -- transport -------------------------------------------------------
    async def _post(self, texts: list[str], kind: str) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/embed", json={"texts": texts, "kind": kind}
            )
            response.raise_for_status()
            vectors = response.json().get("vectors") or []
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding service returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    def _fallback(self) -> EmbeddingProvider:
        if self._local is None:
            from kaede_recall_mcp.embeddings.fastembed import FastEmbedProvider

            logger.warning("loading the embedding model locally as a fallback")
            self._local = FastEmbedProvider(self.model_name)
        return self._local

    # -- EmbeddingProvider ----------------------------------------------
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []
        try:
            return await self._post(documents, "document")
        except Exception as exc:
            logger.warning("embed service failed (%s), using the local model", exc)
            return await self._fallback().embed_documents(documents)

    async def embed_query(self, query: str) -> list[float]:
        try:
            vectors = await self._post([query], "query")
            return vectors[0]
        except Exception as exc:
            logger.warning("embed service failed (%s), using the local model", exc)
            return await self._fallback().embed_query(query)

    def get_vector_name(self) -> str:
        if self._vector_name:
            return self._vector_name
        return f"fast-{self.model_name.split('/')[-1].lower()}"

    def get_vector_size(self) -> int:
        if self._vector_size:
            return self._vector_size
        return self._fallback().get_vector_size()
