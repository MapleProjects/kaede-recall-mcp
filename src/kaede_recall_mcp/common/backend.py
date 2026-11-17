"""Storage backend re-exports.

This is the only module that names the vector database engine. The rest of the
code talks to `AsyncClient`, `Client` and `models`, so the engine can be
replaced without touching the memory layer.
"""

from qdrant_client import AsyncQdrantClient as AsyncClient
from qdrant_client import QdrantClient as Client
from qdrant_client import models

__all__ = ["AsyncClient", "Client", "models"]
