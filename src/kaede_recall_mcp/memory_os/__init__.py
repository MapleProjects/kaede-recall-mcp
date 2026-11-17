"""Memory OS layer for the Kaede Recall.

One collection holds every kind of memory as a typed point:

- nodes: fact, preference, profile, decision, problem, procedure, trace, doc
- relations: stored inside each node payload as edges and backlinks

Retrieval is dense with an exact lane for identifiers and a hard byte budget,
and optional one-hop graph expansion. Nothing is injected unless asked for.
"""

from kaede_recall_mcp.memory_os.schema import (
    DENSE_VECTOR,
    ensure_collection,
)
from kaede_recall_mcp.memory_os.store import MemoryStore, NodeRef
from kaede_recall_mcp.memory_os.recall import RecallHit, recall, walk

__all__ = [
    "DENSE_VECTOR",
    "ensure_collection",
    "MemoryStore",
    "NodeRef",
    "RecallHit",
    "recall",
    "walk",
]
