"""Deterministic point ids.

Both write paths (the legacy document store and the memory OS) derive the
point id from the content digest, so the same memory always lands on the same
key. A retry, a double write or a replay after a crash cannot create a second
copy, and the id is a canonical UUID with dashes, which is what the backend stores
and what every tool returns.
"""

from __future__ import annotations

import uuid

# Fixed namespace: changing it would orphan every existing point.
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kaede-recall-mcp/point")


def point_id_for(*parts: str) -> str:
    """Return the canonical uuid5 (with dashes) for a set of identity parts."""
    payload = "\x00".join(str(part) for part in parts)
    return str(uuid.uuid5(POINT_NAMESPACE, payload))
