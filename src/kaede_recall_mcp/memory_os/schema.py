"""Collection schema for the memory OS.

Named vectors:
  dense - jina-embeddings-v2-base-es, 768 dims, cosine

Payload indexes are what make filtering cheap. Without them the backend scans.
"""

from __future__ import annotations

import os

from kaede_recall_mcp.common.backend import AsyncClient, models

DENSE_VECTOR = "dense"
DENSE_SIZE = 768
DEFAULT_COLLECTION = os.environ.get("MEMORY_OS_COLLECTION", "kaede-memory")

KINDS = (
    "fact",
    "preference",
    "profile",
    "decision",
    "problem",
    "procedure",
    "trace",
    "doc",
    "edge",
)

KEYWORD_FIELDS = (
    "kind",
    "scope",
    "status",
    "project",
    "entity",
    "repo",
    "host",
    "tags",
    "rel",
    "from_id",
    "to_id",
    "procedure_id",
    "hash",
    "source",
    "superseded_by",
    "entities",
)

DATETIME_FIELDS = ("created_at", "updated_at", "valid_from", "valid_to")


def _vectors_config() -> dict[str, models.VectorParams]:
    return {
        DENSE_VECTOR: models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE),
    }


async def ensure_collection(
    client: AsyncClient,
    collection: str,
    *,
    recreate: bool = False,
) -> bool:
    """Create the collection and its payload indexes. Returns True if created."""
    exists = await client.collection_exists(collection)
    if exists and recreate:
        await client.delete_collection(collection)
        exists = False
    if exists:
        return False

    await client.create_collection(
        collection_name=collection,
        vectors_config=_vectors_config(),
        on_disk_payload=True,
    )

    for field in KEYWORD_FIELDS:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    for field in DATETIME_FIELDS:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.DATETIME,
        )
    await client.create_payload_index(
        collection_name=collection,
        field_name="deprecated",
        field_schema=models.PayloadSchemaType.BOOL,
    )
    await client.create_payload_index(
        collection_name=collection,
        field_name="text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
        ),
    )
    return True
