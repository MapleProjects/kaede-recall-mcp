import asyncio
import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from pydantic import BaseModel
from kaede_recall_mcp.common.backend import AsyncClient, models

from kaede_recall_mcp.common.ids import point_id_for
from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.memory_os.client import invalidate_entities
from kaede_recall_mcp.memory_os.entities import extract_identifiers, merge_entities
from kaede_recall_mcp.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]

DOCUMENT_PATH = "document"
# One dedupe field for both write paths. `content_hash` is only read, never
# written, so points created before the unification still dedupe correctly.
CONTENT_HASH_PATH = "hash"
LEGACY_HASH_PATH = "content_hash"
CREATED_AT_PATH = "created_at"
UPDATED_AT_PATH = "updated_at"

# Re-embedding uses the trigger plus the head of the body. The tail of a long
# document adds noise to the vector and costs tokens for no extra recall.
EMBED_HEAD_CHARS = 1200

T = TypeVar("T")

# Errors that mean "the server is not reachable right now" rather than "the
# request is wrong". Only these are retried.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_fingerprint(content: str, metadata: Metadata | None) -> str:
    """Stable hash of the content plus its metadata.

    Two entries with the same text and the same metadata are the same memory.
    The same text under different metadata is a different memory and is kept.
    """
    canonical_metadata = json.dumps(
        metadata or {}, sort_keys=True, ensure_ascii=False, default=str
    )
    payload = f"{_normalize_text(content)}\x00{canonical_metadata}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Entry(BaseModel):
    """
    A single entry in the collection.
    """

    content: str
    metadata: Metadata | None = None
    # The point id. It is populated when an entry is read back via
    # `search`, `get`, `update` or `delete`, and is the handle required to
    # modify an existing entry.
    id: str | None = None
    # Cosine similarity of the entry against the query. Only set by `search`
    # and by query-based `delete`.
    score: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    # True when `store` found an identical memory and returned it instead of
    # writing a second copy.
    duplicate: bool = False


class KaedeConnector:
    """
    Encapsulates the connection to a the backend server and all the methods to interact with it.
    :param backend_url: The URL of the backend server.
    :param backend_api_key: The API key to use for the backend server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param backend_path: The path to the storage directory for the backend client, if local mode is used.
    :param field_indexes: Payload indexes to create with the collection.
    :param dedupe: Skip a write when an identical content and metadata already exist.
    :param retries: Attempts for a client call that fails on a transient connection error.
    """

    def __init__(
        self,
        backend_url: str | None,
        backend_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        backend_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        dedupe: bool = True,
        retries: int = 3,
    ):
        self._backend_url = backend_url.rstrip("/") if backend_url else None
        self._backend_api_key = backend_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncClient(
            location=backend_url,
            api_key=backend_api_key,
            path=backend_path,
            check_compatibility=False,
        )
        self._field_indexes = field_indexes
        self._dedupe = dedupe
        self._retries = max(1, retries)
        self._vector_names: dict[str, str] = {}

    async def _vector_name(self, collection_name: str) -> str:
        """Use the memory OS vector name when the collection has one."""
        cached = self._vector_names.get(collection_name)
        if cached:
            return cached
        name = self._embedding_provider.get_vector_name()
        try:
            info = await self._call(lambda: self._client.get_collection(collection_name))
            vectors = info.config.params.vectors
            if isinstance(vectors, dict) and "dense" in vectors:
                name = "dense"
        except Exception:
            pass
        self._vector_names[collection_name] = name
        return name

    @property
    def client(self) -> AsyncClient:
        """The underlying client, for layers that need raw access."""
        return self._client

    async def _call(self, factory: Callable[[], Awaitable[T]]) -> T:
        """Run a client call, retrying only transient connection failures."""
        delay = 0.25
        last: BaseException | None = None
        for attempt in range(self._retries):
            try:
                return await factory()
            except _TRANSIENT_ERRORS as exc:
                last = exc
                if attempt == self._retries - 1:
                    break
                logger.warning(
                    "the backend call failed (%s), retry %d/%d in %.2fs",
                    type(exc).__name__,
                    attempt + 1,
                    self._retries - 1,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
        assert last is not None
        raise last

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the backend server.
        :return: A list of collection names.
        """
        response = await self._call(self._client.get_collections)
        return [collection.name for collection in response.collections]

    def _resolve_collection(self, collection_name: str | None) -> str:
        resolved = collection_name or self._default_collection_name
        if not resolved:
            raise ValueError(
                "No collection name provided and no default collection configured."
            )
        return resolved

    @staticmethod
    def _entry_from_point(point: models.Record, score: float | None = None) -> Entry:
        payload = point.payload or {}
        return Entry(
            content=payload.get(DOCUMENT_PATH) or payload.get("text", ""),
            metadata=payload.get(METADATA_PATH),
            id=str(point.id),
            score=score,
            created_at=payload.get(CREATED_AT_PATH),
            updated_at=payload.get(UPDATED_AT_PATH),
        )

    async def store(self, entry: Entry, *, collection_name: str | None = None) -> Entry:
        """
        Store some information in the collection, along with the specified metadata.

        When deduplication is enabled, an entry whose content and metadata match an
        existing one is not written again: the existing entry is returned with
        ``duplicate=True``.

        :param entry: The entry to store in the collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        :return: The stored entry, including the generated point id.
        """
        if not entry.content or not entry.content.strip():
            raise ValueError("Cannot store an empty document.")

        collection_name = self._resolve_collection(collection_name)
        await self._ensure_collection_exists(collection_name)

        fingerprint = content_fingerprint(entry.content, entry.metadata)

        if self._dedupe:
            existing = await self._find_by_fingerprint(collection_name, fingerprint)
            if existing is not None:
                existing.duplicate = True
                return existing

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to the backend
        vector_name = await self._vector_name(collection_name)
        point_id = await self._free_point_id(collection_name, fingerprint)
        now = _utc_now()
        payload = {
            DOCUMENT_PATH: entry.content,
            METADATA_PATH: entry.metadata,
            CONTENT_HASH_PATH: fingerprint,
            CREATED_AT_PATH: now,
            UPDATED_AT_PATH: now,
            "text": entry.content,
            "summary": entry.content[:280],
            "kind": "doc",
            "status": "active",
            "entities": extract_identifiers(entry.content),
            "edges": [],
            "backlinks": [],
            "deprecated": False,
        }
        await self._call(
            lambda: self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )
        )
        return Entry(
            content=entry.content,
            metadata=entry.metadata,
            id=point_id,
            created_at=now,
            updated_at=now,
        )

    async def _find_by_fingerprint(
        self, collection_name: str, fingerprint: str
    ) -> Entry | None:
        points, _ = await self._call(
            lambda: self._client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key=CONTENT_HASH_PATH,
                            match=models.MatchValue(value=fingerprint),
                        ),
                        models.FieldCondition(
                            key=LEGACY_HASH_PATH,
                            match=models.MatchValue(value=fingerprint),
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        )
        if not points:
            return None
        return self._entry_from_point(points[0])

    async def _free_point_id(
        self, collection_name: str, fingerprint: str
    ) -> str:
        """Deterministic id for this content, free of a stale occupant.

        The id is derived from the content digest, so a second write of the same
        text lands on the same point. If that point was edited afterwards, it
        holds a different hash; reusing the id would silently undo the edit, so
        the sibling id derived from the occupant's hash is used instead.
        """
        base = point_id_for("doc", collection_name, fingerprint)
        points = await self._call(
            lambda: self._client.retrieve(
                collection_name=collection_name,
                ids=[base],
                with_payload=True,
                with_vectors=False,
            )
        )
        if not points:
            return base
        payload = points[0].payload or {}
        occupant = str(
            payload.get(CONTENT_HASH_PATH) or payload.get(LEGACY_HASH_PATH) or ""
        )
        if not occupant or occupant == fingerprint:
            return base
        return point_id_for("doc", collection_name, fingerprint, occupant)

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[Entry]:
        """
        Find points in the collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :param score_threshold: Minimum similarity score for a result to be returned, optional.

        :return: A list of entries found, each carrying its id and similarity score.
        """
        collection_name = self._resolve_collection(collection_name)
        collection_exists = await self._call(
            lambda: self._client.collection_exists(collection_name)
        )
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = await self._vector_name(collection_name)

        # Search in the backend
        search_results = await self._call(
            lambda: self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
        )

        return [
            self._entry_from_point(result, score=result.score)
            for result in search_results.points
        ]

    async def get(
        self, point_id: str, *, collection_name: str | None = None
    ) -> Entry | None:
        """
        Read a single entry by its point id.
        :param point_id: The id of the point to read.
        :param collection_name: The name of the collection to read from, optional. If not provided,
                                the default collection is used.
        :return: The entry, or None when the collection or the point does not exist.
        """
        collection_name = self._resolve_collection(collection_name)
        if not await self._call(
            lambda: self._client.collection_exists(collection_name)
        ):
            return None

        points = await self._retrieve(collection_name, point_id)
        if not points:
            return None
        return self._entry_from_point(points[0])

    @staticmethod
    def _id_candidates(point_id: str | int) -> list[str]:
        """Return the accepted spellings of a point id, most specific first.

        the backend accepts an unsigned integer or a UUID, and the same UUID can be
        written with or without dashes. The remote server normalizes both forms,
        local mode matches the stored string exactly, so every spelling is tried
        before concluding that the point does not exist.
        """
        if isinstance(point_id, int):
            if point_id < 0:
                raise ValueError(f"'{point_id}' is not a valid point id: negative.")
            return [str(point_id)]

        candidate = str(point_id).strip()
        if candidate.isdigit():
            return [candidate]

        try:
            parsed = uuid.UUID(candidate)
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"'{point_id}' is not a valid point id: expected a UUID or an "
                "unsigned integer."
            ) from exc

        candidates: list[str] = []
        for form in (candidate, parsed.hex, str(parsed)):
            if form not in candidates:
                candidates.append(form)
        return candidates

    async def _retrieve(
        self, collection_name: str, point_id: str | int
    ) -> list[models.Record]:
        for candidate in self._id_candidates(point_id):
            points = await self._call(
                lambda c=candidate: self._client.retrieve(
                    collection_name=collection_name,
                    ids=[c],
                    with_payload=True,
                    with_vectors=False,
                )
            )
            if points:
                return points
        return []

    async def update(
        self,
        point_id: str,
        *,
        content: str | None = None,
        metadata: Metadata | None = None,
        clear_metadata: bool = False,
        collection_name: str | None = None,
    ) -> Entry:
        """
        Update an existing entry in the collection in place, identified by its point id.

        Only the fields that are provided are modified; anything left as ``None`` is untouched.
        A ``content`` change re-embeds the document and rewrites the point in a single upsert,
        so the vector and the payload can never diverge. A ``metadata`` change rewrites the
        payload without touching the vector, so it is cheap and never triggers a re-embedding.

        :param point_id: The id of the point to update. Must refer to an existing point.
        :param content: The new content to store. If provided, the document is re-embedded.
        :param metadata: The new metadata. If provided, it replaces the existing metadata.
        :param clear_metadata: Remove the metadata key entirely. Mutually exclusive with ``metadata``.
        :param collection_name: The name of the collection to update in, optional. If not provided,
                                the default collection is used.
        :return: The updated entry.
        :raises ValueError: If nothing is provided to update, if the collection does not
                            exist, or if no point with the given id exists.
        """
        if content is None and metadata is None and not clear_metadata:
            raise ValueError(
                "Nothing to update: provide 'content', 'metadata', or 'clear_metadata'."
            )
        if clear_metadata and metadata is not None:
            raise ValueError("Provide either 'metadata' or 'clear_metadata', not both.")
        if content is not None and not content.strip():
            raise ValueError("Cannot set an empty document.")

        collection_name = self._resolve_collection(collection_name)

        if not await self._call(
            lambda: self._client.collection_exists(collection_name)
        ):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        existing = await self._retrieve(collection_name, point_id)
        if not existing:
            raise ValueError(
                f"No entry with id '{point_id}' found in collection '{collection_name}'."
            )

        # Use the id exactly as the backend stored it, so the write targets the same
        # key regardless of how the caller spelled the UUID.
        point_id = str(existing[0].id)
        payload = dict(existing[0].payload or {})

        if content is not None:
            payload[DOCUMENT_PATH] = content
            payload["text"] = content
            payload["summary"] = content[:280]
            payload["entities"] = merge_entities(
                payload.get("entities"), extract_identifiers(content)
            )
            # The legacy field would keep matching the old body after an edit.
            payload.pop(LEGACY_HASH_PATH, None)
        if clear_metadata:
            payload.pop(METADATA_PATH, None)
        elif metadata is not None:
            payload[METADATA_PATH] = metadata

        payload[CONTENT_HASH_PATH] = content_fingerprint(
            payload.get(DOCUMENT_PATH, ""), payload.get(METADATA_PATH)
        )
        payload[UPDATED_AT_PATH] = _utc_now()
        payload.setdefault(CREATED_AT_PATH, payload[UPDATED_AT_PATH])

        if content is not None:
            # Single atomic write: vector and payload land together. The vector
            # is built from the trigger plus the head of the body, not the whole
            # document, so a long tail cannot dilute the embedding.
            summary = str(payload.get("summary") or content[:280]).strip()
            embed_input = f"{summary}\n{content[:EMBED_HEAD_CHARS]}".strip()
            embeddings = await self._embedding_provider.embed_documents([embed_input])
            vector_name = await self._vector_name(collection_name)
            await self._call(
                lambda: self._client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector={vector_name: embeddings[0]},
                            payload=payload,
                        )
                    ],
                )
            )
        else:
            # Metadata-only change: rewrite the payload, leave the vector alone.
            await self._call(
                lambda: self._client.overwrite_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[point_id],
                )
            )

        if content is not None:
            # The entity set may have changed, so the daemon's registry is stale.
            await invalidate_entities()

        return Entry(
            content=payload.get(DOCUMENT_PATH) or payload.get("text", ""),
            metadata=payload.get(METADATA_PATH),
            id=str(point_id),
            created_at=payload.get(CREATED_AT_PATH),
            updated_at=payload.get(UPDATED_AT_PATH),
        )

    async def delete(
        self,
        query: str | None = None,
        *,
        point_id: str | None = None,
        collection_name: str | None = None,
        limit: int = 1,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
        dry_run: bool = False,
    ) -> list[Entry]:
        """
        Delete entries either by exact point id or by semantic search.

        Exactly one of ``query`` or ``point_id`` must be provided. Deleting by id is
        deterministic and is the safe path. Deleting by query removes the ``limit``
        closest matches, optionally restricted by ``score_threshold`` and
        ``query_filter``; use ``dry_run`` to preview the matches without removing them.

        :param query: Semantic query used to find the entries to delete.
        :param point_id: Exact id of the entry to delete.
        :param collection_name: The name of the collection to delete from, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to delete when deleting by query.
        :param score_threshold: Minimum similarity score for a match to be deleted.
        :param query_filter: The filter to apply to the query, if any.
        :param dry_run: Return the matches without deleting them.
        :return: The deleted entries, or the matches when ``dry_run`` is set.
        :raises ValueError: If both or neither of ``query`` and ``point_id`` are provided.
        """
        if (query is None) == (point_id is None):
            raise ValueError("Provide exactly one of 'query' or 'point_id' to delete.")

        collection_name = self._resolve_collection(collection_name)
        if not await self._call(
            lambda: self._client.collection_exists(collection_name)
        ):
            return []

        if point_id is not None:
            points = await self._retrieve(collection_name, point_id)
            entries = [self._entry_from_point(point) for point in points]
        else:
            assert query is not None
            query_vector = await self._embedding_provider.embed_query(query)
            vector_name = await self._vector_name(collection_name)
            results = await self._call(
                lambda: self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                )
            )
            entries = [
                self._entry_from_point(result, score=result.score)
                for result in results.points
            ]

        if not entries or dry_run:
            return entries

        point_ids = [entry.id for entry in entries if entry.id is not None]
        await self._call(
            lambda: self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
        )
        return entries

    async def list_entries(
        self,
        *,
        collection_name: str | None = None,
        limit: int = 20,
        offset: str | None = None,
        query_filter: models.Filter | None = None,
    ) -> tuple[list[Entry], str | None]:
        """
        Page through the collection without a semantic query.

        :param collection_name: The name of the collection to read, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param offset: The opaque cursor returned by a previous call, optional.
        :param query_filter: The filter to apply, if any.
        :return: The entries and the cursor for the next page, or None on the last page.
        """
        collection_name = self._resolve_collection(collection_name)
        if not await self._call(
            lambda: self._client.collection_exists(collection_name)
        ):
            return [], None

        points, next_offset = await self._call(
            lambda: self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        )
        return [self._entry_from_point(point) for point in points], (
            str(next_offset) if next_offset is not None else None
        )

    async def count(self, *, collection_name: str | None = None) -> int:
        """
        Count the entries in a collection.
        :param collection_name: The name of the collection to count, optional. If not provided,
                                the default collection is used.
        :return: The number of entries, or 0 when the collection does not exist.
        """
        collection_name = self._resolve_collection(collection_name)
        if not await self._call(
            lambda: self._client.collection_exists(collection_name)
        ):
            return 0
        result = await self._call(
            lambda: self._client.count(collection_name=collection_name, exact=True)
        )
        return result.count

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._call(
            lambda: self._client.collection_exists(collection_name)
        )
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the memory OS vector name so both tool families share one collection
            vector_name = "dense"
            try:
                await self._call(
                    lambda: self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            vector_name: models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE,
                            )
                        },
                    )
                )
            except Exception:
                # Another writer may have created it between the check and the
                # create. Only swallow the error if the collection now exists.
                if not await self._call(
                    lambda: self._client.collection_exists(collection_name)
                ):
                    raise

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._call(
                        lambda f=field_name,
                        t=field_type: self._client.create_payload_index(
                            collection_name=collection_name,
                            field_name=f,
                            field_schema=t,
                        )
                    )

        # The deduplication filter needs an index on the fingerprint field.
        if self._dedupe:
            try:
                await self._call(
                    lambda: self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=CONTENT_HASH_PATH,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                )
            except Exception as exc:  # index already exists or server refused
                logger.debug("content_hash index not created: %s", exc)
