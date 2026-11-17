"""Write path for the memory OS.

Every write is a typed point. Facts dedupe by content hash, contradictions
supersede instead of deleting, and relations live inside the node payload so a
hop costs one retrieve instead of an extra point and an extra query.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from kaede_recall_mcp.common.backend import AsyncClient, models

from kaede_recall_mcp.common.ids import point_id_for
from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.memory_os.entities import extract_identifiers, merge_entities
from kaede_recall_mcp.memory_os.schema import DENSE_VECTOR

_WS = re.compile(r"\s+")

# A fact is only auto-superseded when it is almost the same sentence, in the
# same scope and project, and about exactly the same entities. Anything looser
# would overwrite a different fact from the same team.
SUPERSEDE_THRESHOLD = 0.92


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def content_hash(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\x00{_norm(text)}".encode("utf-8")).hexdigest()


def _project_condition(project: str | None) -> models.Condition:
    """Match an exact project, or the absence of the field when there is none."""
    if project:
        return models.FieldCondition(key="project", match=models.MatchValue(value=project))
    return models.IsEmptyCondition(is_empty=models.PayloadField(key="project"))


@dataclass
class NodeRef:
    id: str
    kind: str
    duplicate: bool = False
    superseded: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(
        self,
        client: AsyncClient,
        collection: str,
        dense: EmbeddingProvider,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense = dense

    async def _dense_vec(self, text: str) -> list[float]:
        return (await self.dense.embed_documents([text[:8000]]))[0]

    async def _find_active_by_hash(
        self,
        kind: str,
        digest: str,
        scope: str,
        project: str | None,
    ) -> str | None:
        must: list[models.Condition] = [
            models.FieldCondition(key="kind", match=models.MatchValue(value=kind)),
            models.FieldCondition(key="hash", match=models.MatchValue(value=digest)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
            models.FieldCondition(key="scope", match=models.MatchValue(value=scope)),
        ]
        must.append(_project_condition(project))
        points, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=must),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return str(points[0].id) if points else None

    async def write_node(
        self,
        kind: str,
        text: str,
        *,
        summary: str | None = None,
        scope: str = "global",
        project: str | None = None,
        status: str = "active",
        confidence: float = 1.0,
        source: str | None = None,
        tags: Iterable[str] | None = None,
        entities: Iterable[str] | None = None,
        links: Iterable[str] | None = None,
        supersedes_id: str | None = None,
        dedupe: bool = True,
        embed_text: str | None = None,
        **extra: Any,
    ) -> NodeRef:
        digest = content_hash(kind, text)
        if dedupe:
            existing = await self._find_active_by_hash(kind, digest, scope, project)
            if existing:
                return NodeRef(id=existing, kind=kind, duplicate=True)

        point_id = point_id_for("memory", kind, digest, scope, project or "")
        now = _now()
        payload: dict[str, Any] = {
            "kind": kind,
            "text": text,
            "summary": summary or text[:280],
            "scope": scope,
            "status": status,
            "confidence": confidence,
            "hash": digest,
            "created_at": now,
            "updated_at": now,
            "valid_from": now,
            "tags": list(tags or []),
            "entities": merge_entities(
                entities,
                extract_identifiers(f"{text} {summary or ''} {embed_text or ''}"),
            ),
            "edges": [],
            "backlinks": [],
            "deprecated": False,
        }
        if source:
            payload["source"] = source
        if project:
            payload["project"] = project
        for key, value in extra.items():
            if value is not None:
                payload[key] = value

        vector = await self._dense_vec(embed_text or text)
        if kind == "fact" and not supersedes_id and payload["entities"]:
            supersedes_id = await self._similar_fact(
                vector, payload["entities"], scope, project
            )

        await self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={DENSE_VECTOR: vector},
                    payload=payload,
                )
            ],
        )

        if supersedes_id:
            await self.supersede(supersedes_id, point_id)
        for spec in links or []:
            rel, _, to_id = str(spec).partition(":")
            if rel and to_id:
                await self.link(point_id, rel.strip(), to_id.strip())
        return NodeRef(
            id=point_id, kind=kind, superseded=supersedes_id, payload=payload
        )

    async def _similar_fact(
        self,
        vector: list[float],
        entities: list[str],
        scope: str,
        project: str | None,
    ) -> str | None:
        """A near-duplicate fact on the same entity is superseded, not kept twice.

        The candidate must live in the same scope and project and carry exactly
        the same entity set. A different fact about the same team is kept.
        """
        must: list[models.Condition] = [
            models.FieldCondition(key="kind", match=models.MatchValue(value="fact")),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
            models.FieldCondition(key="entities", match=models.MatchAny(any=entities)),
            models.FieldCondition(key="scope", match=models.MatchValue(value=scope)),
            _project_condition(project),
        ]
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR,
            query_filter=models.Filter(
                must=must,
                must_not=[
                    models.FieldCondition(key="deprecated", match=models.MatchValue(value=True))
                ],
            ),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        wanted = set(entities)
        for point in response.points:
            if float(point.score or 0) < SUPERSEDE_THRESHOLD:
                continue
            previous = set((point.payload or {}).get("entities") or [])
            if previous == wanted:
                return str(point.id)
        return None

    async def update_node(
        self,
        point_id: str,
        *,
        text: str | None = None,
        summary: str | None = None,
        tags: Iterable[str] | None = None,
        entities: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        """Edit a node in place.

        `text` rewrites the body, recomputes the hash and the entities, and
        re-embeds `summary + first 1200 chars`. `summary` alone re-embeds the
        same way, because the trigger is part of the vector. Tags and entities
        alone only rewrite the payload, with no model call.
        """
        points = await self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = dict(points[0].payload or {})
        kind = str(payload.get("kind", "fact"))
        reembed = text is not None or summary is not None

        if text is not None:
            body = text.strip()
            if not body:
                raise ValueError("Cannot set an empty memory.")
            payload["text"] = body
            payload["summary"] = (summary or body[:280]).strip()
            extracted = extract_identifiers(body)
            payload["entities"] = (
                merge_entities(entities, extracted)
                if entities is not None
                else merge_entities(payload.get("entities"), extracted)
            )
            payload["hash"] = content_hash(kind, body)
        elif summary is not None:
            payload["summary"] = summary.strip()

        if tags is not None:
            payload["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
        if entities is not None and text is None:
            payload["entities"] = merge_entities(entities, [])
        payload["updated_at"] = _now()

        if reembed:
            body = str(payload.get("text", ""))
            trigger = str(payload.get("summary") or body[:280]).strip()
            vector = await self._dense_vec(f"{trigger}\n{body[:1200]}")
            await self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={DENSE_VECTOR: vector},
                        payload=payload,
                    )
                ],
            )
        else:
            await self.client.overwrite_payload(
                collection_name=self.collection,
                payload=payload,
                points=[point_id],
            )
        return payload

    async def delete_nodes(
        self,
        *,
        point_id: str | None = None,
        query: str | None = None,
        limit: int = 1,
        score_threshold: float | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Delete by exact id or by semantic match, with a dry-run preview."""
        if (point_id is None) == (query is None):
            raise ValueError("Provide exactly one of 'point_id' or 'query'.")

        hits: list[dict[str, Any]] = []
        if point_id is not None:
            points = await self.client.retrieve(
                collection_name=self.collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                data = dict(point.payload or {})
                hits.append(
                    {
                        "id": str(point.id),
                        "score": None,
                        "kind": data.get("kind", ""),
                        "preview": str(data.get("summary") or data.get("text", ""))[:160],
                    }
                )
        else:
            assert query is not None
            vector = await self.dense.embed_query(query)
            response = await self.client.query_points(
                collection_name=self.collection,
                query=vector,
                using=DENSE_VECTOR,
                limit=max(1, min(int(limit), 50)),
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            for point in response.points:
                data = dict(point.payload or {})
                hits.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score or 0.0),
                        "kind": data.get("kind", ""),
                        "preview": str(data.get("summary") or data.get("text", ""))[:160],
                    }
                )

        if dry_run or not hits:
            return hits

        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[hit["id"] for hit in hits]),
        )
        return hits

    async def link(self, from_id: str, rel: str, to_id: str) -> bool:
        """Add a relation to both payloads. Idempotent. One retrieve each."""
        if from_id == to_id:
            return False
        points = await self.client.retrieve(
            collection_name=self.collection,
            ids=[from_id, to_id],
            with_payload=True,
            with_vectors=False,
        )
        by_id = {str(p.id): dict(p.payload or {}) for p in points}
        source, target = by_id.get(from_id), by_id.get(to_id)
        if source is None or target is None:
            return False
        outgoing = list(source.get("edges") or [])
        if not any(e.get("rel") == rel and e.get("to") == to_id for e in outgoing):
            outgoing.append({"rel": rel, "to": to_id})
        incoming = list(target.get("backlinks") or [])
        if not any(b.get("rel") == rel and b.get("from") == from_id for b in incoming):
            incoming.append({"rel": rel, "from": from_id})
        now = _now()
        await self.client.set_payload(
            collection_name=self.collection,
            payload={"edges": outgoing, "updated_at": now},
            points=[from_id],
        )
        await self.client.set_payload(
            collection_name=self.collection,
            payload={"backlinks": incoming, "updated_at": now},
            points=[to_id],
        )
        return True

    async def supersede(self, old_id: str, new_id: str, reason: str | None = None) -> None:
        now = _now()
        payload: dict[str, Any] = {
            "status": "superseded",
            "valid_to": now,
            "superseded_by": new_id,
            "updated_at": now,
        }
        if reason:
            payload["supersede_reason"] = reason
        await self.client.set_payload(
            collection_name=self.collection, payload=payload, points=[old_id]
        )
        await self.link(old_id, "superseded_by", new_id)

    async def get(self, point_id: str) -> dict[str, Any] | None:
        points = await self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = dict(points[0].payload or {})
        payload["id"] = str(points[0].id)
        return payload
