"""Read path for the memory OS.

One vectorization per query. Two searches reuse that vector: a global ranking
and an entity slot. Relations live in the payload. Output is progressive and
the second body opens only on a measured tie or a procedure-context pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from kaede_recall_mcp.common.backend import AsyncClient, models

from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.memory_os.entities import (
    BONUS_FLOOR,
    EntityRegistry,
    entity_bonus,
)
from kaede_recall_mcp.memory_os.schema import DENSE_VECTOR

FULL_TEXT_MIN_SCORE = 0.30
MIN_HIT_SCORE = 0.20
SECOND_MIN_BASE = 0.30
DUAL_TIGHT = 0.015
DUAL_MARGIN = 0.047  # p25 of the 30-pair calibration with the dynamic registry
DUAL_MIN_SECOND = 0.33
BODY_DUAL = 450
BODY_SINGLE_RATIO = 0.6
PROCEDURE_FULL_SCORE = 0.42

# The exact lane is a tie-breaker, not a trump card. It adds a small, bounded
# bonus on top of the real similarity, so a path or a digest can win a close
# race but never outranks a strong semantic match.
EXACT_BONUS = 0.08
# An entity node needs this raw similarity before it may take slot 1 on entity
# priority alone. Below it, the global top keeps slot 1.
ENTITY_SLOT_MIN_BASE = 0.33
# A node that does not share the query entities needs this margin over slot 1
# to displace it.
SWAP_GAP = 0.12
SWAP_TIE = 0.01

COMPLEMENT_KINDS = {"procedure"}
STATUS_KINDS = {"fact", "problem", "trace", "decision"}


@dataclass
class RecallHit:
    id: str
    kind: str
    score: float
    text: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    via: str = "search"
    base: float = 0.0

    def trigger(self, limit: int = 160) -> str:
        body = (self.summary or self.text).replace("\n", " ").strip()
        if len(body) > limit:
            body = body[: limit - 1].rstrip() + "…"
        return body

    @property
    def entities(self) -> list[str]:
        return [str(value).lower() for value in (self.payload.get("entities") or [])]


_FILE_EXTS = (
    "py|pyi|rs|dart|yaml|yml|json|jsonc|md|sh|bash|zsh|toml|ini|cfg|conf|txt|"
    "js|mjs|cjs|ts|tsx|jsx|go|java|kt|kts|c|h|cc|cpp|hpp|cs|rb|php|swift|sql|"
    "html|css|scss|xml|csv|log|db|sqlite|ipynb|service|desktop|lock|gradle|"
    "env|proto|tf|nix|vim|lua|pl|jl|ex|exs|erl|hs|ml|clj|scala|vue|svelte"
)

# Only tokens that cannot be produced by a common word qualify for the exact
# lane: a path with a slash, a known file name, a hex digest of 7+ chars.
# Acronyms and short versions (API, NAT, MCP, v2, v3) are deliberately absent:
# they appear in half the collection and would hijack slot 1.
_IDENT = re.compile(
    r"(?P<path>(?:~|\.{1,2})?/[\w.@+-]{2,}(?:/[\w.@+-]+)*)"
    rf"|(?P<file>\b[\w-]+\.(?:{_FILE_EXTS})\b)"
    r"|(?P<hash>\b(?=[0-9a-f]{7,64}\b)(?=[0-9a-f]*\d)[0-9a-f]{7,64}\b)",
    re.IGNORECASE,
)


def _identifier_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in _IDENT.finditer(query):
        term = match.group(0).strip()
        if term and term not in terms:
            terms.append(term)
    return terms[:5]


def _build_filter(
    kinds: Iterable[str] | None,
    scope: str | None,
    project: str | None,
    status: str | None,
    extra: dict[str, Any] | None,
    *,
    include_superseded: bool = False,
    include_deprecated: bool = False,
    entities: Iterable[str] | None = None,
) -> models.Filter | None:
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    if kinds:
        must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
    if scope:
        must.append(models.FieldCondition(key="scope", match=models.MatchValue(value=scope)))
    if project:
        must.append(models.FieldCondition(key="project", match=models.MatchValue(value=project)))
    if status:
        must.append(models.FieldCondition(key="status", match=models.MatchValue(value=status)))
    if entities:
        must.append(
            models.FieldCondition(key="entities", match=models.MatchAny(any=list(entities)))
        )
    for key, value in (extra or {}).items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if not include_superseded:
        must_not.append(
            models.FieldCondition(key="status", match=models.MatchValue(value="superseded"))
        )
    if not include_deprecated:
        must_not.append(
            models.FieldCondition(key="deprecated", match=models.MatchValue(value=True))
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must, must_not=must_not)


async def _exact_hits(
    client: AsyncClient,
    collection: str,
    query: str,
    query_filter: models.Filter | None,
    dense_vec: list[float],
    limit: int = 3,
) -> list[models.ScoredPoint]:
    """Points whose text contains a path, a file name or a hex digest.

    The search runs through the vector index so every hit carries its real
    similarity. The caller adds the bounded exact bonus on top; there is no
    fixed score, so a weak document cannot jump to slot 1 just because it
    mentions a common acronym.
    """
    terms = _identifier_terms(query)
    if not terms:
        return []
    should = [
        models.FieldCondition(key="text", match=models.MatchText(text=term))
        for term in terms
    ]
    must: list[models.Condition] = [query_filter] if query_filter else []
    response = await client.query_points(
        collection_name=collection,
        query=dense_vec,
        using=DENSE_VECTOR,
        query_filter=models.Filter(must=must, should=should),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def _hit(point: models.Record, via: str = "search") -> RecallHit:
    payload = dict(point.payload or {})
    score = float(getattr(point, "score", None) or 0.0)
    return RecallHit(
        id=str(point.id),
        kind=str(payload.get("kind", "doc")),
        score=score,
        text=str(payload.get("text", "")),
        summary=str(payload.get("summary", "")),
        payload=payload,
        via=via,
        base=score,
    )


def _dual_open(first: RecallHit, second: RecallHit) -> bool:
    if second.score < DUAL_MIN_SECOND:
        return False
    gap = max(0.0, first.score - second.score)
    if gap <= DUAL_TIGHT:
        return True
    pair = {first.kind, second.kind}
    complementary = bool(pair & COMPLEMENT_KINDS) and bool(pair & STATUS_KINDS)
    if complementary and gap <= DUAL_MARGIN:
        return True
    # High-confidence pair. Same topic, both strong, slightly outside the
    # calibrated margin: open the second body instead of paying another turn.
    shared = set(first.entities) & set(second.entities)
    return bool(complementary and shared and second.score >= 0.60 and gap <= 0.09)


def _is_dead(payload: dict[str, Any] | None) -> bool:
    """A superseded or deprecated node is never returned by a graph hop."""
    data = payload or {}
    return data.get("status") == "superseded" or data.get("deprecated") is True


async def _touch_hits(
    client: AsyncClient,
    collection: str,
    hits: list["RecallHit"],
    limit: int = 5,
) -> None:
    """Bump usage counters on the nodes actually returned.

    Best effort: a metric failure never breaks a recall, and only the first few
    slots are touched to keep the hook well under its timeout.
    """
    if not hits:
        return
    now = datetime.now(timezone.utc).isoformat()
    for hit in hits[:limit]:
        try:
            points = await client.retrieve(
                collection_name=collection,
                ids=[hit.id],
                with_payload=["hits"],
                with_vectors=False,
            )
            current = int(((points[0].payload or {}) if points else {}).get("hits") or 0)
            await client.set_payload(
                collection_name=collection,
                payload={"hits": current + 1, "last_recalled_at": now},
                points=[hit.id],
            )
        except Exception:
            continue


async def _expand(
    client: AsyncClient,
    collection: str,
    start_ids: list[str],
    depth: int,
    rel: str | None = None,
) -> list[models.Record]:
    """Follow payload edges. One retrieve per level, dead nodes filtered here.

    The ids of a level are grouped into a single retrieve, so a hop costs one
    round trip. Superseded and deprecated neighbours are dropped in Python
    instead of paying a second filtered query per level.
    """
    visited = set(start_ids)
    frontier = list(start_ids)
    collected: list[models.Record] = []
    if depth <= 0:
        return collected
    # One retrieve per level: level 0 resolves the starting nodes without
    # collecting them, every later level is the hop result.
    for level in range(depth + 1):
        if not frontier:
            break
        nodes = await client.retrieve(
            collection_name=collection,
            ids=frontier,
            with_payload=True,
            with_vectors=False,
        )
        alive = [node for node in nodes if not _is_dead(node.payload)]
        if level > 0:
            collected.extend(alive)
        next_ids: list[str] = []
        for node in alive:
            payload = node.payload or {}
            for edge in list(payload.get("edges") or []) + list(payload.get("backlinks") or []):
                relation = edge.get("rel")
                target = edge.get("to") or edge.get("from")
                if not target or target in visited:
                    continue
                if rel and relation != rel:
                    continue
                next_ids.append(str(target))
        visited.update(next_ids)
        frontier = next_ids
    return collected


def _render(hits: list[RecallHit], max_chars: int) -> str:
    if not hits:
        return ""
    dual = len(hits) >= 2 and _dual_open(hits[0], hits[1])
    per_body = min(BODY_DUAL, max(150, (max_chars - 400) // 2)) if dual else max(
        400, int(max_chars * BODY_SINGLE_RATIO)
    )

    lines: list[str] = []
    used = 0
    for index, hit in enumerate(hits):
        line = f"{index + 1}. [{hit.kind} {hit.score:.3f}] {hit.trigger()} id={hit.id}"
        opens_body = (index == 0 and hit.score >= FULL_TEXT_MIN_SCORE) or (
            index == 1
            and (
                dual
                or (hit.kind == "procedure" and hit.score >= PROCEDURE_FULL_SCORE)
            )
        )
        full = ""
        if opens_body:
            room = min(per_body, max_chars - used - len(line) - 2)
            body = hit.text.strip()
            if body and room > 80:
                full = body[:room].rstrip() + ("…" if len(body) > room else "")
        block = line + ("\n" + full if full else "")
        if used + len(block) + 1 > max_chars:
            block = line
            if used + len(block) + 1 > max_chars:
                break
        lines.append(block)
        used += len(block) + 1
    return "\n".join(lines)


def _order_slots(
    hits: list[RecallHit],
    query_entities: list[str],
    limit: int,
) -> list[RecallHit]:
    """Entity relevance first, then a different kind from the same entity.

    No grammar. The order comes from score and kind diversity. An entity node
    only takes slot 1 on entity priority when its raw similarity clears
    ENTITY_SLOT_MIN_BASE; otherwise the global top keeps slot 1. Slot 2 may
    displace slot 1 without sharing the query entity when it wins by SWAP_GAP,
    so a weak entity match cannot bury a strong global procedure.
    """
    if not hits:
        return []
    if query_entities:
        entity_hits = [hit for hit in hits if set(hit.entities) & set(query_entities)]
        strong = [hit for hit in entity_hits if hit.base >= ENTITY_SLOT_MIN_BASE]
        primary = strong[0] if strong else hits[0]
        pool = entity_hits if strong else hits
        second = next(
            (
                hit
                for hit in pool
                if hit.id != primary.id and hit.kind != primary.kind
            ),
            None,
        )
        if second is None:
            second = next((hit for hit in pool if hit.id != primary.id), None)
        if second is None:
            second = next(
                (
                    hit
                    for hit in hits
                    if hit.id != primary.id and hit.kind != primary.kind
                ),
                None,
            )
    else:
        primary = hits[0]
        second = next(
            (hit for hit in hits if hit.id != primary.id and hit.kind != primary.kind),
            None,
        )
        if second is None:
            second = next((hit for hit in hits if hit.id != primary.id), None)

    if second is not None and second.base < SECOND_MIN_BASE:
        second = None
    ordered: list[RecallHit] = [primary]
    if second is not None:
        ordered.append(second)
    for hit in hits:
        if all(hit.id != other.id for other in ordered):
            ordered.append(hit)
    if len(ordered) >= 2:
        first, challenger = ordered[0], ordered[1]
        shares = not query_entities or bool(
            set(challenger.entities) & set(query_entities)
        )
        gap = challenger.score - first.score
        if (shares and gap >= SWAP_TIE) or (not shares and gap >= SWAP_GAP):
            ordered[0], ordered[1] = challenger, first
    return ordered[:limit]


async def recall(
    client: AsyncClient,
    collection: str,
    query: str,
    dense: EmbeddingProvider,
    *,
    kinds: Iterable[str] | None = None,
    scope: str | None = None,
    project: str | None = None,
    status: str | None = "active",
    limit: int = 6,
    max_chars: int = 4000,
    walk_depth: int = 0,
    include_superseded: bool = False,
    include_deprecated: bool = False,
    registry: EntityRegistry | None = None,
    extra_filter: dict[str, Any] | None = None,
) -> tuple[list[RecallHit], str]:
    query_entities = registry.match(query) if registry else []
    dense_vec = await dense.embed_query(query)

    global_filter = _build_filter(
        kinds,
        scope,
        project,
        status,
        extra_filter,
        include_superseded=include_superseded,
        include_deprecated=include_deprecated,
    )
    response = await client.query_points(
        collection_name=collection,
        query=dense_vec,
        using=DENSE_VECTOR,
        query_filter=global_filter,
        limit=max(limit * 5, 30),
        with_payload=True,
        with_vectors=False,
    )

    hits: list[RecallHit] = []
    seen: set[str] = set()
    for point in response.points:
        hit = _hit(point)
        seen.add(hit.id)
        hits.append(hit)

    if query_entities:
        entity_filter = _build_filter(
            kinds,
            scope,
            project,
            status,
            extra_filter,
            include_superseded=include_superseded,
            include_deprecated=include_deprecated,
            entities=query_entities,
        )
        entity_response = await client.query_points(
            collection_name=collection,
            query=dense_vec,
            using=DENSE_VECTOR,
            query_filter=entity_filter,
            limit=max(limit * 3, 15),
            with_payload=True,
            with_vectors=False,
        )
        for point in entity_response.points:
            hit = _hit(point, via="entity")
            if hit.id in seen:
                continue
            seen.add(hit.id)
            hits.append(hit)

    for point in await _exact_hits(
        client, collection, query, global_filter, dense_vec
    ):
        hit = _hit(point, via="exact")
        hit.base = hit.score
        hit.score = min(1.0, hit.base + EXACT_BONUS)
        if hit.id in seen:
            continue
        seen.add(hit.id)
        hits.append(hit)

    for hit in hits:
        if query_entities and hit.base >= BONUS_FLOOR:
            hit.score += entity_bonus(query_entities, hit.entities)
    hits.sort(key=lambda hit: hit.score, reverse=True)
    hits = [hit for hit in hits if hit.score >= MIN_HIT_SCORE]
    hits = _order_slots(hits, query_entities, max(limit, 1))

    if walk_depth > 0 and hits:
        seen = {hit.id for hit in hits}
        for point in await _expand(client, collection, [hit.id for hit in hits[:2]], walk_depth):
            if str(point.id) in seen:
                continue
            seen.add(str(point.id))
            hits.append(_hit(point, via="graph"))

    await _touch_hits(client, collection, hits)
    return hits, _render(hits, max_chars)


async def recall_by_ids(
    client: AsyncClient,
    collection: str,
    ids: Iterable[str],
    *,
    max_chars: int = 4000,
) -> tuple[list[RecallHit], str]:
    """Open known ids without vectorizing. Used to read back slots 2 to 5."""
    wanted = [str(value).strip() for value in ids if str(value).strip()]
    if not wanted:
        return [], ""
    points = await client.retrieve(
        collection_name=collection,
        ids=wanted,
        with_payload=True,
        with_vectors=False,
    )
    by_id = {str(point.id): point for point in points}
    hits = [_hit(by_id[pid]) for pid in wanted if pid in by_id]
    for hit in hits:
        hit.score = 1.0
        hit.base = 1.0
    await _touch_hits(client, collection, hits)
    return hits, _render(hits, max_chars)


async def walk(
    client: AsyncClient,
    collection: str,
    start_id: str,
    *,
    rel: str | None = None,
    depth: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Breadth-first traversal over payload edges."""
    visited = {start_id}
    frontier = [start_id]
    out: list[dict[str, Any]] = []
    for level in range(depth):
        if not frontier or len(out) >= limit:
            break
        nodes = await client.retrieve(
            collection_name=collection,
            ids=frontier,
            with_payload=True,
            with_vectors=False,
        )
        next_frontier: list[str] = []
        for node in nodes:
            payload = node.payload or {}
            for edge in payload.get("edges") or []:
                relation, target = edge.get("rel"), edge.get("to")
                if not target or (rel and relation != rel):
                    continue
                out.append(
                    {
                        "level": level + 1,
                        "from": str(node.id),
                        "rel": relation,
                        "to": str(target),
                    }
                )
                if target not in visited:
                    visited.add(str(target))
                    next_frontier.append(str(target))
        frontier = next_frontier
    return out[:limit]