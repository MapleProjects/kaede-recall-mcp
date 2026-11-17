"""Bootstrap and migration CLI for the memory OS.

Usage:
  python -m kaede_recall_mcp.memory_os.bootstrap create  --collection kaede-memory
  python -m kaede_recall_mcp.memory_os.bootstrap migrate --source agents-memory-v2 --target kaede-memory
  python -m kaede_recall_mcp.memory_os.bootstrap query   "crear cuenta con perfil personal"
  python -m kaede_recall_mcp.memory_os.bootstrap stats   --collection kaede-memory
  python -m kaede_recall_mcp.memory_os.bootstrap dedupe  --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from kaede_recall_mcp.common.backend import AsyncClient, models

from kaede_recall_mcp.embeddings.fastembed import FastEmbedProvider
from kaede_recall_mcp.memory_os.entities import EntityRegistry
from kaede_recall_mcp.memory_os.recall import recall
from kaede_recall_mcp.memory_os.schema import DEFAULT_COLLECTION, ensure_collection
from kaede_recall_mcp.memory_os.store import MemoryStore

DENSE_MODEL = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es")
KAEDE_BACKEND_URL = os.environ.get("KAEDE_BACKEND_URL", "http://localhost:6333")
MAX_EMBED_CHARS = 6000

_PROCEDURE_HINTS = ("skill_index", "skill", "procedimiento", "workflow", "runbook")
_PROBLEM_HINTS = ("troubleshooting", "fix", "repair", "error", "fallo", "failed", "bug")
_DECISION_HINTS = ("configuration", "infrastructure", "decision", "migration", "setup")


def classify(content: str, metadata: dict[str, Any]) -> str:
    blob = " ".join(
        str(metadata.get(key, "")) for key in ("type", "category", "topic", "entity_name")
    ).lower()
    head = content[:400].lower()
    # Legacy skill snapshots stay as docs. The live procedures are ingested
    # from SKILL.md with a clean trigger embedding and would be duplicated here.
    if metadata.get("type") == "skill_index":
        return "doc"
    if any(h in blob for h in _PROCEDURE_HINTS) or head.startswith("# ") and "skill" in head:
        return "procedure"
    if any(h in blob for h in _PROBLEM_HINTS):
        return "problem"
    if any(h in blob for h in _DECISION_HINTS):
        return "decision"
    return "fact"


def _summary(content: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _created_at(metadata: dict[str, Any]) -> str:
    raw = metadata.get("date") or metadata.get("created_at")
    if isinstance(raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return f"{raw}T00:00:00+00:00"
    return datetime.now(timezone.utc).isoformat()


async def cmd_create(args: argparse.Namespace) -> None:
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    created = await ensure_collection(client, args.collection, recreate=args.recreate)
    print(f"{'created' if created else 'already exists'}: {args.collection}")
    await client.close()


async def cmd_migrate(args: argparse.Namespace) -> None:
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    await ensure_collection(client, args.target)
    dense = FastEmbedProvider(DENSE_MODEL)
    store = MemoryStore(client, args.target, dense)

    offset = None
    total = 0
    written = 0
    duplicates = 0
    while True:
        points, offset = await client.scroll(
            collection_name=args.source,
            limit=args.batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = dict(point.payload or {})
            content = str(payload.get("document") or payload.get("content") or "").strip()
            if not content:
                continue
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"raw": metadata}
            kind = classify(content, metadata)
            project = metadata.get("project")
            scope = f"project:{project}" if project else "global"
            ref = await store.write_node(
                kind,
                content,
                summary=_summary(content),
                scope=scope,
                source=args.source,
                project=project,
                entity=metadata.get("entity_name"),
                repo=metadata.get("repo"),
                host=metadata.get("host"),
                tags=[str(metadata.get("topic"))] if metadata.get("topic") else None,
                entities=[
                    value
                    for value in (metadata.get("entity_name"), project, metadata.get("repo"))
                    if value
                ],
                deprecated=metadata.get("type") == "skill_index",
                created_at=_created_at(metadata),
                legacy_id=str(point.id),
                legacy_metadata=json.dumps(metadata, ensure_ascii=False)[:2000],
                embed_text=_summary(content, 1200),
            )
            total += 1
            if ref.duplicate:
                duplicates += 1
            else:
                written += 1
            if total % 100 == 0:
                print(f"  scanned={total} written={written} duplicates={duplicates}", flush=True)
        if offset is None:
            break

    print(f"done. scanned={total} written={written} duplicates={duplicates}")
    await client.close()


def _trigger_text(name: str, description: str, body: str) -> str:
    """The text that should match a user's request, not the whole manual."""
    parts = [name, description]
    match = re.search(
        r"^#{1,3}\s*(when to use|cu[aá]ndo usar|use when|when to apply)\s*$",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        tail = body[match.end() : match.end() + 700]
        parts.append(tail.split("\n## ")[0])
    else:
        parts.append(body[:400])
    return "\n".join(part.strip() for part in parts if part and part.strip())


async def cmd_skills(args: argparse.Namespace) -> None:
    """Ingest SKILL.md files as procedure nodes, one per skill."""
    from pathlib import Path

    root = Path(args.root).expanduser()
    files = sorted(root.rglob("SKILL.md"))
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    await ensure_collection(client, args.collection)
    dense = FastEmbedProvider(DENSE_MODEL)
    store = MemoryStore(client, args.collection, dense)

    written = 0
    updated = 0
    skipped = 0
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        name, description, body = _parse_skill(raw, path)
        if not name or not body.strip():
            skipped += 1
            continue
        text = f"{name}\n\n{description}\n\n{body}".strip()
        category = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "root"

        existing, _ = await client.scroll(
            collection_name=args.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="kind", match=models.MatchValue(value="procedure")),
                    models.FieldCondition(key="procedure_id", match=models.MatchValue(value=name)),
                    models.FieldCondition(key="status", match=models.MatchValue(value="active")),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if existing:
            old = existing[0]
            old_hash = (old.payload or {}).get("hash")
            from kaede_recall_mcp.memory_os.store import content_hash

            if old_hash == content_hash("procedure", text):
                skipped += 1
                continue
            ref = await store.write_node(
                "procedure",
                text,
                summary=description or _summary(body),
                scope="global",
                source="hermes-skills",
                procedure_id=name,
                tags=[category],
                path=str(path),
                dedupe=False,
                entities=[name, category],
                embed_text=_trigger_text(name, description, body),
            )
            await store.supersede(str(old.id), ref.id, reason="skill updated")
            updated += 1
        else:
            await store.write_node(
                "procedure",
                text,
                summary=description or _summary(body),
                scope="global",
                source="hermes-skills",
                procedure_id=name,
                tags=[category],
                path=str(path),
                entities=[name, category],
                embed_text=_trigger_text(name, description, body),
            )
            written += 1
        if (written + updated) % 25 == 0:
            print(f"  written={written} updated={updated} skipped={skipped}", flush=True)

    print(f"skills done. files={len(files)} written={written} updated={updated} skipped={skipped}")
    await client.close()


def _parse_skill(raw: str, path) -> tuple[str, str, str]:
    text = raw.lstrip("\ufeff")
    name = path.parent.name
    description = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            front, body = parts[1], parts[2]
            for line in front.splitlines():
                if line.lower().startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip("\"'") or name
                elif line.lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip("\"'")
    return name, description, body


async def cmd_query(args: argparse.Namespace) -> None:
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    dense = FastEmbedProvider(DENSE_MODEL)
    registry = EntityRegistry()
    await registry.load(client, args.collection)
    kinds = args.kinds.split(",") if args.kinds else None
    hits, block = await recall(
        client,
        args.collection,
        args.query,
        dense,
        kinds=kinds,
        limit=args.limit,
        max_chars=args.max_chars,
        walk_depth=args.walk_depth,
        registry=registry,
    )
    print(f"query: {args.query}")
    print(f"hits: {len(hits)}  chars: {len(block)}")
    print(block or "(no results)")
    await client.close()


async def cmd_stats(args: argparse.Namespace) -> None:
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    info = await client.get_collection(args.collection)
    print(f"points: {info.points_count}")
    for kind in ("fact", "preference", "profile", "decision", "problem", "procedure", "trace", "doc"):
        count = await client.count(
            collection_name=args.collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="kind", match=models.MatchValue(value=kind))]
            ),
            exact=True,
        )
        if count.count:
            print(f"  {kind}: {count.count}")
    await client.close()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def _vector(point) -> list[float] | None:
    stored = point.vector
    if isinstance(stored, dict):
        stored = stored.get("dense") or next(iter(stored.values()), None)
    return list(stored) if stored else None


def _preview(payload: dict[str, Any], limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(payload.get("summary") or payload.get("text") or ""))
    return text[:limit]


def _created(point) -> str:
    return str((point.payload or {}).get("created_at") or "")


async def _active_points(client, collection: str) -> list:
    points: list = []
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value="active")
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="deprecated", match=models.MatchValue(value=True)
                    )
                ],
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None or not batch:
            break
    return points


def find_duplicate_pairs(
    points: list,
    *,
    threshold: float = 0.90,
    entity: str | None = None,
    max_share: float = 0.12,
) -> list[tuple[float, Any, Any, str]]:
    """Pairs of active points that share an entity and score >= threshold."""
    total = len(points)
    # Below 30 points a 12% ceiling would flag every entity, so the floor keeps
    # small collections usable; the share still rules on real ones.
    ceiling = max(3, total * max_share) if total else 0
    by_entity: dict[str, list] = {}
    for point in points:
        for value in set((point.payload or {}).get("entities") or []):
            by_entity.setdefault(str(value).lower(), []).append(point)

    pairs: list[tuple[float, Any, Any, str]] = []
    seen: set[tuple[str, str]] = set()
    for name, group in by_entity.items():
        if entity and name != entity.lower():
            continue
        if len(group) < 2 or (ceiling and len(group) > ceiling):
            continue
        for index, first in enumerate(group):
            first_vec = _vector(first)
            if first_vec is None:
                continue
            for second in group[index + 1 :]:
                first_id, second_id = str(first.id), str(second.id)
                key = (first_id, second_id) if first_id < second_id else (second_id, first_id)
                if key in seen:
                    continue
                second_vec = _vector(second)
                if second_vec is None:
                    continue
                score = _cosine(first_vec, second_vec)
                if score >= threshold:
                    seen.add(key)
                    pairs.append((score, first, second, name))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


async def cmd_dedupe(args: argparse.Namespace) -> None:
    """List or consolidate active duplicates that share an entity.

    Uses the vectors already stored in the backend, so it needs no model. Without
    --dry-run the older point of each pair is superseded by the newer one.
    """
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    if not await client.collection_exists(args.collection):
        print(f"collection not found: {args.collection}")
        await client.close()
        return

    points = await _active_points(client, args.collection)
    pairs = find_duplicate_pairs(
        points, threshold=args.threshold, entity=args.entity
    )[: args.limit]
    print(
        f"active={len(points)} pairs={len(pairs)} threshold={args.threshold:.2f}"
        + (f" entity={args.entity}" if args.entity else "")
    )
    for score, first, second, name in pairs:
        first_payload = first.payload or {}
        second_payload = second.payload or {}
        print(f"{score:.4f}  {first.id}  {second.id}  entity={name}")
        print(f"    A [{first_payload.get('kind')} {_created(first)}] {_preview(first_payload)}")
        print(f"    B [{second_payload.get('kind')} {_created(second)}] {_preview(second_payload)}")

    if args.dry_run:
        print("dry run: nothing written")
        await client.close()
        return

    from kaede_recall_mcp.embeddings.factory import create_embedding_provider
    from kaede_recall_mcp.settings import EmbeddingProviderSettings

    dense = create_embedding_provider(EmbeddingProviderSettings())
    store = MemoryStore(client, args.collection, dense)
    dropped: set[str] = set()
    merged = 0
    for score, first, second, _ in pairs:
        if str(first.id) in dropped or str(second.id) in dropped:
            continue
        keep, drop = (
            (first, second) if _created(first) >= _created(second) else (second, first)
        )
        await store.supersede(str(drop.id), str(keep.id), reason=f"dedupe {score:.3f}")
        dropped.add(str(drop.id))
        merged += 1
    print(f"consolidated={merged} kept={len(pairs) - merged}")
    await client.close()


def find_compact_pairs(
    points: list,
    *,
    threshold: float = 0.88,
    max_share: float = 0.12,
) -> list[tuple[float, Any, Any, str, str]]:
    """Pairs of active facts/problems in the same entity and scope.

    Returns (score, first, second, entity, scope), highest score first.
    """
    total = len(points)
    ceiling = max(3, total * max_share) if total else 0
    by_key: dict[tuple[str, str], list] = {}
    for point in points:
        payload = point.payload or {}
        scope = str(payload.get("scope") or "global")
        for value in set(payload.get("entities") or []):
            by_key.setdefault((str(value).lower(), scope), []).append(point)

    pairs: list[tuple[float, Any, Any, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for (entity, scope), group in by_key.items():
        if len(group) < 2 or (ceiling and len(group) > ceiling):
            continue
        for index, first in enumerate(group):
            first_vec = _vector(first)
            if first_vec is None:
                continue
            for second in group[index + 1 :]:
                first_id, second_id = str(first.id), str(second.id)
                key = (first_id, second_id) if first_id < second_id else (second_id, first_id)
                if key in seen:
                    continue
                second_vec = _vector(second)
                if second_vec is None:
                    continue
                score = _cosine(first_vec, second_vec)
                if score >= threshold:
                    seen.add(key)
                    pairs.append((score, first, second, entity, scope))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


async def cmd_compact(args: argparse.Namespace) -> None:
    """Consolidate near-duplicate facts and problems, offline.

    Similarity comes from the vectors already stored in the backend. Merging and
    re-embedding goes through 127.0.0.1:7451/embed only; the CLI never loads a
    local model. The older node of each pair is marked superseded.
    """
    client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
    if not await client.collection_exists(args.collection):
        print(f"collection not found: {args.collection}")
        await client.close()
        return

    points: list = []
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=args.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="kind",
                        match=models.MatchAny(any=["fact", "problem"]),
                    ),
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value="active")
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="deprecated", match=models.MatchValue(value=True)
                    )
                ],
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None or not batch:
            break

    pairs = find_compact_pairs(points, threshold=args.threshold)[: args.limit]
    print(
        f"active_facts_problems={len(points)} pairs={len(pairs)} "
        f"threshold={args.threshold:.2f}"
    )
    for score, first, second, entity, scope in pairs:
        print(f"{score:.4f}  {first.id}  {second.id}  entity={entity} scope={scope}")
        print(f"    A [{_created(first)}] {_preview(first.payload or {})}")
        print(f"    B [{_created(second)}] {_preview(second.payload or {})}")

    if args.dry_run:
        print("dry run: nothing written")
        await client.close()
        return

    from kaede_recall_mcp.embeddings.remote import ServiceEmbeddingProvider

    dense = ServiceEmbeddingProvider(
        os.environ.get("MEMORY_OS_EMBED_URL", "http://127.0.0.1:7451"), DENSE_MODEL
    )
    if not dense.available:
        print("embedding daemon unavailable at 127.0.0.1:7451; nothing written")
        await client.close()
        return

    store = MemoryStore(client, args.collection, dense)
    dropped: set[str] = set()
    merged = 0
    for score, first, second, entity, scope in pairs:
        if str(first.id) in dropped or str(second.id) in dropped:
            continue
        keep, drop = (
            (first, second) if _created(first) >= _created(second) else (second, first)
        )
        keep_payload = keep.payload or {}
        drop_payload = drop.payload or {}
        keep_text = str(keep_payload.get("text") or "").strip()
        drop_text = str(drop_payload.get("text") or "").strip()
        combined = keep_text
        if drop_text and drop_text not in keep_text:
            candidate = f"{keep_text}\n\n{drop_text}".strip()
            if len(candidate) <= args.max_chars:
                combined = candidate
        try:
            if combined != keep_text:
                await store.update_node(str(keep.id), text=combined)
            await store.supersede(str(drop.id), str(keep.id), reason=f"compact {score:.3f}")
        except Exception as exc:
            print(f"  skip {keep.id}/{drop.id}: {exc}")
            continue
        dropped.add(str(drop.id))
        merged += 1
    print(f"consolidated={merged} kept={len(pairs) - merged}")
    await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory-os")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--collection", default=DEFAULT_COLLECTION)
    create.add_argument("--recreate", action="store_true")
    create.set_defaults(func=cmd_create)

    migrate = sub.add_parser("migrate")
    migrate.add_argument("--source", default="agents-memory-v2")
    migrate.add_argument("--target", default=DEFAULT_COLLECTION)
    migrate.add_argument("--batch", type=int, default=64)
    migrate.set_defaults(func=cmd_migrate)

    skills = sub.add_parser("skills")
    skills.add_argument("--root", default="~/.hermes/skills")
    skills.add_argument("--collection", default=DEFAULT_COLLECTION)
    skills.set_defaults(func=cmd_skills)

    query = sub.add_parser("query")
    query.add_argument("query")
    query.add_argument("--collection", default=DEFAULT_COLLECTION)
    query.add_argument("--kinds", default=None)
    query.add_argument("--limit", type=int, default=6)
    query.add_argument("--max-chars", type=int, default=4000)
    query.add_argument("--walk-depth", type=int, default=0)
    query.set_defaults(func=cmd_query)

    stats = sub.add_parser("stats")
    stats.add_argument("--collection", default=DEFAULT_COLLECTION)
    stats.set_defaults(func=cmd_stats)

    dedupe = sub.add_parser("dedupe")
    dedupe.add_argument("--collection", default=DEFAULT_COLLECTION)
    dedupe.add_argument("--threshold", type=float, default=0.90)
    dedupe.add_argument("--limit", type=int, default=50)
    dedupe.add_argument("--entity", default=None)
    dedupe.add_argument("--dry-run", action="store_true")
    dedupe.set_defaults(func=cmd_dedupe)

    compact = sub.add_parser("compact")
    compact.add_argument("--collection", default=DEFAULT_COLLECTION)
    compact.add_argument("--threshold", type=float, default=0.88)
    compact.add_argument("--limit", type=int, default=50)
    compact.add_argument("--max-chars", type=int, default=6000)
    compact.add_argument("--dry-run", action="store_true")
    compact.set_defaults(func=cmd_compact)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
