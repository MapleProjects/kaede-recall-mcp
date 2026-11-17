"""MCP tools for the memory OS.

Two tools, not five. Recall carries the graph hop and write carries the links,
so a turn never pays for a second schema or a second round trip.
"""

from typing import Annotated

from fastmcp import Context
from pydantic import Field
from kaede_recall_mcp.common.backend import AsyncClient

from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.memory_os.client import invalidate_entities
from kaede_recall_mcp.memory_os.entities import EntityRegistry
from kaede_recall_mcp.memory_os.recall import recall, recall_by_ids
from kaede_recall_mcp.memory_os.schema import ensure_collection
from kaede_recall_mcp.memory_os.store import MemoryStore

MAX_RECALL_CHARS = 8000


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def register_memory_tools(
    server,
    client: AsyncClient,
    collection: str,
    dense: EmbeddingProvider,
    *,
    read_only: bool = False,
) -> None:
    store = MemoryStore(client, collection, dense)
    registry = EntityRegistry()
    ready = {"value": False}

    async def _ensure() -> None:
        if not ready["value"]:
            await ensure_collection(client, collection)
            await registry.load(client, collection)
            ready["value"] = True

    async def memory_recall(
        ctx: Context,
        query: Annotated[
            str | None,
            Field(description="What to look for, in the user's own words. Optional when ids is set"),
        ] = None,
        kinds: Annotated[
            str | None,
            Field(description="Comma separated: fact,preference,profile,decision,problem,procedure,trace,doc"),
        ] = None,
        scope: Annotated[str | None, Field(description="Exact scope, e.g. global or project:foo")] = None,
        project: Annotated[str | None, Field(description="Project id filter")] = None,
        limit: Annotated[int, Field(description="Max memories, default 5")] = 5,
        max_chars: Annotated[int, Field(description="Hard cap for the block, default 3000")] = 3000,
        walk_depth: Annotated[int, Field(description="Follow relations this many hops, default 0")] = 0,
        include_superseded: Annotated[bool, Field(description="Include replaced memories")] = False,
        ids: Annotated[
            str | None,
            Field(description="Comma separated ids to open directly, skipping the search"),
        ] = None,
    ) -> str:
        """Search long-term memory. Use before answering about past work, configs, decisions, problems or how to do a task."""
        await _ensure()
        if ids:
            hits, block = await recall_by_ids(
                client,
                collection,
                _split(ids) or [],
                max_chars=max(500, min(max_chars, MAX_RECALL_CHARS)),
            )
            return block or "No memory with those ids."
        if not query:
            return "Provide a query or a list of ids."
        hits, block = await recall(
            client,
            collection,
            query,
            dense,
            kinds=_split(kinds),
            scope=scope,
            project=project,
            limit=max(1, min(limit, 20)),
            max_chars=max(500, min(max_chars, MAX_RECALL_CHARS)),
            walk_depth=max(0, min(walk_depth, 2)),
            include_superseded=include_superseded,
            registry=registry,
        )
        if not hits:
            return "No relevant memories."
        return block

    async def memory_write(
        ctx: Context,
        kind: Annotated[
            str,
            Field(description="fact, preference, profile, decision, problem, procedure, trace or doc"),
        ],
        text: Annotated[str, Field(description="The memory itself, self contained")],
        summary: Annotated[str | None, Field(description="One line trigger, optional")] = None,
        scope: Annotated[str, Field(description="global or project:<id>")] = "global",
        project: Annotated[str | None, Field(description="Project id, optional")] = None,
        tags: Annotated[str | None, Field(description="Comma separated tags")] = None,
        entities: Annotated[
            str | None,
            Field(description="Comma separated aliases, e.g. rpi,raspberry-pi,systemd"),
        ] = None,
        links: Annotated[
            str | None,
            Field(description="Comma separated relations, e.g. fixed_by:<id>,depends_on:<id>"),
        ] = None,
        supersedes_id: Annotated[str | None, Field(description="Id of the memory this replaces")] = None,
        source: Annotated[str | None, Field(description="Where it came from")] = None,
    ) -> str:
        """Store one durable memory, optionally linked or replacing an older one."""
        await _ensure()
        ref = await store.write_node(
            kind,
            text,
            summary=summary,
            scope=scope,
            project=project,
            tags=_split(tags),
            entities=_split(entities),
            links=_split(links),
            supersedes_id=supersedes_id,
            source=source,
        )
        if ref.duplicate:
            return f"Already stored, id={ref.id}"
        registry.add(ref.payload.get("entities") or [])
        await invalidate_entities()
        return f"Stored {kind}, id={ref.id}"

    async def memory_update(
        ctx: Context,
        point_id: Annotated[
            str,
            Field(description="The id of the memory to edit, as returned by memory-recall"),
        ],
        text: Annotated[
            str | None,
            Field(
                description="New body. Recomputes the hash, re-extracts entities and "
                "re-embeds summary + first 1200 chars."
            ),
        ] = None,
        summary: Annotated[
            str | None,
            Field(description="New one-line trigger. Re-embeds the memory."),
        ] = None,
        tags: Annotated[
            str | None,
            Field(description="Comma separated tags. Replaces the current tags."),
        ] = None,
        entities: Annotated[
            str | None,
            Field(description="Comma separated aliases. Replaces the current entities."),
        ] = None,
    ) -> str:
        """Edit a stored memory in place. Omitted fields are kept unchanged."""
        await _ensure()
        if text is None and summary is None and tags is None and entities is None:
            return "Nothing to update: provide text, summary, tags or entities."
        try:
            updated = await store.update_node(
                point_id,
                text=text,
                summary=summary,
                tags=_split(tags),
                entities=_split(entities),
            )
        except ValueError as exc:
            return f"Could not update {point_id}: {exc}"
        if updated is None:
            return f"No memory with id '{point_id}'."
        registry.add(updated.get("entities") or [])
        await invalidate_entities()
        return f"Updated {updated.get('kind', 'memory')}, id={point_id}"

    async def memory_delete(
        ctx: Context,
        point_id: Annotated[
            str | None,
            Field(
                description="Exact id to delete. Deterministic and preferred. "
                "Mutually exclusive with 'query'."
            ),
        ] = None,
        query: Annotated[
            str | None,
            Field(
                description="Semantic query used to find the memories to delete. "
                "Mutually exclusive with 'point_id'."
            ),
        ] = None,
        limit: Annotated[int, Field(description="Max memories to delete by query, default 1")] = 1,
        score_threshold: Annotated[
            float | None,
            Field(description="Minimum similarity for a match to be deleted"),
        ] = None,
        dry_run: Annotated[
            bool,
            Field(description="Preview the matches without deleting anything"),
        ] = False,
    ) -> str:
        """Delete a memory by exact id or by semantic match."""
        await _ensure()
        try:
            hits = await store.delete_nodes(
                point_id=point_id,
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                dry_run=dry_run,
            )
        except ValueError as exc:
            return f"Could not delete: {exc}"
        if not hits:
            target = point_id if point_id is not None else query
            return f"No matching memory found for: {target}"
        if not dry_run:
            await invalidate_entities()
        verb = "Would delete" if dry_run else "Deleted"
        lines = [f"{verb} {len(hits)} memory(ies):"]
        for hit in hits:
            score = f" (score {hit['score']:.4f})" if hit.get("score") is not None else ""
            lines.append(f"- id {hit['id']}{score}: {hit['preview']}")
        return "\n".join(lines)

    server.tool(
        memory_recall,
        name="memory-recall",
        description="Search long-term memory (facts, decisions, problems, procedures) with a hard size budget.",
    )
    if read_only:
        return
    server.tool(
        memory_write,
        name="memory-write",
        description="Store one durable memory. Dedupes, links and supersedes in one call.",
    )
    server.tool(
        memory_update,
        name="memory-update",
        description="Edit a stored memory in place: text, summary, tags or entities. Re-embeds when the trigger or the body changes.",
    )
    server.tool(
        memory_delete,
        name="memory-delete",
        description="Delete a memory by exact id or by semantic match, with a dry-run preview. The only way to remove a node.",
    )
