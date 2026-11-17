import uuid

import pytest

from kaede_recall_mcp.connector import Entry, KaedeConnector

from .support import DeterministicEmbedding


@pytest.fixture
async def embedding_provider():
    return DeterministicEmbedding()


@pytest.fixture
async def connector(embedding_provider):
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    connector = KaedeConnector(
        backend_url=":memory:",
        backend_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    yield connector


@pytest.mark.asyncio
async def test_store_returns_point_id(connector):
    stored = await connector.store(Entry(content="An entry with an id"))

    assert stored.id
    assert stored.content == "An entry with an id"

    fetched = await connector.get(stored.id)
    assert fetched is not None
    assert fetched.content == "An entry with an id"


@pytest.mark.asyncio
async def test_store_rejects_empty_document(connector):
    with pytest.raises(ValueError, match="empty document"):
        await connector.store(Entry(content="   "))


@pytest.mark.asyncio
async def test_search_returns_scores(connector):
    await connector.store(Entry(content="Python is a programming language"))

    [result] = await connector.search("Python programming")
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_search_score_threshold_filters_results(connector):
    await connector.store(Entry(content="Python is a programming language"))

    results = await connector.search(
        "completely unrelated topic about cooking", score_threshold=0.99
    )
    assert results == []


@pytest.mark.asyncio
async def test_get_unknown_id_returns_none(connector):
    await connector.store(Entry(content="Seed"))

    assert await connector.get(uuid.uuid4().hex) is None


@pytest.mark.asyncio
async def test_get_invalid_id_raises_value_error(connector):
    await connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="not a valid point id"):
        await connector.get("not-a-uuid")


@pytest.mark.asyncio
async def test_delete_by_point_id(connector):
    first = await connector.store(Entry(content="First memory"))
    await connector.store(Entry(content="Second memory"))

    deleted = await connector.delete(point_id=first.id)

    assert len(deleted) == 1
    assert deleted[0].id == first.id
    assert deleted[0].content == "First memory"

    remaining = await connector.search("memory", limit=10)
    assert len(remaining) == 1
    assert remaining[0].content == "Second memory"


@pytest.mark.asyncio
async def test_delete_by_point_id_unknown_returns_empty(connector):
    await connector.store(Entry(content="Seed"))

    assert await connector.delete(point_id=uuid.uuid4().hex) == []


@pytest.mark.asyncio
async def test_delete_requires_exactly_one_selector(connector):
    with pytest.raises(ValueError, match="exactly one"):
        await connector.delete()

    with pytest.raises(ValueError, match="exactly one"):
        await connector.delete("query", point_id=uuid.uuid4().hex)


@pytest.mark.asyncio
async def test_delete_dry_run_keeps_entries(connector):
    await connector.store(Entry(content="Python is a programming language"))

    preview = await connector.delete("Python programming", dry_run=True)

    assert len(preview) == 1
    assert preview[0].score is not None

    still_there = await connector.search("Python programming")
    assert len(still_there) == 1


@pytest.mark.asyncio
async def test_delete_by_query_respects_limit(connector):
    await connector.store(Entry(content="Python is a programming language"))
    await connector.store(Entry(content="Python is also a snake"))
    await connector.store(Entry(content="The Eiffel Tower is in Paris"))

    deleted = await connector.delete("Python", limit=2)

    assert len(deleted) == 2
    remaining = await connector.search("anything", limit=10)
    assert len(remaining) == 1
    assert "Eiffel" in remaining[0].content


@pytest.mark.asyncio
async def test_delete_by_query_respects_score_threshold(connector):
    await connector.store(Entry(content="Python is a programming language"))

    deleted = await connector.delete(
        "completely unrelated cooking topic", score_threshold=0.99
    )

    assert deleted == []
    assert len(await connector.search("Python", limit=10)) == 1


@pytest.mark.asyncio
async def test_update_returns_updated_entry(connector):
    stored = await connector.store(
        Entry(content="Original content", metadata={"status": "draft"})
    )

    updated = await connector.update(
        stored.id, content="Revised content", metadata={"status": "final"}
    )

    assert updated.id == stored.id
    assert updated.content == "Revised content"
    assert updated.metadata == {"status": "final"}

    fetched = await connector.get(stored.id)
    assert fetched.content == "Revised content"
    assert fetched.metadata == {"status": "final"}


@pytest.mark.asyncio
async def test_update_clear_metadata(connector):
    stored = await connector.store(
        Entry(content="Keep me", metadata={"status": "draft"})
    )

    updated = await connector.update(stored.id, clear_metadata=True)

    assert updated.metadata is None
    assert updated.content == "Keep me"


@pytest.mark.asyncio
async def test_update_rejects_metadata_and_clear_together(connector):
    stored = await connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="not both"):
        await connector.update(stored.id, metadata={"a": 1}, clear_metadata=True)


@pytest.mark.asyncio
async def test_update_rejects_empty_content(connector):
    stored = await connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="empty document"):
        await connector.update(stored.id, content="  ")


@pytest.mark.asyncio
async def test_update_keeps_unknown_payload_keys(connector):
    stored = await connector.store(Entry(content="Seed"))

    # Simulate a payload written by another tool.
    await connector._client.set_payload(
        collection_name=connector._default_collection_name,
        payload={"extra": "kept"},
        points=[stored.id],
    )

    await connector.update(stored.id, content="New body")

    points = await connector._client.retrieve(
        collection_name=connector._default_collection_name,
        ids=[stored.id],
        with_payload=True,
    )
    assert points[0].payload["extra"] == "kept"
    assert points[0].payload["document"] == "New body"
