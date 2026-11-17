import uuid

import pytest

from kaede_recall_mcp.connector import Entry, KaedeConnector, content_fingerprint

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


def test_fingerprint_ignores_whitespace_and_case():
    a = content_fingerprint("Hello   World", {"k": 1})
    b = content_fingerprint("hello world", {"k": 1})
    assert a == b


def test_fingerprint_ignores_metadata_key_order():
    a = content_fingerprint("text", {"a": 1, "b": 2})
    b = content_fingerprint("text", {"b": 2, "a": 1})
    assert a == b


def test_fingerprint_changes_with_metadata():
    a = content_fingerprint("text", {"a": 1})
    b = content_fingerprint("text", {"a": 2})
    assert a != b


@pytest.mark.asyncio
async def test_store_deduplicates_identical_memory(connector):
    first = await connector.store(
        Entry(content="The same fact", metadata={"k": "v"})
    )
    second = await connector.store(
        Entry(content="The same fact", metadata={"k": "v"})
    )

    assert second.duplicate is True
    assert second.id == first.id
    assert await connector.count() == 1


@pytest.mark.asyncio
async def test_store_keeps_same_content_with_different_metadata(connector):
    first = await connector.store(
        Entry(content="The same fact", metadata={"k": "v1"})
    )
    second = await connector.store(
        Entry(content="The same fact", metadata={"k": "v2"})
    )

    assert second.duplicate is False
    assert second.id != first.id
    assert await connector.count() == 2


@pytest.mark.asyncio
async def test_store_sets_timestamps(connector):
    stored = await connector.store(Entry(content="Timestamped memory"))

    assert stored.created_at
    assert stored.updated_at
    assert stored.created_at == stored.updated_at

    fetched = await connector.get(stored.id)
    assert fetched.created_at == stored.created_at


@pytest.mark.asyncio
async def test_update_moves_updated_at_and_keeps_created_at(connector):
    stored = await connector.store(Entry(content="Original"))

    updated = await connector.update(stored.id, content="Changed")

    assert updated.created_at == stored.created_at
    assert updated.updated_at >= stored.updated_at


@pytest.mark.asyncio
async def test_update_refreshes_fingerprint(connector):
    stored = await connector.store(Entry(content="Original"))

    await connector.update(stored.id, content="Changed")

    # Storing the old text again must create a new point, not match the updated one.
    again = await connector.store(Entry(content="Original"))
    assert again.duplicate is False
    assert await connector.count() == 2


@pytest.mark.asyncio
async def test_list_entries_pages_through_collection(connector):
    for i in range(5):
        await connector.store(Entry(content=f"Memory number {i}"))

    page1, cursor = await connector.list_entries(limit=2)
    assert len(page1) == 2
    assert cursor is not None

    page2, cursor2 = await connector.list_entries(limit=2, offset=cursor)
    assert len(page2) == 2
    assert {e.id for e in page1}.isdisjoint({e.id for e in page2})

    page3, cursor3 = await connector.list_entries(limit=2, offset=cursor2)
    assert len(page3) == 1
    assert cursor3 is None


@pytest.mark.asyncio
async def test_count_empty_collection(connector):
    assert await connector.count() == 0


@pytest.mark.asyncio
async def test_dedupe_disabled_still_idempotent_by_id(embedding_provider):
    connector = KaedeConnector(
        backend_url=":memory:",
        backend_api_key=None,
        collection_name=f"test_collection_{uuid.uuid4().hex}",
        embedding_provider=embedding_provider,
        dedupe=False,
    )

    # Without the dedupe lookup, the deterministic uuid5 still maps identical
    # content to the same point, so a repeated write is idempotent.
    first = await connector.store(Entry(content="Repeated"))
    second = await connector.store(Entry(content="Repeated"))
    await connector.store(Entry(content="Different"))

    assert first.id == second.id
    assert await connector.count() == 2
