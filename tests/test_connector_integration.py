import uuid

import pytest

from kaede_recall_mcp.connector import Entry, KaedeConnector

from .support import DeterministicEmbedding


@pytest.fixture
async def embedding_provider():
    """Offline deterministic embeddings; no model download."""
    return DeterministicEmbedding()


@pytest.fixture
async def connector(embedding_provider):
    """Fixture to provide a KaedeConnector with in-memory the backend client."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory the backend
    connector = KaedeConnector(
        backend_url=":memory:",
        backend_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )

    yield connector


@pytest.mark.asyncio
async def test_store_and_search(connector):
    """Test storing an entry and then searching for it."""
    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    await connector.store(test_entry)

    # Search for the entry
    results = await connector.search("fox jumps")

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(connector):
    """Test searching in an empty collection."""
    # Search in an empty collection
    results = await connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(connector):
    """Test storing and searching multiple entries."""
    # Store multiple entries
    entries = [
        Entry(
            content="Python is a programming language",
            metadata={"topic": "programming"},
        ),
        Entry(content="The Eiffel Tower is in Paris", metadata={"topic": "landmarks"}),
        Entry(content="Machine learning is a subset of AI", metadata={"topic": "AI"}),
    ]

    for entry in entries:
        await connector.store(entry)

    # Search for programming-related entries
    programming_results = await connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await connector.search(
        "artificial intelligence machine learning"
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_search_returns_point_id(connector):
    """Test that searched entries carry the point id needed to update them."""
    await connector.store(Entry(content="An entry with a retrievable id"))

    results = await connector.search("retrievable id")

    assert len(results) == 1
    assert results[0].id is not None


@pytest.mark.asyncio
async def test_update_content_and_metadata(connector):
    """Test updating both the content and metadata of an existing entry."""
    await connector.store(
        Entry(content="Original content", metadata={"status": "draft"})
    )
    [stored] = await connector.search("Original content")

    await connector.update(
        stored.id,
        content="Revised content",
        metadata={"status": "final"},
    )

    [updated] = await connector.search("Revised content")
    assert updated.id == stored.id
    assert updated.content == "Revised content"
    assert updated.metadata == {"status": "final"}


@pytest.mark.asyncio
async def test_update_metadata_only_preserves_content(connector):
    """Test that a metadata-only update leaves the content untouched."""
    await connector.store(
        Entry(content="Keep this content", metadata={"status": "draft"})
    )
    [stored] = await connector.search("Keep this content")

    await connector.update(stored.id, metadata={"status": "final"})

    [updated] = await connector.search("Keep this content")
    assert updated.content == "Keep this content"
    assert updated.metadata == {"status": "final"}


@pytest.mark.asyncio
async def test_update_unknown_id_raises(connector):
    """Test that updating a non-existent id raises rather than silently inserting."""
    # Create the collection first so the failure is specifically about the missing id.
    await connector.store(Entry(content="Seed entry"))

    with pytest.raises(ValueError):
        await connector.update(
            uuid.uuid4().hex, content="Should not be created"
        )


@pytest.mark.asyncio
async def test_update_requires_a_field(connector):
    """Test that an update with nothing to change raises."""
    await connector.store(Entry(content="Seed entry"))
    [stored] = await connector.search("Seed entry")

    with pytest.raises(ValueError):
        await connector.update(stored.id)


@pytest.mark.asyncio
async def test_ensure_collection_exists(connector):
    """Test that the collection is created if it doesn't exist."""
    # The collection shouldn't exist yet
    assert not await connector._client.collection_exists(
        connector._default_collection_name
    )

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await connector.store(test_entry)

    # Now the collection should exist
    assert await connector._client.collection_exists(
        connector._default_collection_name
    )


@pytest.mark.asyncio
async def test_metadata_handling(connector):
    """Test that metadata is properly stored and retrieved."""
    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1)
    )
    await connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2)
    )

    # Search and verify metadata is preserved
    results = await connector.search("metadata")

    assert len(results) == 2

    # Check that both metadata objects are present in the results
    found_metadata1 = False
    found_metadata2 = False

    for result in results:
        if result.metadata.get("source") == "book":
            assert result.metadata.get("author") == "Jane Doe"
            assert result.metadata.get("year") == 2023
            found_metadata1 = True
        elif result.metadata.get("source") == "article":
            assert "science" in result.metadata.get("tags", [])
            assert "research" in result.metadata.get("tags", [])
            found_metadata2 = True

    assert found_metadata1
    assert found_metadata2


@pytest.mark.asyncio
async def test_entry_without_metadata(connector):
    """Test storing and retrieving entries without metadata."""
    # Store an entry without metadata
    await connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_custom_collection_store_and_search(connector):
    """Test storing and searching in a custom collection."""
    # Define a custom collection name
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    # Store a test entry in the custom collection
    test_entry = Entry(
        content="This is stored in a custom collection",
        metadata={"custom": True},
    )
    await connector.store(test_entry, collection_name=custom_collection)

    # Search in the custom collection
    results = await connector.search(
        "custom collection", collection_name=custom_collection
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata

    # Verify the entry is not in the default collection
    default_results = await connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_multiple_collections(connector):
    """Test using multiple collections with the same connector."""
    # Define two custom collection names
    collection_a = f"collection_a_{uuid.uuid4().hex}"
    collection_b = f"collection_b_{uuid.uuid4().hex}"

    # Store entries in different collections
    entry_a = Entry(
        content="This belongs to collection A", metadata={"collection": "A"}
    )
    entry_b = Entry(
        content="This belongs to collection B", metadata={"collection": "B"}
    )
    entry_default = Entry(content="This belongs to the default collection")

    await connector.store(entry_a, collection_name=collection_a)
    await connector.store(entry_b, collection_name=collection_b)
    await connector.store(entry_default)

    # Search in collection A
    results_a = await connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].content == entry_a.content

    # Search in collection B
    results_b = await connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].content == entry_b.content

    # Search in default collection
    results_default = await connector.search("belongs")
    assert len(results_default) == 1
    assert results_default[0].content == entry_default.content


@pytest.mark.asyncio
async def test_nonexistent_collection_search(connector):
    """Test searching in a collection that doesn't exist."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete_entry(connector):
    """Test storing an entry and then deleting it."""
    test_entry = Entry(
        content="This memory should be deleted",
        metadata={"temporary": True},
    )
    await connector.store(test_entry)

    # Verify it exists
    results = await connector.search("memory should be deleted")
    assert len(results) == 1

    # Delete it
    deleted = await connector.delete("memory should be deleted")
    assert len(deleted) == 1
    assert deleted[0].content == test_entry.content

    # Verify it's gone
    results = await connector.search("memory should be deleted")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_collection(connector):
    """Test deleting from a collection that doesn't exist."""
    nonexistent = f"nonexistent_{uuid.uuid4().hex}"
    deleted = await connector.delete(
        "test query", collection_name=nonexistent
    )
    assert len(deleted) == 0


@pytest.mark.asyncio
async def test_delete_no_match_empty_collection(connector):
    """Test deleting from an empty collection."""
    deleted = await connector.delete("nonexistent memory")
    assert len(deleted) == 0


@pytest.mark.asyncio
async def test_delete_preserves_other_entries(connector):
    """Test that delete only removes the closest match."""
    await connector.store(Entry(content="Python is a programming language"))
    await connector.store(Entry(content="The Eiffel Tower is in Paris"))

    # Delete only the Python entry
    deleted = await connector.delete("Python programming")
    assert len(deleted) == 1
    assert "Python" in deleted[0].content

    # Eiffel Tower entry should still exist
    results = await connector.search("Eiffel Tower Paris")
    assert len(results) == 1
    assert "Eiffel" in results[0].content


@pytest.mark.asyncio
async def test_delete_custom_collection(connector):
    """Test deleting from a custom collection."""
    custom_collection = f"custom_{uuid.uuid4().hex}"

    await connector.store(
        Entry(content="Custom collection entry"),
        collection_name=custom_collection,
    )

    deleted = await connector.delete(
        "custom collection", collection_name=custom_collection
    )
    assert len(deleted) == 1

    results = await connector.search(
        "custom collection", collection_name=custom_collection
    )
    assert len(results) == 0
