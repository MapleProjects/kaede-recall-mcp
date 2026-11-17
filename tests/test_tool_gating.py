"""The memory OS must not expose a second, untyped write path."""

import pytest

from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.mcp_server import KaedeRecallMCPServer
from kaede_recall_mcp.settings import KaedeSettings


class FakeDense(EmbeddingProvider):
    async def embed_documents(self, documents):
        return [[0.1] * 8 for _ in documents]

    async def embed_query(self, query):
        return [0.1] * 8

    def get_vector_name(self):
        return "dense"

    def get_vector_size(self):
        return 8


def _server(read_only: bool = False) -> KaedeRecallMCPServer:
    return KaedeRecallMCPServer(
        backend_settings=KaedeSettings(
            KAEDE_BACKEND_URL=":memory:",
            COLLECTION_NAME="test",
            KAEDE_READ_ONLY="true" if read_only else "false",
        ),
        embedding_provider=FakeDense(),
    )


@pytest.mark.asyncio
async def test_server_exposes_four_tools():
    tools = await _server().get_tools()
    assert set(tools) == {
        "memory-recall",
        "memory-write",
        "memory-update",
        "memory-delete",
    }


@pytest.mark.asyncio
async def test_read_only_exposes_recall_only():
    tools = await _server(read_only=True).get_tools()
    assert set(tools) == {"memory-recall"}

