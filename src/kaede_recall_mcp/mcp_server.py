"""MCP server for the Kaede memory OS.

One collection, one write path, four tools: memory-recall, memory-write,
memory-update and memory-delete. The raw document tools (find, get, list,
store) are gone; they wrote a second, untyped copy of the same memory and
split the agent's view of what it knows.
"""

import logging
from typing import Any, Optional

from fastmcp import FastMCP

from kaede_recall_mcp.common.filters import make_indexes
from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.embeddings.factory import create_embedding_provider
from kaede_recall_mcp.connector import KaedeConnector
from kaede_recall_mcp.settings import (
    EmbeddingProviderSettings,
    KaedeSettings,
)

logger = logging.getLogger(__name__)


class KaedeRecallMCPServer(FastMCP):
    """A MCP server for the vector backend with the typed memory layer only."""

    def __init__(
        self,
        backend_settings: KaedeSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "kaede-recall-mcp",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.backend_settings = backend_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.connector = KaedeConnector(
            backend_settings.location,
            backend_settings.api_key,
            backend_settings.collection_name,
            self.embedding_provider,
            backend_settings.local_path,
            make_indexes(backend_settings.filterable_fields_dict()),
            dedupe=backend_settings.dedupe,
            retries=backend_settings.retries,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def setup_tools(self):
        """Register the four memory tools."""
        from kaede_recall_mcp.memory_os.schema import DEFAULT_COLLECTION
        from kaede_recall_mcp.memory_os.tools import register_memory_tools

        collection = self.backend_settings.collection_name or DEFAULT_COLLECTION
        register_memory_tools(
            self,
            self.connector.client,
            collection,
            self.embedding_provider,
            read_only=self.backend_settings.read_only,
        )