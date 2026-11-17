import os

from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.embeddings.types import EmbeddingProviderType
from kaede_recall_mcp.settings import EmbeddingProviderSettings

_OFF = {"0", "false", "no", "off"}


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        # The daemon owns the model. A client that loads its own copy costs
        # about 878 MB of resident memory for vectors it could ask for over
        # loopback in a few milliseconds.
        remote_enabled = os.environ.get("MEMORY_OS_EMBED_REMOTE", "true").lower() not in _OFF
        if remote_enabled:
            from kaede_recall_mcp.embeddings.remote import (
                DEFAULT_SERVICE_URL,
                ServiceEmbeddingProvider,
            )

            url = os.environ.get("MEMORY_OS_EMBED_URL", DEFAULT_SERVICE_URL)
            provider = ServiceEmbeddingProvider(url, settings.model_name)
            if provider.available:
                return provider

        from kaede_recall_mcp.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
