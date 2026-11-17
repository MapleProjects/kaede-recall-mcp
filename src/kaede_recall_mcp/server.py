from kaede_recall_mcp.mcp_server import KaedeRecallMCPServer
from kaede_recall_mcp.settings import (
    EmbeddingProviderSettings,
    KaedeSettings,
)

mcp = KaedeRecallMCPServer(
    backend_settings=KaedeSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)