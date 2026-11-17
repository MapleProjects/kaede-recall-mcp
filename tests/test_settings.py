import pytest

from kaede_recall_mcp.embeddings.types import EmbeddingProviderType
from kaede_recall_mcp.settings import (
    EmbeddingProviderSettings,
    KaedeSettings,
)


class TestKaedeSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        KaedeSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("KAEDE_BACKEND_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = KaedeSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("KAEDE_BACKEND_URL", "http://backend.example.com:6333")
        monkeypatch.setenv("KAEDE_BACKEND_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("KAEDE_SEARCH_LIMIT", "15")
        monkeypatch.setenv("KAEDE_READ_ONLY", "1")

        settings = KaedeSettings()
        assert settings.location == "http://backend.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("KAEDE_BACKEND_PATH", "/path/to/local/backend")

        settings = KaedeSettings()
        assert settings.local_path == "/path/to/local/backend"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if the backend URL is provided."""
        monkeypatch.setenv("KAEDE_BACKEND_URL", "http://localhost:6333")
        monkeypatch.setenv("KAEDE_BACKEND_PATH", "/path/to/local/backend")

        with pytest.raises(ValueError):
            KaedeSettings()

        monkeypatch.delenv("KAEDE_BACKEND_URL", raising=False)
        monkeypatch.setenv("KAEDE_BACKEND_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            KaedeSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"
