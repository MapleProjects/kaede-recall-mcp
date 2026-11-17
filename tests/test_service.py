"""Tests for the HTTP daemon and the client-side embedding provider.

The model is never loaded here: the runtime is patched, so these tests cover
the transport, the cache and the invalidation signal, not FastEmbed.
"""

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from kaede_recall_mcp.embeddings.remote import ServiceEmbeddingProvider
from kaede_recall_mcp.memory_os import service


@pytest.fixture
def daemon(monkeypatch):
    """Run the real handler on an ephemeral port with a patched runtime."""
    reloads = []
    monkeypatch.setattr(
        service.RUNTIME, "_reload_registry", lambda force=False: reloads.append(force)
    )
    setattr(service.RUNTIME, "_test_reloads", reloads)
    monkeypatch.setattr(service.RUNTIME, "embed", lambda texts, kind: [[0.5, 0.25] for _ in texts])
    monkeypatch.setattr(
        service.RUNTIME,
        "recall",
        lambda query, max_chars, limit: {
            "block": f"hit for {query}",
            "top_score": 0.5,
            "top_base": 0.45,
            "top_kind": "fact",
            "top_id": "abc",
            "entities": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _post(url, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_health_reports_collection(daemon):
    with urllib.request.urlopen(f"{daemon}/health", timeout=5) as response:
        data = json.loads(response.read())
    assert data["ok"] is True
    assert "collection" in data


def test_embed_returns_one_vector_per_text(daemon):
    data = _post(f"{daemon}/embed", {"texts": ["a", "b"], "kind": "document"})
    assert data["vectors"] == [[0.5, 0.25], [0.5, 0.25]]
    assert data["dim"] == 2


def test_invalidate_clears_cache_and_reloads(daemon):
    service.RUNTIME._cache_put("k", {"block": "v"})
    data = _post(f"{daemon}/invalidate", {})
    assert data["ok"] is True
    assert service.RUNTIME._cache_get("k") is None
    assert getattr(service.RUNTIME, "_test_reloads")[-1] is True


def test_recall_returns_block(daemon):
    data = _post(f"{daemon}/recall", {"query": "hola", "max_chars": 500, "limit": 2})
    assert data["block"] == "hit for hola"


def test_cache_is_bounded_and_expires():
    runtime = service.Runtime()
    runtime._cache_put("k", {"block": "v"})
    assert runtime._cache_get("k") == {"block": "v"}
    runtime._cache.clear()
    runtime._cache["old"] = (time.time() - service.CACHE_TTL - 1, {"block": "x"})
    assert runtime._cache_get("old") is None


def test_remote_provider_uses_the_daemon(daemon):
    provider = ServiceEmbeddingProvider(daemon, "fake-model")
    assert provider.available is True
    assert provider.get_vector_name() == "fast-fake-model"


async def test_remote_provider_embeds_through_daemon(daemon):
    provider = ServiceEmbeddingProvider(daemon, "fake-model")
    assert await provider.embed_documents(["a"]) == [[0.5, 0.25]]
    assert await provider.embed_query("q") == [0.5, 0.25]


def test_remote_provider_falls_back_when_daemon_is_down():
    provider = ServiceEmbeddingProvider("http://127.0.0.1:1", "fake-model")
    assert provider.available is False
