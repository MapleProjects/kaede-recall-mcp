"""Tool-level tests: every mutation must tell the daemon to invalidate."""

import pytest
from fastmcp import Client

from kaede_recall_mcp.mcp_server import KaedeRecallMCPServer
from kaede_recall_mcp.memory_os import tools as memory_tools
from kaede_recall_mcp.settings import KaedeSettings

from .support import DeterministicEmbedding


def _server() -> KaedeRecallMCPServer:
    return KaedeRecallMCPServer(
        backend_settings=KaedeSettings(
            KAEDE_BACKEND_URL=":memory:", COLLECTION_NAME="tool-test"
        ),
        embedding_provider=DeterministicEmbedding(size=768),
    )


async def _call(server: KaedeRecallMCPServer, name: str, arguments: dict) -> str:
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
        if isinstance(result, list):
            return "".join(getattr(item, "text", "") for item in result)
        content = getattr(result, "content", None) or []
        return "".join(getattr(item, "text", "") for item in content)


def _point_id(text: str) -> str:
    return text.split("id=")[-1].rstrip("')] ").strip()


@pytest.fixture
def invalidations(monkeypatch):
    calls: list[bool] = []

    async def fake_invalidate() -> bool:
        calls.append(True)
        return True

    monkeypatch.setattr(memory_tools, "invalidate_entities", fake_invalidate)
    return calls


async def test_memory_write_invalidates(invalidations):
    server = _server()
    await _call(server, "memory-write", {"kind": "fact", "text": "una memoria nueva de prueba"})
    assert invalidations


async def test_memory_update_invalidates(invalidations):
    server = _server()
    stored = await _call(server, "memory-write", {"kind": "fact", "text": "memoria para editar"})
    invalidations.clear()
    await _call(server, "memory-update", {"point_id": _point_id(stored), "tags": "uno,dos"})
    assert invalidations


async def test_memory_delete_dry_run_does_not_invalidate(invalidations):
    server = _server()
    stored = await _call(server, "memory-write", {"kind": "fact", "text": "memoria para borrar"})
    invalidations.clear()
    await _call(server, "memory-delete", {"point_id": _point_id(stored), "dry_run": True})
    assert invalidations == []


async def test_memory_delete_invalidates(invalidations):
    server = _server()
    stored = await _call(server, "memory-write", {"kind": "fact", "text": "memoria borrada de verdad"})
    invalidations.clear()
    await _call(server, "memory-delete", {"point_id": _point_id(stored)})
    assert invalidations