"""HTTP recall and embedding service.

The model loads once. The hook posts a query and gets a budgeted block back in
milliseconds instead of paying a process start and an ONNX load per turn, and
every MCP stdio process asks this daemon for vectors instead of loading its own
copy of the model.

Endpoints:
  GET  /health      liveness, collection, vector size and cache stats
  POST /recall      {"query", "max_chars", "limit"} -> block and top scores
  POST /embed       {"texts": [...], "kind": "document"|"query"} -> vectors
  POST /invalidate  drop the entity registry and the recall cache after a write
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kaede_recall_mcp.common.backend import AsyncClient

from kaede_recall_mcp.embeddings.fastembed import FastEmbedProvider
from kaede_recall_mcp.memory_os.entities import EntityRegistry
from kaede_recall_mcp.memory_os.recall import recall

KAEDE_BACKEND_URL = os.environ.get("KAEDE_BACKEND_URL", "http://localhost:6333")
COLLECTION = os.environ.get("MEMORY_OS_COLLECTION", "kaede-memory")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es")
HOST = os.environ.get("MEMORY_OS_HOST", "127.0.0.1")
PORT = int(os.environ.get("MEMORY_OS_PORT", "7451"))

# The recall cache lives here, in memory, instead of a JSON file on disk. One
# process owns it, so there is no lock file, no partial write and no stale file
# after a crash.
CACHE_TTL = float(os.environ.get("MEMORY_OS_CACHE_TTL", "900"))
CACHE_MAX = int(os.environ.get("MEMORY_OS_CACHE_MAX", "256"))
REGISTRY_TTL = float(os.environ.get("MEMORY_OS_REGISTRY_TTL", "60"))

logger = logging.getLogger("memory-os-service")


class Runtime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True, name="memory-os-loop")
        self.ready = threading.Event()
        self.client: AsyncClient | None = None
        self.dense: FastEmbedProvider | None = None
        self.registry = EntityRegistry()
        self.loaded_at = 0.0
        # One lock for every registry reload: the HTTP threads and the recall
        # path can both ask for a refresh, and two concurrent loads would race
        # on the compiled pattern. Reentrant so invalidate() can reload while
        # holding it.
        self._reload_lock = threading.RLock()
        self._dirty = threading.Event()
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(120):
            raise RuntimeError("memory service did not initialise in time")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._init())
        self.ready.set()
        self.loop.run_forever()

    async def _init(self) -> None:
        self.client = AsyncClient(url=KAEDE_BACKEND_URL, check_compatibility=False)
        self.dense = FastEmbedProvider(EMBEDDING_MODEL)
        await self.registry.load(self.client, COLLECTION)
        self.loaded_at = time.time()

    # -- entity registry -------------------------------------------------
    def invalidate(self) -> None:
        """Drop the entity registry and the recall cache, then reload.

        Both live under the reload lock, so a concurrent recall either sees the
        old snapshot or the new one, never a half-cleared registry.
        """
        with self._reload_lock:
            with self._cache_lock:
                self._cache.clear()
            self._dirty.set()
            self._reload_registry(force=True)

    def _reload_registry(self, force: bool = False) -> None:
        with self._reload_lock:
            fresh = time.time() - self.loaded_at <= REGISTRY_TTL
            if not force and not self._dirty.is_set() and fresh:
                return
            assert self.client is not None
            self.registry.reset()
            future = asyncio.run_coroutine_threadsafe(
                self.registry.load(self.client, COLLECTION), self.loop
            )
            try:
                future.result(timeout=15)
                self.loaded_at = time.time()
                self._dirty.clear()
            except Exception:
                logger.warning("entity registry reload failed", exc_info=True)

    # -- recall cache ----------------------------------------------------
    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            stamp, value = entry
            if time.time() - stamp > CACHE_TTL:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = (time.time(), value)
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_MAX:
                self._cache.popitem(last=False)

    # -- work ------------------------------------------------------------
    def recall(self, query: str, max_chars: int, limit: int) -> dict[str, Any]:
        assert self.client is not None and self.dense is not None
        limit = max(1, min(limit, 10))
        max_chars = max(500, min(max_chars, 6000))
        key = f"{limit}:{max_chars}:{query}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self._reload_registry()
        future = asyncio.run_coroutine_threadsafe(
            recall(
                self.client,
                COLLECTION,
                query,
                self.dense,
                limit=limit,
                max_chars=max_chars,
                registry=self.registry,
            ),
            self.loop,
        )
        hits, block = future.result(timeout=20)
        top = hits[0] if hits else None
        result = {
            "block": block,
            "top_score": float(top.score) if top else 0.0,
            "top_base": float(top.base) if top else 0.0,
            "top_kind": top.kind if top else "",
            "top_id": top.id if top else "",
            "entities": self.registry.match(query),
        }
        self._cache_put(key, result)
        return result

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        assert self.dense is not None
        if not texts:
            return []
        if kind == "query":
            coro = self._embed_queries(texts)
        else:
            coro = self.dense.embed_documents(texts)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=60)

    async def _embed_queries(self, texts: list[str]) -> list[list[float]]:
        assert self.dense is not None
        return [await self.dense.embed_query(text) for text in texts]

    def info(self) -> dict[str, Any]:
        size = 0
        name = ""
        if self.dense is not None:
            try:
                size = self.dense.get_vector_size()
                name = self.dense.get_vector_name()
            except Exception:
                pass
        with self._cache_lock:
            cached = len(self._cache)
        return {
            "ok": True,
            "collection": COLLECTION,
            "model": EMBEDDING_MODEL,
            "vector_size": size,
            "vector_name": name,
            "entities": len(self.registry.values),
            "cache_entries": cached,
            "cache_ttl": CACHE_TTL,
        }


RUNTIME = Runtime()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, RUNTIME.info())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/recall":
                payload = self._body()
                query = str(payload.get("query", "")).strip()
                if not query:
                    self._send(400, {"error": "query is required"})
                    return
                result = RUNTIME.recall(
                    query,
                    int(payload.get("max_chars", 1800)),
                    int(payload.get("limit", 4)),
                )
                self._send(200, result)
                return

            if self.path == "/embed":
                payload = self._body()
                texts = payload.get("texts") or []
                if isinstance(texts, str):
                    texts = [texts]
                texts = [str(text) for text in texts]
                kind = str(payload.get("kind", "document")).lower()
                vectors = RUNTIME.embed(texts, "query" if kind == "query" else "document")
                self._send(200, {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0})
                return

            if self.path == "/invalidate":
                RUNTIME.invalidate()
                self._send(200, {"ok": True})
                return

            self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover - service boundary
            logger.exception("request failed: %s", self.path)
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        logger.debug(fmt, *args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RUNTIME.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    logger.info("memory-os service on %s:%s collection=%s", HOST, PORT, COLLECTION)
    server.serve_forever()


if __name__ == "__main__":
    main()
