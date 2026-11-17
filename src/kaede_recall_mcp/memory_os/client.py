"""Client side of the memory OS daemon.

Writes happen in a stdio MCP process, but the daemon owns the entity registry
and the recall cache. After a write the client tells the daemon that its view is
stale, so the next query sees the new aliases instead of waiting for the TTL.

The call is best effort and has a circuit breaker: when the daemon is not there,
the first failure disables the notification for the life of the process, so a
write never pays the timeout twice.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SERVICE_URL = os.environ.get("MEMORY_OS_EMBED_URL", "http://127.0.0.1:7451")
NOTIFY_TIMEOUT = float(os.environ.get("MEMORY_OS_NOTIFY_TIMEOUT", "0.6"))

_disabled = False


async def invalidate_entities(timeout: float = NOTIFY_TIMEOUT) -> bool:
    """Ask the daemon to reload entities and drop its recall cache."""
    global _disabled
    if _disabled:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{SERVICE_URL.rstrip('/')}/invalidate")
            return response.status_code == 200
    except Exception as exc:
        _disabled = True
        logger.info("memory service unavailable, entity invalidation disabled: %s", exc)
        return False
