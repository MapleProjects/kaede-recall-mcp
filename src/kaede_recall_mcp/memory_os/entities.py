"""Dynamic entity registry.

No project names, synonyms or language-specific words live in this module.
Aliases live in the `entities` payload of each point. The registry is
built from that field at startup and grows on every write.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

BONUS_FIRST = 0.04
BONUS_SECOND = 0.02
BONUS_CAP = 0.06
BONUS_FLOOR = 0.30

_IDENT_PATTERNS = (
    re.compile(r"[/\\][\w./\\-]{2,}"),                       # paths
    re.compile(r"\b[\w.-]+/[\w.-]+\b"),                       # owner/repo
    re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b"),         # kebab-case
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),         # snake_case
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),     # CamelCase
    re.compile(r"\b[a-zA-Z]+\d+[a-zA-Z0-9]*\b"),              # esp32, v3, h264
    re.compile(r"\b[A-Z]{2,}\b"),                             # acronyms
)
_EDGE = re.compile(r"^[\s.,;:!?()\[\]{}'\"`]+|[\s.,;:!?()\[\]{}'\"`]+$")
_WS = re.compile(r"\s+")

# Noise that must never become an entity: pure numbers, measurements and
# hexadecimal digests. They are unique per document, so they add no signal and
# they pollute the entity slot with tokens nobody queries for.
_NUMERIC = re.compile(r"^\d+(?:[.,]\d+)*$")
_HEX = re.compile(r"^(?:0x)?[0-9a-f]{7,}$", re.IGNORECASE)
_UNIT = re.compile(
    r"^\d+(?:[.,]\d+)?\s*"
    r"(?:ms|us|ns|s|sec|secs|min|mins|h|hr|hrs|kb|mb|gb|tb|kib|mib|gib|tib|"
    r"b|bit|bits|px|fps|hz|khz|mhz|ghz|w|kw|v|mv|a|ma|mah|wh|kwh|mm|cm|km|"
    r"kg|mg|ml|l|mbps|gbps|kbps|rpm|dpi|ppi|%|ºc|°c|c|f)$",
    re.IGNORECASE,
)

# A facet value present in more than this share of the collection is a stop
# word, not an entity. `http`, `json` or `api` appear everywhere, so matching
# them would hand the entity slot to an arbitrary document.
DEFAULT_MAX_ENTITY_SHARE = 0.12


def _is_noise(token: str) -> bool:
    if not token or len(token) < 2:
        return True
    if token.isdigit() or _NUMERIC.match(token):
        return True
    if _HEX.match(token):
        return True
    if _UNIT.match(token):
        return True
    return False



def normalize_entity(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _WS.sub(" ", text)
    return _EDGE.sub("", text)


def extract_identifiers(text: str) -> list[str]:
    """Structural identifiers: kebab, snake, CamelCase, alphanumeric, acronyms.

    Measurements, bare numbers and hex digests are dropped, and a token glued
    to a number on its left (`100MS`, `32GB`) is treated as a unit suffix.
    """
    found: list[tuple[int, str]] = []
    for pattern in _IDENT_PATTERNS:
        for match in pattern.finditer(text):
            token = normalize_entity(match.group(0))
            if _is_noise(token):
                continue
            before = text[match.start() - 1] if match.start() else ""
            if before and (before.isdigit() or before in ".,"):
                continue
            found.append((match.start(), token))
    found.sort(key=lambda item: (item[0], -len(item[1])))
    out: list[str] = []
    for _, token in found:
        if token not in out:
            out.append(token)
    return out[:20]


def entity_bonus(query_entities: Iterable[str], node_entities: Iterable[str]) -> float:
    """+0.04 for the first shared alias, +0.02 for a second, capped at +0.06."""
    shared = set(query_entities) & set(node_entities)
    if not shared:
        return 0.0
    if len(shared) == 1:
        return BONUS_FIRST
    return min(BONUS_CAP, BONUS_FIRST + BONUS_SECOND)


def merge_entities(existing: Iterable[str] | None, extra: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in list(existing or []) + list(extra):
        token = normalize_entity(value)
        if token and token not in out:
            out.append(token)
    return out


class EntityRegistry:
    """Aliases read from the backend, plus anything new written at runtime."""

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._pattern: re.Pattern[str] | None = None

    @property
    def values(self) -> list[str]:
        return sorted(self._values)

    def add(self, values: Iterable[str]) -> int:
        before = len(self._values)
        for value in values:
            token = normalize_entity(value)
            if token:
                self._values.add(token)
        if len(self._values) != before:
            self._pattern = None
        return len(self._values) - before

    def reset(self) -> None:
        """Drop every alias. A reload after this is authoritative: entities
        that disappeared from the collection stop matching."""
        self._values.clear()
        self._pattern = None

    def _compiled(self) -> re.Pattern[str] | None:
        if self._pattern is None and self._values:
            ordered = sorted(self._values, key=len, reverse=True)
            body = "|".join(re.escape(value) for value in ordered)
            self._pattern = re.compile(
                rf"(?<![\w-])(?:{body})(?![\w-])", re.IGNORECASE
            )
        return self._pattern

    def match(self, text: str) -> list[str]:
        pattern = self._compiled()
        if pattern is None:
            return []
        hits: list[tuple[int, str]] = []
        for match in pattern.finditer(text):
            hits.append((match.start(), normalize_entity(match.group(0))))
        hits.sort(key=lambda item: item[0])
        out: list[str] = []
        for _, token in hits:
            if token not in out:
                out.append(token)
        return out[:12]

    async def load(
        self,
        client: Any,
        collection: str,
        *,
        max_share: float = DEFAULT_MAX_ENTITY_SHARE,
    ) -> int:
        """Read every alias in the collection. Facet first, scroll as fallback.

        Values present in more than ``max_share`` of the points are dropped:
        a term that appears in 12% of the collection is a stop word, and
        matching it would hand the entity slot to an arbitrary document.
        """
        total = 0
        try:
            counted = await client.count(collection_name=collection, exact=True)
            total = int(getattr(counted, "count", 0) or 0)
        except Exception:
            total = 0
        ceiling = total * max_share if total else None

        values: list[str] = []
        try:
            response = await client.facet(
                collection_name=collection, key="entities", limit=20000, exact=True
            )
            for hit in response.hits:
                if ceiling is not None and int(getattr(hit, "count", 0) or 0) > ceiling:
                    continue
                values.append(str(hit.value))
        except Exception:
            counts: dict[str, int] = {}
            offset = None
            while True:
                points, offset = await client.scroll(
                    collection_name=collection,
                    limit=512,
                    offset=offset,
                    with_payload=["entities"],
                    with_vectors=False,
                )
                if not points:
                    break
                for point in points:
                    for value in (point.payload or {}).get("entities") or []:
                        token = normalize_entity(value)
                        if token:
                            counts[token] = counts.get(token, 0) + 1
                if offset is None:
                    break
            values = [
                token
                for token, count in counts.items()
                if ceiling is None or count <= ceiling
            ]
        return self.add(values)