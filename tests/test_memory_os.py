import hashlib

import pytest
from kaede_recall_mcp.common.backend import AsyncClient

from kaede_recall_mcp.embeddings.base import EmbeddingProvider
from kaede_recall_mcp.memory_os.bootstrap import _trigger_text, classify
from kaede_recall_mcp.memory_os.entities import EntityRegistry, entity_bonus, extract_identifiers
from kaede_recall_mcp.memory_os.recall import (
    RecallHit,
    _dual_open,
    _order_slots,
    recall,
    recall_by_ids,
    walk,
)
from kaede_recall_mcp.memory_os.schema import ensure_collection
from kaede_recall_mcp.memory_os.store import MemoryStore, content_hash


class FakeDense(EmbeddingProvider):
    def __init__(self, size: int = 768):
        self.size = size

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return [((seed >> (i % 32)) & 0xFF) / 255.0 for i in range(self.size)]

    async def embed_documents(self, documents):
        return [self._vec(d) for d in documents]

    async def embed_query(self, query):
        return self._vec(query)

    def get_vector_name(self):
        return "dense"

    def get_vector_size(self):
        return self.size


@pytest.fixture
async def store():
    client = AsyncClient(location=":memory:", check_compatibility=False)
    await ensure_collection(client, "test-memory")
    s = MemoryStore(client, "test-memory", FakeDense())
    yield client, s
    await client.close()


async def test_hash_is_stable_and_normalises_whitespace():
    assert content_hash("fact", "Hola   mundo") == content_hash("fact", " hola mundo ")
    assert content_hash("fact", "a") != content_hash("decision", "a")


async def test_write_dedupes_identical_content(store):
    _, s = store
    first = await s.write_node("fact", "El gateway corre en systemd.")
    second = await s.write_node("fact", "El gateway corre en systemd.")
    assert first.duplicate is False
    assert second.duplicate is True
    assert first.id == second.id


async def test_write_keeps_different_content(store):
    _, s = store
    a = await s.write_node("fact", "hecho uno")
    b = await s.write_node("fact", "hecho dos")
    assert a.id != b.id


async def test_supersede_marks_old_and_links(store):
    client, s = store
    old = await s.write_node("decision", "usar SQLite para memoria")
    new = await s.write_node("decision", "usar the backend para memoria")
    await s.supersede(old.id, new.id, reason="mejor busqueda")
    payload = await s.get(old.id)
    assert payload["status"] == "superseded"
    assert payload["superseded_by"] == new.id
    assert any(e["rel"] == "superseded_by" and e["to"] == new.id for e in payload["edges"])
    target = await s.get(new.id)
    assert any(b["rel"] == "superseded_by" and b["from"] == old.id for b in target["backlinks"])
    paths = await walk(client, "test-memory", old.id, depth=1)
    assert any(p["rel"] == "superseded_by" and p["to"] == new.id for p in paths)


async def test_write_with_links_creates_both_sides(store):
    _, s = store
    fix = await s.write_node("problem", "el servicio no responde tras actualizar")
    proc = await s.write_node(
        "procedure",
        "reiniciar el servicio y revisar el log",
        links=[f"fixed_by:{fix.id}"],
        entities=["rpi", "raspberry-pi"],
    )
    proc_payload = await s.get(proc.id)
    assert any(e["rel"] == "fixed_by" and e["to"] == fix.id for e in proc_payload["edges"])
    assert "rpi" in proc_payload["entities"]
    fix_payload = await s.get(fix.id)
    assert any(b["rel"] == "fixed_by" and b["from"] == proc.id for b in fix_payload["backlinks"])


async def test_recall_skips_deprecated(store):
    client, s = store
    await s.write_node("doc", "snapshot viejo de una skill", deprecated=True)
    hits, block = await recall(client, "test-memory", "snapshot viejo", FakeDense())
    assert hits == []
    assert block == ""


async def test_recall_returns_budgeted_block(store):
    client, s = store
    await s.write_node("procedure", "Registrar una cuenta en un portal con perfil personal")
    hits, block = await recall(
        client, "test-memory", "crear cuenta en un portal", FakeDense(), limit=5, max_chars=300
    )
    assert hits
    assert len(block) <= 300
    assert "procedure" in block


async def test_recall_returns_nothing_when_empty(store):
    client, _ = store
    hits, block = await recall(client, "test-memory", "cualquier cosa", FakeDense())
    assert hits == []
    assert block == ""


def test_classify_reads_metadata_hints():
    assert classify("# skill", {"type": "skill_index"}) == "doc"
    assert classify("# skill", {"category": "skill"}) == "procedure"
    assert classify("fallo al arrancar", {"category": "troubleshooting"}) == "problem"
    assert classify("cambio de config", {"category": "configuration"}) == "decision"
    assert classify("dato suelto", {}) == "fact"


def test_trigger_text_prefers_when_to_use_section():
    body = "## When to use\nWhen filling forms or registering accounts.\n\n## Steps\n1. open"
    trigger = _trigger_text("web-form", "filling forms", body)
    assert "filling forms" in trigger
    assert "registering accounts" in trigger
    assert "## Steps" not in trigger


def test_entity_extraction_is_structural():
    ids = extract_identifiers("server1-20-1 MapleMeshRender esp32 v3 /home/maple/app.py")
    assert "server1-20-1" in ids
    assert "maplemeshrender" in ids
    assert "esp32" in ids
    assert "/home/maple/app.py" in ids


def test_registry_matches_aliases_from_payload():
    registry = EntityRegistry()
    registry.add(["RPi", "raspberry", "server1-20-1"])
    assert registry.match("la raspberry del rpi") == ["raspberry", "rpi"]
    assert registry.match("server1-20-1 no responde") == ["server1-20-1"]


def test_entity_bonus_is_additive_and_capped():
    assert entity_bonus(["rpi"], ["rpi", "hermes"]) == 0.04
    assert entity_bonus(["complexivo", "rpi"], ["complexivo", "rpi"]) == 0.06
    assert entity_bonus(["rpi"], ["hyprland"]) == 0.0


def _hit(kind: str, score: float) -> RecallHit:
    return RecallHit(id=f"{kind}-{score}", kind=kind, score=score, text="", summary="")


def test_dual_body_opens_on_tight_tie_of_any_kind():
    assert _dual_open(_hit("fact", 0.474), _hit("fact", 0.469))
    assert _dual_open(_hit("problem", 0.50), _hit("problem", 0.49))


def test_dual_body_opens_on_procedure_context_pair():
    assert _dual_open(_hit("procedure", 0.50), _hit("fact", 0.47))
    assert not _dual_open(_hit("fact", 0.50), _hit("fact", 0.45))


def test_dual_body_requires_second_score_floor():
    assert not _dual_open(_hit("procedure", 0.40), _hit("fact", 0.32))


def test_order_slots_prefers_different_kind():
    hits = [_hit("procedure", 0.50), _hit("fact", 0.48), _hit("fact", 0.47)]
    ordered = _order_slots(hits, [], 3)
    assert [hit.kind for hit in ordered[:2]] == ["procedure", "fact"]


def test_order_slots_swaps_when_second_scores_higher():
    hits = [_hit("fact", 0.40), _hit("procedure", 0.45)]
    ordered = _order_slots(hits, [], 2)
    assert ordered[0].kind == "procedure"


def test_order_slots_prefers_entity_node():
    def ehit(kind, score, entities):
        return RecallHit(
            id=f"{kind}-{score}-{entities[0]}",
            kind=kind,
            score=score,
            text="",
            summary="",
            payload={"entities": entities},
            base=score,
        )

    hits = [
        ehit("fact", 0.45, ["other"]),
        ehit("fact", 0.40, ["rpi"]),
        ehit("problem", 0.39, ["rpi"]),
    ]
    ordered = _order_slots(hits, ["rpi"], 3)
    assert ordered[0].payload["entities"] == ["rpi"]


async def test_write_auto_extracts_entities(store):
    _, s = store
    ref = await s.write_node("fact", "El servicio server1-20-1 usa esp32 y v3")
    assert "server1-20-1" in ref.payload["entities"]
    assert "esp32" in ref.payload["entities"]


async def test_recall_by_ids_opens_body_without_search(store):
    client, s = store
    ref = await s.write_node("fact", "dato que se abre por id", summary="trigger")
    hits, block = await recall_by_ids(client, "test-memory", [ref.id], max_chars=400)
    assert hits and hits[0].id == ref.id
    assert "dato que se abre por id" in block


# -- exact lane ---------------------------------------------------------


def test_identifier_terms_ignore_acronyms_and_short_versions():
    from kaede_recall_mcp.memory_os.recall import _identifier_terms

    assert _identifier_terms("revisa la API y el NAT del MCP en v2 y v3") == []
    assert _identifier_terms("mira el commit 3f9a2c1") == ["3f9a2c1"]


def test_identifier_terms_keep_paths_and_known_files():
    from kaede_recall_mcp.memory_os.recall import _identifier_terms

    terms = _identifier_terms("falla /home/maple/app/recall.py y config.yaml")
    assert "/home/maple/app/recall.py" in terms
    assert "config.yaml" in terms


def test_extract_identifiers_drops_units_numbers_and_hex():
    ids = extract_identifiers("latencia 100ms, 32GB de ram, 3.14, hash 3f9a2c1b")
    assert "100ms" not in ids
    assert "32gb" not in ids
    assert "3.14" not in ids
    assert "3f9a2c1b" not in ids


# -- slot ordering ------------------------------------------------------


def _ehit(kind, score, entities, base=None):
    return RecallHit(
        id=f"{kind}-{score}-{'-'.join(entities) or 'none'}",
        kind=kind,
        score=score,
        text="",
        summary="",
        payload={"entities": entities},
        base=score if base is None else base,
    )


def test_order_slots_ignores_weak_entity_node():
    hits = [
        _ehit("procedure", 0.52, []),
        _ehit("fact", 0.30, ["rpi"]),
    ]
    ordered = _order_slots(hits, ["rpi"], 2)
    assert ordered[0].kind == "procedure"


def test_order_slots_swaps_global_node_on_wide_margin():
    hits = [
        _ehit("fact", 0.40, ["rpi"]),
        _ehit("procedure", 0.55, []),
    ]
    ordered = _order_slots(hits, ["rpi"], 2)
    assert ordered[0].kind == "procedure"


def test_order_slots_keeps_entity_node_on_small_margin():
    hits = [
        _ehit("fact", 0.40, ["rpi"]),
        _ehit("procedure", 0.45, []),
    ]
    ordered = _order_slots(hits, ["rpi"], 2)
    assert ordered[0].payload["entities"] == ["rpi"]


# -- registry ceiling ---------------------------------------------------


class _FacetHit:
    def __init__(self, value, count):
        self.value = value
        self.count = count


class _FacetResponse:
    def __init__(self, hits):
        self.hits = hits


class _CountResponse:
    def __init__(self, count):
        self.count = count


class _FacetClient:
    def __init__(self, total, hits):
        self.total = total
        self.hits = hits

    async def count(self, collection_name, exact=True):
        return _CountResponse(self.total)

    async def facet(self, collection_name, key, limit, exact=True):
        return _FacetResponse(self.hits)


async def test_registry_load_drops_ubiquitous_entities():
    client = _FacetClient(
        total=1000,
        hits=[
            _FacetHit("http", 400),
            _FacetHit("api", 130),
            _FacetHit("rpi", 40),
            _FacetHit("systemd", 12),
        ],
    )
    registry = EntityRegistry()
    await registry.load(client, "test-memory")
    assert registry.values == ["rpi", "systemd"]


# -- graph hop ----------------------------------------------------------


class _RetrieveClient:
    def __init__(self, nodes):
        self.nodes = nodes
        self.calls = 0

    async def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        self.calls += 1
        return [self.nodes[i] for i in ids if i in self.nodes]


async def test_expand_filters_dead_neighbours_in_one_retrieve_per_level():
    from kaede_recall_mcp.common.backend import models

    from kaede_recall_mcp.memory_os.recall import _expand

    nodes = {
        "a": models.Record(
            id="a",
            payload={"edges": [{"rel": "fixed_by", "to": "b"}, {"rel": "fixed_by", "to": "c"}]},
        ),
        "b": models.Record(id="b", payload={"status": "superseded", "edges": []}),
        "c": models.Record(id="c", payload={"status": "active", "edges": []}),
    }
    client = _RetrieveClient(nodes)
    out = await _expand(client, "test-memory", ["a"], 1)
    assert [str(node.id) for node in out] == ["c"]
    assert client.calls == 2  # level 0 (start) + level 1 (neighbours)


async def test_write_node_id_is_deterministic_uuid5(store):
    _, s = store
    first = await s.write_node("fact", "id determinista", dedupe=False)
    second = await s.write_node("fact", "id determinista", dedupe=False)
    assert first.id == second.id
    assert first.id.count("-") == 4


# -- in-place edit and delete -------------------------------------------


async def test_update_node_reembeds_and_rehashes(store):
    client, s = store
    ref = await s.write_node("fact", "texto original de prueba", entities=["rpi"])
    before = await client.retrieve(
        "test-memory", ids=[ref.id], with_payload=True, with_vectors=True
    )
    updated = await s.update_node(
        ref.id, text="texto nuevo de prueba", tags=["tag-nuevo"]
    )
    after = await client.retrieve(
        "test-memory", ids=[ref.id], with_payload=True, with_vectors=True
    )
    assert updated is not None
    assert updated["text"] == "texto nuevo de prueba"
    assert updated["tags"] == ["tag-nuevo"]
    assert updated["hash"] != before[0].payload["hash"]
    assert after[0].vector["dense"] != before[0].vector["dense"]


async def test_update_node_tags_only_keeps_hash_and_vector(store):
    client, s = store
    ref = await s.write_node("fact", "texto estable de prueba")
    before = await client.retrieve(
        "test-memory", ids=[ref.id], with_payload=True, with_vectors=True
    )
    updated = await s.update_node(ref.id, tags=["solo-tag"])
    after = await client.retrieve(
        "test-memory", ids=[ref.id], with_payload=True, with_vectors=True
    )
    assert updated is not None
    assert updated["hash"] == before[0].payload["hash"]
    assert after[0].vector["dense"] == before[0].vector["dense"]


async def test_delete_nodes_dry_run_keeps_the_point(store):
    client, s = store
    ref = await s.write_node("fact", "no borrar todavia")
    hits = await s.delete_nodes(point_id=ref.id, dry_run=True)
    assert [hit["id"] for hit in hits] == [ref.id]
    assert (await client.retrieve("test-memory", ids=[ref.id])) != []


async def test_delete_nodes_by_id(store):
    client, s = store
    ref = await s.write_node("fact", "borrar por id")
    hits = await s.delete_nodes(point_id=ref.id)
    assert [hit["id"] for hit in hits] == [ref.id]
    assert (await client.retrieve("test-memory", ids=[ref.id])) == []


async def test_delete_nodes_by_query(store):
    client, s = store
    text = "borrar por consulta exacta"
    ref = await s.write_node("fact", text)
    hits = await s.delete_nodes(query=text, limit=1, score_threshold=0.9)
    assert [hit["id"] for hit in hits] == [ref.id]
    assert (await client.retrieve("test-memory", ids=[ref.id])) == []


async def test_delete_nodes_requires_exactly_one_selector(store):
    _, s = store
    with pytest.raises(ValueError):
        await s.delete_nodes()
    with pytest.raises(ValueError):
        await s.delete_nodes(point_id="x", query="y")


async def test_recall_records_usage_metrics(store):
    client, s = store
    text = "nodo con metricas de uso"
    ref = await s.write_node("fact", text)
    await recall(client, "test-memory", text, FakeDense(), limit=3)
    points = await client.retrieve("test-memory", ids=[ref.id], with_payload=True)
    assert points[0].payload["hits"] >= 1
    assert "last_recalled_at" in points[0].payload

