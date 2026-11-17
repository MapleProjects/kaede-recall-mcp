"""Offline tests for the dedupe scan."""

from types import SimpleNamespace

from kaede_recall_mcp.memory_os.bootstrap import find_duplicate_pairs


def _point(point_id, vector, entities, kind="fact", created="2026-01-01T00:00:00+00:00", scope="global"):
    return SimpleNamespace(
        id=point_id,
        vector={"dense": vector},
        payload={
            "entities": entities,
            "kind": kind,
            "created_at": created,
            "text": point_id,
            "scope": scope,
        },
    )


def test_find_duplicate_pairs_above_threshold():
    points = [
        _point("a", [1.0, 0.0], ["rpi"]),
        _point("b", [0.95, 0.05], ["rpi"]),
        _point("c", [0.0, 1.0], ["rpi"]),
    ]
    pairs = find_duplicate_pairs(points, threshold=0.90)
    assert len(pairs) == 1
    score, first, second, entity = pairs[0]
    assert {first.id, second.id} == {"a", "b"}
    assert score >= 0.90
    assert entity == "rpi"


def test_find_duplicate_pairs_requires_a_shared_entity():
    points = [
        _point("a", [1.0, 0.0], ["rpi"]),
        _point("b", [1.0, 0.0], ["dell"]),
    ]
    assert find_duplicate_pairs(points, threshold=0.90) == []


def test_find_duplicate_pairs_skips_ubiquitous_entities():
    points = [_point(str(index), [1.0, 0.0], ["http"]) for index in range(10)]
    # http appears in 100% of the points: a stop word, not a dedupe signal.
    assert find_duplicate_pairs(points, threshold=0.90) == []


def test_find_duplicate_pairs_filters_by_entity():
    points = [
        _point("a", [1.0, 0.0], ["rpi"]),
        _point("b", [1.0, 0.0], ["rpi"]),
        _point("c", [1.0, 0.0], ["dell"]),
        _point("d", [1.0, 0.0], ["dell"]),
    ]
    pairs = find_duplicate_pairs(points, threshold=0.90, entity="dell")
    assert len(pairs) == 1
    assert {pairs[0][1].id, pairs[0][2].id} == {"c", "d"}


def test_find_duplicate_pairs_handles_missing_vectors():
    points = [
        SimpleNamespace(id="a", vector=None, payload={"entities": ["rpi"]}),
        _point("b", [1.0, 0.0], ["rpi"]),
    ]
    assert find_duplicate_pairs(points, threshold=0.90) == []


def test_find_compact_pairs_requires_same_entity_and_scope():
    from kaede_recall_mcp.memory_os.bootstrap import find_compact_pairs

    points = [
        _point("a", [1.0, 0.0], ["rpi"], scope="global"),
        _point("b", [1.0, 0.0], ["rpi"], scope="global"),
        _point("c", [1.0, 0.0], ["rpi"], scope="project:other"),
    ]
    pairs = find_compact_pairs(points, threshold=0.88)
    assert len(pairs) == 1
    _, first, second, entity, scope = pairs[0]
    assert {first.id, second.id} == {"a", "b"}
    assert entity == "rpi"
    assert scope == "global"


def test_find_compact_pairs_orders_by_score():
    from kaede_recall_mcp.memory_os.bootstrap import find_compact_pairs

    points = [
        _point("a", [1.0, 0.0], ["rpi"]),
        _point("b", [0.90, 0.10], ["rpi"]),
        _point("c", [1.0, 0.02], ["rpi"]),
    ]
    pairs = find_compact_pairs(points, threshold=0.88)
    assert pairs[0][0] >= pairs[-1][0]