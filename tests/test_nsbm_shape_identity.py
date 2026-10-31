from __future__ import annotations

import json
import random

import pytest

from semmap_haken.shape_identity import Shape, ShapeRegistry


def permute(shape, order):
    kinds = [None] * len(order)
    for old, new in enumerate(order):
        kinds[new] = shape.node_types[old]
    return Shape(tuple(kinds), tuple((order[u], order[v], r) for u, v, r in shape.edges))


def test_same_symbol_for_any_node_order_and_discovery_level():
    registry = ShapeRegistry('fixture')
    shape = Shape(('ATOM',) * 4, ((0, 1, 'R'), (0, 1, 'R'), (1, 2, 'S'), (3, 3, 'L')))
    symbol, _ = registry.intern(shape)
    rng = random.Random(41)
    for level in range(20):
        order = list(range(4))
        rng.shuffle(order)
        found, mapping = registry.intern(permute(shape, order))
        assert found == symbol
        assert sorted(mapping) == list(range(4))
    assert len(registry.shapes) == 1


def test_modes_are_nested_and_do_not_merge_exact_symbols():
    registry = ShapeRegistry('fixture')
    out = Shape(('ATOM',) * 3, ((0, 1, 'R'), (0, 2, 'R')))
    inside = Shape(('ATOM',) * 3, ((1, 0, 'R'), (2, 0, 'R')))
    typed = Shape(('ATOM',) * 3, ((0, 1, 'S'), (0, 2, 'S')))
    a = registry.intern(out)[0]
    b = registry.intern(inside)[0]
    c = registry.intern(typed)[0]
    d = registry.intern(Shape((f'S:{a}', 'ATOM', 'ATOM'), out.edges))[0]
    assert len({a, b, c, d}) == 4
    assert len(set(registry.classes('topology').values())) == 1
    assert len(set(registry.classes('directed').values())) == 2
    assert len(set(registry.classes('typed').values())) == 3
    assert len(set(registry.classes('recursive').values())) == 4
    assert len(registry.shapes) == 4


def test_recursive_children_use_global_ids_not_levels():
    registry = ShapeRegistry('fixture')
    child = registry.intern(Shape(('ATOM',) * 2, ((0, 1, 'R'),)))[0]
    parent = Shape((f'S:{child}', 'ATOM'), ((0, 1, 'S'),))
    expected = registry.intern(parent)[0]
    for level in (2, 4, 17):
        assert registry.intern(permute(parent, (1, 0)))[0] == expected
    assert len(registry.shapes) == 2


def test_parallel_relation_multisets_even_with_same_global_histogram():
    registry = ShapeRegistry('fixture')
    # Node loops distinguish the endpoint pairs; all global relation counts agree.
    common = ((0, 0, 'X'), (1, 1, 'Y'), (2, 2, 'Z'))
    a = common + ((0, 1, 'R'), (0, 1, 'R'), (0, 1, 'S'),
                  (1, 2, 'R'), (1, 2, 'S'), (1, 2, 'S'))
    b = common + ((0, 1, 'R'), (0, 1, 'S'), (0, 1, 'S'),
                  (1, 2, 'R'), (1, 2, 'R'), (1, 2, 'S'))
    assert registry.intern(Shape(('ATOM',) * 3, a))[0] != registry.intern(Shape(('ATOM',) * 3, b))[0]
    assert len(set(registry.classes('directed').values())) == 1


def test_collision_in_prefilter_does_not_merge_nonisomorphic_graphs(monkeypatch):
    import semmap_haken.shape_identity as module
    monkeypatch.setattr(module, '_bucket', lambda graph: ())
    registry = ShapeRegistry('fixture')
    one = registry.intern(Shape(('ATOM',) * 3, ((0, 1, 'R'), (1, 2, 'R'))))[0]
    two = registry.intern(Shape(('ATOM',) * 3, ((0, 1, 'R'), (0, 2, 'R'))))[0]
    assert one != two


def test_relation_integer_ids_are_not_global_identity():
    registry = ShapeRegistry('fixture')
    shape = {'shape_id': 0, 'node_types': ['ATOM', 'ATOM'], 'edges': [[0, 1, 0]]}
    a = registry.references([shape], ['R'])[0]['symbol']
    shape['edges'] = [[0, 1, 1]]
    b = registry.references([shape], ['Q', 'R'])[0]['symbol']
    assert a == b
    assert registry.expand(registry.references([shape], ['Q', 'R']), ['Q', 'R']) == [shape]


def test_reference_permutation_reconstructs_edge_slots_and_types():
    registry = ShapeRegistry('fixture')
    first = {'shape_id': 4, 'node_types': ['ATOM'] * 3,
             'edges': [[0, 1, 0], [0, 1, 0], [1, 2, 1], [2, 2, 2]]}
    registry.references([first], ['R', 'S', 'L'])
    # Relations and vertices are renumbered independently in the second archive.
    second = {'shape_id': 9, 'node_types': ['ATOM'] * 3,
              'edges': [[0, 1, 0], [1, 1, 1], [2, 0, 2], [2, 0, 2]]}
    refs = registry.references([second], ['S', 'L', 'R'])
    assert refs[0]['symbol'] == 0
    assert 'order' in refs[0]
    assert registry.expand(refs, ['S', 'L', 'R']) == [second]


def test_resume_preserves_namespace_definitions_and_future_ids():
    registry = ShapeRegistry('fixture')
    a = Shape(('ATOM',) * 2, ((0, 1, 'R'),))
    registry.intern(a)
    restored = ShapeRegistry.loads(registry.dumps())
    assert restored.dumps() == registry.dumps()
    assert restored.intern(permute(a, (1, 0)))[0] == 0
    b = Shape(('S:0', 'ATOM'), ((0, 1, 'Q'),))
    assert restored.intern(b)[0] == registry.intern(b)[0] == 1
    assert restored.dumps() == registry.dumps()


@pytest.mark.parametrize('kind', ['S:0', 'S:-1', 'S:00', 'S:x', 'L0_S0'])
def test_unknown_forward_or_legacy_children_are_rejected(kind):
    with pytest.raises(ValueError):
        ShapeRegistry().intern(Shape((kind, 'ATOM'), ((0, 1, 'R'),)))


@pytest.mark.parametrize('edges', [((-1, 0, 'R'),), ((0, 2, 'R'),), ((0, 1, 0),)])
def test_invalid_shapes(edges):
    with pytest.raises(ValueError):
        ShapeRegistry().intern(Shape(('ATOM', 'ATOM'), edges))


def test_bad_contract_duplicates_and_ref_permutation_are_rejected():
    registry = ShapeRegistry('fixture')
    registry.intern(Shape(('ATOM',) * 2, ((0, 1, 'R'),)))
    payload = json.loads(registry.dumps())
    payload['shapes'] *= 2
    with pytest.raises(ValueError):
        ShapeRegistry.loads(json.dumps(payload))
    payload['contract'] = 'ignore-all-labels'
    with pytest.raises(ValueError):
        ShapeRegistry.loads(json.dumps(payload))
    with pytest.raises(ValueError):
        registry.expand([{'local': 0, 'symbol': 0, 'order': [0, 0]}], ['R'])
    with pytest.raises(ValueError):
        registry.classes('approximately')
