"""A symbol reappears at a later scale; exact payload and legacy decode survive."""
import json
import zipfile

import pytest

from semmap_haken.grammar_codec import encode_grammar
from semmap_haken.graphex_components import EdgeRecord
from semmap_haken.hypothesis.recursive_codec import (
    SHARED_FORMAT, _assert_records, _register_transition, _state, _write_archive,
    contract_records, decode_recursive,
)
from semmap_haken.shape_identity import ShapeRegistry


def test_global_reuse_permutation_recursion_and_resume(tmp_path):
    records = (
        EdgeRecord(0, 0, 1, 'R', -0.0), EdgeRecord(1, 0, 1, 'R', 0.5),
        EdgeRecord(2, 3, 2, 'R', 2.0), EdgeRecord(3, 3, 2, 'R', 3.0),
        EdgeRecord(4, 1, 4, 'I', 4.0), EdgeRecord(5, 3, 5, 'I', 5.0),
        EdgeRecord(6, 6, 7, 'R', 6.0), EdgeRecord(7, 7, 7, 'L', 7.5),
        EdgeRecord(8, 5, 6, 'R', 9.0), EdgeRecord(9, 7, 1, 'R', 10.0),
    )
    ids = {'node_ids': [f'n{i}' for i in range(8)],
           'assertion_ids': [f'a{i}' for i in range(10)],
           'relation_names': ['R', 'I', 'L'], 'graph_sha256': 'fixture'}
    registry = ShapeRegistry('fixture')
    accepted = tmp_path / 'base.zip'
    _write_archive(None, original_ids=ids, n=8, edges=records,
                   original_count=8, original_assertions=10,
                   output=accepted, registry=registry)
    n, edges, types = 8, records, {}
    for level, figures in enumerate(([(0, 1)], [(1, 2)], [(0, 1)])):
        # Simulate a fresh process before every level.
        with zipfile.ZipFile(accepted) as z:
            registry = ShapeRegistry.loads(z.read('dictionary.json'))
        transition = tmp_path / f't{level}.zip'
        encode_grammar(n, edges, figures, transition, node_types=types,
                       symbol_coding='huffman')
        refs = _register_transition(transition, registry)
        n, edges, _, types = contract_records(n, edges, figures, types, level,
                                             transition, shape_refs=refs)
        output = tmp_path / f'l{level + 1}.zip'
        _write_archive(accepted, original_ids=ids, n=n, edges=edges,
                       original_count=8, original_assertions=10, output=output,
                       transition=transition, registry=registry, shape_refs=refs)
        restored_n, decoded, restored_ids = decode_recursive(output)
        assert restored_n == 8 and restored_ids == ids
        _assert_records(decoded, records)
        state = _state(output)
        assert state[0] == n and state[1] == edges and state[3] == types
        if level == 0:
            assert types[0] == 'S:0'
        if level == 1:
            assert types[0] == types[1] == 'S:0'
            assert len(registry.shapes) == 1
            assert refs[0]['order'] == [1, 0]
        accepted = output
    assert len(registry.shapes) == 2
    assert registry.shapes[1].node_types == ('S:0', 'S:0')
    with zipfile.ZipFile(accepted) as z:
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['format'] == SHARED_FORMAT
        assert z.namelist().count('dictionary.json') == 1
        assert not any(p.endswith('/shapes.json') or p.endswith('/rules.json') for p in z.namelist())
        assert json.loads(z.read('transition_000_001/shape_refs.json'))[0]['symbol'] == 0
        assert json.loads(z.read('transition_001_002/shape_refs.json'))[0]['symbol'] == 0
        assert json.loads(z.read('transition_002_003/shape_refs.json'))[0]['symbol'] == 1


def test_shared_writer_rejects_changed_dictionary_namespace(tmp_path):
    registry = ShapeRegistry('first')
    base = tmp_path / 'base.zip'
    _write_archive(None, original_ids={}, n=1, edges=(), original_count=1,
                   original_assertions=0, output=base, registry=registry)
    with pytest.raises(ValueError, match='append-only'):
        _write_archive(base, original_ids={}, n=1, edges=(), original_count=1,
                       original_assertions=0, output=tmp_path / 'bad.zip',
                       registry=ShapeRegistry('other'))


def test_shared_writer_rejects_invalid_permutation_before_publication(tmp_path):
    records = (EdgeRecord(0, 0, 1, 'R', 1.0),)
    transition = tmp_path / 'transition.zip'
    encode_grammar(2, records, [(0, 1)], transition, symbol_coding='huffman')
    registry = ShapeRegistry('fixture')
    refs = _register_transition(transition, registry)
    refs[0]['order'] = [0, 0]
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError):
        _write_archive(None, original_ids={}, n=1, edges=(), original_count=2,
                       original_assertions=1, output=output, transition=transition,
                       registry=registry, shape_refs=refs)
    assert not output.exists()
