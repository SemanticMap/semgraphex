"""Explicit, bounded export of a published text graph to existing PreparedGraph."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .text_corpus import _json, open_corpus, sha256


def export_prepared(corpus: str | Path, doc_id: str, output: str | Path, *,
                    level: str = 'occurrence', directed: bool = True,
                    max_edges: int = 2_000_000) -> dict:
    """Distance stays an attribute; adjacency uses sum(1/(1+distance_chars)).

    The full provenance/occurrence graph stays in Parquet. PreparedGraph holds
    in-memory selected_edges, so export has a separate explicit memory guard.
    """
    if level not in {'occurrence', 'concept'} or max_edges < 1:
        raise ValueError('Invalid export level or edge budget')
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    db = open_corpus(corpus)
    try:
        if db.execute('SELECT count(*) FROM documents WHERE doc_id=?', [doc_id]).fetchone()[0] != 1:
            raise ValueError(f'Unknown document: {doc_id}')
        if level == 'occurrence':
            keys = [r[0] for r in db.execute('SELECT occurrence_id FROM occurrences '
                         'WHERE doc_id=? ORDER BY occurrence_id', [doc_id]).fetchall()]
            import hashlib
            prefix = 'urn:semmap:' + hashlib.sha256(doc_id.encode('utf-8')).hexdigest()
            node_ids = tuple(f'{prefix}:word:{i:012d}' for i in keys)
            source, target = 'source_occurrence_id', 'target_occurrence_id'
        else:
            keys = [r[0] for r in db.execute('SELECT DISTINCT concept_uri FROM occurrences '
                    'WHERE doc_id=? AND concept_uri IS NOT NULL ORDER BY concept_uri',
                    [doc_id]).fetchall()]
            node_ids = tuple(keys)
            source, target = 'source_concept', 'target_concept'
        # Only constants chosen above enter SQL identifiers; doc_id is parameterized.
        query = f'''SELECT {source} AS source, {target} AS target, relation_uri,
                    sum(proximity_weight), count(*), count(DISTINCT assertion_id),
                    min(distance_chars), avg(distance_chars), max(distance_chars),
                    list(DISTINCT assertion_id ORDER BY assertion_id)
             FROM edges WHERE doc_id=? GROUP BY {source},{target},relation_uri
             ORDER BY {source},{target},relation_uri'''
        count = db.execute(f'SELECT count(*) FROM ({query})', [doc_id]).fetchone()[0]
        if count > max_edges:
            raise ValueError(f'{count} typed edges exceed export --max-edges={max_edges}; '
                             'use concept level or raise the budget explicitly')
        rows = db.execute(query, [doc_id]).fetchall()
    finally:
        db.close()
    import numpy as np
    from scipy import sparse
    from .graph_build import PreparedGraph, save_prepared_graph
    index = {key: i for i, key in enumerate(keys)}
    rr, cc, vv, selected = [], [], [], []
    histogram = Counter()
    for a, b, relation, weight, pairs, assertions, minimum, mean, maximum, ids in rows:
        i, j = index[a], index[b]
        rr.append(i)
        cc.append(j)
        vv.append(weight)
        histogram[relation.rsplit('/', 1)[-1]] += pairs
        selected.append({'start_uri': node_ids[i], 'end_uri': node_ids[j],
            'relation_uri': relation, 'relation': relation.rsplit('/', 1)[-1],
            'aggregate_contribution': weight, 'assertion_count': assertions,
            'occurrence_pair_count': pairs, 'assertion_ids': ids,
            'min_distance_chars': minimum, 'mean_distance_chars': mean,
            'max_distance_chars': maximum, 'dataset': None, 'sources': None, 'license': None})
    matrix = sparse.csr_matrix((np.asarray(vv, dtype=np.float64), (rr, cc)),
                               shape=(len(keys), len(keys)), dtype=np.float64)
    if not directed:
        matrix = (matrix + matrix.T - sparse.diags(matrix.diagonal())).tocsr()
    loops = int(np.count_nonzero(matrix.diagonal()))
    report = {'node_count': len(keys), 'edge_count': int(matrix.nnz if directed else
              (matrix.nnz-loops)//2+loops), 'adjacency_nnz': int(matrix.nnz),
              'self_loop_count': loops, 'self_loop_policy': 'include',
              'relation_histogram': dict(sorted(histogram.items())),
              'selection_policy': 'all occurrence vertices, including unmapped/isolates'
                                   if level == 'occurrence' else 'all mapped concept vertices',
              'projection': level, 'directed': directed,
              'weight_transform': 'sum(1/(1+abs(target_start-source_start)))',
              'symmetrization': 'none' if directed else 'sum directed contributions once',
              'missing_provenance': ['dataset', 'sources', 'license']}
    root = Path(corpus).resolve()
    corpus_manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(r for r in corpus_manifest['documents'] if r['identity']['doc_id'] == doc_id)
    origin = root / 'documents' / entry['identity']['key']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.text-export-', dir=destination.parent) as work:
        staging = Path(work) / 'prepared'
        save_prepared_graph(PreparedGraph(matrix, node_ids, report, tuple(selected)), staging,
                            metadata={'doc_id': doc_id, 'corpus_run_sha256': corpus_manifest['run_sha256']},
                            resolved_config={'text_projection': report})
        metadata = json.loads((staging / 'graph_metadata.json').read_text(encoding='utf-8'))
        for name in ('occurrences.parquet', 'text.txt'):
            shutil.copyfile(origin / name, staging / name)
            metadata['checksums'][name] = sha256(staging / name)
        _json(staging / 'graph_metadata.json', metadata)
        os.replace(staging, destination)
    return {'output': str(destination), **report}
