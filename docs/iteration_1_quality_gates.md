# Iteration 1 M0 quality gates

The active M0 verification suite is [`tests/test_semmap_haken_quality_gates.py`](../tests/test_semmap_haken_quality_gates.py). It complements the focused unit and integration tests under [`tests/`](../tests/).

## Notebook-to-CLI equivalence

The notebook gate executes [`notebooks/01_data_smoke_and_sparse_graph.ipynb`](../notebooks/01_data_smoke_and_sparse_graph.ipynb) with an isolated deterministic offline configuration, then executes the CLI `prepare` workflow with the same fixture in a separate root. Equivalence means the ordered node URIs and CSR `indptr`, `indices`, and `data` are identical; the scientific portions of the resolved configuration agree; and sparse-artifact checksums agree. Run IDs, timestamps, host details, storage roots, and the notebook-specific provenance envelope are intentionally not compared.

The notebook records its execution metadata into [`manifest.json`](../src/semmap_haken/manifest.py) after it delegates preparation to the public CLI workflow. The CLI remains the reproducible replay interface.

## Required gates

| Gate | Verification |
| --- | --- |
| Offline acquisition | Manual fixture paths prevent network access; verified cache reuse is asserted. |
| Download integrity | Interrupted transfers restart from zero; checksum mismatch removes both destination and partial files. |
| Sparse preparation | The preparation path is guarded against `toarray` and `todense` access. |
| Artifact integrity | Every manifest artifact path exists and the manifest and graph-metadata checksums match bytes on disk. |
| Notebook provenance | Headless notebook output records notebook environment metadata and declares CLI checksum equivalence. |

The checks are hermetic: fixtures and in-process transport doubles replace external data services. This prevents nondeterministic CI failures while preserving the observable M0 contracts.
