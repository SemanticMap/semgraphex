# ConceptNet-100k: full Wishart reruns at k=4 and k=5

Notebook: [notebooks/07_wishart_100k_k4_k5_shared_shapes_colab.ipynb](../notebooks/07_wishart_100k_k4_k5_shared_shapes_colab.ipynb). Run in Colab with a **GPU runtime** and a mounted Google Drive. Change DATASET_DRIVE to the actual Drive-relative path of the raw ConceptNet assertions TSV; the notebook validates its existence rather than assuming a prepared dataset exists.

The notebook performs **two independent full runs** using the existing semmap-wishart-colab runner. For each k, raw ConceptNet is parsed anew with dataset.max_nodes=100000 and dictionary.frequency_scan=full at every completed compression level. Discovery remains sampled (candidate_limit=5000), and MDL selects nonoverlapping occurrences. The exact number of loaded vertices is reported by the run; 100000 is a cap, not a guarantee about the source file. k=4 and k=5 do not reuse each other's checkpoints, dictionary, cluster labels or coarsening hierarchy. Device=auto prefers CUDA for typed-WL kNN when available; VF2 and sparse graph diagnostics remain CPU operations.

The new config raises max_dictionary_size from 20000 to 100000 so the prior dictionary-size limit does not automatically stop both runs at level four. This is resource-intensive and does not override other scientifically meaningful stop reasons (no positive MDL, min nodes, max levels, etc.). Each run checkpoints every completed level to Drive. RUN_MODE=RESUME uses the original config/input/Git revision and device and restores the latest valid checkpoint. Do not change these between NEW and RESUME.

## Factorized representation

After each full run completes, python -m semmap_haken.wishart_shared_shapes exports:

- dictionary_factorized/internal_shapes.jsonl: each exact internal directed, relation- and child-symbol-aware topology once, with VF2 verification (WL is only a prefilter);
- dictionary_factorized/type_variants.jsonl: exact type -> shape ID, shape-to-prototype node permutation, and per-node external port signatures;
- dictionary_factorized/type_metadata.jsonl: original graph-type metadata without duplicate prototype;
- transition_XXX_YYY/external_connections.jsonl.gz: each **directed entry** in the source per-relation sparse layers crossing the boundary of an accepted compression figure, including original fine endpoints, coarse endpoints, local prototype port positions, relation and weight;
- transition_XXX_YYY/figure_dynamics.jsonl: each compressed figure's external flow, stationary mass before and after contraction, exit probabilities, target slow-mode coordinates and eigenvalues, sampled betweenness of its macro node, sampled distances touching it and original concept concatenation;
- transition_XXX_YYY/graph_metrics.json: source/target graph-level mean degree, second degree moment, spreading and percolation proxies, spectral gap, synchronizability, Monte-Carlo MFPT, congestion proxy, sampled macro clustering and sampled intercluster distances, as well as caveats.

The export checks exact prototype roundtrip for every dictionary type, including node positions, relation labels and port signatures. It measures compact JSONL and gzip sizes for the legacy dictionary vs the factored dictionary. The **existing** run's legacy graph_types.jsonl, adjacency/relations matrices and checkpoints remain in place for resume, diagnostics and backward compatibility. Therefore the exported factored representation does not replace the run's entire persisted footprint, and the gzip comparison is **dictionary only**, not a full-graph codec. Internal edge weights and edges between non-figure nodes are retained in source level matrices, not encoded by the new export. To build a standalone full-graph codec, those must also be encoded and decoded exactly.

All dynamic diagnostics follow the existing runner's conventions: directed inputs are symmetrized for dynamics; MFPT is Monte Carlo with a finite cap; betweenness, clustering and distances are sampled; thresholds are documented proxies; eigenvector signs/bases may differ across independent runs. The exported per-figure slow-mode coordinate is the **target macro node's** coordinate, not a separately estimated eigenmode of its internal subgraph.

Outputs are under Drive runs/wishart-cn100k-k4-shared-v1 and runs/wishart-cn100k-k5-shared-v1; factored analysis is under analysis/<run-name>/shared-shapes-v1. Source data is staged under /content and not modified.

Local regression: pytest -q tests/test_semmap_haken_shared_shapes.py. Colab end-to-end on the 100k input still needs to be executed in the user's runtime.
