# nSBM → exact graph dictionary → Huffman: Colab guide

Open the [10k/100k Colab notebook](https://colab.research.google.com/github/SemanticMap/semgraphex/blob/feature/nsbm-huffman-grammar-cn10k/notebooks/20_nsbm_huffman_grammar_cn10k_cn100k_colab.ipynb).

1. Runtime → Change runtime type → GPU T4 or L4 (High-RAM recommended at 100k).
2. Mount Drive and verify the 10k TSV at `MyDrive/SemanticMap/colab/hypothesis/inputs/h1-cn10k-typed-full-01/conceptnet_en_10k_relations.tsv`. Set `INPUT_PATHS` if the actual location differs. This is a prepared **selected** ConceptNet subgraph, not a raw prefix.
3. `PROFILE="10k"`, `RUN_MODE="NEW"`, `GPU_REQUIRED=True`. Run all.
4. The notebook stages inputs to `/content`, pins Git revision, installs graph-tool in micromamba Python 3.12, enables CuPy on available GPU, checks code and runs CPU/OpenMP nSBM plus bounded threaded CSR/process VF2 grammar discovery.
5. Each run saves `manifest.json`, nSBM restarts, W-only empirical projection, varint and Huffman exact grammar ZIPs, full bundles with node/assertion IDs, compact baseline, `report.json` and `COMPLETED`. Only `COMPLETED` marks a fully published Drive checkpoint.
6. For interruption: `RUN_MODE="RESUME"`, same `SOURCE_REVISION`, `PROFILE`, `RUN_ID`, seed and settings. Never edit a stored manifest.
7. For 100k: `PROFILE="100k"`, set `INPUT_PATHS["100k"]` to a prepared relation-stratified 100k TSV. The notebook uses a distinct run folder automatically; `RUN_MODE="NEW"`.
8. The experiment does not assert nSBM gain from node count reduction or entropy proxies. Primary metric: complete bundle ZIP bytes over a compact exact baseline at the same record/identifier fidelity.

**GPU honesty:** CUDA handles degree-batch ranking and W block-intensity projection, not graph-tool inference or VF2. Smaller 10k graphs can be CPU bound. Exact semantic assertion/weight storage may dominate. The `wishart` mode is a first-level typed-WL + Wishart density-mode **proposal** using the identical exact codec and source graph. It is not the older recursive Wishart pipeline from PR #13. That full matched comparison remains a separate follow-up; legacy runs with aggregated edges are not comparable as exact assertion archives.

Source document: [experiment plan](nsbm_huffman_compression_plan.md).

## Version 2 priorities
- **Independent controls**: `wishart` uses no nSBM membership, `nsbm_wishart` is explicitly a hybrid. All methods scan the same full-graph candidate space and use the same discovery limit; search time is reported separately.
- **Measured acceptance**: after non-overlap resolution, whole symbol families enter the grammar only when the actual exact Huffman ZIP shrinks. Source-level port records, post-selection support, and code lengths are reported. ZIP/DEFLATE remains a separate ablation.
- **Recursion**: up to three informative nSBM scales, source-level `EdgeRecord` IDs and binary64 weights preserved, each reverse transition verified. A next level is retained only if its consolidated exact hierarchy archive is smaller. If no level wins, this is an explicit negative result, not a bug.
- **Recovery**: per-4k-centre VF2 scan shards, SHA-256 of every published grammar bundle and recursive checkpoint. Corruption, partial temporary files and revision mismatch never count as successful RESUME.
