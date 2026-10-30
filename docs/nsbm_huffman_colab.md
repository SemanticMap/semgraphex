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

**GPU honesty:** CUDA handles degree-batch ranking and W block-intensity projection, not graph-tool inference or VF2. Smaller 10k graphs can be CPU bound. Exact semantic assertion/weight storage may dominate. A full Wishart matched comparison on this same assertion-preserving graph is a separate follow-up; legacy Wishart outputs with aggregated edges cannot be used as the primary lossless baseline.

Source document: [experiment plan](nsbm_huffman_compression_plan.md).
