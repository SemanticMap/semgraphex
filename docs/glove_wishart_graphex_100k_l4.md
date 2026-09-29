# ConceptNet 100k → GloVe → Wishart → finite Graphex W/S/I/R → exact decoder → Haken

**Status:** runnable 100k pilot. Two Colab notebooks in this branch: 10 is alignment/GPU-kNN/Wishart; 11 is finite empirical graphex, exact codec and dynamics. Run on L4 (High-RAM when available), on the same Google account that owns `MyDrive/SemanticMap/colab/wishart`.

## Immutable input and exactness contract

Default is the *actual 100,000-vertex* `level_000` from the completed `runs/wishart-dictionary-100k-gpu-gpu` experiment. Before computation, notebooks assert `adjacency.shape==(100000,100000)`. The experiment fingerprints adjacency, membership, all relation layers, GloVe text, code commit and settings. OOV and noise are retained as separate nodes during quotient formation. Only English ConceptNet URI lexical labels are aligned; phrases use a mean of normalized matched token embeddings; coverage flags distinguish full, partial and OOV. The official `glove.6B.100d.txt` is cached at `data/glove.6B.100d.txt`.

**Losslessness scope is the prepared graph, not the original ConceptNet TSV.** Saved `level_000` may already be symmetrized, weighted, or have parallel assertions merged. The exact codec preserves the indexed list of concept memberships, original `adjacency.npz` bytes, *each directed weighted record from each stored typed CSR layer*, and stable record IDs for those stored records. It cannot recover information lost in prior preprocessing.

## Actual implemented methods

1. PyTorch L4 FP32 query-tiled exact cosine top-k, 4,096-query restart shards; a shard is published atomically to the Drive run as it finishes. Do not treat FP32 near-ties as bitwise cross-device deterministic.
2. The existing `semmap_haken.wishart_cluster.wishart_cluster` forms GloVe density families; unembedded nodes remain OOV. Large clusters are not automatically taken as exact graph types.
3. A **bounded two-port** structural dictionary accepts only disjoint, repeated directed/typed edge-pair motifs within the same Wishart family. This is deliberately narrower than the complete recursive graph grammar. The number and minimum support are configurable.
4. An empirical **directed typed block-intensity** model gives finite W statistics. Its values are observed record count divided by the ordered block-pair exposure; because multiedges are possible, this is intensity, not necessarily a Bernoulli probability or a fitted infinite exchangeable graphex. Per-family incoming/outgoing external interfaces are reported separately by ConceptNet relation.
5. The existing `graphex_components.classify_edges` separates W motif-internal, S true-leaf, I isolated-pair and R remaining exact records. The archive uses canonical Huffman for W shape-occurrence symbols and the full per-node cluster-label stream; prototypes reconstruct internal endpoints/relation. Columnar source-delta varint streams preserve the exact S/I/R records; W stores exact edge IDs and binary64 weights. The complete archive includes W statistics, external connector profiles, memberships, adjacency, codebook, shapes, placements and residuals. No predictor is silently substituted for observed edges.
6. The independently written baseline ZIP encodes the **same exact graph and memberships** using per-record JSONL plus identical adjacency bytes. Actual whole-ZIP bytes, rather than the older `mdl_ratio_proxy`, decide which of no motif, 1,024 motifs or configured maximum is retained. All candidate reports are saved, including negative compression.
7. The Haken diagnostic uses `J=-alpha I+beta S`, with the **same alpha/beta** on original and quotient, compares the largest normalized-adjacency eigenvalues, excludes eigenvalues ~1 from principal-angle calculations, reports stationarity TV and retains directed quotient edges. Symmetrization is diagnostic-only.

## Checkpoint and resume

Both notebooks work on `/content`; stage output is incrementally synced to `MyDrive/SemanticMap/colab/wishart/runs/wishart-glove-graphex-cn100k-01`. The first notebook saves `embedding`, CUDA `knn` shards and Wishart checkpoints. The second restores the same run, rejects changed source/GloVe/config/Git revision, executes full-size codec tests and Haken, then writes `COMPLETED` **only after** exact decoder verification. A corrupted checkpoint raises an exception rather than resuming silently. `COMPLETED` is copied last to Drive.

## Success gates and limitations

- Mandatory: exact decoded equality for prepared CSR records, relation labels, directions, weights, membership URIs and the original adjacency file; no drop of OOV nodes.
- Measure physical archive bytes including dictionaries, side information and residual. It may be larger than the baseline; 10% saving is an experimental target, **not** an asserted outcome.
- Report nontrivial slow eigenspace angles and external connector profiles. Zero eigenvalue error for the exact decoded archive is a correctness check; nonzero quotient error is a measured approximation cost.
- CPU CI uses a tiny graph only. A full 100k L4 run is **not** claimed until both notebooks complete and `verification.json`, `codec_report.json`, `dynamics/haken.json` and `COMPLETED` are present on Drive.
- This version does **not** implement general multi-level grammar, a fitted infinite graphex limit, exact restoration of pre-preparation TSV rows, nonlinear center-manifold validation, or multi-seed statistical comparison. Those require separate experiments; do not infer them from this pilot.
