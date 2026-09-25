# Graphex Exact v1 — offline CSR-record pilot

This branch introduces a **read-only pilot**, not the full online Wishart
selection algorithm and not an estimator of a limiting probabilistic graphex.

## Run

```bash
python -m semmap_haken.graphex_offline \
  --run /content/drive/MyDrive/SemanticMap/semgraphex/runs/<completed-run> \
  --level 0 --output /content/graphex-level-000.zip
```

Run separately with `--level N` on each saved source level. The input must
have `COMPLETED`, `level_NNN/adjacency.npz`,
`level_NNN/relations/index.json`, and
`transition_NNN_NNN+1/figure_occurrences.jsonl`. The pilot does not modify
old checkpoints or declare older checkpoint formats to be graphex mode.

## Semantics and guarantees

* `W`: records internal to an accepted figure. Repeated internal shapes
  are matched with relation- and node-type-aware VF2, not merely by degree.
  A deterministic port profile selects an isomorphism on symmetries.
* `S`: directed and typed records with a true leaf outside all figures.
  The leaf has exactly **one distinct structural neighbor** (not one CSR
  record); reciprocal directed rows, multiple relation labels and weights
  remain separately encoded. Cross-figure boundaries are **never** silently
  relabeled as stars.
* `I`: all exact stored edge records in an isolated two-vertex component
  of the underlying simple, undirected support graph at the saved level.
  A reciprocal CSR pair counts as one structural connection, but both records
  survive decoding. An isolated pair with several relation records is a typed
  finite-codec pattern, **not** automatically a classical simple-graphex
  dust realization. Rarity alone never makes an edge dust.
* `R`: every other record, including connections between figures.
  Every record is assigned exactly once. Parallel records have distinct IDs.

The archive stores version, vertices, complete W-shape registry, placements,
exact edge records and a **bit-packed canonical Huffman stream** of W/S/I/R
component symbols. Component and conditional S|W frequencies are reported.
The decoder checks format, code stream and IDs and returns directed, typed,
weighted records; the encoder asserts equality with the original records.

The report's `archive_bytes` is the actual on-disk size, including the ZIP
container, codebook, shapes, placements and all payloads. `baseline_bytes`
is an independently DEFLATE-compressed exact-graph container. A negative
`net_saved_bytes` is a valid experimental result. The pilot reconstructs W topology, orientation and relations from shared
shape prototypes and occurrence maps; per-W-record payload contains record
ID, weight and its prototype-edge position. S/I/R still retain their exact
addresses and relation labels. It establishes a correctness and measurement
baseline, **not** a fully optimized conditional/joint graph-word codec.

**Losslessness scope:** saved *per-relation CSR entries*. If source ConceptNet
rows were aggregated during preparation, their original multiplicities and
individual weights cannot be reconstructed from CSR. This limitation must
not be represented as raw ConceptNet TSV losslessness.

**Hierarchy scope:** encode one level at a time. Do not sum per-level
frequencies as if all levels were independent or assert a globally optimal
hierarchical code. Levels above zero represent the stored coarse graph.

## Required before enabling online MDL

1. Optimize the W stream further with compact permutations, typed child
   symbols and corrections for non-prototype records.
2. Add explicit S patterns with leaf-target address coding and I pattern
   dictionaries, plus joint words, conditional S|W and marginal streams.
3. Compare **complete serialized** lengths for exact-type, independent
   components, conditional and joint codes, including dictionary one-time
   costs and hierarchy/child links.
4. Implement global cross-level edge ownership, exact expansion through
   child words and preserved original row IDs where available.
5. Add versioned `graphex_exact_v1` config/checkpoints and reject old
   checkpoint versions rather than silently migrating.
6. Couple measured *marginal* code length to conflict-aware online Wishart
   selection and prove RESUME produces identical archives.
7. Evaluate statistical graphex `(W,S,I)` separately using an appropriate
   sampling design, not two compression settings of the same ConceptNet.

The online `graph_mdl.py` `bits_proxy` is **unchanged** in this pilot and
must not be described as an actual encoded length.

## Structural-degree diagnostic (Wishart CN100k k4)

The previous first offline run produced `S=0` on every level because the
classifier counted **records** rather than structural neighbors. In the saved
`level_000` all six relation-layer CSR matrices are symmetric. An undirected
leaf therefore has two directed records and never passed `record_degree==1`.
Direct inspection of that level found 4,021 degree-one vertices and 426 eligible
figure-to-leaf structural pairs, represented by **852 directed relation records**.
The underlying graph had no isolated two-vertex components at level zero.

The CPU and CUDA classifiers now deduplicate unordered neighbor pairs for
*eligibility* without deduplicating any records in the encoded payload. The
encoder's roundtrip assertion is unchanged. For a corrected experiment use a
**new RUN_ID**, e.g. `wishart-cn100k-k4-graphex-exact-v1-structural-degree-v2`;
existing archives are not migrated or overwritten.

Note that symmetry was introduced by the saved graph representation: one
cannot reconstruct original ConceptNet row orientation merely by decoding
the symmetric relation-layer CSR matrices. To make a raw ConceptNet lossless
claim, preserve row provenance before this symmetrization.
