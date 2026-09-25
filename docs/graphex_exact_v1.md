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
  Cross-figure boundaries are **never** silently relabeled as stars.
* `I`: exactly one non-loop edge record forming an isolated pair in the
  graph of the saved level. Rarity alone never makes an edge dust.
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
`net_saved_bytes` is a valid experimental result. The current pilot retains
full edge addresses and labels even for W: it establishes a correctness and
measurement baseline, **not** an optimized structure-eliding codec.

**Losslessness scope:** saved *per-relation CSR entries*. If source ConceptNet
rows were aggregated during preparation, their original multiplicities and
individual weights cannot be reconstructed from CSR. This limitation must
not be represented as raw ConceptNet TSV losslessness.

**Hierarchy scope:** encode one level at a time. Do not sum per-level
frequencies as if all levels were independent or assert a globally optimal
hierarchical code. Levels above zero represent the stored coarse graph.

## Required before enabling online MDL

1. Replace redundant per-edge W payloads with canonical prototype topology
   plus exact permutations, relation/weight payload and optional corrections.
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
