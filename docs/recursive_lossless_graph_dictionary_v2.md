# Recursive lossless graph dictionary v2

This development path separates exact graph storage from semantic/statistical
analysis.

## Invariants

1. Every emitted transition grammar archive must round-trip its source
   relation-layer CSR entries exactly before contraction.
2. Internal topology and external interface are different identities:
   GraphType remains the full exact type, while shape_id identifies reusable
   internal structure and interface_variant_id identifies its boundary pattern.
3. GloVe/Wishart may rank structural candidates but must not be required for a
   motif to exist in the lossless dictionary.
4. Coarse matrices produced by P^T A P are derived analysis views. They are not
   sufficient evidence of reversibility.
5. Report measured archive bytes separately from the existing structural MDL
   proxy.

## New artifacts

When dictionary.codec.emit_transition_archives=true:

    transition_XXX_YYY/
      grammar_exact_v2.zip

The archive contains only decoder-relevant data:

    manifest.json
    shapes.json
    variants.json
    rules.json
    occurrences.bin
    internal.bin
    ports.bin
    residual.bin

Wishart labels, GloVe vectors, graphex statistics and Haken diagnostics remain
normal analysis artifacts outside the exact archive.

The persistent dictionary also writes:

    dictionary/internal_shapes.jsonl
    dictionary/interface_variants.jsonl

so boundary variants can reuse one exact internal shape.

## Current recursion contract

Higher-level EgoCandidate.node_types already carry lower-level GraphType IDs.
grammar_exact_v2 stores those node types in its rule child_symbols. Therefore
transition archives expose recursive composition while each transition remains
independently decodable. A later consolidation step can encode only the final
start graph plus the shared rule DAG instead of retaining independent
transition archives.

## Compression measurement

Each exact transition report records:

- archive_bytes;
- baseline_bytes;
- compression_ratio;
- shapes and interface variants;
- internal, port and residual record counts;
- compressed bytes by archive entry.

The existing mdl_gain_bits_proxy remains available as a baseline selection
objective and must not be reported as measured storage compression.


## Consolidated recursive archive

When exact transition archives are enabled, the run now also builds:

    hierarchy_exact_v1.zip
    hierarchy_codec_report.json

The consolidated archive stores the final coarse relation graph once and keeps
only transition grammar deltas for contracted figures. During reverse decoding:

1. coarse edges between two untouched singleton nodes are inherited to the
   previous level;
2. every edge touching a contracted figure is discarded from the coarse view
   and restored from internal/port payload;
3. this is repeated from the final level down to level 0.

The current consolidated format requires:

    wishart.aggregation: sum

because unchanged singleton-to-singleton edge weights then have a one-to-one
preimage. A run publishes COMPLETED only after the consolidated archive decodes
exactly to the level-0 relation-layer records.

## Selection objective

The grammar-v2 configuration uses:

    dictionary:
      selection:
        objective: grammar_v2_logical

This logical cost includes:

- amortized rule definition;
- binary occurrence placement;
- exact internal-edge payload;
- interface/port payload.

It is a selection model aligned with the codec layout, but it is still not the
measured DEFLATE container size. The exact transition and hierarchy archive
reports remain the authoritative storage measurements.

Port-heavy motifs are therefore penalized before contraction rather than only
after archive construction.

## Incremental full-graph census

The grammar-v2 configuration uses conservative incremental frequency scanning:

    frequency_scan: incremental
    frequency_full_rescan_every: 4

Only exact matches whose centers are farther than radius+1 from every contracted
node are carried to the next level. All affected centers and all previous
no-match centers are scanned again. Every fourth level forces a complete scan.
After RESUME the first census is also complete because the transient occurrence
cache is intentionally not part of the checkpoint contract.

The reusable data structure is in:

    src/semmap_haken/occurrence_index.py

This optimization must preserve exact type counts; it is not allowed to change
the scientific clustering input.

## Grammar-induced graphon/graphex projection

Each accepted transition writes:

    transition_XXX_YYY/dictionary_graphex.json

The file is deliberately described as a finite empirical grammar-induced typed
block model, not as proof of convergence to an exchangeable graphon/graphex
limit.

Mass is propagated from the number of original level-0 concepts represented by
each current node, not from the number of coarse nodes:

    mu(T) = original nodes represented by T / N0

For each relation r the projection records an empirical block intensity between
grammar symbols. The current implementation estimates the W-like block kernel.
S and I are explicitly reported as "not estimated"; codec W/S/I/R labels are
not reinterpreted as mathematical graphex components.

## Adjacency redundancy diagnostic

The exact transition report includes an adjacency-from-relations diagnostic.
The codec does not store adjacency.npz. It records whether the current adjacency
can be reproduced numerically as the sum of relation layers and reports maximum
absolute error and weight sums.

This is an integrity observation, not an assumption: if the diagnostic fails,
adjacency must not be claimed to be redundant for that run.

## Colab / L4 execution

Notebook:

    notebooks/13_recursive_lossless_grammar_v2_colab.ipynb

The notebook contains only parameters, environment setup, the existing
semmap-wishart-colab invocation and publication checks. Heavy computation stays
in package modules. NEW and RESUME runs inherit the existing strict config,
input, Git-revision and compute-device lineage checks.

The notebook validates per-transition exact codec reports after publication.
The hierarchy run itself is responsible for the stronger consolidated
final-to-level0 roundtrip before COMPLETED is written.
