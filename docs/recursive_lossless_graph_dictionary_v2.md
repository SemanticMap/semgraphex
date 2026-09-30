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

The existing mdl_gain_bits_proxy is retained for candidate selection and must
not be reported as measured storage compression.
