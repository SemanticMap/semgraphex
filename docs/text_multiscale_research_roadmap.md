# Text corpora → multiscale semantic-network research (dissertation v5)

> **P1 slice 1 implemented in follow-on PR #28 (not the whole P1):** optional
> group-key train/val/test split in the existing H1 runner; a source-aware
> RESUME contract; an observed/shuffled text-distance control; and a thin
> `semmap_haken.text_h1` CLI. Independent text-only edge extraction,
> cross-corpus validation, and the full P1 null experiment remain open.

First bounded text/ConceptNet H1 run after completing the P0 audit:

```bash
python -m semmap_haken.text_h1 \
    --corpus /content/local-corpus \
    --doc-id 'Txt/example.txt' \
    --output /content/research-h1-example \
    --projection concept --max-vertices 10000 --max-edges 200000 \
    --restarts 10 --nulls 99 --workers auto
```

The H1 fitter uses graph-tool on **CPU/OpenMP**; CUDA is not a model fitting
backend. Distance-sensitive occurrence H1 requires
`--projection occurrence --max-distance-chars 500`.
Position shuffling requires a distance limit because the current H1 nSBM
does not model edge weights: shuffling without a window is an ineffective
topological null, which the CLI now rejects. NEW/RESUME manifests are
bound to the origin corpus and the grouping scheme, so changes cannot
silently reuse old fits.


This work starts from
`feature/text-corpus-colab-drive-resume` at
`f4203587b24d0d23bafa0cd6067244f70caab3bb`, without modifying
that production Colab processing branch. The dissertation's central
hypothesis is a **reproducible multilevel organization across sampling
scales**, requiring convergent statistical, structural-information,
dynamic, and semantic evidence. Finding nested blocks alone is not a
verification.

## Critical methodological distinction

Text–ConceptNet graphs in this repository are **induced by ConceptNet**:
a pair of occurrences is linked precisely when their mapped concepts
have an assertion in the snapshot. A document may produce millions of
edges from one assertion and high token frequencies. Therefore this
network is a text-conditioned *projection of ConceptNet*, NOT the
independent semantic network requested for external validation.

The induced full graph is determined by the occurrence table, the
assertion table and the Cartesian expansion rule excluding self
occurrences. Capped `edges_part_*.parquet` are an archival choice,
not evidence of absent links; their prefix bias must not be used in
primary H1 fitting.

## P0 — evidence and research views (implemented in this PR)

1. `text_research.audit_corpus`: checks root `COMPLETED` and run SHA,
   per-document manifests, checksums and consistency; states
   `full`, `capped`, `unknown_legacy`. Only SHA-verified full
   published graphs are admitted to H1 *if using archived edges*.
   Verification cannot be turned off while claiming research eligibility.
2. `open_research_views`: opens SHA-verified Parquet sources on a
   temporary DuckDB connection; defines lazy `research_occurrences`,
   `research_assertions`, `research_concept_projection`,
   `research_expanded_edges`, and a seeded position-shuffle control.
   The concept projection obtains exact edge counts from frequencies
   without expanding an O(n²) relation. Scale views select edges by
   absolute original Unicode character distance. Capped archives are
   reconstructible in full because **all original occurrences and
   assertions are retained**.
3. `load_typed_text_graph`: directly converts a *bounded, complete
   requested view* to the existing H1 `TypedGraph` without writing a
   new graph database. Refuses exceeding `max_vertices`/`max_edges`
   rather than taking a prefix. Preserves direction, types,
   original assertion provenance, occurrence IDs, distances,
   and `holdout_groups` that unite all occurrences/reciprocals of
   the same unordered concept pair and relation.

Example (run on a completed Drive-staged local corpus):

```bash
python -m semmap_haken.text_research /content/local-corpus
python -m semmap_haken.text_research /content/local-corpus \
    --doc-id 'Txt/example.txt' --scale 100 --scale 1000
```

An explicit `--no-verify` gives a **metadata-only unchecked audit**,
never an admissible scientific result. Do not use it with `--doc-id`.

Python API:

```python
from semmap_haken.text_research import audit_corpus, open_research_views
from semmap_haken.text_research_adapter import load_typed_text_graph

audit = audit_corpus('/content/local-corpus')  # SHA-256 verified
with open_research_views('/content/local-corpus', 'Txt/example.txt') as v:
    n_all = v.full_edge_count()    # does not enumerate all word pairs
    window = v.distance_view(100) # lazy view, materialize only when queried
typed = load_typed_text_graph(
    '/content/local-corpus', 'Txt/example.txt',
    projection='concept', max_vertices=10000, max_edges=200000)
```

P0 limitations: enumerating a full occurrence graph at huge scales can still
be costly, despite lazy views; the TypedGraph adapter deliberately
refuses oversized inputs. No model fitting or independent text network is
claimed. Existing long-document Google Drive checkpoint and resume
formats remain unchanged. Full SHA audit of very large corpora is I/O heavy.

**P0 acceptance gates:** no false admission of capped or unverified
legacy archives, counts agree with exact small-graph expansion, text
distance windows match direct enumeration, source assertion identity
is retained, no silent prefix clipping, and both source layouts are
covered by CI.

## P1 — statistical hierarchy (next, blocked on P0 gates)

1. Implement grouped relation- and concept-pair-stratified train/val/test
   BEFORE nSBM fitting, ensuring all occurrence duplicates/reciprocals
   fall into one split. Do not leak held-out ConceptNet assertions through
   repeated occurrences.
2. Plug bounded research projections into existing
   `hypothesis/runner.py` and `inference.py`, retaining direction,
   multiplicity and relation type. Compare SBM/DC-SBM/nested DC-SBM at
   identical seeds/restart budget, MDL, held-out ranking and calibrated
   nulls (both configuration and flat DC-SBM).
3. Add position-permutation and frequency/ConceptNet controls. On a
   plain unweighted complete occurrence graph, position shuffling must
   preserve connectivity modulo vertex renaming; distance-sensitive
   windows *may* change. Separate positional and structural effects.
4. Implement truly independent corpus-only semantic graph
   (co-occurrence and/or dependency relations extracted without
   ConceptNet edge filtering); use ConceptNet only for post hoc
   annotation. Test corpus/domain-held-out transfer separately.
5. Assess stability across restarts, perturbations, p-sampling and
   independent documents; compare partitions on intersecting original
   concepts, not by arbitrary block indices.

**P1 gate:** at least two nontrivial nested levels plus model comparison,
independent held-out prediction, null significance with multiplicity
correction, reproducible membership stability; negative/inconclusive is
a valid scientific conclusion.

## P2 — independent channels + W model family

- Independent typed motifs/WL and exact graph dictionary + Huffman,
  comparing archive bytes to a fair non-grammar baseline INCLUDING
  "occurrences + assertions + deterministic expansion" baseline.
- Independent Markov Stability, directed/typed transfer and Haken slow
  subspace comparisons (linear diagnostics are not nonlinear slaving).
- Semantic GloVe/SVD and external topic validation on held-out labels,
  with frequency/degree controls and no use of validation labels in
  block discovery.
- For the accepted nested partitions estimate the finite, directed,
  relation-specific degree-aware step kernels `W_l`; test original-mass
  exposure and independent held-out prediction plus scale alignment.
  Reuse `hypothesis/projection.py`; do not claim a limiting graphon
  just because level matrices can be plotted.

**P2 gate:** every channel reports its own evidence, null uncertainty,
disagreements and failure outcomes without circular validation.

## P3 — joint W/S/I with exact residual

Build a joint probabilistic generative estimator, not a heuristic
degree-1 classifier:
`P(z_e=k | G), k∈{W,S,I}` with separately modeled directed/type
multiplicity. Distinguish true star/dust generators from low observed
degree in a finite W network. Keep exact residual `Δ_l` **distinct from
I/dust** and trace all block parameters back to level-0 vertices.

**P3 gate:** posterior calibration and predictive comparison vs W-only,
p-sampling consistency, missingness/sparsity sensitivity and honest
model selection. An unsuccessful W/S/I fit must not be presented as
support for additional components.

## Reuse / non-goals

Use existing `hypothesis/{graph,runner,inference,nulls,statistics,
hierarchy_validation,projection,selfsim_metrics}.py`,
`graph_dictionary.py`, `grammar_codec.py`, Wishart/typed WL,
`text_corpus.py`, `text_chunk_graph.py` and DuckDB/Parquet.
No new remote DB, service framework or copy of the H1 pipeline.
No changes to existing Colab experiment outputs or input data.
