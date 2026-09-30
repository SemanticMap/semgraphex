# Recursive Wishart compression-figure discovery

This branch adds an experimental, sparse-first framework for discovering
**unknown compression figures** in ConceptNet.  The figure vocabulary is not
predefined.  Radius-`r` ego-subgraphs are compared by a selected graph metric,
Wishart mode analysis discovers dense structural types, non-overlapping
instances of those types are contracted, and the procedure repeats on the
quotient graph.

## Methodological status

This is an experimental synthesis.  Wishart clustering discovers modes in the
space induced by the chosen metric; it does **not** prove that a cluster is a
semantic primitive.  Each accepted ego occurrence is contracted independently:
two distant but structurally similar occurrences are never collapsed into the
same supernode.

The implemented Wishart sweep follows the k-nearest-neighbour mode-analysis
logic: candidates are processed from high to low local density and two
established density modes are not joined across a sufficiently low saddle.
The implementation uses `-log(r_k)` as a dimension-free density-height proxy.

Primary Wishart references:

- D. Wishart, "Numerical Classification Method for deriving Natural Classes",
  *Nature* 221, 97-98 (1969), DOI: https://doi.org/10.1038/221097a0.
- D. Wishart, "Mode analysis: A generalization of nearest neighbour which
  reduces chaining effects", *Numerical Taxonomy* (1969), pp. 282-311.

## Metrics

Set `wishart.metric` to one of:

- **`typed_wl`** — hashed Weisfeiler-Lehman subtree features.  Edge relation
  types are part of the WL messages; concept text is not used.  Fast baseline
  for large candidate sets.  Reference: Shervashidze et al.,
  "Weisfeiler-Lehman Graph Kernels", JMLR 12 (2011):
  https://www.jmlr.org/papers/v12/shervashidze11a.html.
- **`graphlet`** — pre-sampled typed induced graphlet/motif frequencies.
  Sampling bounds cost when ego graphs grow.  Reference: Shervashidze et al.,
  "Efficient Graphlet Kernels for Large Graph Comparison", AISTATS 2009:
  https://proceedings.mlr.press/v5/shervashidze09a.html.
- **`lowrank_gw`** — POT
  `lowrank_gromov_wasserstein_samples` using topology-only spectral
  coordinates of ego nodes.  Reference: Scetbon, Peyre & Cuturi,
  "Linear-Time Gromov-Wasserstein Distances using Low Rank Couplings and
  Costs", ICML 2022:
  https://proceedings.mlr.press/v162/scetbon22b.html.
- **`fgw`** — Fused Gromov-Wasserstein: shortest-path structure plus
  relation-incidence node features, evaluated on deterministic degree
  landmarks.  Reference: Vayer et al., "Optimal Transport for structured data
  with application on graphs", ICML 2019:
  https://proceedings.mlr.press/v97/titouan19a.html.
- **`relation_js`** — square-root Jensen-Shannon distance between normalized
  relation histograms.  This is the cheapest relation-semantic channel but does
  not by itself encode topology.

GW/FGW are optional because of their cost:

```bash
pip install -c requirements/constraints.txt -e '.[dev,wishart]'
```

The remaining metrics need only the base package.

## End-to-end run

The runner can use a completed `semmap-haken prepare` artifact:

```bash
python scripts/run_wishart_coarsening.py \
  --config configs/wishart_conceptnet_small.yaml \
  --prepared tmp/runs/prepare-smoke \
  --output tmp/runs/wishart-smoke
```

or read the ConceptNet assertion file referenced by `dataset.path` directly:

```bash
python scripts/run_wishart_coarsening.py \
  --config configs/wishart_conceptnet_small.yaml
```

## What is saved at every level

Each `level_XXX/` directory contains:

- `adjacency.npz` — current sparse graph;
- `relations/*.npz` — relation-specific sparse layers;
- `membership.json` — original ConceptNet concepts represented by every
  current node, plus the full `concept_concat` string;
- `dynamic_metrics.json`;
- `dynamic_vectors.npz` — stationary mass, slow eigenvalues/eigenvectors,
  approximate betweenness and sampled macro distances.

The dynamic snapshot records:

- mean degree and degree second moment;
- heterogeneous-mean-field spreading-threshold proxy;
- configuration-model percolation-threshold proxy;
- spectral gap;
- Laplacian synchronizability ratio;
- Monte-Carlo MFPT and hit rate;
- sampled betweenness, maximum betweenness and congestion-threshold proxy;
- macro clustering;
- sampled inter-node/inter-cluster path lengths.

These are diagnostics, not universal invariants.  The formulas and approximation
caveats are persisted in `dynamic_metrics.json`.

Dynamic-process background: Barrat, Barthelemy & Vespignani,
*Dynamical Processes on Complex Networks*, Cambridge University Press (2008).
For preserving slow dynamics under coarse graining see Gfeller & De Los Rios,
"Spectral Coarse Graining of Complex Networks", PRL 99, 038701 (2007):
https://doi.org/10.1103/PhysRevLett.99.038701.

## What is saved at every contraction

Each `transition_XXX_YYY/` directory contains:

- `wishart.json` — metric backend, mode counts, peak densities and noise;
- `candidate_labels.npz` — Wishart labels, kNN radii and density proxy;
- `figure_occurrences.jsonl` — each selected non-overlapping compression
  figure, its type, fine nodes, original concepts and `concept_concat`;
- `fine_to_coarse.npy` — exact contraction map;
- `cluster_dynamics.jsonl` — per-supernode:
  - external flow,
  - external flow by ConceptNet relation,
  - stationary mass,
  - sparse exit-probability distribution,
  - original concepts,
  - figure type when the node was created by a discovered figure.

## Interpretation of the recursion

At level `s`:

```text
G_s
  -> ego candidates H_i
  -> chosen metric d(H_i, H_j)
  -> Wishart density modes
  -> non-overlapping figure occurrences
  -> quotient G_{s+1}
  -> recompute metrics and repeat
```

The figure vocabulary is recomputed at every level.  Persistent types across
levels are therefore empirical evidence for cross-scale structural recurrence,
not an artifact of reusing a fixed motif dictionary.

## Scaling notes

- `typed_wl` and `graphlet` build feature vectors and use kNN search without
  an all-pairs distance matrix.
- `relation_js` is currently pairwise over candidates.
- `lowrank_gw` and `fgw` are pairwise transport metrics and therefore
  enforce `transport_max_candidates`.
- MFPT, betweenness, clustering and macro distances are sampled/Monte-Carlo
  diagnostics, with sample counts controlled by YAML.
- No dense full-ConceptNet adjacency is created.
