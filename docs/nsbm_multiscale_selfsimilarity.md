# Four independent tests of nSBM multiscale self-similarity (SemMap)

## Goal

On exactly the saved ConceptNet-10k typed, directed **assertion** graph, test whether
patterns repeat across 22→7→3 nested-DC-SBM levels, not merely that nested
partitions exist. All four tests use the *same original-vertex-to-block mapping*
and two negative controls where applicable. Four diagnostics, not one score.

### 1. Structural: directed motifs

- Build weighted typed adjacency **only** for 22, 7 and 3 blocks; source 10k stays sparse.
- For each scale sample `MOTIF_SAMPLES` rooted connected wedges with center
  probability proportional to \(\binom{k_i}{2}\), take two distinct neighbors,
  label the induced 3-node directed triad up to all 6 vertex permutations.
- Report per-scale directed-motif distribution, sampled wedge closure rate,
  internalized assertion fraction and adjacent JS distances.
- At B=3 only one possible triple exists: low statistical power.
- Quotient aggregation induces new edges by construction. No tautological
  interpretation as fractality.

### 2. Typed block graphon

- For each relation \(r\) and original-vertex block \(a,b\), count directed
  assertion multiplicities \(E^{(r)}_{ab}\), with **block masses**
  \(n_a n_b\) in the denominator. Loops are allowed in our model.
- Normalize relation-wise intensity by empirical global relation density,
  sample the **same original vertex pairs** across the three levels, compare
  log-intensity distributions with JS and RMSE.
- Supplement with an 80/20 **unique (subject, relation, object)** holdout and
  relation-conditioned uniform absent-triple negatives. Report AUC.
- CRITICAL: the supplied nSBM labels were inferred from the FULL graph.
  Such AUC has partition leakage. It is diagnostic only; independent
  predictive evaluation requires re-fitting nSBM on training edges.

### 3. Haken/slow diffusion

- Construct symmetric, unit-multiplicity adjacency on the same original graph
  (sum of both directions plus one identity self-loop per original vertex).
- Use normalized diffusion \(S=D^{-1/2}WD^{-1/2}\), with teleport
  \(S_\epsilon=(1-\epsilon)S+\epsilon qq^T\) and
  \(q\propto\sqrt d\). This distinguishes nontrivial slow modes.
- The linear Haken-inspired Jacobian \(J=-\alpha I+\beta S_\epsilon\)
  gives relaxation rates. Compare **nontrivial eigenvalues** and the
  squared principal cosines between original and correctly mass-lifted
  coarse eigenspaces: \(L=D^{1/2} M D_B^{-1/2}\).
- Not a test of nonlinear slaving. Direction and relation types are
  symmetrized solely for this probe. n=3 admits at most two nontrivial modes.

### 4. Independent radius-ball box covering + multi-q

- On undirected sparse support, greedy radius-\(r\) burning covers every
  original vertex exactly once, with each assigned ball diameter at most
  \(2r\). This yields a **heuristic upper bound** on optimum box count.
- Repeat randomized degree-priority cover, radius \(r=0,…,5\), with
  \(\ell=2r+1\); regress log box count vs log(ell) (power law) and ell
  (exponential) with comparable two-parameter AIC scores. Only a
  descriptive preference, not significance or proof.
- Mass \(\mu_i=n_i/n\); partition function \(Z(q,\ell)=\sum_i\mu_i^q\),
  q=-2,-1,0,1,2. \(D_q=\tau(q)/(q-1)\) for q≠1 and Shannon entropy
  slope for q=1. Negative q sensitive to rare singleton boxes.
- Box covering does **not** derive from nSBM partitions. This is crucial for
  testing real scale dependence, independently of the model hierarchy.

## Controls

1. **degree-stratified nested-label permutations:** one vertex permutation
   applied to all original-vertex levels; exact hierarchy and block sizes
   preserved. Conditional control, not resampled nSBM inference.
2. **relation-wise directed stub matching:** permute targets among assertions
   of the same relation. Preserves each vertex's in-/out-degree per relation,
   global multiplicity and weights collection. Allows self-loops/multiedges;
   not a uniform simple-graph null. The original fixed hierarchy is reused.
3. Compare observed score to both ensembles with empirical upper-tail
   \(p=(1+\#\{T_{null}\geq T_{obs}\})/(1+R)\), without multiple-test
   correction. For box covering the label-shuffle control is meaningless;
   use stub matching only.
4. At least 19 nulls for minimum p=0.05; 99+ for confirmatory thresholds.
   Validate across several independent nSBM restart fits for strong claims.

## Reproducible Colab

[Run ready notebook](https://colab.research.google.com/github/SemanticMap/semgraphex/blob/feature/nsbm-multiscale-self-similarity/notebooks/21_nsbm_self_similarity_four_tests_colab.ipynb)

Notebook reads original `fit_data/{ids.json,source.npy,target.npy,relation.npy,weight.npy}`,
`nsbm_partitions.npz` and original `manifest.json` from
`MyDrive/SemanticMap/colab/hypothesis/runs/nsbm-huffman-10k-v3`.
It validates original SHA and exact original vertex order; no retraining.
It pins one Git SHA, runs a small smoke, computes locally, atomically writes
per-null SHA checkpoints, periodically syncs to Drive. RESUME requires
identical input, partition, config and code revision.

Primary CLI:

```bash
semmap-nsbm-selfsim --graph-dir /path/to/fit_data \
  --partitions /path/to/nsbm_partitions.npz \
  --source-manifest /path/to/nsbm-huffman-10k-v3/manifest.json \
  --output /tmp/selfsimilarity-run --mode NEW \
  --null-repeats 19 --jobs 8 --motif-samples 2500 \
  --pair-samples 12000 --box-repeats 3 --box-radii 0,1,2,3,4,5
```

Final artifacts: `manifest.json`, `original_checkpoint.json`,
`null_permuted_partitions_*.json`, `null_typed_degree_rewire_*.json`,
`result_checkpoint.json`, `report.json`, `COMPLETED`.

## Scientific interpretation

**Hierarchy ≠ self-similarity ≠ fractality ≠ multifractality.**
Cross-level similarity is not enough, since quotient blocks are nested
by construction. Need null-overperformance and independent scale law.

References:
- Song, Havlin, Makse (2005), Self-similarity of complex networks.
  https://doi.org/10.1038/nature03248
- Gfeller & De Los Rios (2007), Spectral coarse graining of complex networks.
  https://doi.org/10.1103/PhysRevLett.99.038701
