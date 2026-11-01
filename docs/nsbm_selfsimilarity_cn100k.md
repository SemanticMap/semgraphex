## ConceptNet-100k: nSBM fit only → all four self-similarity tests

[Open the integrated 100k Colab notebook](https://colab.research.google.com/github/SemanticMap/semgraphex/blob/feature/nsbm-multiscale-self-similarity/notebooks/22_nsbm_self_similarity_cn100k_colab.ipynb).

### Two independent stages
**A. Fit nSBM-only** (graph-tool CPU/OpenMP, no VF2 or Huffman):
`semmap-nsbm-fit-only --input conceptnet_en_100k_relations.tsv --vertices 100000 --output fit_dir --restarts 3 --cpu-budget 12 --mode NEW`.

Writes `fit_data/{ids.json,source.npy,target.npy,relation.npy,weight.npy}`,
`nsbm_partitions.npz`, `fit_summary.json`, `manifest.json`,
`tasks/compression_nested_restart_*.{json,npz}`, `COMPLETED`.
The independent model fits and FIT task checkpoints are from the existing H1 path.
nSBM graph-tool is **CPU**, so select high-RAM, not necessarily T4/L4.

**B. Run four tests** in the same 100k notebook with `semmap-nsbm-selfsim`,
reusing only the verified 100k `fit_data` and `nsbm_partitions.npz`.
This stage uses SciPy sparse, not graph-tool; it runs on standard CPU Colab.

### Input preparation
A real, filtered five-column ConceptNet assertions file must represent exactly
100,000 **distinct** original concept URIs; "100k" in the Wishart run ID
is not a reliable vertex count. The previously calculated 10k nSBM labels
are **not portable** across distinct 100k node indices.

If the prepared 100k file is absent, the notebook optionally calls existing:
`semmap-conceptnet-10k --input SOURCE.tsv[.gz] --nodes 100000
--relations RelatedTo,IsA,PartOf,UsedFor,HasProperty,HasA
--max-assertions 2000000 --output conceptnet_en_100k_relations.tsv`.

This performs degree-seeded, relation-stratified BFS on a *bounded* filtered
ConceptNet source; it does not select a random sample. A cap of
`max_assertions` fails rather than selecting an arbitrary prefix.
For larger sources prepare a bounded, documented subcorpus or increase
the cap with adequate RAM; never silently truncate.

### Drive locations
`MyDrive/SemanticMap/colab/hypothesis/runs/nsbm-fit-cn100k-01/`
and
`MyDrive/SemanticMap/colab/hypothesis/runs/nsbm-selfsimilarity-cn100k-01/`.

The notebook stages everything on `/content`, synchronizes complete
checkpoint files every 120 seconds, and sets `COMPLETED` last.
Resume with `RUN_MODE='RESUME'` and *identical* scientific parameters and code.
After success, adding more nSBM restarts or control repeats requires a
new immutable run ID. One graph-tool restart cannot be recovered from the
middle of a single unfinished fit, but completed restarts are reused.

### Scale and scientific limitations
At 100k, exact nSBM training and 19 x 2 null-model calls can be costly;
pilot `BOX_REPEATS=2`, `MAX_SIMULTANEOUS_NULLS=2` conserves RAM.
Radii 0–4 give only exploratory box-cover scaling; 99+ controls plus
multiple independently refitted nested-SBMs are needed for stronger evidence.
The graphon holdout AUC is conditional and **leaky** because labels were
fitted on the full original graph. Do not interpret it as independent
predictive quality.

See `docs/nsbm_multiscale_selfsimilarity.md` for scientific definitions.
