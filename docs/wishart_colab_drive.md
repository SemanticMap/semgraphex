# Recursive Wishart in Google Colab with Google Drive

This branch keeps the scientific Wishart pipeline identical to the local CLI and
adds a Colab storage/execution adapter.

## Design

Do **not** run the sparse hierarchy directly inside `/content/drive`.

Google Drive in Colab is mounted through a FUSE layer and is much slower than
the ephemeral local filesystem for repeated small/random reads and writes.
The Colab runner therefore uses:

```text
Google Drive input
        |
        v
/content/semmap-wishart/inputs     <- one incremental staging copy
        |
        v
local recursive Wishart compute
        |
        +--> level_000
        +--> transition_000_001
        +--> level_001
        ...
        |
        v
incremental checkpoint sync
        |
        v
/content/drive/MyDrive/SemanticMap/colab/wishart/runs/<run-name>
```

`COMPLETED` is synchronized only after the final files have been copied.

## Install in Colab

From the checked-out repository:

```bash
pip install -c requirements/constraints-colab.txt -e '.[wishart,notebook,gpu]'
```

Mount Drive in a notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Recommended Drive layout

```text
MyDrive/
  SemanticMap/
    colab/
      wishart/
        config/
          experiment.yaml
        data/
          conceptnet_en_100k.tsv
        prepared/
        prepare-xxxxxxxxxxxx/
          adjacency.npz
          ...
          COMPLETED
        runs/
```

## Run from a prepared graph on Drive

This is the preferred path because the ConceptNet parse/build stage is reused:

```bash
semmap-wishart-colab \
  --config configs/wishart_conceptnet_colab.yaml \
  --drive-root /content/drive/MyDrive/SemanticMap/colab/wishart \
  --prepared-drive prepared/prepare-xxxxxxxxxxxx \
  --run-name wishart-typed-wl-01
```

The prepared directory is copied once to local scratch:

```text
/content/semmap-wishart/inputs/prepared/prepare-xxxxxxxxxxxx
```

and all heavy computation runs there.

## Run from a ConceptNet file on Drive

```bash
semmap-wishart-colab \
  --config configs/wishart_conceptnet_colab.yaml \
  --drive-root /content/drive/MyDrive/SemanticMap/semgraphex \
  --dataset-drive data/conceptnet_en_100k.tsv \
  --run-name wishart-typed-wl-01
```

The dataset is staged to local scratch before parsing.

## Checkpoint behavior

After every completed `level_XXX` and `transition_XXX_YYY` the runner copies
only new/changed files to Drive.

Drive contains:

```text
runs/wishart-typed-wl-01/
  input.json
  DRIVE_CHECKPOINT.json
  level_000/
  transition_000_001/
  level_001/
  ...
  hierarchy.json
  COLAB_RUN.json
  COMPLETED
```

If the run raises an exception, the runner writes `FAILED.json` locally and
syncs all completed partial artifacts without a `COMPLETED` marker.

The graph-dictionary runner additionally saves a checksummed, immutable
`checkpoints/level_XXX/state.pkl.gz` and `manifest.json` after each fully
contracted level. Re-run with the same `--run-name` plus `--resume` to restore
the latest verified state. The CLI checks the original input SHA256, YAML SHA256,
Git revision, and CPU/CUDA device family. Never combine `--resume` and
`--overwrite`; never load a pickle checkpoint from an untrusted source.

## Scratch cleanup

By default a successful run is removed from local `/content` after the final
Drive synchronization.

Keep it for interactive inspection with:

```bash
semmap-wishart-colab ... --keep-scratch
```

## File comparison mode

Default:

```bash
--compare-mode size_mtime
```

This avoids rereading large NPZ files during every checkpoint.

For more expensive checksum validation:

```bash
--compare-mode sha256
```

## CPU/thread behavior

Most current Wishart metrics are CPU-bound. The Colab runner defaults to:

```bash
--blas-threads 1
```

to avoid nested BLAS/OpenMP oversubscription on small Colab CPU allocations.

This does not change the scientific metric. It only limits native thread pools.

## Colab-specific memory optimizations

### Typed WL

Incoming relation layers are transposed to CSR once per candidate rather than
performing repeated sparse column extraction inside every WL iteration.

### Relation histogram + sqrt(JS)

The old implementation materialized an `N x N` all-pairs distance matrix.

The Colab branch computes top-k neighbors in blocks:

```text
working memory ~ O(block_size * N + N * k)
```

Configure:

```yaml
wishart:
  relation_js_block_size: 128
```

### Dynamic diagnostics

The Wishart dynamics code no longer converts the complete sparse graph into a
NetworkX object for sampled diagnostics.

It uses:
- CSR-native source-sampled Brandes betweenness;
- sparse sampled clustering;
- SciPy sparse shortest paths;
- the existing sparse Monte-Carlo MFPT.

This avoids the large Python-object overhead of a full NetworkX graph.

### Explicit garbage collection

Candidate ego graphs, neighbor structures and transition-local objects are
released after every completed contraction before the next scale.

## Transport metrics

`lowrank_gw` and `fgw` are still pairwise transport calculations.

Keep:

```yaml
transport_max_candidates: 256
```

or another explicitly justified small value. Do not silently raise it to the
same candidate count as typed WL.

## Colab resource preflight

Before staging, the runner measures:
- local scratch free space;
- approximate source size;
- available RAM;
- Drive free space when available.

The default scratch recommendation is conservative:

```text
max(3 * input_size, input_size + 2 GiB)
```

If local scratch is insufficient, the run fails before copying/compute.

## Reproducibility

The durable `input.json` records:
- config path;
- selected metric;
- source Drive path;
- local scratch path;
- staging statistics;
- preflight result;
- BLAS thread limit;
- storage strategy.

`COLAB_RUN.json` records the final local/Drive locations and hierarchy summary.

The actual graph/metric/coarsening functions are the same functions used by
`semmap-wishart`; the Colab layer does not modify `k`, `significance`,
`candidate_limit` or other scientific parameters automatically.

## Notebook

Use:

```text
notebooks/04_wishart_colab_drive.ipynb
```

It mounts Drive explicitly, installs the package and constructs the
`semmap-wishart-colab` command from editable variables.


## GPU-first execution (Wishart graph dictionary)

Use `notebooks/05_wishart_graph_dictionary_colab.ipynb` from
`feature/wishart-gpu-first-parallel`. In Colab select **Runtime → Change
runtime type → GPU**. The notebook installs the explicit Torch extra with
`constraints-colab.txt` and probes CUDA before building any graph. The local
scientific/CI constraints (`requirements/constraints.txt`) are not used on
Colab: they can force an incompatible downgrade of the preinstalled NumPy
stack. Project `pyproject.toml` bounds were relaxed accordingly.

The default YAML remains:

```yaml
colab:
  device: auto
  cpu_workers: 4
  gpu_batch_size: 128
  min_gpu_types: 128
  checkpoint_every_levels: 1
```

`auto` selects and **freezes** CUDA when available for `typed_wl` and
`graphlet` neighbor search. No scientific parameters are altered. VF2,
recursive dictionary registration, CSR extraction and dynamics still use CPU.
CUDA nearest-neighbor search is float32; CPU sklearn search uses float64.
To preserve one numerical backend throughout a run, an unrecoverable GPU OOM
fails the current run rather than silently switching all later levels to CPU.
CUDA tiles are automatically reduced before failure; tune the GPU batch size
and start a **new** run if it remains too large.

The CLI always prints `{"stage":"device_selected", ...}` to stdout, with
requested/selected device, GPU model/memory when available, and effective
`cpu_workers`. If auto cannot access CUDA, it warns on stderr with
`cuda_diagnostics()` and proceeds on CPU. Explicit `--device cuda` fails
if unavailable. For type spaces below `min_gpu_types`, the ordinary
`auto` search helper may elect CPU and writes a visible
`small_type_space` reason; the Colab CLI freezes the resolved CUDA device
for the complete experiment, so a CUDA-selected Colab run always uses CUDA
for eligible kNN levels with at least two types.

The notebook uses `wishart-dictionary-100k-gpu` and
`wishart-dictionary-100k-cpu` run-name suffixes. This prevents an
interrupted float64 CPU experiment from being resumed with float32 CUDA.
Inspect `runs/<name>/input.json` and
`transition_XXX_YYY/wishart.json` for requested device,
`metric.backend` (`torch_cuda` or `sklearn_cpu`) and any
`fallback_reason`.

To run directly with a Drive YAML:

```bash
semmap-wishart-colab \
  --drive-root /content/drive/MyDrive/SemanticMap/colab/wishart \
  --config-drive config/experiment.yaml \
  --prepared-drive prepared/<prepared-id> \
  --run-name wishart-dictionary-100k-gpu \
  --device auto --blas-threads 1
```

## Maximum safe Python parallelism

`colab.cpu_workers` is the bounded worker budget. The CLI clamps this to
the currently visible CPU count rather than oversubscribing a small Colab
VM; `--blas-threads 1` avoids nested OpenMP/BLAS pools.

- **Threads:** independent SciPy CSR ego extraction, relation contractions,
  transition relation blocks, sampled local clustering and sampled shortest
  paths. These return results in deterministic input order.
- **Spawn processes:** independent WL fingerprints, read-only exact
  dictionary/VF2 full-scan matches, typed-WL/graphlet feature batches and
  sampled Brandes source contributions. Use `spawn` exclusively after CUDA
  initialization; shared dictionary snapshots are loaded once per worker,
  in-flight full-scan batches are bounded and all counters/type IDs are
  modified only by the parent process.
- **Caching:** prototype NetworkX graph LRU is transient (excluded from
  checkpoint pickle); per-type MDL prototype costs are memoized; relation
  degrees and CSR views are reused across scan batches.
- **Serial by design:** density-ordered Wishart merging, order-dependent MDL
  set packing, Huffman code generation and contraction levels themselves.
  MFPT retains its original sequential RNG stream unless a separately
  validated, explicitly opt-in seed-splitting experiment is introduced.

`hierarchy.json.levels_detail[*].phase_timing_seconds` records the
per-level cost of dynamic diagnostics, discovery, full-scan matching,
feature/kNN/Wishart, MDL scoring, transition metrics, contraction, and
checkpoint I/O. Compare these profiles and overall wall time on the **same
100k input** with CPU worker counts 1, 2, and 4 before claiming speedups.
The CPU-only GitHub Actions suite tests scientific equality and checkpoint
behavior; a clean Colab GPU run, OOM handling and actual T4/L4 performance
must still be validated on the chosen Colab hardware.

See `docs/wishart_gpu_first_parallelization.md` for the phase-by-phase
implementation ledger and remaining profiling-gated candidates.
