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
/content/drive/MyDrive/SemanticMap/semgraphex/runs/<run-name>
```

`COMPLETED` is synchronized only after the final files have been copied.

## Install in Colab

From the checked-out repository:

```bash
pip install -c requirements/constraints.txt -e '.[wishart,notebook]'
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
    semgraphex/
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
  --drive-root /content/drive/MyDrive/SemanticMap/semgraphex \
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

The current branch checkpoints partial evidence but does not automatically
resume a partially completed hierarchy. Restart with a new run name or use
`--overwrite` to restart the same name cleanly.

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
