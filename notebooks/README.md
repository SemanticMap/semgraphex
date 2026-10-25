# Notebook workflows

The notebooks are **thin front ends** over the public [`semmap_haken`](../src/semmap_haken/) package and CLI. They do not reimplement ConceptNet parsing or sparse graph construction.

## Clean Colab setup

In a fresh Colab runtime, clone the public repository and install the optional notebook tooling:

```bash
git clone <repository-url> semmap-haken
%cd semmap-haken
    pip install -e '.[notebook]'
```

Open [`00_colab_setup_and_conceptnet.ipynb`](00_colab_setup_and_conceptnet.ipynb) first, then [`01_data_smoke_and_sparse_graph.ipynb`](01_data_smoke_and_sparse_graph.ipynb), [`02_linear_modes_and_dynamics.ipynb`](02_linear_modes_and_dynamics.ipynb), and [`03_one_step_haken_coarsening.ipynb`](03_one_step_haken_coarsening.ipynb).

Notebook 03 is an offline-first, library/CLI-backed M2 demonstration. It creates a local 12-node ring only to satisfy the iterative spectral smoke contract, then runs the same `prepare` → `run` path as [`configs/haken_one_step_smoke.yaml`](../configs/haken_one_step_smoke.yaml). It displays only persisted partition, compression, subspace/eigenvalue, and lifted-trajectory distortion summaries, separating observed outputs from interpretation caveats. It is not a ConceptNet result.

For an NVIDIA CUDA 12 runtime, optionally install `pip install -e '.[cuda,notebook]'`. Notebook 02 displays the selected backend and fallback reason; CUDA remains strict when explicitly requested and is only an acceleration experiment until its recorded parity checks pass. M2 enabled configurations always force CPU float64 reference execution before calculating their M1/M2 evidence.

## Safe defaults and persistence

Both notebooks default to the committed tiny TSV fixture and set `ALLOW_PRODUCTION_DOWNLOAD = False`; they do not clone, install, access the network, or mount Drive when executed by headless tests. A manual dataset path is preferred for a user-provided dump. Production acquisition is an explicit opt-in after resource preflight.

The active workspace remains local for the run. Drive mounting is an explicit notebook callback, not a core import. If durable storage is requested, [`persist_paths()`](../src/semmap_haken/notebook.py) atomically copies selected light metadata after the run; it does not automatically copy heavy CSR artifacts.

Each workflow displays the source, checksum/cache status, preflight result, run ID, and package-produced artifact paths. The resulting manifest/config/artifacts remain CLI-replayable through [`semmap-haken prepare`](../src/semmap_haken/cli.py).

## H1 — Colab CPU/OpenMP + CUDA

Open [`18_h1_nsbm_nulls_l4_12cpu_colab.ipynb`](18_h1_nsbm_nulls_l4_12cpu_colab.ipynb) to reproduce H1 on a 12-CPU budget (if allocated) with a separate micromamba `graph-tool` environment and CUDA batch block projections. The notebook fixes source SHA, runs genuine graph-tool integration tests, stages raw assertion TSV locally, persists sharded checkpoints to Drive during long experiments, and supports SMOKE/PILOT/FULL plus optional synthetic-calibration runs. **CUDA does not accelerate graph-tool nSBM**, and notebook presence is not a clean Colab Run All verification. See `docs/h1_scientific_protocol.md`.


## ConceptNet-10k relation-aware H1 experiment

Use [19_h1_conceptnet_10k_relations_colab.ipynb](19_h1_conceptnet_10k_relations_colab.ipynb) to prepare **exactly 10,000 English ConceptNet vertices** from a genuine five-column ConceptNet assertion TSV/TSV.GZ, preserving assertion IDs, directions, weights and independent relation labels. The default eight relation layers are RelatedTo, IsA, PartOf, UsedFor, HasProperty, HasA, Synonym, Causes. Missing input relations fail clearly; remove unavailable relations in the configuration instead of silently substituting data.

The notebook is pinned to a new tested code revision, uses CPU/OpenMP on at most 12 granted cores for the layer-aware SBM/DC-SBM/nSBM, and GPU/CuPy only for independently validated W-only block-projection calculations. Two control ensembles, restart stability ARI, per-relation MRR/Hits@10 and optional planted-hierarchy recovery calibration are included.

Change CONCEPTNET_TSV_ON_DRIVE in the first configuration cell to your raw source file. QUICK (2 restarts, 3 nulls per ensemble) is an **engineering test only**; VALIDATION (3, 49) and FULL (10, 99) are much more expensive and need separate power/false-positive calibration. All runs get a new RUN_ID and can resume from Drive snapshots; prior 100k pilot artifacts are never reused.
