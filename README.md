# semmap-haken

`semmap-haken` is the active, sparse-first research foundation for testing the Haken-coarsening ConceptNet hypotheses. The approved design is in [`docs/haken_coarsening_roadmap.md`](docs/haken_coarsening_roadmap.md). It is **notebook-first, library-backed, and CLI-reproducible**: notebooks will call package APIs, while the CLI is the replayable interface for every reportable run.

## Installation

Install the active core package with:

```bash
python -m pip install -e .
```

For notebook validation and optional headless notebook execution, install the isolated extra rather than adding Jupyter packages to the research core:

```bash
python -m pip install -e '.[notebook]'
```

The legacy [`semgraphex/`](semgraphex/) package remains importable during migration. Its corpus-search dependencies are deliberately isolated in the `legacy` optional group:

```bash
python -m pip install -e '.[legacy]'
```

## Data preparation (M0)

- [`src/semmap_haken/config.py`](src/semmap_haken/config.py) loads and validates a YAML file once, then exposes a resolved, serializable configuration shared by notebook and CLI callers.
- [`src/semmap_haken/data_manager.py`](src/semmap_haken/data_manager.py) acquires a versioned dump from a verified cache, explicit manual path, or HTTP URL. HTTP downloads stream through `*.part`, verify size/SHA-256, then atomically rename. Interrupted downloads restart rather than trust partial content; the core has no Colab import.
- [`src/semmap_haken/conceptnet.py`](src/semmap_haken/conceptnet.py) streams plain or gzip five-field assertion TSV, preserves full URIs/direction/provenance, and reports filtering and invalid-record counters. ConceptNet `weight` is a heuristic confidence/informativeness weight, never a probability.
- [`src/semmap_haken/graph_build.py`](src/semmap_haken/graph_build.py) constructs a SciPy CSR adjacency without dense NxN allocation and persists a round-trippable prepared artifact.

Prepared artifacts are published atomically only after artifact checksums and a `COMPLETED` marker are written. Each run persists its configured local input path, declared source/version/URL, input format, row limit, selected-edge provenance in `selected_edges.jsonl`, and replay metadata. The default `graph.self_loop_policy` is `exclude`. Input download, decompression, and source-file checksum verification are deliberately outside `prepare`.

Use the tested (not hash-locked) constraints in [`requirements/constraints.txt`](requirements/constraints.txt): `python -m pip install -c requirements/constraints.txt -e '.[dev,notebook]'`, then run `python -m pytest -q` in a clean environment.
- [`src/semmap_haken/manifest.py`](src/semmap_haken/manifest.py) persists resolved config, provenance, checksums, stage states, and notebook/Colab metadata—including resource and Drive-cache fields.

Starter inputs are [`configs/conceptnet_en_smoke.yaml`](configs/conceptnet_en_smoke.yaml), [`configs/conceptnet_en_small.yaml`](configs/conceptnet_en_small.yaml), and the profiles under [`configs/resource_profiles/`](configs/resource_profiles/).

## A1 sparse spectral diagnostics

[`src/semmap_haken/operators.py`](src/semmap_haken/operators.py) provides sparse undirected normalized adjacency \(S=D^{-1/2}AD^{-1/2}\), preserving zero-degree nodes as zero rows and never materializing an NxN dense array. [`src/semmap_haken/modes.py`](src/semmap_haken/modes.py) uses iterative `eigsh` to emit *slow-mode candidates*—not established order parameters—with residuals, growth/decay rates for \(J=-\alpha I+\beta S\), stable-mode relaxation times, IPR/participation and localization diagnostics, degree correlations, eigengaps, timescale gaps, and a compact JSON/NPZ artifact.

For `beta: auto_critical`, A1 requires a connected, nonnegative undirected normalized-adjacency graph without isolates. It excludes the unique Perron/stationary \(\lambda=1\) mode from candidate selection, requests \(\beta=(\alpha-m)/\lambda_*\) for the leading eligible positive nontrivial \(\lambda_*\), then clips to \(\beta\le\alpha-m\) so the full Jacobian remains stable. The artifact persists requested/selected beta, target eigenvalue, margin, spectral abscissa, and the clipping caveat. The `r` diagnostic compares maximum eigengap and timescale-gap proposals; agreement is used, otherwise eigengap is the deterministic fallback.

## M1 linear dynamics workflow

[`src/semmap_haken/dynamics.py`](src/semmap_haken/dynamics.py) evolves \(\dot{x}=(-\alpha I+\beta S)x\) with sparse [`expm_multiply()`](src/semmap_haken/dynamics.py:160), never a dense matrix exponential. It produces seeded single-node, Gaussian, random-sparse, and hub-targeted perturbations. For each trajectory it projects onto the selected nontrivial A1 eigenvectors and records relative RMSE \(\|x-\hat{x}\|_F/\|x\|_F\) and max-amplitude NRMSE. These outputs are **slow-mode candidate diagnostics, not proof of order parameters**.

Prepare a graph, set `spectral.prepared_graph_dir` in [`configs/haken_linear_smoke.yaml`](configs/haken_linear_smoke.yaml) or [`configs/haken_linear_small.yaml`](configs/haken_linear_small.yaml), then run:

```bash
python -m semmap_haken prepare --config configs/conceptnet_en_smoke.yaml
python -m semmap_haken run --config configs/haken_linear_smoke.yaml
```

The `run` directory contains atomic spectral and dynamics NPZ/JSON/CSV artifacts, checksums and manifest references, plus spectrum, relaxation-time, eigengap, and reconstruction PNG plots through the optional `notebook` extra. The storage policy is configurable: `all` writes trajectories (smoke), while `summaries` writes initial states, modal amplitudes, and metrics only (small profile). Local-neighborhood and relation-group perturbations remain deferred because the prepared artifact has no inexpensive semantic-group index.

## Acceleration foundation

[`src/semmap_haken/compute.py`](src/semmap_haken/compute.py) resolves the typed `execution` section in the linear configs. `cpu` is strict and never imports CuPy; `cuda` is strict and fails when CuPy or the selected NVIDIA device is unavailable; `auto` prefers a usable CUDA device and records an explicit CPU fallback reason otherwise. The selected backend, CPU/GPU inventory, dtype, workers, batch size, solver method, timing, and fallback information are persisted in spectral diagnostics, dynamics summaries, and the run manifest.

The base installation remains CPU-only. In a CUDA 12 Colab runtime, install the optional extra only after confirming the runtime's CUDA compatibility:

```bash
pip install -e '.[cuda,notebook]'
python -m semmap_haken run --config configs/haken_linear_smoke.yaml
```

CPU propagation remains SciPy `expm_multiply` and supports deterministic, stable-index process chunks. CUDA uses lazy CuPy CSR eigensolving and an explicitly labelled sparse RK4 propagation fallback because CuPy does not provide an equivalent sparse `expm_multiply`; it is an accuracy-controlled approximation, not an exact CPU-equivalent propagator. CUDA float64 parity must be measured before interpreting CUDA results; float32 is an ablation. No performance claim is made without a recorded benchmark.

## Colab-first quick start

Use [`notebooks/00_colab_setup_and_conceptnet.ipynb`](notebooks/00_colab_setup_and_conceptnet.ipynb) in a clean Colab runtime, then run [`notebooks/01_data_smoke_and_sparse_graph.ipynb`](notebooks/01_data_smoke_and_sparse_graph.ipynb) and [`notebooks/02_linear_modes_and_dynamics.ipynb`](notebooks/02_linear_modes_and_dynamics.ipynb). The new notebook preflights resources, uses the offline fixture by default, delegates to the package CLI, and displays beta selection, `r` candidates, and trajectory reconstruction error.

Safety is deliberate: both notebooks default to the committed offline tiny fixture; `ALLOW_PRODUCTION_DOWNLOAD` is `False`; Drive use is disabled; and no headless test clones, installs packages, mounts Drive, or accesses the network. A production dump requires explicit opt-in or a manually supplied local path after the shared resource preflight reports available RAM/disk against the selected profile.

Notebook execution is **active-local first**. The helpers in [`src/semmap_haken/notebook.py`](src/semmap_haken/notebook.py) resolve the same workspace/data/cache/runs roots as [`src/semmap_haken/config.py`](src/semmap_haken/config.py), capture an execution snapshot, and can atomically copy selected light metadata to a user-selected durable destination. They never automatically persist heavy sparse artifacts or assume a personal Drive path.

The notebook↔CLI contract is strict: notebooks use package APIs and the public `prepare` workflow; they display source, checksum/cache status, preflight, `run_id`, and artifact paths. The resulting [`manifest.json`](src/semmap_haken/manifest.py) and resolved config are the provenance record, while [`semmap-haken prepare`](src/semmap_haken/cli.py) remains the reproducible replay interface.

## CLI surface

```bash
python -m semmap_haken --help
python -m semmap_haken download --help
python -m semmap_haken prepare --help
python -m semmap_haken run --help
python -m semmap_haken evaluate --help
```

`prepare` requires `dataset.path` to reference an already-downloaded, already-decompressed ConceptNet assertions TSV/CSV file, such as `../conceptnet-assertions-5.7.0.csv`. It does not download, decompress, or verify the source file. Set `dataset.max_rows: 1000` to parse only the first 1,000 physical rows before malformed-row handling and filtering; use `null` to parse the entire file. This prefix limit is deterministic but order-sensitive, not a semantic sample. Set `dataset.relations: []` to accept all ConceptNet relation types, or list names such as `RelatedTo` and `IsA` to whitelist only those types. Similarly, `dataset.language: ""` disables language filtering. The starter configurations write ephemeral run artifacts beneath `./tmp/runs/prepare-*/`: `adjacency.npz`, `nodes.json`, `graph_metadata.json`, `resolved_config.json`, and `manifest.json`. The repository-local [`tmp/`](tmp/) directory is ignored by Git and is the supported location for temporary configs and smoke-run output.

URI indices are lexicographic and therefore independent of input order. With `component: largest`, ties select the component whose lexicographically smallest URI is smallest; `max_nodes` then retains the first URI-sorted nodes and induces the corresponding sparse subgraph. This is reproducible but not a semantic sampling rule. Large dumps remain streamed, yet graph construction holds accepted edge aggregates and must be budgeted for available RAM/disk.

## Development verification

```bash
python -m pytest tests
```

The historical prototype remains available only for compatibility and is not evidence for the Haken/ConceptNet research program.
