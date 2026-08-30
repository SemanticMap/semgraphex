# semmap-haken

`semmap-haken` is the active, sparse-first research foundation for testing the Haken-coarsening ConceptNet hypotheses. The approved design is in [`docs/haken_coarsening_roadmap.md`](docs/haken_coarsening_roadmap.md). It is **notebook-first, library-backed, and CLI-reproducible**: notebooks will call package APIs, while the CLI is the replayable interface for every reportable run.

## Installation

Install the active core package with:

```bash
python -m pip install -e .
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
- [`src/semmap_haken/manifest.py`](src/semmap_haken/manifest.py) persists resolved config, provenance, checksums, stage states, and notebook/Colab metadata—including resource and Drive-cache fields.

Starter inputs are [`configs/conceptnet_en_smoke.yaml`](configs/conceptnet_en_smoke.yaml), [`configs/conceptnet_en_small.yaml`](configs/conceptnet_en_small.yaml), and the profiles under [`configs/resource_profiles/`](configs/resource_profiles/).

## CLI surface

```bash
python -m semmap_haken --help
python -m semmap_haken download --help
python -m semmap_haken prepare --help
python -m semmap_haken run --help
python -m semmap_haken evaluate --help
```

`download` accepts a configured/manual source and writes the versioned cache below `data/cache/datasets/conceptnet/<version>/<checksum>/`. `prepare` requires `dataset.path` for a local/offline fixture or manual dump; it creates `runs/prepare-*/adjacency.npz`, `nodes.json`, `graph_metadata.json`, `resolved_config.json`, and `manifest.json`.

URI indices are lexicographic and therefore independent of input order. With `component: largest`, ties select the component whose lexicographically smallest URI is smallest; `max_nodes` then retains the first URI-sorted nodes and induces the corresponding sparse subgraph. This is reproducible but not a semantic sampling rule. Large dumps remain streamed, yet graph construction holds accepted edge aggregates and must be budgeted for available RAM/disk.

## Development verification

```bash
python -m pytest tests
```

The historical prototype remains available only for compatibility and is not evidence for the Haken/ConceptNet research program.
