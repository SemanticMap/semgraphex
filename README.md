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

## Foundation contracts

- [`src/semmap_haken/config.py`](src/semmap_haken/config.py) loads and validates a YAML file once, then exposes a resolved, serializable configuration shared by notebook and CLI callers.
- [`src/semmap_haken/artifacts.py`](src/semmap_haken/artifacts.py) defines versioned metadata contracts for downloads and future sparse `GraphArtifact` payloads. Sparse payload I/O is intentionally deferred to Workstream 2.
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

The command grammar is stable and handler registration is extensible. Commands return a clear “not yet implemented” result until their owning workstream supplies an implementation. This foundation does **not** download ConceptNet, parse assertions, build graphs, or make scientific claims.

## Development verification

```bash
python -m pytest tests/test_semmap_haken_foundation.py tests/test_semmap_haken_cli.py
```

The historical prototype remains available only for compatibility and is not evidence for the Haken/ConceptNet research program.
