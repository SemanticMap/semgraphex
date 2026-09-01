# Notebook workflows

The notebooks are **thin front ends** over the public [`semmap_haken`](../src/semmap_haken/) package and CLI. They do not reimplement ConceptNet parsing or sparse graph construction.

## Clean Colab setup

In a fresh Colab runtime, clone the public repository and install the optional notebook tooling:

```bash
git clone <repository-url> semmap-haken
%cd semmap-haken
pip install -e '.[notebook]'
```

Open [`00_colab_setup_and_conceptnet.ipynb`](00_colab_setup_and_conceptnet.ipynb) first, then [`01_data_smoke_and_sparse_graph.ipynb`](01_data_smoke_and_sparse_graph.ipynb).

## Safe defaults and persistence

Both notebooks default to the committed tiny TSV fixture and set `ALLOW_PRODUCTION_DOWNLOAD = False`; they do not clone, install, access the network, or mount Drive when executed by headless tests. A manual dataset path is preferred for a user-provided dump. Production acquisition is an explicit opt-in after resource preflight.

The active workspace remains local for the run. Drive mounting is an explicit notebook callback, not a core import. If durable storage is requested, [`persist_paths()`](../src/semmap_haken/notebook.py) atomically copies selected light metadata after the run; it does not automatically copy heavy CSR artifacts.

Each workflow displays the source, checksum/cache status, preflight result, run ID, and package-produced artifact paths. The resulting manifest/config/artifacts remain CLI-replayable through [`semmap-haken prepare`](../src/semmap_haken/cli.py).
