from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from semmap_haken.notebook import (
    NotebookPreflightError,
    detect_environment,
    make_execution_metadata,
    persist_paths,
    resolve_notebook_paths,
    run_resource_preflight,
)


def test_detect_environment_does_not_import_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    environment = detect_environment(modules={"IPython": object()})
    assert environment.kind == "local_jupyter"
    assert "google.colab" not in sys.modules


def test_detect_environment_identifies_colab_from_loaded_module() -> None:
    assert detect_environment(modules={"google.colab": object()}).kind == "colab"
    assert detect_environment(modules={}).kind == "cli"


def test_paths_resolve_from_config_and_explicit_overrides(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths: {workspace_root: workspace, data_root: data, cache_root: cache, runs_root: runs}
dataset: {source: fixture, language: en, relations: [RelatedTo], min_weight: 1.0, max_nodes: 10, component: largest}
graph: {directed: false, weight_transform: raw, operator: normalized_adjacency}
""",
        encoding="utf-8",
    )
    paths = resolve_notebook_paths(config, cache_root=tmp_path / "external-cache")
    assert paths.workspace_root == tmp_path / "workspace"
    assert paths.cache_root == tmp_path / "external-cache"
    assert paths.runs_root == tmp_path / "workspace" / "runs"


def test_resource_preflight_passes_and_fails_for_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("recommended_ram_gib: 1\nrecommended_disk_gib: 1\n", encoding="utf-8")
    monkeypatch.setattr("semmap_haken.notebook.available_memory_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr("semmap_haken.notebook.available_disk_bytes", lambda _: 2 * 1024**3)
    result = run_resource_preflight(profile, tmp_path)
    assert result.ok is True
    monkeypatch.setattr("semmap_haken.notebook.available_memory_bytes", lambda: 0)
    with pytest.raises(NotebookPreflightError, match="RAM"):
        run_resource_preflight(profile, tmp_path)


def test_persist_paths_copies_atomically_and_returns_records(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "persisted"
    records = persist_paths([source / "manifest.json"], source_root=source, destination_root=destination)
    assert records[0].destination == destination / "manifest.json"
    assert records[0].destination.read_text(encoding="utf-8") == "{}"
    assert not list(destination.rglob("*.part"))


def test_execution_metadata_captures_notebook_and_persistence_state(tmp_path: Path) -> None:
    metadata = make_execution_metadata(
        notebook_name="00_colab_setup_and_conceptnet.ipynb",
        environment=detect_environment(modules={}),
        paths=resolve_notebook_paths(tmp_path),
        drive_enabled=False,
        persisted_paths=[tmp_path / "runs" / "prepare-demo"],
    )
    serialized = json.dumps(metadata)
    assert metadata["notebook_name"] == "00_colab_setup_and_conceptnet.ipynb"
    assert metadata["drive"]["enabled"] is False
    assert "prepare-demo" in serialized


@pytest.mark.parametrize("name", ["00_colab_setup_and_conceptnet.ipynb", "01_data_smoke_and_sparse_graph.ipynb", "02_linear_modes_and_dynamics.ipynb"])
def test_notebook_is_thin_and_offline_safe_by_default(name: str) -> None:
    notebook = Path("notebooks") / name
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])
    assert document["nbformat"] == 4
    assert "semmap_haken" in source
    assert "ALLOW_PRODUCTION_DOWNLOAD = False" in source
    assert "stream_assertions(" not in source
    assert "build_sparse_graph(" not in source


@pytest.mark.parametrize("name", ["00_colab_setup_and_conceptnet.ipynb", "01_data_smoke_and_sparse_graph.ipynb", "02_linear_modes_and_dynamics.ipynb"])
def test_notebook_executes_offline_when_optional_tools_are_installed(name: str) -> None:
    nbformat = pytest.importorskip("nbformat")
    from nbclient import NotebookClient

    notebook = nbformat.read(Path("notebooks") / name, as_version=4)
    NotebookClient(notebook, timeout=60, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}).execute()
