"""Notebook contract: static validation, never execute Colab mounts or downloads."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat


NOTEBOOK = (Path(__file__).resolve().parents[1] /
            "notebooks" / "19_h1_conceptnet_10k_relations_colab.ipynb")


def test_notebook_structure_and_all_python_cells_compile():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 18
    for i, cell in enumerate(notebook.cells):
        assert cell.source.strip()
        if cell.cell_type == "code":
            assert not cell.outputs
            assert cell.execution_count is None
            ast.parse(cell.source, filename=f"ConceptNet 10k Colab cell #{i}")


def test_notebook_has_valid_typed_10k_cpu_gpu_and_resume_contracts():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    full = "\n".join("".join(c["source"]) for c in nb["cells"])
    for token in (
        "SOURCE_REVISION", "REPO_BRANCH", "h1-cn10k-relation-aware-validation",
        "TARGET_NODES = 10000", "CPU_BUDGET = 12", "MAX_QUALIFYING_ASSERTIONS",
        "RELATIONS", "IsA", "RelatedTo", "PartOf", "UsedFor",
        "conceptnet_10k", "graph-tool-base=2.98", "graph_tool",
        "semmap_haken.hypothesis.h1_cli", "device=\"cuda\"",
        "device=\"cpu\"", "np.isclose", "taskset", "restore_run",
        "persist_run", "RUN_CALIBRATION", "RUN_ID", "COMPLETED",
        "restart_stability", "by_relation", "hits_at_10",
        "CONCEPTNET_TSV_ON_DRIVE", "VALIDATION", "QUICK",
    ):
        assert token in full, token
    assert "planted_typed_graph" not in nb["cells"][9]["source"][0]
    assert all(c["cell_type"] != "code" or not c["outputs"] for c in nb["cells"])
