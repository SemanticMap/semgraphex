"""Static contract for the Colab H1 workflow; does not contact Drive or install packages."""
from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "18_h1_nsbm_nulls_l4_12cpu_colab.ipynb"


def test_h1_colab_notebook_is_valid_and_code_cells_compile():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 12
    for index, cell in enumerate(data["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            body = "".join(cell["source"])
            ast.parse(body, filename=f"colab-cell-{index}")
        else:
            assert "".join(cell["source"]).strip()


def test_h1_colab_notebook_preserves_reproducibility_and_honest_devices():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    required = (
        "SOURCE_REVISION", "RUN_PROFILE", "CPU_BUDGET = 12",
        "graph-tool-base=2.98", "semmap_haken.hypothesis.h1_cli",
        "COMPLETED",
        "restore_run", "persist_run", "SOURCE_REVISION",
        "device=\"cuda\"", "device=\"cpu\"", "np.isclose",
        "--mode", "--restarts", "--nulls", "--workers",
        "taskset", "CUDA", "CONCEPTNET_TSV_ON_DRIVE",
    )
    for item in required:
        assert item in code, item
    assert code.count("graph-tool-base=2.98") == 1
    assert "asyncio" not in code
    assert all(c["cell_type"] != "code" or not c["outputs"] for c in nb["cells"])
