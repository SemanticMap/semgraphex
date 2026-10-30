"""Colab CUDA bootstrap must validate cuBLAS, not only GPU visibility.

This is a static CI guard. Actual T4/L4 CUDA execution must be tested in Colab.
"""
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/20_nsbm_huffman_grammar_cn10k_cn100k_colab.ipynb")


def test_cupy_conda_cuda12_and_cublas_preflight():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    setup = next(s for s in cells if "CUDA_PROBE =" in s)
    smoke = next(s for s in cells if "check_env = dict(" in s)
    config = next(s for s in cells if "SOURCE_REVISION =" in s)

    assert 'packages += ["cupy", "cuda-version=12"]' in setup
    assert '"cupy", "cuda-version=12"], check=True' in setup
    assert '"cupy-cuda12x"], check=True' in setup
    assert "result = a @ a" in setup
    assert "np.allclose(cp.asnumpy(result)" in setup
    assert "cp.cuda.Stream.null.synchronize()" in setup
    assert 'str(ENV_ROOT / "lib")' in setup
    assert "env=GPU_SUBPROCESS_ENV" in setup
    assert "check_env = dict(GPU_SUBPROCESS_ENV" in smoke
    assert "PYTHONWARNINGS=" in smoke
    # No experiment/revision identity change is needed for environment repair.
    assert 'SOURCE_REVISION = "02a03e035e0268e96544f59895752fc22f60f97a"' in config
    assert 'RUN_ID = f"nsbm-huffman-{PROFILE}-v3"' in config
