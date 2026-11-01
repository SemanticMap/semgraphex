"""Colab import bootstrap regression: package not visible in Jupyter sys.path.

Notebook uses one pinned Git checkout but Colab can lose editable-install
visibility between sessions; graph-tool fitting must never rerun for this.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest


NOTEBOOK = (Path(__file__).resolve().parents[1] /
            "notebooks/22_nsbm_self_similarity_cn100k_colab.ipynb")


def _load_helper():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    config = nb.cells[1].source
    parsed = ast.parse(config)
    helper = next(
        n for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "require_semmap_code"
    )
    repo = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    namespace = {"Path": Path, "REPO": repo, "SOURCE_REVISION": sha}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(NOTEBOOK),
                 "exec"), namespace)
    return nb, namespace["require_semmap_code"], namespace


def test_notebook_imports_before_staging_and_verification():
    nb, _, _ = _load_helper()
    setup, stage, verify = (nb.cells[i].source for i in (4,5,7))
    assert "require_semmap_code()" in setup
    assert "require_semmap_code().stage_fit_snapshot" in stage
    assert "require_semmap_code().load_verified_fit_bundle" in verify
    assert setup.index("require_semmap_code()") < len(setup)


def test_bootstrap_restores_source_path_without_package_reinstall(monkeypatch):
    _, helper, ns = _load_helper()
    repo = ns["REPO"]
    root = str(repo / "src")
    monkeypatch.setattr(
        sys, "path", [x for x in sys.path if x != root]
    )
    mod = helper()
    assert str(repo / "src") in sys.path
    assert Path(mod.__file__).resolve() == (
        repo / "src/semmap_haken/hypothesis/selfsim_colab_io.py"
    ).resolve()


def test_bootstrap_fails_before_import_when_repo_missing(tmp_path):
    _, helper, ns = _load_helper()
    # Rebind REPO in the helper's own globals.
    ns["REPO"] = tmp_path / "not_cloned"
    with pytest.raises(RuntimeError, match="ячейку 4"):
        helper()


def test_bootstrap_refuses_mismatched_source_revision():
    _, helper, ns = _load_helper()
    ns["SOURCE_REVISION"] = "0" * 40
    with pytest.raises(RuntimeError, match="Git-ревизия"):
        helper()
