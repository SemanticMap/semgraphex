from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from semmap_haken.dynamics_compare import propagate_model, reconstruct_fine_to_coarse
from semmap_haken.multiscale_config import load_research_extensions


def test_three_dynamics_models_return_finite_equal_shapes() -> None:
    operator = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.0],
            ]
        )
    )
    initial = np.array([[1.0, 0.0, 0.0], [0.0, -0.5, 0.5]])
    times = np.linspace(0.0, 1.0, 6)
    outputs = {
        model: propagate_model(
            operator,
            initial,
            times,
            model=model,
            alpha=1.0,
            beta=0.8,
            cubic_g=1.0,
            nonlinear_max_step=0.02,
        )
        for model in ("linear", "cubic_haken", "tanh")
    }
    assert {value.shape for value in outputs.values()} == {(2, 6, 3)}
    assert all(np.all(np.isfinite(value)) for value in outputs.values())
    assert np.allclose(outputs["linear"][:, 0], initial)
    assert np.allclose(outputs["cubic_haken"][:, 0], initial)
    assert np.allclose(outputs["tanh"][:, 0], initial)
    assert not np.allclose(outputs["linear"][:, -1], outputs["cubic_haken"][:, -1])
    assert not np.allclose(outputs["linear"][:, -1], outputs["tanh"][:, -1])


def test_reconstruct_fine_to_coarse_from_nested_memberships(tmp_path: Path) -> None:
    fine = {"0": ["a", "b"], "1": ["c"], "2": ["d", "e"]}
    coarse = {"0": ["a", "b", "c"], "1": ["d", "e"]}
    fine_path = tmp_path / "fine.json"
    coarse_path = tmp_path / "coarse.json"
    fine_path.write_text(json.dumps(fine), encoding="utf-8")
    coarse_path.write_text(json.dumps(coarse), encoding="utf-8")
    mapping = reconstruct_fine_to_coarse(fine_path, coarse_path)
    assert mapping.tolist() == [0, 0, 1]


def test_m54_config_exposes_three_models() -> None:
    config = Path(__file__).parents[1] / "configs" / "haken_m54_dynamics_compare_33k_to50.yaml"
    options = load_research_extensions(config).dynamics_comparison
    assert options.enabled
    assert options.models == ("linear", "cubic_haken", "tanh")
    assert options.node_targets[0] == 33000
    assert options.node_targets[-1] == 50
