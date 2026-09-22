from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

from semmap_haken.dynamics import (
    DynamicsConfigurationError,
    build_perturbations,
    run_linear_dynamics,
    save_dynamics_result,
)
from semmap_haken.modes import analyze_normalized_adjacency
from semmap_haken.operators import normalized_adjacency


def _path_graph(size: int) -> sparse.csr_matrix:
    rows = np.arange(size - 1)
    return sparse.csr_matrix(
        (np.ones(2 * (size - 1)), (np.concatenate((rows, rows + 1)), np.concatenate((rows + 1, rows)))),
        shape=(size, size),
    )


def test_sparse_linear_solver_matches_tiny_dense_reference() -> None:
    operator, degrees = normalized_adjacency(_path_graph(5))
    modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=4)
    initial = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    result = run_linear_dynamics(
        operator,
        modes,
        initial_states=initial,
        time_grid=np.array([0.0, 0.25, 0.5]),
        storage_policy="all",
    )

    jacobian = -np.eye(5) + 0.5 * operator.toarray()  # Dense is test-only reference.
    expected = np.stack([expm(time * jacobian) @ initial[0] for time in result.time_grid])
    assert np.allclose(result.trajectories[0], expected, atol=1e-10)
    assert np.all(np.isfinite(result.relative_rmse))
    assert result.selected_r == modes.selection.selected_r


def test_perturbations_are_deterministic_and_cover_requested_kinds() -> None:
    degrees = np.array([1.0, 4.0, 2.0, 3.0])
    first = build_perturbations(
        node_count=4,
        degrees=degrees,
        seed=1729,
        amplitude=2.0,
        per_kind=2,
        sparse_fraction=0.5,
    )
    second = build_perturbations(
        node_count=4,
        degrees=degrees,
        seed=1729,
        amplitude=2.0,
        per_kind=2,
        sparse_fraction=0.5,
    )

    assert first.labels == second.labels
    assert np.array_equal(first.initial_states, second.initial_states)
    assert {label.split("-")[0] for label in first.labels} == {"single_node", "gaussian", "random_sparse", "hub_targeted"}
    hub_index = first.labels.index("hub_targeted-0")
    assert np.argmax(first.initial_states[hub_index]) == 1


def test_dynamics_rejects_invalid_shapes_and_never_uses_dense_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    operator, degrees = normalized_adjacency(_path_graph(5))
    modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=4)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda _: (_ for _ in ()).throw(AssertionError("dense conversion")))

    with pytest.raises(DynamicsConfigurationError, match="time grid"):
        run_linear_dynamics(operator, modes, initial_states=np.ones((1, 5)), time_grid=np.array([0.0]))
    result = run_linear_dynamics(operator, modes, initial_states=np.ones((1, 5)), time_grid=np.array([0.0, 1.0]))
    assert result.trajectories is None
    assert np.all(np.isfinite(result.modal_amplitudes))


def test_dynamics_artifact_is_atomic_and_checksum_verified(tmp_path) -> None:
    operator, degrees = normalized_adjacency(_path_graph(5))
    modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=4)
    result = run_linear_dynamics(operator, modes, initial_states=np.ones((1, 5)), time_grid=np.array([0.0, 1.0]), storage_policy="all")

    paths = save_dynamics_result(result, tmp_path / "dynamics", resolved_config={"dynamics": {"model": "linear"}}, input_artifacts={"spectral": "abc"})
    assert paths["numeric"].is_file()
    assert paths["summary"].is_file()
    summary = __import__("json").loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["numeric_checksum_sha256"]
    assert summary["aggregate"]["trajectory_count"] == 1
