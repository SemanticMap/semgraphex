from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.modes import (
    SpectralConfigurationError,
    analyze_normalized_adjacency,
    load_mode_result,
    save_mode_result,
    select_auto_critical_beta,
    select_slow_mode_dimension,
)
from semmap_haken.operators import OperatorInputError, normalized_adjacency


def _path_graph(size: int) -> sparse.csr_matrix:
    rows = np.arange(size - 1)
    return sparse.csr_matrix(
        (np.ones(2 * (size - 1)), (np.concatenate((rows, rows + 1)), np.concatenate((rows + 1, rows)))),
        shape=(size, size),
    )


def test_normalized_adjacency_is_sparse_symmetric_and_safe_for_isolates() -> None:
    adjacency = sparse.block_diag((_path_graph(3), sparse.csr_matrix((1, 1))), format="csr")
    operator, degrees = normalized_adjacency(adjacency)

    assert sparse.isspmatrix_csr(operator)
    assert np.array_equal(degrees, np.array([1.0, 2.0, 1.0, 0.0]))
    assert (operator - operator.T).nnz == 0
    assert np.all(np.isfinite(operator.data))
    assert operator[3, 3] == 0.0


def test_normalized_adjacency_rejects_non_square_nonfinite_and_nonsymmetric_input() -> None:
    with pytest.raises(OperatorInputError, match="square"):
        normalized_adjacency(sparse.csr_matrix(np.ones((2, 3))))
    with pytest.raises(OperatorInputError, match="finite"):
        normalized_adjacency(sparse.csr_matrix([[0.0, np.nan], [np.nan, 0.0]]))
    with pytest.raises(OperatorInputError, match="symmetric"):
        normalized_adjacency(sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]))


def test_iterative_spectrum_has_small_residuals_and_sign_invariant_subspace() -> None:
    operator, degrees = normalized_adjacency(_path_graph(8))
    first = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta="auto_critical", top_k=5)
    second = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta="auto_critical", top_k=5)

    assert np.all(first.residual_norms < 1e-8)
    assert np.all(np.diff(first.eigenvalues) <= 0)
    assert first.beta_selection.spectral_abscissa <= 1e-10
    assert np.allclose(first.eigenvectors @ first.eigenvectors.T, second.eigenvectors @ second.eigenvectors.T, atol=1e-8)


def test_auto_critical_excludes_trivial_mode_and_refuses_degenerate_graph() -> None:
    operator, degrees = normalized_adjacency(_path_graph(6))
    result = select_auto_critical_beta(np.array([1.0, 0.5, -0.5]), alpha=1.0, margin=0.05)
    assert result.target_eigenvalue == pytest.approx(0.5)
    # The requested 1.9 would make the excluded Perron mode unstable. A1
    # records and applies the deterministic full-state stability clip instead.
    assert result.requested_beta == pytest.approx(1.9)
    assert result.beta == pytest.approx(0.95)
    assert result.spectral_abscissa == pytest.approx(-0.05)
    with pytest.raises(SpectralConfigurationError, match="eligible nontrivial"):
        select_auto_critical_beta(np.array([1.0]), alpha=1.0, margin=0.05)
    with pytest.raises(SpectralConfigurationError, match="at least three"):
        analyze_normalized_adjacency(operator[:2, :2], degrees=degrees[:2], alpha=1.0, beta="auto_critical", top_k=1)


def test_auto_critical_refuses_disconnected_or_isolated_graphs() -> None:
    disconnected = sparse.block_diag((_path_graph(3), _path_graph(3)), format="csr")
    operator, degrees = normalized_adjacency(disconnected)
    with pytest.raises(SpectralConfigurationError, match="connected"):
        analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta="auto_critical", top_k=4)


def test_dimension_selection_consensus_and_localization_diagnostics() -> None:
    selection = select_slow_mode_dimension(
        eigenvalues=np.array([0.9, 0.8, 0.2, 0.1]),
        growth_rates=np.array([-0.1, -0.2, -0.8, -0.9]),
        max_r=3,
    )
    assert selection.selected_r == 2
    assert selection.eigengap_r == 2
    assert selection.timescale_gap_r == 2
    assert selection.bootstrap_deferred is True

    operator, degrees = normalized_adjacency(sparse.csr_matrix(np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)))
    result = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=3)
    assert np.all(result.participation_ratios >= 1.0)
    assert np.all(result.inverse_participation_ratios > 0.0)
    assert np.all(np.isfinite(result.degree_correlations) | np.isnan(result.degree_correlations))


def test_mode_artifact_round_trip_is_atomic_and_preserves_arrays(tmp_path) -> None:
    operator, degrees = normalized_adjacency(_path_graph(6))
    original = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta="auto_critical", top_k=4)
    paths = save_mode_result(original, tmp_path / "spectral", resolved_config={"graph": {"operator": "normalized_adjacency"}}, input_checksums={"adjacency.npz": "a" * 64})
    restored = load_mode_result(tmp_path / "spectral")

    assert paths["numeric"].is_file()
    assert np.allclose(restored.eigenvalues, original.eigenvalues)
    assert np.allclose(restored.eigenvectors @ restored.eigenvectors.T, original.eigenvectors @ original.eigenvectors.T)
    assert restored.selection.selected_r == original.selection.selected_r
