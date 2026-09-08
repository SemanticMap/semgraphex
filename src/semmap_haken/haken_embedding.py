"""Topology-only candidate slow-mode coordinates for one-step M2 experiments.

Coordinates deliberately derive only from the spectral arrays.  They are not
claimed to be unique order parameters: signs and rotations must be handled by
subspace comparisons at later cross-scale boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .modes import ModeResult


@dataclass(frozen=True)
class HakenEmbedding:
    coordinates: np.ndarray
    selected_mode_indices: tuple[int, ...]
    weighting: Literal["none", "relaxation_time"]
    caveats: tuple[str, ...]


def build_haken_embedding(modes: ModeResult, *, weighting: Literal["none", "relaxation_time"] = "none") -> HakenEmbedding:
    """Return ``modes.eigenvectors[:, 1:1+r]`` without semantic inputs."""
    selected_r = min(modes.selection.selected_r, modes.eigenvectors.shape[1] - 1)
    if selected_r < 1:
        raise ValueError("at least one selected nontrivial slow-mode candidate is required")
    if weighting not in {"none", "relaxation_time"}:
        raise ValueError("weighting must be none or relaxation_time")
    coordinates = np.asarray(modes.eigenvectors[:, 1 : 1 + selected_r], dtype=np.float64).copy()
    if weighting == "relaxation_time":
        weights = np.asarray(modes.relaxation_times[1 : 1 + selected_r], dtype=np.float64)
        coordinates *= np.where(np.isfinite(weights), weights, 0.0)
    return HakenEmbedding(
        coordinates=coordinates,
        selected_mode_indices=tuple(range(1, 1 + selected_r)),
        weighting=weighting,
        caveats=(
            "Coordinates are candidate slow-mode coordinates, not proven order parameters.",
            "Eigenvector sign and degenerate-subspace rotations are not canonical; compare subspaces at cross-scale boundaries.",
        ),
    )
