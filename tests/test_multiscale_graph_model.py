"""Tests for aligned multiscale grammar graph diagnostics."""
from __future__ import annotations

import pytest

from semmap_haken.multiscale_graph_model import aligned_projection_distance


def _model(mass_a=0.5, intensity=0.25):
    return {
        "mass": {
            "GT_A": {"probability_mass": mass_a},
            "GT_B": {"probability_mass": 1.0 - mass_a},
        },
        "blocks": [
            {
                "relation": "r",
                "source_symbol": "GT_A",
                "target_symbol": "GT_B",
                "record_intensity": intensity,
                "weight_intensity": 2.0 * intensity,
            }
        ],
    }


def test_identical_multiscale_projection_distance_is_zero():
    result = aligned_projection_distance(_model(), _model())
    assert result["mass_total_variation"] == 0.0
    assert result["record_intensity_weighted_l1"] == 0.0
    assert result["weight_intensity_weighted_l1"] == 0.0


def test_mass_and_block_changes_are_reported_separately():
    result = aligned_projection_distance(
        _model(mass_a=0.5, intensity=0.25),
        _model(mass_a=0.7, intensity=0.5),
    )
    assert result["mass_total_variation"] == pytest.approx(0.2)
    assert result["record_intensity_weighted_l1"] > 0
    assert result["weight_intensity_weighted_l1"] > 0
