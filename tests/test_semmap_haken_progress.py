from semmap_haken.hierarchy import (
    _estimate_planned_contractions,
    _format_duration,
    _progress_bar,
)


def test_progress_plan_matches_floor_quantized_two_percent_run_to_100() -> None:
    assert _estimate_planned_contractions(
        100_000,
        min_nodes=100,
        max_levels=360,
        target_reduction=0.02,
    ) == 355


def test_progress_helpers_are_stable() -> None:
    assert _format_duration(0) == "0s"
    assert _format_duration(65) == "1m 05s"
    assert _format_duration(3661) == "1h 01m 01s"
    assert _progress_bar(0.5, width=10) == "█████░░░░░"
