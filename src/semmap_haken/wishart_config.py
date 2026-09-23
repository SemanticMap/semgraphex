"""Configuration for recursive Wishart discovery/coarsening.

The Wishart section is intentionally separate from the established Haken config
contract.  The base YAML is still parsed by :func:`semmap_haken.config.load_config`
for ConceptNet/data/graph settings; this module validates only the additional
`wishart` mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Any

import yaml


MetricName = Literal[
    "typed_wl",
    "graphlet",
    "lowrank_gw",
    "fgw",
    "relation_js",
]


@dataclass(frozen=True)
class WishartOptions:
    metric: MetricName = "typed_wl"
    radius: int = 1
    max_ego_nodes: int = 48
    candidate_limit: int = 2000
    k_neighbors: int = 12
    significance: float = 0.7
    min_cluster_size: int = 3
    min_cluster_mass: float = 3.0
    clustering_domain: Literal["candidates", "canonical_types"] = "canonical_types"
    density_weight: Literal["uniform", "occurrence_frequency"] = "occurrence_frequency"
    min_figure_nodes: int = 2
    max_figures_per_level: int = 1000
    max_levels: int = 6
    min_graph_nodes: int = 50
    aggregation: Literal["sum", "mean_density"] = "sum"
    random_seed: int = 0

    wl_iterations: int = 2
    feature_dim: int = 512
    graphlet_size: int = 3
    graphlet_samples: int = 256
    relation_js_block_size: int = 128

    transport_rank: int = 24
    transport_max_candidates: int = 256
    fgw_alpha: float = 0.5

    slow_modes: int = 8
    mfpt_pairs: int = 24
    mfpt_walks_per_pair: int = 8
    mfpt_max_steps: int = 500
    betweenness_samples: int = 64
    clustering_samples: int = 256
    distance_samples: int = 128

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WishartOptions":
        metric = str(raw.get("metric", "typed_wl"))
        allowed = {"typed_wl", "graphlet", "lowrank_gw", "fgw", "relation_js"}
        if metric not in allowed:
            raise ValueError(f"wishart.metric must be one of {sorted(allowed)}")

        kwargs = dict(
            metric=metric,
            radius=int(raw.get("radius", 1)),
            max_ego_nodes=int(raw.get("max_ego_nodes", 48)),
            candidate_limit=int(raw.get("candidate_limit", 2000)),
            k_neighbors=int(raw.get("k_neighbors", 12)),
            significance=float(raw.get("significance", 0.7)),
            min_cluster_size=int(raw.get("min_cluster_size", 3)),
            min_cluster_mass=float(raw.get("min_cluster_mass", 3.0)),
            clustering_domain=str(raw.get("clustering_domain", "canonical_types")),
            density_weight=str(raw.get("density_weight", "occurrence_frequency")),
            min_figure_nodes=int(raw.get("min_figure_nodes", 2)),
            max_figures_per_level=int(raw.get("max_figures_per_level", 1000)),
            max_levels=int(raw.get("max_levels", 6)),
            min_graph_nodes=int(raw.get("min_graph_nodes", 50)),
            aggregation=str(raw.get("aggregation", "sum")),
            random_seed=int(raw.get("random_seed", 0)),
            wl_iterations=int(raw.get("wl_iterations", 2)),
            feature_dim=int(raw.get("feature_dim", 512)),
            graphlet_size=int(raw.get("graphlet_size", 3)),
            graphlet_samples=int(raw.get("graphlet_samples", 256)),
            relation_js_block_size=int(raw.get("relation_js_block_size", 128)),
            transport_rank=int(raw.get("transport_rank", 24)),
            transport_max_candidates=int(raw.get("transport_max_candidates", 256)),
            fgw_alpha=float(raw.get("fgw_alpha", 0.5)),
            slow_modes=int(raw.get("slow_modes", 8)),
            mfpt_pairs=int(raw.get("mfpt_pairs", 24)),
            mfpt_walks_per_pair=int(raw.get("mfpt_walks_per_pair", 8)),
            mfpt_max_steps=int(raw.get("mfpt_max_steps", 500)),
            betweenness_samples=int(raw.get("betweenness_samples", 64)),
            clustering_samples=int(raw.get("clustering_samples", 256)),
            distance_samples=int(raw.get("distance_samples", 128)),
        )
        result = cls(**kwargs)
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "radius": self.radius,
            "max_ego_nodes": self.max_ego_nodes,
            "candidate_limit": self.candidate_limit,
            "k_neighbors": self.k_neighbors,
            "min_cluster_size": self.min_cluster_size,
            "min_figure_nodes": self.min_figure_nodes,
            "max_figures_per_level": self.max_figures_per_level,
            "max_levels": self.max_levels,
            "min_graph_nodes": self.min_graph_nodes,
            "wl_iterations": self.wl_iterations,
            "feature_dim": self.feature_dim,
            "graphlet_size": self.graphlet_size,
            "graphlet_samples": self.graphlet_samples,
            "relation_js_block_size": self.relation_js_block_size,
            "transport_rank": self.transport_rank,
            "transport_max_candidates": self.transport_max_candidates,
            "slow_modes": self.slow_modes,
            "mfpt_pairs": self.mfpt_pairs,
            "mfpt_walks_per_pair": self.mfpt_walks_per_pair,
            "mfpt_max_steps": self.mfpt_max_steps,
            "betweenness_samples": self.betweenness_samples,
            "clustering_samples": self.clustering_samples,
            "distance_samples": self.distance_samples,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"wishart values must be positive: {', '.join(bad)}")
        if self.aggregation not in {"sum", "mean_density"}:
            raise ValueError("wishart.aggregation must be sum or mean_density")
        if self.significance < 0:
            raise ValueError("wishart.significance must be non-negative")
        if self.min_cluster_mass <= 0:
            raise ValueError("wishart.min_cluster_mass must be positive")
        if self.clustering_domain not in {"candidates", "canonical_types"}:
            raise ValueError(
                "wishart.clustering_domain must be candidates or canonical_types"
            )
        if self.density_weight not in {"uniform", "occurrence_frequency"}:
            raise ValueError(
                "wishart.density_weight must be uniform or occurrence_frequency"
            )
        if not 0 <= self.fgw_alpha <= 1:
            raise ValueError("wishart.fgw_alpha must be in [0, 1]")
        if self.graphlet_size not in {3, 4}:
            raise ValueError("wishart.graphlet_size must be 3 or 4")


def load_wishart_options(path: str | Path) -> WishartOptions:
    source = Path(path)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("config must be a mapping")
    raw = document.get("wishart", {})
    if not isinstance(raw, Mapping):
        raise ValueError("wishart must be a mapping")
    return WishartOptions.from_mapping(raw)


@dataclass(frozen=True)
class DictionaryOptions:
    """Persistent graph-dictionary and MDL selection options."""

    enabled: bool = True
    boundary_sensitive: bool = True
    frequency_scan: Literal["full", "discovery"] = "full"
    frequency_scan_batch_size: int = 5000
    min_support: int = 3
    min_mdl_gain_bits: float = 0.0
    local_improvement: bool = True
    family_match_jaccard: float = 0.5
    max_dictionary_size: int = 20000
    huffman: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DictionaryOptions":
        selection = raw.get("selection", {})
        if not isinstance(selection, Mapping):
            raise ValueError("dictionary.selection must be a mapping")
        limits = raw.get("limits", {})
        if not isinstance(limits, Mapping):
            raise ValueError("dictionary.limits must be a mapping")
        result = cls(
            enabled=bool(raw.get("enabled", True)),
            boundary_sensitive=bool(raw.get("boundary_sensitive", True)),
            frequency_scan=str(raw.get("frequency_scan", "full")),
            frequency_scan_batch_size=int(raw.get("frequency_scan_batch_size", 5000)),
            min_support=int(raw.get("min_support", 3)),
            min_mdl_gain_bits=float(selection.get("min_gain_bits", 0.0)),
            local_improvement=bool(selection.get("local_improvement", True)),
            family_match_jaccard=float(raw.get("family_match_jaccard", 0.5)),
            max_dictionary_size=int(limits.get("max_dictionary_size", 20000)),
            huffman=bool(raw.get("huffman", True)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.frequency_scan not in {"full", "discovery"}:
            raise ValueError("dictionary.frequency_scan must be full or discovery")
        if self.frequency_scan_batch_size <= 0:
            raise ValueError("dictionary.frequency_scan_batch_size must be positive")
        if self.min_support <= 0:
            raise ValueError("dictionary.min_support must be positive")
        if self.max_dictionary_size <= 0:
            raise ValueError("dictionary.max_dictionary_size must be positive")
        if not 0.0 <= self.family_match_jaccard <= 1.0:
            raise ValueError("dictionary.family_match_jaccard must be in [0, 1]")


def load_dictionary_options(path: str | Path) -> DictionaryOptions:
    source = Path(path)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("config must be a mapping")
    raw = document.get("dictionary", {})
    if not isinstance(raw, Mapping):
        raise ValueError("dictionary must be a mapping")
    return DictionaryOptions.from_mapping(raw)
