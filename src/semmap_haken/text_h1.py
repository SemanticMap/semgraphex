"""Text-driven H1 entry point, reusing the existing nSBM/MDL/null runner.

All occurrence copies of one ConceptNet concept-pair relation are held out
together. Fit the hierarchy ONLY on the training split; ConceptNet-induced
text graphs remain internal controls, not independent semantic validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hypothesis.runner import H1Config, run_h1
from .text_research import audit_corpus
from .text_research_adapter import load_typed_text_graph


def run_text_h1(
    corpus: str | Path, doc_id: str, output: str | Path, *,
    config: H1Config | None = None,
    projection: str = "concept",
    max_distance_chars: int | None = None,
    position_shuffle_seed: int | None = None,
    max_vertices: int = 10_000,
    max_edges: int = 200_000,
    resume: bool = False,
    fit_fn=None,
    flat_sampler=None,
) -> dict:
    """Bounded, reproducible text projection → existing H1 scientific driver.

    The adapter rejects oversized graphs, rather than taking a biased prefix.
    Position-shuffled controls share their ConceptNet-induced unweighted
    topology; only distance-dependent weights/window edges can change.
    """
    if position_shuffle_seed is not None and max_distance_chars is None:
        raise ValueError(
            "Position shuffle without a distance window changes only weights; "
            "the current H1 nSBM ignores edge weights. Set max_distance_chars."
        )
    audit = audit_corpus(corpus, doc_id=doc_id)
    row = audit["documents"][0]
    typed = load_typed_text_graph(
        corpus, doc_id, projection=projection,
        max_distance_chars=max_distance_chars,
        position_shuffle_seed=position_shuffle_seed,
        max_vertices=max_vertices, max_edges=max_edges,
    )
    source_contract = {
        "schema": "text-h1-sources-v1",
        "corpus_run_sha256": audit["run_sha256"],
        "doc_id": doc_id,
        "document_identity_key": row["key"],
        "stored_graph_state": row["state"],
        "implicit_full_graph_reconstructed": True,
        "projection": typed.metadata(),
    }
    # Runner owns H1 model fitting, restarts, independent train/val/test
    # null ensembles, checkpointing and conservative scientific verdict.
    return run_h1(
        typed.graph, output,
        config=config if config is not None else H1Config(),
        resume=resume,
        group_keys=typed.holdout_groups,
        source_contract=source_contract,
        fit_fn=fit_fn,
        flat_sampler=flat_sampler,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run ConceptNet-induced text H1 (not independent text evidence)"
    )
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--doc-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--projection", choices=("concept", "occurrence"),
                   default="concept")
    p.add_argument("--max-distance-chars", type=int)
    p.add_argument("--position-shuffle-seed", type=int)
    p.add_argument("--max-vertices", type=int, default=10_000)
    p.add_argument("--max-edges", type=int, default=200_000)
    p.add_argument("--mode", choices=("NEW", "RESUME"), default="NEW")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--nulls", type=int, default=99)
    p.add_argument("--workers", default="auto")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--negatives", type=int, default=100)
    args = p.parse_args(argv)
    workers = args.workers
    if workers != "auto":
        try:
            workers = int(workers)
        except ValueError:
            p.error("--workers must be auto or a positive integer")
    config = H1Config(
        seed=args.seed, restarts=args.restarts, null_replicates=args.nulls,
        workers=workers, bootstrap_repetitions=args.bootstrap,
        negatives=args.negatives,
    )
    result = run_text_h1(
        args.corpus, args.doc_id, args.output, config=config,
        projection=args.projection,
        max_distance_chars=args.max_distance_chars,
        position_shuffle_seed=args.position_shuffle_seed,
        max_vertices=args.max_vertices, max_edges=args.max_edges,
        resume=args.mode == "RESUME",
    )
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "decision": result["decision"],
        "source_scope": "ConceptNet-induced text projection (not independent)",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
