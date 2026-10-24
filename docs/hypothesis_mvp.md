# Multiscale hypothesis — MVP-1

This first slice starts implementation of the dissertation's experiments 3.1–3.7. It is **not** the finished hypothesis-validation framework and makes **no claim** that hierarchy or graphex limits have been established.

## New code

| Area | Module | Status |
|---|---|---|
| Typed directed assertion graph (retains assertion URI, relation, weight and parallel records) | `hypothesis/graph.py` | Initial working slice |
| Directed relation-specific in/out-degree-preserving controls | `hypothesis/nulls.py` | Initial working slice; no proven chain-mixing convergence |
| Vertex-induced p-sampling and edge dropout | `hypothesis/graph.py` | Initial working slice |
| Sparse original-mass block intensity projection, CPU or batched CuPy CUDA | `hypothesis/projection.py` | Finite empirical **W-only** kernel, **not** a graphex limit |
| Ordinary SBM / DC-SBM / nested DC-SBM with relation layers | `hypothesis/inference.py` | Optional `graph-tool` adapter; needs graph-tool integration and scale verification |
| End-to-end pilot and parallel null fits | `hypothesis/cli.py` | Reproducible smoke path; does not test held-out prediction |

## Run

```bash
python -m pip install -e '.[dev]'
semmap-hypothesis pilot \
  --input tests/fixtures/conceptnet_tiny.tsv \
  --output tmp/runs/hypothesis-smoke-01 \
  --nulls 4 --workers auto --device cpu
```

`--device auto` tries CuPy CUDA first and records actual backend and fallback reason. For Colab with compatible CUDA 12, install the existing `cuda` extra; `--device cuda` fails instead of silently using CPU. GPU acceleration is used for sparse block aggregation **only**. `graph-tool` Bayesian inference uses CPU/OpenMP and is installed separately using a compatible operating-system or conda environment; it is not a pip dependency of the base package.

To fit actual models and control ensembles (requires `graph-tool`):

```bash
semmap-hypothesis pilot \
  --input path/to/conceptnet-assertions.tsv \
  --output tmp/runs/hypothesis-fit-01 \
  --nulls 99 --workers auto --device auto --fit
```

Outputs: `typed_graph.npz` (full assertion IDs and typed edge records), optional `graphon_level_*.json` for every inferred nSBM level, `summary.json` (input SHA-256, hardware choice, worker budget, per-relation controls, optional MDL scores), `COMPLETED` written last. Run with a new output name rather than silently overwriting old evidence. For a quick prefix smoke, use `--max-rows N` (physical input rows; **biased**, not suitable for evidence). Input is currently materialized in RAM — start with a 10k-node pilot; large 100k/1M memory scaling is **not yet verified**. Use fast Colab-local scratch storage and sync only completed artifacts to Drive.

## Concurrency contract

- Null controls are independent processes (`spawn` to avoid CUDA/fork hazards). The auto worker count is bounded by CPUs, available memory estimate, and number of tasks.
- `graph-tool` internal OpenMP threads share the CPU budget with the null-worker pool. No nested `workers × all CPU threads` oversubscription.
- CuPy operates in bounded edge batches, transferring only aggregated keys back to host. There is no dense vertex-by-vertex matrix.
- Reproducible per-control seeds and the actual CPU/GPU choice appear in the summary; seed alone does not guarantee bitwise deterministic CUDA/graph-tool results.

## Required scientific follow-ups

1. Verify graph-tool LayeredBlockState compatibility and MDL on a dedicated environment (CI without graph-tool covers import/error handling only). Include multiple inference initializations and held-out evaluation.
2. Validate null mixing/acceptance and empirical significance calibration on ground-truth synthetic graphs. Degree-preserving control cannot test excess **total** degree-one nodes.
3. Add honest split protocols that group parallel assertions and avoid duplicate/reverse leakage.
4. Add paired perturbation experiments and hierarchical partition alignment/AMI/VI.
5. Fit **generative** `(W,S,I)` with uncertainty and model-selection baselines; current projection **does not** estimate S or I. Keep exact codec residual Δ separate.
6. Evaluate p-sampling across multiple p/seeds and independently acquired networks (ConceptNet, RuWordNet and corpus-derived graph). One p-sample is only a smoke check.
7. Wire existing grammar compression, GloVe/SVD and dynamics as independent validators only after the primary hierarchy detector is finalized.
8. Move full assertion provenance metadata (sources/dataset/license) into a dedicated immutable ledger instead of only retaining URI and source checksum.

Do not treat `COMPLETED` as scientific confirmation: it means artifacts were persisted.
