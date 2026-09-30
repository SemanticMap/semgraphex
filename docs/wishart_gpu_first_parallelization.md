# GPU-first Wishart + maximum safe Python CPU concurrency

This document accompanies the user-supplied `gpu_first_remediation_plan.md`.
It records implemented steps, reproducibility boundaries and remaining
profiling-gated optimizations. All source references are relative to this
repository and the `feature/wishart-gpu-first-parallel` branch.

## Implementation ledger

| Original plan | Status | Code / verification |
| --- | --- | --- |
| S1 Torch `gpu` optional extra | Implemented | `pyproject.toml`; imported lazily in `wishart_gpu.py` |
| S2 Colab-only relaxed constraints | Implemented | `requirements/constraints-colab.txt`; remove conflicting base `numpy<2`, `scipy<1.14`, notebook matplotlib upper bound |
| S3 install/probe before computation | Implemented | notebook installs `.[wishart,notebook,gpu]`, imports scientific stack and displays GPU/driver diagnostic |
| S4 `-gpu` / `-cpu` runs | Implemented | notebook stable device-suffixed run names |
| S5 graphlet forwards GPU flags | Implemented | `wishart_metrics.build_neighbor_graph` |
| S6 CPU fallback visible | Implemented | `cuda_diagnostics`, CLI `device_selected` and warning, kNN backend metadata |
| S7 device mismatch guidance | Implemented | `_assert_resume_device_consistent` |
| S8 CPU CI regression suite | Implemented | `tests/test_semmap_haken_colab_resume_gpu.py` and notebook test |
| S9 docs/config comments | Implemented | this doc, Colab how-to, and dictionary YAML |
| S10 real Colab GPU/manual benchmarks | Requires a user-authorized Colab runtime | no T4/L4 performance numbers claimed |
| P1 cache prototype graphs / MDL costs | Implemented | transient LRU, reusable candidate graph, per-type cost cache |
| P2 process fingerprint discovery | Implemented | `ordered_fingerprints` + serial type-ID allocation |
| P3 process full-scan VF2 | Implemented | spawned read-only dictionary workers, ordered bounded `Future` queue |
| P4 vectorize prepared relation-layer build | Profiling-gated | first measure one-time preparation vs per-level costs; preserve duplicate weight summation order |
| P5 sparse relation contractions and flow blocks | Implemented | threads + shared membership CSR |
| P6 typed-WL and graphlet feature batches | Implemented | ordered spawned chunks, SciPy CSR vertical assembly |
| P7b source-sampled Brandes | Implemented | spawned processes + ordered per-source vector reduction |
| P7c/d shortest paths / clustering | Implemented | bounded ordered threads + unchanged RNG stream |
| P7a MFPT seed-splitting | Not enabled | would change RNG streams; requires explicit scientific ablation |
| P7e ARPACK eigsh concurrency | Profiling-gated | numerical/library thread-safety and convergence first |
| P8 parallel Google Drive FUSE sync | Not enabled | throughput uncertain; `COMPLETED` must remain the last durable write |
| P9 single effective worker budget | Implemented | `colab.cpu_workers` capped by visible vCPU count; BLAS = 1 |
| P10 parallel-vs-serial equality tests | Implemented for active parallel paths | CPU CI: worker counts, exact arrays and resume |
| P11 phase timing | Implemented | `levels_detail[*].phase_timing_seconds` |
| P12 order-dependent phases left serial | Preserved | Wishart density sweep, MDL packing, Huffman, level recurrence |

## Actual parallelization model

The graph hierarchy is **not** parallelized across levels: level `s+1`
depends on the quotient computed at level `s`. The parallel boundary is
inside a level. For independent SciPy sparse operations, bounded threads
avoid copying a 100k graph. For GIL-heavy NetworkX VF2/WL and Python Brandes,
`ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'))`
allocates independent Python interpreters. This is essential after CUDA has
been probed in the parent. Only read-only dictionary snapshots enter workers;
all mutations of the persistent type registry, overlap selection, frequency
counts and checkpoint state happen in the main process.

Parallel full scan stages only `2 * cpu_workers` batches concurrently, then
consumes the oldest future. This implements backpressure rather than eager
scheduling of 100k ego candidates. Worker results contain compact
`(type_id, prototype_mapping)` tuples instead of complete `GraphType`
objects. The original center order fixes `candidate_index`, de-duplication
and type frequencies. The extraction and match pools do not run nested sets
of `cpu_workers` threads/processes: while spawn matching workers are active,
the parent extracts each batch with one thread.

For feature generation, per-batch sparse matrices are assembled by
`scipy.sparse.vstack` in submitted chunk order. The graphlet RNG remains
seeded per candidate center. Prototype graphs are kept in a bounded,
transient LRU and excluded from checkpoint pickle; workers construct their
own cache. Relation-degree CSR arrays are computed once per scan instead
of once per candidate.

## GPU-first device contract

`colab.device: auto` is intentionally **not** replaced with `cuda` in
the scientific YAML: a CPU-only Colab runtime must still work. During
bootstrap, the notebook installs the Torch extra using Colab-specific
constraints and prints an explicit CUDA probe; the CLI freezes the resolved
CPU/CUDA backend for the entire run. Under a CUDA-selected run, every
eligible `typed_wl`/`graphlet` kNN step with at least two types uses
CUDA. GPU query tiles shrink on OOM. If the smallest tile fails, the run
fails explicitly so its scientific lineage is not silently mixed with
CPU float64 kNN. Unsupported metrics (`relation_js`, GW/FGW) remain CPU.

The `input.json` manifest records the requested and selected device,
GPU name/VRAM where available, effective CPU workers, config hash, source
fingerprint and Git SHA. Distinct `-gpu` and `-cpu` run names make both
lineages independently resumable. Restoring a checkpoint verifies its
version, payload SHA256, config SHA256, input SHA256 and Git revision.

## Benchmark protocol for the 100k graph

Run the same prepared graph and science YAML with a *new run name* for each
backend/worker configuration. Compare:

1. CPU serial: `--device cpu --cpu-workers 1 --blas-threads 1`.
2. CPU parallel: `--device cpu --cpu-workers 2` (or all visible cores).
3. GPU-first: `--device auto --cpu-workers <visible cores>`.

Record `input.json`, transition `wishart.json.metric.backend`,
`hierarchy.json` `phase_timing_seconds`, peak RSS/VRAM if available,
and output scientific summaries. Compare bit-identical dictionary/occurrence
outputs only **within the same CPU numerical backend**. GPU float32 cosine
may differ close to ties, so evaluate GPU-vs-CPU changes in cluster label
stability, MDL, coverage and slow-mode distortion rather than asserting
identical labels.

Only report wall-time gains after observing the hardware-specific profile.
Python process spawn/import, dictionary snapshot copies, Colab CPU quotas,
GPU feature-transfer overhead and Drive FUSE may dominate small workloads.

## Next safe optimizations (profile first)

* Prepared relation-layer construction: aggregate URI-indexed edges into
  typed COO/CSR with stable grouping and test exact duplicate-weight
  summation order before enabling it.
* Heavy MFPT random walks: add a separate opt-in experiment with
  `SeedSequence([seed, pair_index, walk_index])` and distinct baseline
  metrics. Do **not** silently replace the existing sequential RNG.
* Sampled betweenness: consider shared-memory or memory-mapped CSR arrays
  if the process-initializer dictionary/topology copy becomes the bottleneck.
  Do not share mutable sparse data across processes.
* GPU kNN: profile float32 dense feature transfer, VRAM and query tile size
  on the actual T4/L4; consider chunked database tiles if unique type-space
  no longer fits VRAM. Never allocate the full `N_types^2` distance matrix.
* Drive I/O: only after profiling, thread at most 2–4 independent files,
  leave `COMPLETED`/manifest ordering sequential, compare sha256 when
  corruption detection matters.
* ARPACK, order-dependent density merging and MDL set packing: retain
  serial semantics until dedicated deterministic alternatives are proven.

## Verification boundary

GitHub Actions runs CPU regression tests, including spawned workers,
file-based checkpoints and bit-for-bit CPU worker equality. It does **not**
have access to the user's authorized Drive or an actual Colab GPU runtime.
The hardware probe, CUDA backend metadata, reset-and-resume rehearsal, and
100k timings must be recorded from a clean GPU Colab session before treating
this as a measured performance optimization.
