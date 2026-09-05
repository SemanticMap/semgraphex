# Rapid scientific delivery plan

## Decision and scope

This document supersedes the **execution cadence** of the detailed [Haken coarsening roadmap](haken_coarsening_roadmap.md), not the scientific requirements, falsification criteria, or reproducibility obligations in [AGENTS.md](../AGENTS.md). M0 is complete at `b6bf091` after the listed M0 implementation commits, with 37 active tests reported passing. The next objective is a sequence of runnable scientific increments, each producing evidence rather than process artifacts.

The active architecture remains notebook-first, library-backed, and CLI-reproducible: thin Colab notebooks call public [semmap_haken](../src/semmap_haken/) library functions or the same CLI paths, while YAML, checksum, seed, manifest, and artifact schema make every scientific result replayable. Sparse matrices and iterative solvers are mandatory; no dense N-by-N materialization is permitted for prepared ConceptNet graphs.

```mermaid
flowchart LR
  A[M1 linear evidence] --> B[M2 one-step coarsening]
  B --> C[M3 multi-scale hierarchy]
  C --> D[M4 synthetic plateau test]
  D --> E[M5 ConceptNet-small evidence]
  E --> F[Claim-boundary scientific review]
```

## Delivery policy

### GPU-first and multicore-fallback execution

Google Colab NVIDIA GPU is the preferred backend for sparse spectral analysis, batched dynamics, embedding distances, and later bootstrap/baseline workloads. Multicore CPU is the mandatory fallback and must use all safely available physical cores for independent trajectories, replicates, baselines, nulls, parameter sweeps, and other parallel hard-math tasks.

The execution backend changes performance, not scientific semantics: CPU and GPU consume the same sparse artifacts and emit the same versioned result schemas. Backend choice, hardware, dtype, batching, tolerances, and fallback reason are persisted in every run.

```mermaid
flowchart TD
  C[Execution config] --> H[Hardware detection]
  H --> G[CuPy CUDA backend]
  H --> P[Multicore SciPy backend]
  G --> K[Shared scientific kernels]
  P --> K
  K --> W[Within run batching]
  W --> S[Across run scheduler]
  S --> A[Artifacts and telemetry]
```

**Numerical backend.** Add `src/semmap_haken/compute.py` with a `ComputeContext` containing requested and selected backend, device, dtype, worker count, thread limits, memory budget, deterministic seed policy, capability flags, and fallback reason. Adapters expose sparse CSR construction/transfer, sparse matrix products, symmetric partial eigensolve, batched linear evolution, reductions, and host conversion at artifact boundaries.

**Backend selection.** `auto` selects CUDA only when CuPy loads, a CUDA device is visible, required sparse primitives are supported, and VRAM preflight succeeds; otherwise it selects multicore CPU and records why. `cuda` is strict and fails rather than silently falling back. `cpu` is strict and never imports CuPy.

**Within-run parallelism.** CUDA keeps the CSR operator resident, stacks perturbations as multiple right-hand sides, and processes VRAM-bounded batches. Use CuPy memory pools, pinned transfer buffers, and CUDA streams only where profiling shows useful overlap. CPU uses vectorized multi-RHS sparse kernels first, then process-level chunking for independent perturbations/replicates; BLAS/OpenMP threads are capped per worker to prevent oversubscription.

**Across-run parallelism.** From Increment C, seeds, bootstrap samples, baselines, null models, relation variants, and parameter sweeps run concurrently. One GPU-heavy task per device is the default; CPU-only work may run concurrently within reserved-core and RAM limits. Levels of one coarsening chain remain sequential, while independent methods and seeds at a level run in parallel.

**Configuration contract:**

```yaml
execution:
  backend: auto
  device: 0
  dtype: float64
  workers: auto
  reserved_cpu_cores: 1
  threads_per_worker: 1
  gpu_memory_fraction: 0.80
  batch_size: auto
  deterministic: true
  allow_auto_fallback: true
```

**Memory and scheduling rules:**

- Never copy the sparse graph into every CPU worker; use read-only inherited memory where safe or shared/memory-mapped CSR buffers under spawn runtimes.
- Never submit one tiny CUDA job per trajectory; batch perturbations and reduce on device.
- Avoid nested parallelism. The outer scheduler owns processes; each worker gets explicit BLAS/OpenMP limits.
- Estimate CSR, eigenvector, trajectory, reconstruction, and serialization memory before dispatch. Reduce batch size before fallback/failure.
- Synchronize CUDA only for timing and artifact boundaries; reuse memory pools and free cached blocks before unrelated large stages.
- Derive seeds from run seed plus stable task identity, never worker completion order. Merge result shards in stable task-ID order.

**Telemetry.** Record logical/physical CPU counts, selected workers/threads, GPU model/compute capability/VRAM, CuPy/CUDA versions, dtype, batch size, peak RAM/VRAM, transfer time, solver/kernel time, wall time, throughput, and fallback reason.

**Scientific parity.** CPU float64 is the tiny analytic reference. CUDA float64 must agree on eigenvalues, residuals, invariant subspaces, growth/stability decisions, and aggregate trajectory errors within configured tolerances. Raw eigenvectors may differ by sign or rotation. CUDA float32 remains a performance ablation until `r` and downstream decisions are shown stable.

**Acceleration acceptance:**

1. Serial CPU, multicore CPU, and CUDA outputs agree within declared scientific tolerances.
2. Multicore execution shows useful speedup for at least eight independent tasks without breaking the RAM budget.
3. CUDA shows end-to-end speedup on a representative small graph including transfers; `auto` may keep tiny graphs on CPU.
4. No backend materializes dense node-by-node matrices.
5. Automatic batching survives constrained Colab VRAM and records the chosen size.
6. Forced-CUDA failure, automatic fallback, worker failure, and interrupted shard merge produce explicit failed/resumable states.

### Lean quality and review cadence

- Each implementation subtask adds focused tests for its new public contract, numerical invariants, determinism, and sparse behavior.
- Each increment ends with one integration smoke that runs the notebook and its CLI route against the same small input and records usable artifacts.
- Normal increments do **not** create a separate reviewer-agent workstream and do not require review after every task. The implementing Code specialist validates its own work against the stated acceptance criteria.
- A scientific review is required only before a claim based on ConceptNet evidence and before publication. It checks hypothesis wording, provenance, leakage, matched-compression comparisons, null/baseline interpretation, and the distinction between Observed and Interpretation.
- An optional Test Engineer is engaged only to resolve a failing integration smoke, unstable numerical test, or reproducibility regression—not as routine handoff work.
- Commit once at the end of each increment after its tests and smoke pass; do not fragment commits by internal microtask.

### Deferred until the core evidence loop works

Defer the directed branch, all remaining merge variants, the full six-baseline catalog, nonlinear/slaving experiments, graphon family modeling, relation-specific graphs, multiplex graphs, and medium-scale ConceptNet. Workstream 5C is skipped entirely. These remain future requirements from [AGENTS.md](../AGENTS.md), not cancelled research questions; they return only after Increment E produces a reproducible small-graph evidence bundle and identifies the highest-value ablations.

## Vertical increments

Each increment has at most two implementation subtasks, a single commit boundary, and a runnable notebook plus CLI deliverable.

### Increment A — M1 linear spectral and dynamics baseline

**Scientific question:** Does a prepared sparse graph exhibit a reproducibly measurable slow/critical linear spectral regime under an explicit, stable parameter rule?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| A1 — operators, modes, configuration | Add sparse normalized adjacency, iterative eigensolver diagnostics, explicit `beta: auto_critical` selection, and a consensus `r` decision from eigengap, timescale-gap, and elbow signals. Likely files: [operators.py](../src/semmap_haken/operators.py), [modes.py](../src/semmap_haken/modes.py), [config.py](../src/semmap_haken/config.py), [haken_linear.yaml](../configs/haken_linear.yaml), and focused tests under [tests](../tests/). | Operates directly on CSR/CSC input; rejects invalid or non-converged configurations clearly; reports eigenvalues, growth rates, relaxation times, participation/localization diagnostics, individual heuristic choices, consensus `r`, and uncertainty/fallback reason. The auto-critical rule is documented, deterministic for a fixed config/seed, and preserves linear stability. |
| A2 — dynamics, run route, artifacts | Add linear perturbation trajectories, plots/artifacts, a `run` CLI route, and thin [02_linear_spectral_dynamics.ipynb](../notebooks/02_linear_spectral_dynamics.ipynb). Likely files: [dynamics.py](../src/semmap_haken/dynamics.py), [cli.py](../src/semmap_haken/cli.py), [artifacts.py](../src/semmap_haken/artifacts.py), [notebook.py](../src/semmap_haken/notebook.py), and [tests](../tests/). | Runs on both a synthetic sparse graph and an M0-prepared ConceptNet graph; saves resolved config, manifest-linked numeric tables, deterministic perturbation seeds, trajectories, spectrum/eigengap/relaxation plots, and an Observed-only summary. Notebook and CLI produce equivalent scientific arrays/metrics for the same config. |

**Deliverable:** [02_linear_spectral_dynamics.ipynb](../notebooks/02_linear_spectral_dynamics.ipynb) and `python -m semmap_haken run --config configs/haken_linear.yaml`, yielding an M1 artifact directory with diagnostics and trajectory figures.

**Tests and smoke:** unit tests for normalization, eigensolver ordering/sign-invariant subspace output, beta stability rule, `r` consensus, and known-mode analytic trajectory; one notebook-to-CLI smoke on a tiny synthetic graph plus prepared fixture.

**Commit boundary:** `feat: add linear spectral dynamics baseline`.

**Acceleration amendment:** finish the current CPU functional A2 workflow first, then add one focused acceleration-foundation commit before Increment B. It introduces `ComputeContext`, CuPy sparse `eigsh` parity where supported, GPU-resident batched dynamics, multicore CPU trajectory chunking, execution telemetry, and Colab benchmark output without changing M1 formulas or result schemas.

### Increment B — M2 one-step Haken coarsening

**Scientific question:** At matched one-step compression, can slow-mode coordinates define a contraction with measurable dynamic loss?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| B1 — embedding and merges | Add a Haken embedding from the selected slow subspace; implement one scalable connectivity-constrained merge and one simple unconstrained comparator. Likely files: [haken_embedding.py](../src/semmap_haken/haken_embedding.py), [coarsen.py](../src/semmap_haken/coarsen.py), [config.py](../src/semmap_haken/config.py), and [tests](../tests/). | No label/relation text enters the partition. Each node belongs to one cluster; connected mode merges only adjacent nodes; target reduction is achieved or the shortfall is recorded; seed and tie-breaking are deterministic. |
| B2 — quotient, lifting, distortion | Build the quotient and reversible node-to-supernode mapping, then calculate lifted fine/coarse slow-subspace and trajectory distortion. Likely files: [quotient.py](../src/semmap_haken/quotient.py), [metrics.py](../src/semmap_haken/metrics.py), [cli.py](../src/semmap_haken/cli.py), and [03_one_step_haken_coarsening.ipynb](../notebooks/03_one_step_haken_coarsening.ipynb). | Quotient weights, membership mapping, masses, and provenance are persisted; lifted comparison is sign/rotation invariant; run emits compression, slow-subspace distance, slow-eigenvalue error, and trajectory error for both merges. |

**Deliverable:** [03_one_step_haken_coarsening.ipynb](../notebooks/03_one_step_haken_coarsening.ipynb) and the same `run` CLI with a coarsening config.

**Tests and smoke:** hand-checkable quotient fixture, partition/mapping conservation, deterministic merge test, and synthetic plus prepared-graph one-step smoke.

**Commit boundary:** `feat: add one-step haken coarsening`.

### Increment C — M3 minimal multi-scale hierarchy

**Scientific question:** Can the one-step contract run as a restartable hierarchy while retaining the measurements needed to decide whether to continue?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| C1 — hierarchy runner | Add a checkpointable/resumable loop over the M2 contraction. Likely files: [coarsen.py](../src/semmap_haken/coarsen.py), [manifest.py](../src/semmap_haken/manifest.py), [cli.py](../src/semmap_haken/cli.py), and [tests](../tests/). | Produces at least five levels when data permits; resume from an interrupted completed checkpoint produces the same hierarchy as a fresh fixed-seed run; all parent-child mappings are retained. |
| C2 — scale evidence | Add per-scale metrics, simple stop signals, and [04_multiscale_hierarchy.ipynb](../notebooks/04_multiscale_hierarchy.ipynb). Likely files: [metrics.py](../src/semmap_haken/metrics.py), [plateau.py](../src/semmap_haken/plateau.py), and [tests](../tests/). | Records node/edge counts, compression, `r`, eigengap, slow-subspace distance, slow-eigenvalue error, trajectory error, runtime, and explicit stop reason. Stops on a configured signal rather than a fixed final cluster count, while retaining the full completed trajectory. |

**Deliverable:** [04_multiscale_hierarchy.ipynb](../notebooks/04_multiscale_hierarchy.ipynb) and a resumable CLI run directory.

**Tests and smoke:** resume equivalence, five-level tiny synthetic hierarchy, mapping conservation, and CLI/notebook smoke.

**Commit boundary:** `feat: add resumable multiscale hierarchy`.

### Increment D — M4 minimal synthetic plateau detector

**Scientific question:** Does the detection rule separate a deliberately hierarchical positive control from a no-separation negative control?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| D1 — two generators | Add one sparse hierarchical positive generator and one matched-size no-timescale-separation negative generator. Likely files: [synthetic.py](../src/semmap_haken/synthetic.py), [configs](../configs/), and [tests](../tests/). | Both are seeded, sparse, parameter-recorded, and expose their intended ground-truth condition without leaking labels into coarsening. |
| D2 — detector and report | Implement the minimum configured consecutive-level plateau rule and synthetic result table/plot. Likely files: [plateau.py](../src/semmap_haken/plateau.py), [metrics.py](../src/semmap_haken/metrics.py), [cli.py](../src/semmap_haken/cli.py), and [tests](../tests/). | Positive control yields the intended plateau candidate under its declared parameter regime; negative control does not yield an equivalent candidate. Failures remain valid empirical outputs and are reported as such. |

**Deliverable:** CLI synthetic experiment configuration and a compact artifact table/plot, integrated into notebook 04 or a small follow-up section.

**Tests and smoke:** deterministic generator tests, positive/negative detector regression tests, and one end-to-end synthetic CLI smoke.

**Commit boundary:** `feat: add minimal synthetic plateau validation`.

### Increment E — M5 first ConceptNet-small evidence bundle

**Scientific question:** On the prepared ConceptNet-small graph, does the topology-only Haken path preserve selected dynamics better than a small, matched-compression control set and survive basic null checks?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| E1 — fast comparisons | Add exactly three fast baselines—random matching, heavy-edge matching, and spectral coarsening—and two nulls—degree-preserving rewiring and edge-weight shuffle. Likely files: [baselines.py](../src/semmap_haken/baselines.py), [null_models.py](../src/semmap_haken/null_models.py), [metrics.py](../src/semmap_haken/metrics.py), and [tests](../tests/). | Every comparator uses the same prepared graph, seed ledger, compression schedule, and distortion metrics; failed/infeasible baseline runs are visible rather than silently excluded. |
| E2 — evidence bundle and post-hoc interpretation | Produce the reportable ConceptNet-small run and post-hoc cluster summaries without label leakage. Likely files: [semantic_labels.py](../src/semmap_haken/semantic_labels.py), [cli.py](../src/semmap_haken/cli.py), [05_conceptnet_small_evidence.ipynb](../notebooks/05_conceptnet_small_evidence.ipynb), and [tests](../tests/). | Artifacts include provenance, per-scale matched-compression table, null results, representative nodes, relation histogram, and an explicit Observed/Interpretation/Alternative explanations/Status report. No causal or semantic claim is made before the claim-boundary scientific review. |

**Deliverable:** [05_conceptnet_small_evidence.ipynb](../notebooks/05_conceptnet_small_evidence.ipynb), CLI evidence run, and a concise report artifact.

**Tests and smoke:** matched-compression contract tests, deterministic null tests, no-label-input partition test, and one full small-config CLI/notebook smoke.

**Commit boundary:** `feat: add conceptnet small evidence bundle`.

## Immediate execution — Increment A

Use two sequential Code-specialist calls with a single integration point. Do not schedule normal review work.

1. **Code specialist 1 — spectral core:** implement A1 only: sparse normalized operator, iterative spectrum diagnostics, deterministic stable auto-critical beta rule, three-signal `r` selection, YAML validation, and focused unit tests. Return the public data contracts and a config runnable by the next call.
2. **Code specialist 2 — runnable experiment:** build on A1 without redesigning it; implement A2: linear trajectory solver, artifact writers and plots, `run` CLI integration, notebook 02, synthetic and prepared-graph demonstrations, and the increment-end notebook-to-CLI smoke.
3. **Conditional Test Engineer:** call only if the end smoke fails, numerical behavior is flaky, or notebook/CLI scientific arrays differ. The fix target is the concrete failing contract, followed by rerunning the single increment smoke.
4. **Code specialist 3 — acceleration foundation:** add optional CUDA dependencies, `ComputeContext`, strict `auto|cuda|cpu` selection, CuPy spectral/dynamics adapters, multicore fallback, telemetry, CPU/CUDA parity tests, and a benchmark section in notebook 02. CUDA absence skips accelerator tests and never fails CPU CI.

Increment A is complete only when the shared run produces the promised evidence artifacts from both inputs, all new focused tests pass, and the one integration smoke passes. Its output is an M1 baseline, not evidence that ConceptNet has Haken order parameters.

## Milestone claim boundary

Before calling Increment E results scientific evidence, perform one targeted scientific audit against [AGENTS.md](../AGENTS.md): verify provenance and replay, sparse-control interpretation, no semantic leakage, matched-compression fairness, null/baseline limitations, uncertainty, and the required Observed/Interpretation separation. Repeat this audit only at publication preparation or when a core hypothesis claim materially changes.
