# Rapid scientific delivery plan

## Decision and scope

This document supersedes the **execution cadence** of the detailed [Haken coarsening roadmap](haken_coarsening_roadmap.md), not the scientific requirements, falsification criteria, or reproducibility obligations in [AGENTS.md](../AGENTS.md). M0 is complete at `b6bf091`; M1 and its initial acceleration foundation are present at HEAD `1fab45b`. The next objective is M2 and the subsequent runnable scientific increments, each producing evidence rather than process artifacts.

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

### CPU-reference delivery with retained acceleration architecture

Further parallelism and acceleration hardening, acceptance benchmarking, and promotion of CUDA results are **deferred, not cancelled**. M2–M5 proceed by default on the scientifically trusted strict CPU float64 reference path. This is a delivery sequencing choice, not removal of the backend-neutral [`ComputeContext`](../src/semmap_haken/compute.py:37), CUDA adapters, execution telemetry, batching seams, or future within-run and across-run scheduling hooks introduced at `1fab45b`.

The execution backend may change performance, never scientific semantics. Every new scientific module must accept or pass through the shared execution context, avoid direct backend assumptions in its public contract, operate on sparse data, derive randomness from stable seeded task identities, merge batchable work deterministically, and preserve versioned artifact schemas. Levels in one coarsening chain remain sequential; independent merges, perturbations, seeds, baselines, nulls, and parameter variants must remain expressible as stable batches so parallel execution can resume without redesign.

For M2–M5 delivery, the reference configuration is:

**Configuration contract:**

```yaml
execution:
  backend: cpu
  dtype: float64
  workers: 1
  threads_per_worker: 1
  batch_size: auto
  deterministic: true
```

CPU-reference artifacts still record the selected backend, dtype, worker/batch settings, deterministic policy, wall time, memory information available from the current telemetry contract, and any fallback reason. New schemas must allow additive acceleration telemetry later without changing scientific field meanings.

### Deferred acceleration re-entry gate

Acceleration hardening may re-enter after M2 or M3 when it has clear delivery value, and must re-enter before medium-scale runs. Before claiming multicore/CUDA speedup or using CUDA-derived outputs as scientific evidence, close the blocker classes in the concise gate below; the detailed requirements and evidence checklist remain in the [M1 acceleration audit](m1_acceleration_improvement_plan.md#6-acceleration-acceptance-review-gate-to-increment-b).

1. Restore and verify a green, locked CPU float64 reference environment.
2. Complete deterministic solver seeding, narrow capability/error handling, and the backend-adapter seam.
3. Make multicore chunking reachable without sparse-graph copies and persist explicit failed/resumable shard states.
4. Add VRAM-bounded CUDA batching, resident sparse operators, and deterministic out-of-memory reduction/failure behavior.
5. Validate CUDA eigenvalue, residual, invariant-subspace, growth/stability, and trajectory-error parity against the CPU reference; fixed-step RK4 or reduced modal evolution must remain explicitly distinct from reference full-system propagation.
6. Complete memory/timing telemetry and representative end-to-end acceptance benchmarks, including transfers and honest negative results.

Until this gate passes, the retained CUDA and parallel paths are experimental infrastructure: they must not support speedup claims or CUDA-derived scientific conclusions.

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

**Acceleration status:** the initial backend-neutral foundation landed at `1fab45b`, while the [M1 acceleration audit](m1_acceleration_improvement_plan.md) records unresolved hardening and acceptance work. That work is deferred under the policy above and is not a prerequisite for CPU-reference M2 delivery; the current parallel/CUDA paths are not yet accepted as speedup evidence or scientific evidence.

### Increment B — M2 one-step Haken coarsening

**Scientific question:** At matched one-step compression, can slow-mode coordinates define a contraction with measurable dynamic loss?

| Subtask | Scope and key files | Acceptance criteria |
| --- | --- | --- |
| B1 — embedding and merges | Add a Haken embedding from the selected slow subspace; implement one scalable connectivity-constrained merge and one simple unconstrained comparator. Likely files: [haken_embedding.py](../src/semmap_haken/haken_embedding.py), [coarsen.py](../src/semmap_haken/coarsen.py), [config.py](../src/semmap_haken/config.py), and [tests](../tests/). | No label/relation text enters the partition. Each node belongs to one cluster; connected mode merges only adjacent nodes; target reduction is achieved or the shortfall is recorded; seed and tie-breaking are deterministic. |
| B2 — quotient, lifting, distortion | Build the quotient and reversible node-to-supernode mapping, then calculate lifted fine/coarse slow-subspace and trajectory distortion. Likely files: [quotient.py](../src/semmap_haken/quotient.py), [metrics.py](../src/semmap_haken/metrics.py), [cli.py](../src/semmap_haken/cli.py), and [03_one_step_haken_coarsening.ipynb](../notebooks/03_one_step_haken_coarsening.ipynb). | Quotient weights, membership mapping, masses, and provenance are persisted; lifted comparison is sign/rotation invariant; run emits compression, slow-subspace distance, slow-eigenvalue error, and trajectory error for both merges. |

**Execution contract:** B1 and B2 use CPU float64 by default but accept/pass through [`ComputeContext`](../src/semmap_haken/compute.py:37). Their public contracts remain backend-neutral, sparse, deterministic, batchable by stable task identity, and additive to the M1 artifact schema.

**Deliverable:** [03_one_step_haken_coarsening.ipynb](../notebooks/03_one_step_haken_coarsening.ipynb) and the same `run` CLI with a coarsening config.

**Tests and smoke:** focused tests cover embedding shape and selected-mode provenance, no-label partition input, exact partition coverage, adjacency constraints, deterministic ties, target-reduction shortfall, hand-checkable sparse quotient weights, mass/mapping conservation and reversibility, sign/rotation-invariant lifting, distortion metrics, execution-context pass-through, and artifact-schema stability. End with one synthetic plus prepared-graph one-step notebook-to-CLI smoke using the CPU float64 reference path.

**Documentation:** public contracts and the notebook must state the embedding/weighting rule, merge and tie-break rule, shortfall behavior, quotient aggregation and self-loop policy, lift/projection convention, distortion formulas, provenance fields, and CPU-reference execution status. They must not describe slow modes as established ConceptNet order parameters or imply acceleration acceptance.

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

## Immediate execution — Increment B

Use two sequential Code-specialist calls with one M2 commit boundary after B2. B1 hands its tested public contracts directly to B2 without an intermediate commit. Do not schedule a routine review workstream.

1. **B1 — embedding and merges:** add a Haken embedding from the selected slow subspace; implement one scalable connectivity-constrained merge and one simple unconstrained comparator. Verify no semantic label/relation input, exact node coverage, connected-mode adjacency, deterministic seed/tie behavior, and achieved target reduction or an explicit shortfall. Pass the execution context through and return sparse, backend-neutral contracts plus focused tests for B2.
2. **B2 — quotient, lifting, distortion:** build the quotient and reversible node-to-supernode mapping on B1 without redesigning its contracts; calculate lifted fine/coarse slow-subspace and trajectory distortion; persist quotient weights, masses, provenance, compression and distortion metrics; add [03_one_step_haken_coarsening.ipynb](../notebooks/03_one_step_haken_coarsening.ipynb), the CLI route, required contract documentation, and the single CPU-reference notebook-to-CLI smoke.

Increment B is complete only when the focused M2 tests and integration smoke pass and the artifacts satisfy the execution, provenance, reversibility, and schema requirements above. Then create the single `feat: add one-step haken coarsening` commit, including this plan update; do not commit the plan separately.

## Milestone claim boundary

Before calling Increment E results scientific evidence, perform one targeted scientific audit against [AGENTS.md](../AGENTS.md): verify provenance and replay, sparse-control interpretation, no semantic leakage, matched-compression fairness, null/baseline limitations, uncertainty, and the required Observed/Interpretation separation. Repeat this audit only at publication preparation or when a core hypothesis claim materially changes.
