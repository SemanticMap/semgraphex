# M1 acceleration audit and improvement plan

Audit target: commit `1fab45b` (`feat: add sparse execution acceleration foundation`, HEAD of `haken`), reviewed against the GPU-first/multicore-fallback delivery policy in [rapid_scientific_delivery_plan.md](rapid_scientific_delivery_plan.md) lines 18–82, the acceleration amendment at line 116, and the Code-specialist-3 scope at line 185. This document is an audit and plan only; no production code was modified.

## 1. Audit scope, method, and environment evidence

Inspected at `1fab45b` with a clean tree (only untracked, user-owned `docs/dialogs*` files, preserved untouched):

- [compute.py](../src/semmap_haken/compute.py) (new), [dynamics.py](../src/semmap_haken/dynamics.py), [modes.py](../src/semmap_haken/modes.py), [operators.py](../src/semmap_haken/operators.py), [config.py](../src/semmap_haken/config.py), [cli.py](../src/semmap_haken/cli.py), [manifest.py](../src/semmap_haken/manifest.py)
- [test_semmap_haken_acceleration.py](../tests/test_semmap_haken_acceleration.py) (new) and the neighboring CLI/notebook/quality-gate suites
- [haken_linear_small.yaml](../configs/haken_linear_small.yaml), [haken_linear_smoke.yaml](../configs/haken_linear_smoke.yaml), [pyproject.toml](../pyproject.toml), [constraints.txt](../requirements/constraints.txt), [notebook 02](../notebooks/02_linear_modes_and_dynamics.ipynb), README acceleration section

Environment probes executed with `.venv/bin/python3` (Python 3.12.3):

| Probe | Result |
| --- | --- |
| `cupy`/`cupyx` | **MISSING** — local venv is CPU-only; every CUDA branch at this commit is locally unexecuted; [test at line 94](../tests/test_semmap_haken_acceleration.py:94) skips |
| `psutil`, `threadpoolctl` | present in venv but **undeclared** in [pyproject.toml](../pyproject.toml:20) |
| `numpy` / `scipy` | **2.5.2 / 1.18.1** — both exceed the ceilings in [constraints.txt](../requirements/constraints.txt:2) (`numpy<2.0`, `scipy<1.14`) and [pyproject.toml](../pyproject.toml:21) |
| Active-package tests at HEAD | `4 failed, 53 passed, 1 skipped` (excluding legacy `semgraphex` collection errors) |
| Same tests at parent `c423a08` (temp worktree) | same failures plus one more (notebook 01) — **failures pre-date `1fab45b`; root cause is environment drift, not this commit** |

Attribution detail: [test_m1_cli_prepare_to_run_persists_checked_spectral_and_dynamics_artifacts](../tests/test_semmap_haken_data_cli.py:61) and the notebook-02 execution test fail with `SpectralConfigurationError: iterative eigsh diagnostics require a graph with at least three nodes`; the gzip-parser and nonfinite-weight tests also fail identically at the parent commit. The acceleration commit is not the regression source, but the red baseline blocks any acceleration acceptance claim (see F1).

## 2. Requirements checklist (plan lines 18–82, 116, 185)

| # | Requirement (source line) | Status at `1fab45b` | Evidence |
| --- | --- | --- | --- |
| R1 | `ComputeContext` with backend/device/dtype/workers/threads/memory budget/seed policy/capability flags/fallback reason (38) | **Partial** — most fields present; capability flags and seed policy absent | [ComputeContext](../src/semmap_haken/compute.py:37) |
| R2 | Adapter surface: CSR transfer, sparse products, partial eigensolve, batched evolution, reductions, host conversion (38) | **Partial** — no adapter layer; CuPy calls inlined in [modes.py](../src/semmap_haken/modes.py:116) and [dynamics.py](../src/semmap_haken/dynamics.py:139) | — |
| R3 | `auto` gates on CuPy load + device + required sparse primitives + VRAM preflight (40) | **Partial** — import/device/VRAM checked; primitive support unchecked; see F9 | [create()](../src/semmap_haken/compute.py:94) |
| R4 | `cuda` strict fails; `cpu` strict never imports CuPy (40) | **Met** — enforced and tested with an import guard | [test at 55](../tests/test_semmap_haken_acceleration.py:55) |
| R5 | Execution config contract YAML (46–60) | **Met** — typed, validated, matches contract field-for-field | [ExecutionConfig](../src/semmap_haken/config.py:74) |
| R6 | CUDA: CSR resident, multi-RHS batches, VRAM-bounded (42) | **Not met** — auto batch = all states in one chunk; budget computed but never consumed; CSR re-transferred per chunk | [dynamics.py:206](../src/semmap_haken/dynamics.py:206), [150](../src/semmap_haken/dynamics.py:150) |
| R7 | CPU: vectorized multi-RHS first, then process chunking; thread caps per worker (42) | **Partial** — multi-RHS `expm_multiply` first is correct; pool path only reachable via manual `batch_size`; caps depend on undeclared `threadpoolctl` | [_cpu_trajectory_chunk()](../src/semmap_haken/dynamics.py:127) |
| R8 | Never copy sparse graph into every worker; shared/mmap CSR (64) | **Not met** — Jacobian pickled per submitted chunk | [dynamics.py:212](../src/semmap_haken/dynamics.py:212) |
| R9 | Never one tiny CUDA job per trajectory; batch and reduce on device (65) | Met for current single-chunk behavior; fragile once batching lands (per-chunk host reduction) | [dynamics.py:153](../src/semmap_haken/dynamics.py:153) |
| R10 | Avoid nested parallelism; explicit BLAS/OpenMP limits (66) | Partial — `threadpool_limits` used when importable, but application is unrecorded and dependency undeclared (F10) |
| R11 | Pre-dispatch memory estimate; reduce batch before failure (67) | **Not met** — only `storage_policy: all` guard exists | [dynamics.py:197](../src/semmap_haken/dynamics.py:197) |
| R12 | CUDA sync only at timing/artifact boundaries; memory pools (68) | Partial — `cp.asnumpy` per time step synchronizes; no pool setup | [dynamics.py:153](../src/semmap_haken/dynamics.py:153) |
| R13 | Seeds from run seed + stable task identity; merge in task-ID order (69) | Met for dynamics (stable chunk index merge); eigensolver start vector unseeded (F8) | [dynamics.py:213](../src/semmap_haken/dynamics.py:213) |
| R14 | Telemetry: counts, GPU/CUDA versions, dtype, batch, peak RAM/VRAM, transfer vs kernel time, throughput, fallback reason (71) | Partial — inventory, dtype, batch, elapsed, throughput, fallback recorded; **no transfer/kernel split, no peak memory** | [telemetry()](../src/semmap_haken/compute.py:119) |
| R15 | Scientific parity: CUDA float64 agrees on eigenvalues, residuals, subspaces, growth decisions, aggregate trajectory errors (73) | **Not demonstrated** — parity test covers eigenvalues + Gram matrix only; no residual/decision/trajectory assertions | [test at 95](../tests/test_semmap_haken_acceleration.py:95) |
| R16 | Acceptance 1–6 (75–82) | 1 unproven (R15), 2 unproven (R7), 3 unmeasured (no benchmark), 4 met, 5 unmet (R6/R11), 6 mostly unmet (F7) | §3 |
| R17 | Amendment: Colab benchmark output without changing M1 schemas (116) | Partial — notebook benchmark cell prints a timestamp only; schemas preserved (additive keys) | notebook 02 diff |
| R18 | Specialist-3 scope: parity tests skip cleanly without CUDA (185) | **Met** — CPU CI unaffected by CuPy absence | [skipif at 94](../tests/test_semmap_haken_acceleration.py:94) |

## 3. Findings

Severity semantics: **Blocker** = acceptance criterion or explicit delivery-policy rule violated; **High** = correctness/robustness/determinism risk; **Optimization** = performance or quality opportunity. Anchors are clickable.

### 3.1 Blockers

- **F1 — Red, environment-drifted baseline blocks all acceptance claims.** The active venv runs `numpy 2.5.2`/`scipy 1.18.1`, beyond both [constraints.txt](../requirements/constraints.txt:2) and [pyproject.toml](../pyproject.toml:21) ceilings; four active tests fail at HEAD and identically at the parent commit, and legacy [test_basic.py](../tests/test_basic.py)/[test_graphex.py](../tests/test_graphex.py) break whole-suite collection (`spacy` missing). Acceptance criterion 1 ("serial CPU, multicore CPU, CUDA outputs agree…") presumes a trusted serial CPU reference; none exists in this environment. Root cause predates `1fab45b`, but the commit's claim of a passing increment boundary is not reproducible here.
- **F2 — `batch_size: auto` is not VRAM-bounded.** [run_linear_dynamics()](../src/semmap_haken/dynamics.py:206) sets `chunk_size = len(states)` under `auto`, so the CUDA path stages the full RHS block regardless of the budget computed at [compute.py:108](../src/semmap_haken/compute.py:108) (`memory_budget_bytes` has no consumer). Acceptance criterion 5 ("automatic batching survives constrained Colab VRAM and records the chosen size") is unimplementable in the current design; a T4-class GPU with a 10k-node graph and many perturbations can OOM with no recovery path. Related: no pre-dispatch memory estimation exists (plan line 67); the only guard is the `storage_policy: all` storage check at [dynamics.py:197](../src/semmap_haken/dynamics.py:197).
- **F3 — Multicore is unreachable under shipped defaults.** The process-pool branch requires `compute.workers > 1 and len(chunks) > 1` ([dynamics.py:211](../src/semmap_haken/dynamics.py:211)), but with the default `workers: auto, batch_size: auto` there is exactly one chunk, so the run is serial. Acceptance criterion 2 (useful speedup for ≥8 independent tasks) cannot be demonstrated without hand-editing `batch_size`; chunking must be derived from the worker count, not only from an explicit batch size.
- **F4 — CPU workers receive a pickled copy of the sparse graph per chunk.** [dynamics.py:212](../src/semmap_haken/dynamics.py:212) submits the CSR Jacobian as a task argument, serializing the full graph through the call queue once per chunk. This directly violates "Never copy the sparse graph into every CPU worker; use read-only inherited memory where safe or shared/memory-mapped CSR buffers under spawn runtimes" (plan line 64), and makes the multicore path memory-proportional to `chunk_count × nnz`.
- **F5 — Trajectory parity between CUDA RK4 and CPU `expm_multiply` is asserted nowhere.** [test_cuda_eigenspace_and_trajectory_parity_when_available](../tests/test_semmap_haken_acceleration.py:95) is named for trajectory parity but stops at eigenvalue and Gram-matrix checks (line 109); residuals, growth/stability decisions, and aggregate trajectory errors required by plan line 73 are unasserted, and no runtime parity harness (CLI or notebook) exists. Acceptance criterion 1 is therefore unproven for dynamics.
- **F6 — Failure states are not explicit or resumable.** A strict-CUDA failure raises out of [ComputeContext.create()](../src/semmap_haken/compute.py:113) before any manifest exists; a worker failure propagates from `future.result()` at [dynamics.py:214](../src/semmap_haken/dynamics.py:214) with no FAILED marker, no partial-shard ledger, and no resume path; [RunManifest.failure_reason](../src/semmap_haken/manifest.py:32) is defined but never written by the `run` route. Only automatic fallback is recorded. Acceptance criterion 6 is largely unmet.

### 3.2 High

- **F7 (relabel of F1's fix prerequisite) — see F1.** *(kept as pointer: environment lock repair is the prerequisite increment INC-0).*
- **F8 — `deterministic: true` is not enforced in the eigensolver.** Neither SciPy nor CuPy `eigsh` receives `v0` ([modes.py:122](../src/semmap_haken/modes.py:122), [125](../src/semmap_haken/modes.py:125)); both use random start vectors, so near-degenerate clusters can rotate run-to-run. The ComputeContext contract requires a deterministic seed policy (plan line 38), and AGENTS.md §32 requires reproducible modes. Derive `v0` from `runtime.random_seed` via a stable task-identity rule.
- **F9 — Over-broad exception handling rebrands defects as environment facts.** [compute.py:112](../src/semmap_haken/compute.py:112) `except (ImportError, ComputeConfigurationError, Exception)` and [modes.py:126](../src/semmap_haken/modes.py:126) `except (ArpackNoConvergence, Exception)` collapse genuine CuPy API/keyword mismatches (e.g., `tol`/`maxiter` semantics differ between SciPy and `cupyx.scipy.sparse.linalg.eigsh`) into "unavailable"/"did not converge" messages, or into silent CPU fallback under `auto`. Catch narrowly (`ImportError`, `AttributeError`, `RuntimeError`, `cupy.cuda.runtime.CUDARuntimeError`), probe required sparse primitives explicitly (plan line 40: "required sparse primitives are supported"), and record a capability-flag map in the context.
- **F10 — Undeclared runtime dependencies with silent degradation.** [threadpoolctl](../src/semmap_haken/dynamics.py:130) and [psutil](../src/semmap_haken/compute.py:29) are optional imports absent from [pyproject.toml](../pyproject.toml:20); on a fresh Colab install the per-worker thread cap silently no-ops (oversubscription) and `physical_cpu_count` becomes `None`, with telemetry never recording whether either limit was actually applied. Declare them (core or an `accel` extra) and record `thread_limits_applied: bool`.
- **F11 — CUDA operator is not resident and syncs per time step.** The CSR Jacobian is rebuilt and re-transferred inside every chunk call ([dynamics.py:150](../src/semmap_haken/dynamics.py:150)), violating "CUDA keeps the CSR operator resident" (plan line 42) — harmless today only because auto batching produces one chunk. `cp.asnumpy` per output time ([153](../src/semmap_haken/dynamics.py:153), [164](../src/semmap_haken/dynamics.py:164)) synchronizes the stream ~T times per chunk instead of at artifact boundaries (plan line 68); keep device-side output buffers and copy once per chunk.
- **F12 — RK4 error control is a magic constant, not a tolerance.** The substep rule `ceil(interval * max(norm_bound, 1.0) / 0.02)` at [dynamics.py:156](../src/semmap_haken/dynamics.py:156) hardcodes a 0.02 scaled-step target that is neither user-configurable nor tied to a declared error tolerance, and no per-run error estimate versus the CPU reference is computed or recorded. The docstring's "accuracy-controlled" claim ([140](../src/semmap_haken/dynamics.py:140)) is currently a norm-based heuristic; see §5 for the required validation protocol.
- **F13 — `workers: auto` uses logical CPUs, policy says physical.** [compute.py:91](../src/semmap_haken/compute.py:91) sizes workers from `os.cpu_count()` (SMT-inclusive) while the policy demands "all safely available **physical** cores" (plan line 22); with `threads_per_worker > 1` this oversubscribes. Prefer `physical or logical` as the auto basis and keep both counts in telemetry (already present).

### 3.3 Optimization opportunities

- **F14 — Telemetry lacks stage split and peak memory.** Add transfer-time vs kernel-time split and peak host/device memory to [aggregate](../src/semmap_haken/dynamics.py:235) and solver dicts (plan line 71); CuPy memory pools (`cp.get_default_memory_pool()`) should be initialized and their usage reported (plan line 68).
- **F15 — Host-memory duplication of trajectories.** [dynamics.py:219](../src/semmap_haken/dynamics.py:219) concatenates all chunk arrays next to the per-chunk lists; under `storage_policy: summaries` metrics could be reduced per chunk and freed, halving peak host memory for large runs.
- **F16 — Notebook benchmark cell is a stub.** The new "benchmark control" cell in [notebook 02](../notebooks/02_linear_modes_and_dynamics.ipynb) prints `perf_counter()` only; the amendment (line 116) expects a Colab benchmark output. Upgrade to a serial-CPU / multicore-CPU / CUDA comparison table (end-to-end including transfers) writing a `benchmark.json` artifact next to the run.
- **F17 — Minor telemetry formatting.** `cuda_version` renders the raw integer runtime version ([compute.py:111](../src/semmap_haken/compute.py:111), e.g., `"12040"`); format as `"12.4"`. `elapsed_seconds` and `throughput` sample the clock twice ([dynamics.py:245](../src/semmap_haken/dynamics.py:245)); compute throughput from the recorded elapsed value.
- **F18 — Mixed-precision residual semantics unlabeled.** Residuals are computed on host against the float64 operator even for CUDA float32 runs ([modes.py:131](../src/semmap_haken/modes.py:131)) — a sound independent check, but the solver dict should state `residual_reference_dtype: float64` so parity reports are unambiguous. Similarly, CPU `eigsh` under `dtype: float32` ([modes.py:125](../src/semmap_haken/modes.py:125)) silently changes diagnostics dtype; record `solver_dtype` explicitly.
- **F19 — No adapter seam for backend growth.** Inlining CuPy in `modes`/`dynamics` (R2) will duplicate transfer/caching logic in Increment C's across-run scheduler; introduce a thin `compute`-owned CSR/eigsh/propagate adapter module before the coarsening increment needs it.

### 3.4 Positive observations (keep and reinforce)

- Strict CPU isolation is genuinely enforced and *tested* with an import guard — [test_cpu_context_never_imports_cupy_and_emits_telemetry](../tests/test_semmap_haken_acceleration.py:55) is exactly the right contract test for plan line 40.
- The execution YAML contract is reproduced faithfully and validated defensively in [ExecutionConfig](../src/semmap_haken/config.py:74)/[load_config()](../src/semmap_haken/config.py:208).
- Solver semantics are honestly labeled end-to-end: `propagation_method` distinguishes `cuda_rk4_sparse_accuracy_controlled` from `scipy_expm_multiply_*` ([dynamics.py:210](../src/semmap_haken/dynamics.py:210)), and the README explicitly disclaims RK4 as "an accuracy-controlled approximation, not an exact CPU-equivalent propagator".
- The parity test's Gram-matrix comparison is the correct sign/rotation-invariant invariant for subspaces.
- CPU ordering follows the policy: vectorized multi-RHS `expm_multiply` first, stable chunk-index merge for determinism ([dynamics.py:213](../src/semmap_haken/dynamics.py:213)).
- Artifact hygiene is strong: atomic staged writes with checksums and full execution telemetry embedded in the manifest ([cli.py:159](../src/semmap_haken/cli.py:159)).

## 4. Improvement plan — small increments

Execution order is strict: **INC-0 → INC-1 → INC-2 → INC-3 → INC-4 → INC-5 → acceptance review**. Increment B (one-step Haken coarsening) must not start until the §6 acceptance review passes. Each increment ends with its tests plus one notebook-02/CLI smoke and a single commit (delivery cadence, plan line 91).

### INC-0 — Restore a green, locked CPU baseline

- **Objective:** make acceptance criterion 1 evaluable by restoring a trusted serial CPU reference environment.
- **Dependencies:** none.
- **Touched files:** [constraints.txt](../requirements/constraints.txt), [pyproject.toml](../pyproject.toml), possibly [conceptnet.py](../src/semmap_haken/conceptnet.py)/[graph_build.py](../src/semmap_haken/graph_build.py) for the two parser-test fixes, `tests/fixtures/conceptnet_tiny.tsv` or the tiny-config graph size, pytest collection config.
- **Steps:** (1) decide the target stack (pin venv to declared `numpy<2.0, scipy<1.14`, or deliberately raise ceilings and re-validate); (2) rebuild `.venv` from the constraints; (3) fix or explicitly xfail-with-reason the four failing tests (tiny-fixture graph must have ≥3 nodes for [modes.py:109](../src/semmap_haken/modes.py:109); gzip/nonfinite parser tests per numpy-2 semantics); (4) gate legacy `semgraphex` tests behind an extras marker so `pytest tests/` collects cleanly; (5) record the resolved versions in run manifests.
- **Tests:** full suite green under the locked environment; a new `test_environment_matches_constraints` asserting installed `numpy`/`scipy` satisfy the declared ceilings.
- **Acceptance:** `pytest tests/ -q` fully green locally; no behavior change in M1 outputs (checksums of a smoke run unchanged versus a pre-drift reference, or any difference explained).
- **Telemetry:** manifest `software` block gains numpy/scipy versions (extend [telemetry()](../src/semmap_haken/compute.py:136)).
- **Rollback/fallback:** pure environment/test change; revert restores prior state with no data impact.
- **Commit:** `fix: restore locked green cpu baseline for acceleration acceptance`.

### INC-1 — Capability flags, narrow failures, deterministic seeding

- **Objective:** make backend selection honest and reproducible (R1, R3; F8, F9, F10, F13, F18).
- **Dependencies:** INC-0.
- **Touched files:** [compute.py](../src/semmap_haken/compute.py), [modes.py](../src/semmap_haken/modes.py), [config.py](../src/semmap_haken/config.py) (optional `execution.physical_core_basis`), [pyproject.toml](../pyproject.toml) (declare `psutil`/`threadpoolctl`), tests.
- **Steps:** (1) add `capabilities: dict[str, bool]` (csr_matvec, csr_matmul, sparse_eigsh, expm_action) and `seed_policy: str` to `ComputeContext`; probe `cupyx.scipy.sparse` primitives at selection time and let missing capabilities disqualify `auto`→CUDA with a recorded reason; (2) replace the `except (..., Exception)` clauses at [compute.py:112](../src/semmap_haken/compute.py:112) and [modes.py:126](../src/semmap_haken/modes.py:126) with narrow tuples and re-raise programming errors; verify CuPy `eigsh` `tol`/`maxiter` signature at probe time rather than trusting kwargs; (3) derive `v0 = rng(seed + stable_task_id)` in `_largest_eigenpairs` for both backends; (4) size `workers: auto` from physical cores when `psutil` reports them; (5) format `cuda_version` as major.minor; record `thread_limits_applied` and `solver_dtype`/`residual_reference_dtype`.
- **Tests:** capability probe unit test with a fake `cupy` module; strict-cuda failure now reports the precise missing primitive; seeded-`v0` reproducibility test (two calls → identical eigenvectors, same process); telemetry field assertions.
- **Acceptance:** `auto` on a CUDA host either selects CUDA with `capabilities` all true or falls back with a machine-checkable reason; `deterministic: true` yields bit-identical repeat solves in-process.
- **Telemetry:** capabilities, seed policy, thread-limit application flag in every artifact.
- **Rollback/fallback:** feature is additive to the context; `backend: cpu` behavior unchanged.
- **Commit:** `feat: add compute capability flags and deterministic eigensolver seeding`.

### INC-2 — Worker-aware CPU chunking, shared CSR, explicit failure states

- **Objective:** satisfy acceptance criteria 2 and 6 on CPU (F3, F4, F6).
- **Dependencies:** INC-1.
- **Touched files:** [dynamics.py](../src/semmap_haken/dynamics.py), [cli.py](../src/semmap_haken/cli.py), [manifest.py](../src/semmap_haken/manifest.py) usage, [compute.py](../src/semmap_haken/compute.py) (chunk planner), tests, notebook 02.
- **Steps:** (1) auto-derive chunk plan: when `workers > 1` and `len(states) >= 8`, target `chunks ≈ min(workers, ceil(states / per_chunk_min))` so the pool actually engages under shipped configs (respecting an explicit `batch_size` when set); (2) stop pickling the graph: write the Jacobian once to a `.npz`-style memmap (or use a fork-inherited module-level global on fork start-methods) and load it in a pool `initializer`; workers receive only `(indices, times, threads)`; (3) write per-chunk shard files keyed by stable chunk ID; on worker failure mark run `FAILED`, persist `failure_reason` + completed shard ledger in the manifest, and support `--resume` that re-executes only missing shards merged in chunk-ID order; (4) strict-CUDA and any fatal `run` error writes a FAILED manifest instead of a bare traceback.
- **Tests:** ≥8-task timing test asserting `pool_wall < serial_wall` on a multicore host (marked slow, skipped in CI-short); inject a raising worker (bad shard) and assert FAILED manifest + ledger; resume produces byte-identical outputs to a fresh run; shared-memory test asserts worker RSS does not grow with chunk count.
- **Acceptance:** multicore speedup demonstrated within RAM budget (criterion 2); forced worker failure and interrupted-then-resumed merge produce explicit states (criterion 6, CPU half).
- **Telemetry:** per-stage wall time, workers, chunk plan, shard status, peak host RSS (via `psutil` or `resource`).
- **Rollback/fallback:** `workers: 1` restores the serial vectorized path exactly; resume machinery is additive.
- **Commit:** `feat: add worker-aware cpu chunking with shared csr and failure states`.

### INC-3 — VRAM-bounded CUDA batching with a resident operator

- **Objective:** satisfy acceptance criterion 5 and plan lines 42/64/67 (F2, F11, F15 partially).
- **Dependencies:** INC-2 (chunk planner shared with CPU).
- **Touched files:** [compute.py](../src/semmap_haken/compute.py) (adapter: CSR upload/keep-alive, memory estimator, pool init), [dynamics.py](../src/semmap_haken/dynamics.py) (`_cuda_trajectory_chunk` refactor), [config.py](../src/semmap_haken/config.py) (optional `execution.cuda_output_buffer_mb`), tests, notebook 02.
- **Steps:** (1) estimate dispatch bytes = CSR(device dtype) + state block + RK stage temporaries (4 k-arrays) + device output buffer, compare against `memory_budget_bytes`, and solve for the largest safe batch (the recorded "chosen size"); (2) hoist CSR upload out of the chunk loop into a context-scoped cache keyed by `(matrix identity, device, dtype)`; (3) allocate one device output tensor `(chunk, T, n)` per batch, run all substeps without host sync, copy once per batch; free temporaries via the CuPy pool between batches; (4) on `cp.cuda.OutOfMemoryError`, halve the batch and retry, recording each reduction; (5) keep per-time host copies only at the artifact boundary.
- **Tests:** CPU-only unit tests of the estimator/planner (pure arithmetic); a fake-GPU integration test injecting a tiny budget asserting ≥2 batches and recorded size; OOM-halving logic test with a stubbed raising matmul.
- **Acceptance:** a constrained-budget run completes with `batch_size` recorded and CSR transferred once (assert via transfer counter); no dense N×N anywhere (criterion 4 stays green).
- **Telemetry:** transfer seconds, kernel seconds, chosen/reduced batch sizes, peak VRAM (`memGetInfo` delta), pool statistics.
- **Rollback/fallback:** estimator upper-bounds conservatively; on repeated OOM below a minimum batch the run fails explicitly with a resumable state (INC-2 machinery) rather than looping.
- **Commit:** `feat: add vram-bounded cuda batching with resident operator`.

### INC-4 — Parity and propagation-error validation (scientific gate)

- **Objective:** satisfy acceptance criterion 1 and plan line 73; make RK4 claims measurable (F5, F12).
- **Dependencies:** INC-3.
- **Touched files:** [dynamics.py](../src/semmap_haken/dynamics.py) (validation block, optional Krylov propagator), [compute.py](../src/semmap_haken/compute.py), [config.py](../src/semmap_haken/config.py) (`execution.cuda_propagator: rk4|krylov`, `execution.propagation_tolerance`), [test_semmap_haken_acceleration.py](../tests/test_semmap_haken_acceleration.py), notebook 02.
- **Steps:** (1) extend the CUDA parity test to assert residual agreement, identical growth/stability sign decisions, and `max |mean_relative_rmse_CPU − mean_relative_rmse_CUDA|` within tolerance; (2) always compute, on a fixed seeded subsample (e.g., first trajectory per perturbation kind), the reference `expm_multiply` solution and record `propagation_validation = {reference_method, max_relative_trajectory_error, per_kind_deltas, rk4_step_target, dtype}` inside the dynamics aggregate — artifacts then carry solver semantics, not just a label; (3) promote the RK4 `0.02` constant to the configured tolerance-linked step target and add step-halving refinement until the subsample error is under tolerance; (4) implement the Lanczos/Krylov `expm(hJ)V` action (small tridiagonal eigendecomposition per interval, multi-RHS batched) as the `krylov` option with a posteriori residual estimate, per §5.
- **Tests:** parity test as above; RK4 validation block present-and-finite test; `krylov` vs CPU `expm_multiply` trajectory agreement `atol` per configured tolerance; float32 ablation test asserting `r` and growth-decision stability or an explicit recorded instability.
- **Acceptance:** criterion 1 demonstrable on a CUDA host (Colab) with recorded numbers; every CUDA dynamics artifact self-describes its solver, tolerance, and measured error.
- **Telemetry:** validation block above plus solver wall times per method for the §5 decision.
- **Rollback/fallback:** `cuda_propagator: rk4` remains the default; `krylov` ships behind the flag until its parity and speed both pass.
- **Commit:** `feat: add cuda propagation validation and krylov option`.

### INC-5 — Telemetry completion and Colab benchmark output

- **Objective:** close R14/R17 (F14, F16, F17) and produce the amendment's benchmark deliverable.
- **Dependencies:** INC-4.
- **Touched files:** [compute.py](../src/semmap_haken/compute.py), [dynamics.py](../src/semmap_haken/dynamics.py), [modes.py](../src/semmap_haken/modes.py), notebook 02, [notebooks/README.md](../notebooks/README.md), README acceleration section.
- **Steps:** (1) add transfer/kernel/wall splits, peak host RSS and peak VRAM, pool stats, and throughput-from-recorded-elapsed to all stage dicts; (2) replace the notebook's stub benchmark cell with a serial/multicore/CUDA comparison on the smoke graph (end-to-end including transfers) that writes `benchmark.json` and renders a table; `auto` may keep tiny graphs on CPU — record that decision; (3) add a Colab install-verification cell (`pip install -e '.[cuda,notebook]'` + capability printout) feeding the notebook benchmark; (4) README updates pointing at the benchmark artifact.
- **Tests:** telemetry-schema test asserting all plan-line-71 keys present in both stage dicts; notebook smoke executes with the benchmark cell and artifact present.
- **Acceptance:** criterion 3 measurable (CUDA end-to-end speedup recorded or the honest negative recorded); every run artifact contains the complete telemetry set.
- **Telemetry:** this increment *is* telemetry; benchmark JSON schema documented in README.
- **Rollback/fallback:** benchmark cell degrades to a printed table if matplotlib is absent; no library behavior change.
- **Commit:** `feat: complete acceleration telemetry and colab benchmark output`.

### Deferred (explicitly not now)

- Across-run scheduler (plan line 44) belongs to Increment C per the delivery plan; INC-2/3 designs (stable task IDs, shard ledger, one-GPU-task-per-device) are chosen so it can attach later without rework.
- CUDA streams/pinned buffers beyond single-copy-per-batch: only after profiling shows useful overlap (plan line 42).

## 5. Scientific recommendation — CUDA full-system linear evolution

**Position: CPU float64 `expm_multiply` remains the reference semantics. The current CUDA fixed-step RK4 is acceptable only as an explicitly labeled approximation with a recorded, per-run validation error; it must not be described or treated as equivalent to `expm_multiply`. Truncated modal propagation is a third, distinct semantics and is not a substitute for either when full-system reconstruction error is the metric.**

Reasoning and required protocol:

1. **Semantics differ in kind.** `expm_multiply` (Al-Mohy–Higham scaling-and-squaring on the action) computes `exp(tJ)·v` with a backward-error-controlled tolerance. The CUDA path integrates `ẋ = Jx` with classical RK4 at a scaled step `h ≈ 0.02/‖J‖∞` ([dynamics.py:156](../src/semmap_haken/dynamics.py:156)): global error is `O(h⁴)` with problem-dependent constants, there is no stiffness safeguard, and error is never estimated against the reference. Equating the two would be a silent scientific-semantics change — exactly what plan line 24 forbids ("The execution backend changes performance, not scientific semantics").
2. **Required convergence/error validation (lands in INC-4).** Every CUDA dynamics run must (a) solve a fixed seeded subsample of trajectories with the CPU float64 reference, (b) record `max relative trajectory error` and per-perturbation-kind aggregate deltas against the tolerance, and (c) refine the RK4 step (step-halving/Richardson or an embedded RK pair) until under tolerance or fail explicitly. The tolerance, step rule, solver order, and measured error become artifact fields (`propagation_validation`), so downstream `r` decisions and trajectory-error metrics can always be traced to solver semantics.
3. **Recommended exactness path on GPU.** Because `J = −αI + βS` with `S` symmetric, the structure favors a Krylov/Lanczos propagator: per time interval, build a small tridiagonal from `m` Lanczos steps on the multi-RHS block, exponentiate the projected matrix, and map back — with an a posteriori residual estimate to grow `m` adaptively. This is the standard expm-action approach, batches across right-hand sides on GPU, and carries a real error estimate (unlike fixed-step RK4). Chebyshev expansion over `σ(J) ⊂ [−α+βλ_min, −α+βλ_max]` is a viable cheaper alternative with coefficient-decay stopping criteria. Ship as `execution.cuda_propagator: rk4|krylov`, default `rk4` until INC-4's parity + speed gate promotes `krylov`.
4. **Truncated modal propagation** (`x(t) ≈ Φ_r e^{Γ_r t} Φ_rᵀ x₀`) is a rank-`r` *model reduction*, not a propagation scheme; it may be used for slow-subspace diagnostics but must never be reported as full-system trajectory error. Artifacts must keep the three semantics (reference expm-action, RK4 approximation, rank-`r` modal truncation) distinguishable by name.
5. **Current honest labeling is good and must be preserved.** The `propagation_method` string and the README's explicit disclaimer are the right pattern; INC-4 extends them from a label to measured evidence.

## 6. Acceleration acceptance review (gate to Increment B)

Before starting Increment B, verify against plan lines 75–82 and record the evidence bundle under `reports/`:

1. Serial vs multicore vs CUDA outputs agree within declared tolerances (INC-4 parity artifact). 
2. Multicore speedup for ≥8 independent tasks within RAM budget (INC-2 timing artifact). 
3. CUDA end-to-end speedup on the smoke/small graph including transfers, or the recorded honest negative (INC-5 benchmark.json). 
4. No dense node-by-node materialization on any backend (grep + test review). 
5. VRAM-constrained auto batching with recorded chosen size and OOM recovery (INC-3 test + artifact). 
6. Forced-CUDA failure, auto fallback, worker failure, interrupted-and-resumed merge each produce explicit failed/resumable states (INC-2 artifacts). 

If any item fails, the failure is a valid empirical output: record it, adjust, and re-run the gate. Increment B (M2 one-step coarsening) starts only after this checklist is green — consistent with the task's instruction that coarsening work waits for acceleration acceptance.
