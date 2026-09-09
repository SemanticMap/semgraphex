# Start experiment implementation checklist

This document defines the concrete remaining work required to move from the current one-step Haken prototype to the first falsifiable multiscale computation.

## Scope implemented in this increment

### 1. Align ConceptNet preparation and run semantics

- Use the same language filter (`en`) in `prepare`, one-step, and multiscale configs.
- Use the same relation whitelist: `RelatedTo`, `IsA`, `PartOf`, `HasA`, `UsedFor`, `HasProperty`.
- Keep the same `min_weight`, `max_nodes`, component policy, directionality, loop policy, and weight transform.
- Reject a prepared graph when its persisted `resolved_config.json` differs from the run config on any adjacency-defining field.
- Ignore only local source path differences; input identity is controlled by hash/size instead.

### 2. Bind every prepared graph to exact ConceptNet input bytes

- Compute SHA-256 and byte size of the decompressed assertions file before parsing.
- Persist them in `source_identity` inside graph metadata and the run manifest.
- Treat prepared artifacts without this fingerprint as pre-contract artifacts that must be regenerated.
- Keep output artifact checksums in addition to input identity.

### 3. Remove manual prepared-graph path editing

- `run` accepts `--prepared-graph` and overrides the YAML placeholder.
- The resolved run config records the actual selected prepared graph path.
- The prepared graph contract is validated before spectrum or dynamics are calculated.

### 4. Mandatory one-step start-smoke gate

The initial computational path is:

`prepare -> run(one-step) -> evaluate`.

`evaluate` requires:

- `COMPLETED` run marker;
- completed spectral, dynamics, and coarsening stages;
- both `connectivity_matching` and `unconstrained_matching` M2 outputs;
- a nontrivial contraction for each method;
- at least one cross-scale distortion measurement for each method.

A small ConceptNet run must pass this gate before a 10k run is treated as usable evidence.

### 6. Recursive M3 hierarchy

- Reapply the tested M2 contraction recursively.
- Recompute normalized operator, candidate slow modes, perturbation dynamics, and Haken embedding at every scale.
- Use level-specific deterministic perturbation/coarsening seeds derived from the configured base seeds.
- Persist each sparse adjacency matrix and the mapping from every supernode back to original node IDs.
- Record per-level node count, selected `r`, eigengap diagnostic, beta/stability information, and rank-r reconstruction error.
- Record per-transition compression, slow-subspace distance, slow-eigenvalue error, and lifted trajectory error.
- Stop on `max_levels`, `min_nodes`, inability to compute a valid spectral state, or zero feasible merges.

Current first implementation intentionally does not yet provide interruption/resume checkpoints inside one hierarchy. This remains a follow-up hardening item before long medium-scale runs.

### 7. M4 plateau detector

A transition qualifies only when all configured gates pass:

- `|r_s - r_{s+1}| <= r_tolerance`;
- achieved reduction is above the configured minimum;
- slow-subspace projection distance is below threshold;
- trajectory distortion is below threshold;
- slow-eigenvalue error is below threshold.

A plateau candidate is a run of at least `min_consecutive` qualifying transitions. The report records start/end scale, total compression, `r` values, and worst accepted distortions.

Thresholds are experimental parameters, not scientific constants. Sensitivity analysis remains mandatory before a claim.

### 8. Synthetic positive/negative controls

Two sparse controls are implemented:

- `hierarchical_positive`: strong within-block ring/chord connectivity with weak bridges between macro-blocks; intended to contain explicit timescale separation;
- `homogeneous_negative`: matched-size homogeneous circulant graph with no planted block hierarchy.

Both controls pass through exactly the same hierarchy and plateau detector. The positive condition is not hard-coded into the detector; failure to detect a plateau is a valid negative result.

## Commands

### ConceptNet start smoke

```bash
python -m semmap_haken prepare --config configs/conceptnet_en_smoke.yaml
python -m semmap_haken run --config configs/haken_one_step_smoke.yaml --prepared-graph <prepare-run-dir>
python -m semmap_haken evaluate --run <run-dir>
```

### ConceptNet multiscale

```bash
python -m semmap_haken.research_cli multiscale \
  --config configs/haken_multiscale_small.yaml \
  --prepared-graph <prepare-run-dir>
```

### Synthetic falsification smoke

```bash
python -m semmap_haken.research_cli synthetic \
  --config configs/synthetic_plateau_smoke.yaml
```

## Remaining items before first ConceptNet scientific claim

These are deliberately not implemented in the present start-experiment increment:

1. Run the smoke and 10k experiments in a locked clean environment and archive actual numerical artifacts.
2. Add resumable/checkpointed hierarchy execution before medium-scale jobs.
3. Add bootstrap stability for selected slow-subspace dimension `r`.
4. Add the first matched-compression baselines: random matching, heavy-edge matching, spectral coarsening.
5. Add the first nulls: degree-preserving rewiring and weight shuffle.
6. Run threshold sensitivity for plateau detection; a plateau that exists only at one hand-selected threshold is not evidence.
7. Run at least several seeds/perturbation ledgers on synthetic positive and negative controls.
8. Require positive-control sensitivity and negative-control specificity before interpreting ConceptNet plateaus.
9. Add post-hoc semantic cluster interpretation only after topology-only partitions are frozen.
10. Perform claim-boundary review separating Observed / Interpretation / Alternative explanations / Status.

## Deferred beyond the start experiment

- nonlinear Haken dynamics and explicit slaving regression `b = h(a)`;
- graphon family modeling `W(x,y; a_1,...,a_r)`;
- directed and multiplex ConceptNet;
- relation-specific ablations;
- graphex/sparse-limit comparison;
- medium/full ConceptNet scaling and accepted CUDA acceleration.
