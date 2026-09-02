# Iteration 1 Workstream 5A Review

**Review target:** commits `6c686ff`, `8e1f75b`, `284719c`, and `63ae73c`, as incorporated by `d4bb30b`.

**Scope:** the active [`semmap_haken`](../src/semmap_haken/) M0 preparation path only. The legacy [`semgraphex`](../semgraphex/) package was treated as deliberately isolated compatibility code, not as scientific evidence.

## Executive verdict

**Request changes — Iteration 1 must not close before remediation.**

The notebook-first/library-backed/CLI-replay architecture is a sound foundation. In particular, the thin-notebook boundary, deterministic URI ordering, offline defaults, atomic HTTP download promotion, and hermetic equivalence gate are all strong decisions. However, the current `prepare` workflow does not bind a prepared graph to the exact input bytes or retain the selected graph's provenance. In addition, it silently retains self-loops despite the declared default policy, and it accepts non-finite weights. These issues can make later M1 spectral results irreproducible or scientifically ambiguous.

This verdict does **not** reject the M0 approach or the documented M7 streaming limitation. It blocks closure until the correctness and provenance gaps below are fixed and tested.

## Findings

| ID | Priority | Finding | Required remediation acceptance criteria |
| --- | --- | --- | --- |
| R1 | **Blocker** | Prepared runs do not record or verify the checksum of the input dump. | `prepare` records the SHA-256, byte size, declared dataset/version/source, and acquisition/cache status for the exact file parsed; the manifest and graph metadata bind that dataset identity to every adjacency artifact. Replay rejects a changed input at the recorded path unless an explicit new run is created. Add tests for manual/offline and downloaded inputs, including a same-path content mutation. |
| R2 | **Blocker** | The graph artifact irreversibly drops assertion provenance and misreports relation composition after graph selection. | Preserve a deterministic, auditable selected-edge/provenance representation or sufficient lossless aggregation keyed by directed endpoints and relation/dataset/source/license. The reported relation histogram must describe the retained induced graph, not all pre-selection assertions. Add an LCC/`max_nodes` fixture proving selected provenance and counts are exact. |
| R3 | **High** | The advertised `exclude` self-loop policy is not implemented; self-loops are retained and undirected edge counts are wrong when they occur. | Make the self-loop policy explicit in config and metadata, implement the approved default, and calculate edge counts correctly for both directed and undirected graphs. Add fixtures with self-loops and assert adjacency, count, policy, and replay behavior. |
| R4 | **High** | Invalid numeric weights can enter the adjacency. | Reject non-finite (`NaN`, `Infinity`, `-Infinity`) values, and preferably boolean values, before filtering/transformation. In `skip_invalid` mode increment a specific counter; in fail-fast mode report the line. Cover plain and gzip inputs. |
| R5 | **High** | Cache verification is optional by default, so a cache path labelled `unverified` may permanently reuse arbitrary bytes for the declared version. | For reportable ConceptNet acquisition, require an expected SHA-256 and size from a controlled descriptor/lock, key cache storage by the observed digest only after verification, and record the official source URL plus validation result. Manual files may remain supported, but must be explicitly labelled unverified and must not satisfy a verified ConceptNet replay claim. |
| R6 | **High** | The declared supported dependency range permits an environment where the M0 suite cannot collect. | Add a tested dependency lock/constraints path (and, if necessary, compatible upper bounds) for the declared Python versions. CI must install it from a clean environment and run the M0 quality gates. The package installation and test command documented in [`README.md`](../README.md:7) must reproduce that environment. |
| R7 | **Medium** | Prepared-artifact and manifest writes are individually direct, not run-directory atomic. | Stage files in a temporary sibling directory, fsync/validate checksums, write the manifest last, then atomically promote the completed run directory (or mark failed runs unambiguously). Add fault-injection tests proving no completed-looking partial run is discoverable. |
| R8 | **Medium** | The run manifest leaves reproducibility fields unpopulated despite claiming replayability. | Populate `git_commit`, configured seed(s), and a resolved environment/lock identifier; define whether absolute local paths are informational only. Tests must assert these fields for CLI and notebook runs and preserve the existing scientific-artifact equivalence rule. |
| R9 | **Low** | Runtime config accepts arbitrary `profile` labels and does not validate the referenced resource profile during CLI preparation. | Validate profile semantics and resource-profile structure centrally, or document that the profile is notebook advisory only and omit it from replay guarantees. Add invalid-profile tests if validation is selected. |

### R1 — Input identity is absent from a reportable run

The `prepare` path opens `dataset.path` directly and persists only its string path in metadata ([`prepare()`](../src/semmap_haken/cli.py:78), [`metadata`](../src/semmap_haken/cli.py:87)). It then computes checksums only for output artifacts ([`checksums`](../src/semmap_haken/cli.py:88)). Neither [`RunManifest`](../src/semmap_haken/manifest.py:16) nor [`save_prepared_graph()`](../src/semmap_haken/graph_build.py:79) receives a source-dump digest. A path is not stable provenance: the bytes at that path can change without changing the config or manifest.

**Scientific impact:** M1 mode spectra and any later cross-scale comparison could be attributed to a source that cannot be reconstructed. This invalidates the current strong manifest/replay claim for reportable work.

### R2 — Parser-level provenance does not survive to the selected graph

[`Assertion`](../src/semmap_haken/conceptnet.py:18) correctly captures full assertion URI, directed endpoints, dataset, sources, and license. The builder then aggregates only endpoint weights ([`build_sparse_graph()`](../src/semmap_haken/graph_build.py:49)) and retains only a global relation histogram counted before component and node selection ([`relation_counts`](../src/semmap_haken/graph_build.py:56), [`selection`](../src/semmap_haken/graph_build.py:66)). No output stores source/dataset/license or an edge-level relation aggregation.

**Scientific impact:** later source/dataset-effect ablations, relation-specific analysis, and provenance audits cannot determine what produced the actual adjacency. The full-URI and directed parser correctness is valuable, but it is not enough if the prepared artifact loses the evidence before M1.

### R3 — Self-loop contract and graph statistics disagree with implementation

[`GraphArtifactMetadata`](../src/semmap_haken/artifacts.py:34) declares `self_loop_policy="exclude"` by default, yet the graph builder inserts `(start, end)` even when `start == end` ([`edge accumulation`](../src/semmap_haken/graph_build.py:57)). In undirected mode it only mirrors non-self edges ([`mirror guard`](../src/semmap_haken/graph_build.py:61)), while the report divides every nonzero by two ([`edge_count`](../src/semmap_haken/graph_build.py:67)).

**Scientific impact:** unreported loops alter degree-normalized operators and therefore M1 eigenmodes. The stated edge count is also false whenever loops occur.

### R4 — Non-finite weights bypass the configured minimum

The parser accepts any Python `int` or `float` ([`numeric check`](../src/semmap_haken/conceptnet.py:82)) and coerces it to `float` ([`construction`](../src/semmap_haken/conceptnet.py:84)). A JSON `NaN` accepted by Python's decoder is not less than `min_weight`, so it reaches [`math.log1p()`](../src/semmap_haken/graph_build.py:32) and the CSR payload. This is an input-validity defect rather than a normal malformed-line decision.

**Scientific impact:** a single non-finite value can contaminate normalization and spectral calculations without a clear parser failure.

### R5 — Download integrity is only conditional

The downloader's restart/cleanup and atomic promotion are good ([`acquire_dataset()`](../src/semmap_haken/data_manager.py:69)). But the default descriptor provides no expected hash or size ([`DatasetDescriptor`](../src/semmap_haken/data_manager.py:20)); cache storage therefore uses the permanent `unverified` key ([`versioned_cache_path()`](../src/semmap_haken/data_manager.py:48)), while verification only compares expectations when they are non-null ([`_verify()`](../src/semmap_haken/data_manager.py:53)).

This is a documented implementation gap, not an allegation about the official endpoint. It means a network/proxy/cache failure that returns syntactically valid bytes can become a cache hit and still be called a ConceptNet versioned dataset.

### R6 — The executable evidence cannot run in the reviewed environment

The package only specifies lower bounds for NumPy and SciPy ([`dependencies`](../pyproject.toml:20)). In the review environment, NumPy `2.4.4` loaded ahead of a system SciPy built for NumPy 1.x, causing `numpy.core.multiarray failed to import` during test collection. This is an environment compatibility failure, not a product-code functional failure, but it prevents execution of the required M0 gates and needs a reproducible supported environment.

### R7–R9 — Reliability and contract hardening

Output files are written directly in succession ([`save_prepared_graph()`](../src/semmap_haken/graph_build.py:79)); [`RunManifest.write_json()`](../src/semmap_haken/manifest.py:62) is likewise a direct write. The unique run ID lowers collision risk but does not make failure atomic. Also, [`RunManifest.create()`](../src/semmap_haken/manifest.py:36) supports commit and seed provenance but the CLI does not populate them ([`manifest creation`](../src/semmap_haken/cli.py:93)). Finally, [`RuntimeConfig`](../src/semmap_haken/config.py:63) accepts a free-form profile and optional profile path without centrally validating the resource document.

## Positive observations

- The intended architecture is well separated: notebooks are thin interfaces over shared package/CLI logic ([`notebook contract`](../notebooks/README.md:1)) and the executable quality gate compares nodes, CSR arrays, scientific config, and checksums ([`equivalence test`](../tests/test_semmap_haken_quality_gates.py:79)).
- URI ordering and LCC tie-breaking are deterministic ([`_select_nodes()`](../src/semmap_haken/graph_build.py:37)); the README honestly identifies URI-prefix truncation as non-semantic sampling ([`selection caveat`](../README.md:57)).
- The parser preserves full ConceptNet URIs and direction at ingestion ([`Assertion`](../src/semmap_haken/conceptnet.py:18)), supports explicit language/relation filtering ([`filtering`](../src/semmap_haken/conceptnet.py:90)), and explicitly avoids interpreting weight as probability ([`stream_assertions()`](../src/semmap_haken/conceptnet.py:67)).
- HTTP acquisition correctly uses a partial file, re-verifies bytes, cleans interrupted transfers, and atomically replaces the destination ([`download flow`](../src/semmap_haken/data_manager.py:96)).
- Colab behavior is appropriately conservative: production download and Drive are opt-in, the core does not import Colab, and persistence uses atomic copies ([`notebook helpers`](../src/semmap_haken/notebook.py:1), [`persist_paths()`](../src/semmap_haken/notebook.py:156)).
- Legacy dependencies are isolated from the active package extras ([`legacy isolation`](../pyproject.toml:28)), which reduces accidental coupling with the old prototype.

## Accepted risks and explicitly deferred work

These are **not findings blocking M0 closure** when clearly retained as limitations:

1. [`build_sparse_graph()`](../src/semmap_haken/graph_build.py:52) materializes accepted assertions and edge aggregates. This contradicts a future medium-scale streaming goal, but it is explicitly documented as the M0 memory cost ([`README limitation`](../README.md:57)). M7 must replace this with bounded-memory external aggregation/partitioning and benchmark peak RAM.
2. `capped` and `relation_normalized` are accepted config literals but intentionally not implemented in the M0 builder ([`transform_weight()`](../src/semmap_haken/graph_build.py:27)). They must fail clearly (as they do), or be removed from the M0 schema until the relevant workstream implements them.
3. LCC plus lexicographic `max_nodes` sampling is deterministic but biased. It is acceptable for an M0 smoke/small baseline only if every later scientific report carries the policy, evaluates its sensitivity, and never presents it as representative semantic sampling.
4. Graphon estimation, multiplex relations, nonlinear dynamics, M1 slow-mode selection, and M7 resource scaling are future work; their absence is consistent with the approved staged roadmap.

## Checks run

1. Reviewed the requested commits and confirmed the review baseline at `d4bb30b`; the uncommitted roadmap edit was not modified.
2. Read the approved roadmap, quality-gate evidence, active package, configs, notebook helpers/notebooks, and focused tests.
3. Ran `git diff --check`: no whitespace errors were reported.
4. Ran `python -m compileall -q src/semmap_haken`: completed successfully.
5. Attempted targeted M0 tests, including the quality gates and CLI/data tests. Collection was blocked by the NumPy/SciPy ABI mismatch documented in R6. This is recorded as a verification limitation, not treated as passing evidence.

## Closure decision

**Do not close Iteration 1 before R1–R6 are remediated and their acceptance tests pass in a clean, locked environment.** R7–R9 should be scheduled before publication-grade M1 work; R7 and R8 are strongly recommended for the same remediation change because they affect the credibility of the replay contract.

After remediation, re-run the complete quality-gate suite plus new adversarial fixtures for non-finite values, loops, source mutation, cache verification, selected-edge provenance, and interrupted artifact persistence. The M0 source-of-truth then becomes the updated manifest/provenance contract and passing clean-environment evidence—not this review document.

## Workstream 5B remediation appendix

This appendix records implementation evidence only; it does not alter the review verdict above.

| Finding | Remediation | Acceptance coverage |
| --- | --- | --- |
| R1 | Exact input SHA-256, byte size, declared source identity, and verification status are persisted in metadata and manifest resumability. | `tests/test_semmap_haken_quality_gates.py`, `tests/test_semmap_haken_remediation.py` |
| R2 | Deterministic post-selection `selected_edges.jsonl` preserves directed endpoints, relation, contribution, dataset/source/license fields; metadata references its checksum. | `tests/test_semmap_haken_remediation.py` |
| R3 | Configurable `exclude`/`include` loop policy is enforced; reports expose simple-edge count, adjacency NNZ, and loop count. | `tests/test_semmap_haken_remediation.py` |
| R4 | Parser rejects boolean and non-finite weights with an `invalid_weight` counter in skip mode. | `tests/test_semmap_haken_remediation.py` |
| R5 | Unpinned official acquisition/cache reuse is rejected; verified status requires a matching expected digest. | `tests/test_semmap_haken_remediation.py` |
| R6 | NumPy/SciPy bounds and documented tested constraints are in `pyproject.toml` and `requirements/constraints.txt`. | clean environment instructions in `README.md` |

Artifact publication stages a sibling directory, validates files, writes `COMPLETED`, and atomically promotes it while retaining a previous valid directory until promotion succeeds. Loader validation rejects incomplete sets and checksum mismatches. Manifest creation records git revision, host executable, runtime seed, source identity, and the constraints reference; resource profiles are parsed and checked against runtime profile and `dataset.max_nodes`.
