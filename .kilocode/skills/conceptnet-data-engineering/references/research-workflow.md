# ConceptNet Research Data Workflow

## 1. Define the dataset contract

Specify before reading data:

- pinned ConceptNet version and artifact URL;
- allowed node namespaces;
- language policy for both endpoints;
- relation whitelist/blacklist;
- minimum raw weight;
- source or dataset filters;
- treatment of sense-specific URIs;
- directed, undirected, or relation-layer output;
- duplicate and self-loop policy;
- weight transform;
- component selection;
- node cap and deterministic sampling/ranking rule.

Do not make these choices implicitly in parser code.

## 2. Download reproducibly

Store:

- original artifact filename;
- source URL;
- retrieval timestamp;
- byte size;
- SHA-256 checksum;
- expected ConceptNet version;
- license and attribution note.

Keep the raw file immutable. Write transformed data to a separate interim or processed location.

## 3. Normalize assertions

Use an internal record with fields similar to:

```text
assertion_uri: str
relation_uri: str
start_uri: str
end_uri: str
weight_raw: float
dataset_uri: str | null
license_uri: str | null
sources: structured list
surface_text: str | null
input_line: int
```

Validate:

- exactly five TSV fields in dump rows;
- relation URI begins with `/r/`;
- accepted concept endpoints begin with `/c/` unless the contract permits otherwise;
- metadata is a JSON object;
- weight is finite and positive;
- sources is a list when present.

Quarantine failures with line number and reason. Never silently repair scientific input unless the repair rule is documented and counted.

## 4. Filter in a stable order

Recommended order:

1. row/schema validity;
2. endpoint namespace;
3. endpoint languages;
4. relation policy;
5. source/dataset policy;
6. raw-weight threshold;
7. self-loop policy;
8. optional node-selection policy;
9. connected-component policy.

Count removals at every stage. A stable order makes reports comparable and prevents ambiguous filter accounting.

## 5. Select nodes deterministically

For a capped dataset, avoid taking the first `N` rows from the dump because file ordering can encode source, language, or build artifacts.

Use a declared strategy such as:

- seeded expansion from fixed concept URIs;
- top nodes under a declared weighted-degree rule followed by an induced graph;
- deterministic hash sampling followed by component selection;
- relation-stratified sampling;
- a fixed URI list versioned with the experiment.

Report the selection rule and seed. Compare important results against at least one alternative selection strategy.

## 6. Project to graph form

### Directed graph

Store each assertion contribution at `(start_id, end_id)` and retain relation layers or edge metadata.

### Symmetrized baseline

For every retained assertion contribution from `i` to `j`, add the declared transformed weight to an unordered pair or to both matrix entries. State whether the final adjacency uses:

```text
A[i,j] = A[j,i] = sum(assertion contributions between i and j)
```

or another rule. Do not double-count merely because a symmetric relation is represented by one assertion.

### Relation-aware representation

Build one sparse adjacency per relation before combining layers. If constructing `A = sum(alpha_r * A_r)`, version and report all `alpha_r` values.

### Provenance sidecar

Keep information that a numeric sparse matrix cannot represent:

- assertion URIs;
- relation histogram;
- source and dataset histograms;
- licenses;
- raw and transformed weights;
- directed endpoint orientation;
- aggregation counts.

## 7. Validate graph invariants

Check and report:

- stable URI-to-ID and ID-to-URI bijection;
- matrix shape equals node-map size;
- finite matrix data;
- no unintended negative weights;
- expected symmetry or asymmetry;
- duplicate coalescing;
- self-loop count;
- connected-component distribution;
- isolated node count;
- edge count under a clearly stated convention;
- sparse serialization round trip;
- deterministic checksums on node map and edge data.

## 8. Audit confounds

At minimum compare:

- including versus excluding `RelatedTo`;
- including versus excluding `FormOf`;
- relation-specific graphs;
- directed versus symmetrized graphs;
- binary, raw, and transformed weights;
- all hubs versus hub-capped or hub-reported views;
- source and dataset distributions;
- sense-specific versus explicitly collapsed nodes;
- all components versus largest component.

Do not use text labels to create a topology-only partition. Label and relation summaries are valid after the partition is fixed.

## 9. Produce a data report

Include:

```text
Observed
- Measured counts, distributions, checksums, and validation outcomes.

Interpretation
- What the measurements may imply for graph construction.

Alternative explanations
- Ordering, source, relation, hub, language, and filtering confounds.

Status
- supports / weakly supports / inconclusive / contradicts the relevant data assumption.
```

Never mix observed measurements with interpretation.

## 10. Handoff to downstream graph research

Provide downstream agents with:

- sparse matrix or relation layers;
- immutable node map;
- normalized edge/provenance table;
- data contract and filter report;
- checksum manifest;
- license/attribution note;
- known limitations and failed validations;
- explicit weight semantics;
- exact definition of direction and edge aggregation.

Block downstream claims when provenance is missing, node IDs are unstable, filtering cannot be reproduced, or weights have been mislabeled as probabilities.
