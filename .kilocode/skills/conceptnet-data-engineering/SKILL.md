---
name: conceptnet-data-engineering
description: This skill should be used when downloading, querying, parsing, validating, filtering, or transforming ConceptNet 5 data; working with ConceptNet assertion dumps, edge JSON, relation URIs, concept URIs, sources, weights, and licenses; or building deterministic sparse ConceptNet graphs for scientific experiments. Apply it to ConceptNet API clients, dump ingestion, relation-specific datasets, multilingual filtering, provenance-preserving graph construction, and ConceptNet data audits. Do not use it as the primary skill for downstream spectral dynamics, graph coarsening, or graphon estimation after the ConceptNet graph contract is already established.
---

# ConceptNet Data Engineering

## Purpose

Build traceable, reproducible ConceptNet datasets without silently changing the meaning of concepts, relations, weights, direction, provenance, or licensing. Prefer the official ConceptNet 5.7 assertion dump for experiments and use the Web API for exploration, smoke tests, and bounded enrichment.

## Load References Selectively

- Read `references/conceptnet-schema.md` before implementing parsers, URI handling, relation filters, direction rules, or API queries.
- Read `references/research-workflow.md` before producing a research dataset, sparse graph, validation report, or reproducibility manifest.
- Re-check the official wiki when behavior may have changed. Treat the bundled references as a project-focused synthesis, not a replacement for authoritative documentation.

## Choose the Data Access Path

1. Use the 5.7.0 gzipped assertion dump for bulk processing, publication-grade experiments, complete scans, and reproducible subsets.
2. Use API lookup when inspecting one known ConceptNet URI.
3. Use API query when retrieving edges constrained by `start`, `end`, `node`, `other`, `rel`, or `sources`.
4. Use `/related` or `/relatedness` only for Numberbatch-based semantic similarity. Do not treat these endpoints as graph adjacency or assertion evidence.
5. Respect API pagination and design sustained traffic below one request per second; the documented limit is 3600 requests/hour with bursts of 120 requests/minute, and relatedness endpoints count double.

## Ingest the Assertion Dump

1. Pin the exact download URL and dataset version.
2. Stream-decompress the gzip file; do not load the complete dump into memory.
3. Split each record into exactly five tab-separated fields:
   - assertion URI;
   - relation URI;
   - start URI;
   - end URI;
   - JSON metadata.
4. Parse JSON explicitly and retain at least `weight`, `dataset`, `sources`, and `license`; retain `surfaceText` when present and useful.
5. Reject or quarantine malformed rows rather than silently truncating them.
6. Compute and persist a cryptographic checksum for the downloaded artifact.
7. Record accepted, rejected, and filtered row counts by reason.

## Preserve URI Semantics

1. Treat the full URI as the canonical node identifier. A concept generally follows `/c/{language}/{term}` with optional part-of-speech and sense segments.
2. Filter language by parsed URI segments, not by substring search.
3. Preserve sense-specific nodes such as `/c/en/example/n`; do not collapse them into `/c/en/example` unless the experiment declares and logs that ablation.
4. Do not automatically merge `FormOf`, `Synonym`, spelling variants, translations, or normalized labels.
5. Keep external linked-data URIs distinct from `/c/` concept nodes unless an explicit integration contract includes them.
6. Treat relation identifiers as canonical `/r/CamelCase` URIs.

## Apply Relation and Direction Semantics

1. Preserve `start` and `end` in the normalized edge table even when building an undirected baseline.
2. Use the relation catalog to identify symmetric relations; never infer symmetry merely because reverse edges appear.
3. For an undirected projection, document how directed assertions are symmetrized and how parallel assertions are aggregated.
4. Keep relation-specific adjacency layers when relation meaning matters.
5. Treat `RelatedTo` as broad and potentially dominant; report its share and run inclusion/exclusion ablations.
6. Treat `FormOf` as morphological evidence, not automatic semantic identity.

## Interpret Weights Conservatively

1. Treat ConceptNet `weight` as positive assertion strength or believability determined by source aggregation and reliability.
2. Do not call it an empirical frequency, probability, transition probability, calibrated confidence, or causal strength.
3. Preserve the raw weight and record every transform separately.
4. Make binary, raw, `log1p`, capped, and relation-normalized transforms configurable.
5. Recompute stochastic normalization from the constructed adjacency when a random-walk operator is needed; never assume raw ConceptNet weights already define one.

## Build a Deterministic Sparse Graph

1. Filter normalized assertion records before assigning node IDs.
2. Build a deterministic node map by sorting canonical URIs, or use another stable rule that is fully documented.
3. Aggregate duplicate projected edges with an explicit policy such as sum; preserve provenance counts and relation histograms outside the sparse matrix.
4. Construct sparse COO data first, coalesce duplicates, then convert to CSR/CSC for computation.
5. Avoid dense `N x N` materialization for medium or full ConceptNet.
6. Preserve an edge table alongside the matrix containing relation, direction, raw weight, transformed weight, dataset, sources, license, and assertion URI.
7. If selecting the largest connected component, record node and edge counts before and after selection.

## Validate the Dataset Contract

Require all of the following before downstream modeling:

- dataset version, source URL, checksum, retrieval time, and data license;
- exact language, relation, weight, source, dataset, and component filters;
- malformed-row and JSON-parse statistics;
- node and edge counts before and after every major filter;
- relation, weight, license, dataset, and source distributions;
- self-loop, duplicate, zero-degree, and connected-component statistics;
- deterministic node-map checksum;
- sparse matrix shape, nonzero count, dtype, symmetry status, and round-trip validation;
- explicit statement that raw weight is not a Markov probability;
- user-visible attribution and share-alike review for redistributed derived data.

## Apply Project-Specific Guardrails

For the Haken-coarsening project:

1. Start with English-only, relation-whitelisted, weight-filtered data and the largest connected component.
2. Retain full ConceptNet URIs as node identities.
3. Keep topology/dynamics-only partitions independent of labels; use labels only for post-hoc interpretation.
4. Preserve source and relation metadata through every contraction so supernodes remain auditable.
5. Produce separate relation-specific datasets before attempting multiplex or supra-adjacency modeling.
6. Keep a sparse-control branch. Do not densify ConceptNet merely to satisfy a graphon method.

## Avoid Common Failure Modes

- Do not parse the `.csv.gz` filename as comma-separated data; the assertion file is tab-separated.
- Do not split or reconstruct assertion URIs when dedicated relation/start/end fields already exist.
- Do not normalize natural-language text and then substitute it for canonical ConceptNet URIs.
- Do not assume all relations are symmetric.
- Do not mix API response objects with dump rows without a normalized internal schema.
- Do not omit provenance or licenses from intermediate products.
- Do not report semantic similarity from Numberbatch as an observed ConceptNet edge.
- Do not claim that ConceptNet facts are uniformly true, unbiased, appropriate, or complete.

## Produce Completion Artifacts

Return or create:

1. a pinned ingestion configuration;
2. a normalized assertion schema;
3. a deterministic node map;
4. a sparse adjacency or relation-layer collection;
5. a provenance-preserving edge table;
6. a validation and filtering report;
7. a reproducibility manifest containing checksum, code revision, environment, and random seed;
8. attribution and license notes for any distributed output.

## Source Authority

Base factual ConceptNet behavior on `commonsense/conceptnet5`, the official repository with approximately 3,000 stars and more than 350 forks at the time this skill was created. Prefer its wiki pages for Downloads, Edges, Relations, URI hierarchy, API, Languages, and Copying and sharing ConceptNet.
