# Wishart graph dictionary experiment

This branch turns recursive Wishart coarsening into a graph-dictionary
experiment.

The core separation is:

```text
exact GraphType       = WL bucket + exact VF2 identity
Wishart GraphFamily   = density-based family of distinct exact GraphTypes
Huffman code          = frequency code for accepted/reusable GraphTypes
MDL selection         = decision rule for contraction
recursive grammar     = GraphTypes whose nodes include older GraphTypes
```

## Pipeline

At level `s`:

```text
G_s
  -> sampled ego discovery
  -> persistent exact GraphTypes
  -> batched full-graph scan for frequencies
  -> Wishart on unique GraphType representatives
  -> persistent Wishart families
  -> canonical Huffman lengths
  -> structural MDL gain per occurrence
  -> non-overlapping MDL set packing
  -> symbolic quotient G_(s+1)
```

The discovery sample and the frequency scan are intentionally distinct. A
sample can open candidate types cheaply, but Huffman/MDL must be based on a
frequency estimate that is not merely the discovery sample. The default
dictionary experiment therefore scans every current node in bounded batches.

## Exact type identity

`GraphDictionary` uses a fast permutation-invariant WL fingerprint only as a
bucket. Equality is then verified by exact VF2 isomorphism over:

- directed internal topology;
- ConceptNet relation labels;
- recursive child-symbol labels;
- optional typed boundary signature.

WL hashing is therefore not treated as a proof of graph isomorphism.

## Boundary signature

A candidate stores typed incoming/outgoing edge counts crossing the candidate
boundary. With `dictionary.boundary_sensitive: true`, two internally
isomorphic motifs with different external interfaces become different exact
GraphTypes.

This is deliberately stricter than topology-only motif discovery and is an
ablation target.

## Recursive symbols

Every contracted supernode stores its `dictionary_type_id`. Candidate
extraction on the next level includes those type IDs as node labels. Therefore
a new GraphType may have:

```json
{
  "type_id": "GT_000431",
  "children": ["GT_000017", "GT_000024"]
}
```

The global `dictionary/grammar.jsonl` is the derivation DAG.

## Wishart families

Wishart no longer defines exact symbol identity. It clusters unique exact types.
Occurrence frequency can be used as density mass.

Level-local modes are matched across scales with Jaccard overlap of their
member GraphType IDs and receive persistent `WF_XXXXXX` identifiers.

## MDL objective

The current implementation uses an explicit structural MDL proxy:

- raw cost: encode every internal directed edge using two global node
  references plus a relation code;
- dictionary-coded cost: encode occurrence node placement plus a GraphType
  Huffman code and amortized prototype cost;
- boundary edges are omitted from both sides because they remain represented by
  the quotient/residual graph.

This proxy is used for ranking and stopping. It is not claimed to equal the byte
size of a finalized lossless binary codec. `hierarchy.json` and
`dictionary_metrics.json` label these fields with `_proxy`.

## New artifacts

Global:

```text
dictionary/
  graph_types.jsonl
  families.jsonl
  grammar.jsonl
  huffman.json
  statistics.json
```

Per level:

```text
level_XXX/
  symbolic_nodes.jsonl
```

Per transition:

```text
transition_XXX_YYY/
  wishart.json
  type_labels.npz
  dictionary_metrics.json
  figure_occurrences.jsonl
  fine_to_coarse.npy
  cluster_dynamics.jsonl
```

## Primary metrics

The experiment now tracks:

- dictionary size;
- supported/new/reused type counts;
- dictionary novelty;
- reuse rate;
- recursive type count;
- type entropy;
- mean Huffman code length;
- raw/encoded structural MDL proxy;
- MDL gain and ratio;
- covered nodes;
- ordinary node compression;
- the existing dynamic-distortion diagnostics.

A successful dictionary regime should show decreasing novelty and increasing
reuse/recursive composition while structural MDL continues to improve.

## Run

```bash
semmap-wishart \
  --config configs/wishart_conceptnet_dictionary.yaml \
  --prepared <prepared-graph> \
  --output tmp/runs/wishart-dictionary-100k
```

The existing `feature/wishart-recursive-coarsening` branch remains the
contraction-only baseline.
