# semgraphex

Prototype implementation of a pipeline to extract concepts from a text corpus, build local co-occurrence graphs, approximate them with simple non-parametric graphon estimators, derive vector descriptors, and index them for similarity search.

This is a research prototype: algorithms are intentionally simple & modular so they can be swapped for more advanced variants.

## Pipeline Overview

1. Preprocess corpus (tokenize, lemmatize, sentence-split, stopword removal)
2. Extract candidate concepts (NER + statistical keyness + optional embedding clustering)
3. For each concept: gather context windows and build weighted co-occurrence graph
4. Estimate a graphon W (piecewise-constant block model smoothing) plus optional S(x) (degree-based signal)
5. Derive descriptor vector (spectral + density + degree stats + sampled W blocks)
6. Index descriptors in FAISS (fallback to Annoy) for approximate similarity search
7. Query: run same pipeline for query text and retrieve nearest concepts.

## Graphex (Global Concept Network)

After fitting concept-level graphons you can aggregate them into a global concept network (graphex):

1. Take all concept descriptor vectors produced during `fit`.
2. Compute pairwise cosine similarities, threshold to form inter-concept edges.
3. Embed concepts into a low-dimensional latent space with PCA (scaled to [0,1]^2).
4. Estimate a discretized W(x,y) over the latent square by binning/smoothing edge weights.
5. Compute S(x) as normalized weighted degree (hubness) of each concept.
6. Collect prominent edges I (those above similarity threshold).

Code:

```python
from semgraphex import ConceptGraphonIndexer
indexer = ConceptGraphonIndexer().fit(corpus)
grx = indexer.build_graphex(similarity_threshold=0.55)
print(grx.top_hubs())
```

`GraphexRepresentation` provides:

- `concepts`: list of concept labels
- `coords`: latent coordinates (n,2) in [0,1]
- `W_grid`: discretized kernel-smoothed estimate of W
- `S`: hubness signal per concept
- `I`: list of (concept_i, concept_j, weight) edges above threshold

This forms a simple semantic map for downstream visualization or clustering.

## Quick Start

(After installing dependencies) see `examples/demo_basic.py` once created.

## Disclaimer

Graphon estimation here is a simplified approximation (histogram/block model smoothing). For rigorous estimation consider methods like Universal Singular Value Thresholding (USVT), neighborhood smoothing, or stochastic block model fitting.
