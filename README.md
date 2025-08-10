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

## Quick Start

(After installing dependencies) see `examples/demo_basic.py` once created.

## Disclaimer

Graphon estimation here is a simplified approximation (histogram/block model smoothing). For rigorous estimation consider methods like Universal Singular Value Thresholding (USVT), neighborhood smoothing, or stochastic block model fitting.
