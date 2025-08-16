from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional
import numpy as np
from .concepts import extract_ner_concepts, extract_statistical_terms, merge_concept_lists, collect_concept_contexts, ConceptCandidate
from .graph_builder import build_cooccurrence_graph
from .graphon import estimate_graphon
from .index import VectorIndex
from .graphex import build_graphex_from_concepts, GraphexRepresentation

@dataclass
class ConceptSearchResult:
    concept: str
    score: float

class ConceptGraphonIndexer:
    def __init__(self, grid_size: int = 64, smooth_sigma: float = 1.0, descriptor_blocks: int = 16, max_concepts: int = 200):
        self.grid_size = grid_size
        self.smooth_sigma = smooth_sigma
        self.descriptor_blocks = descriptor_blocks
        self.max_concepts = max_concepts
        self._concept_vectors: Dict[str, np.ndarray] = {}
        self._index: Optional[VectorIndex] = None

    def fit(self, texts: Iterable[str]):
        texts = list(texts)
        ner = extract_ner_concepts(texts)
        stat = extract_statistical_terms(texts)
        concepts = merge_concept_lists(ner, stat, max_items=self.max_concepts)
        contexts = collect_concept_contexts(texts, concepts)
        vectors: Dict[str, np.ndarray] = {}
        for c in concepts:
            ctx = contexts.get(c.text.lower(), [])
            if not ctx:
                continue
            G = build_cooccurrence_graph(ctx)
            rep = estimate_graphon(G, grid_size=self.grid_size, smooth_sigma=self.smooth_sigma)
            rep.concept = c.text
            vec = rep.descriptor(sample_blocks=self.descriptor_blocks)
            vectors[c.text] = vec
        if not vectors:
            raise RuntimeError("No concept vectors extracted")
        # unify dim
        dims = {v.shape[0] for v in vectors.values()}
        if len(dims) != 1:
            target = max(dims)
            for k,v in vectors.items():
                if v.shape[0] < target:
                    vectors[k] = np.pad(v, (0, target - v.shape[0]))
        dim = next(iter(vectors.values())).shape[0]
        index = VectorIndex(dim)
        for k,v in vectors.items():
            index.add(k, v)
        index.build()
        self._concept_vectors = vectors
        self._index = index
        return self

    def _build_query_vector(self, query_text: str) -> np.ndarray:
        # treat query as its own mini corpus
        concepts = [ConceptCandidate(query_text, 1.0)]
        contexts = {query_text.lower(): [query_text]}
        G = build_cooccurrence_graph(contexts[query_text.lower()])
        rep = estimate_graphon(G, grid_size=self.grid_size, smooth_sigma=self.smooth_sigma)
        rep.concept = query_text
        vec = rep.descriptor(sample_blocks=self.descriptor_blocks)
        dim = next(iter(self._concept_vectors.values())).shape[0]
        if vec.shape[0] < dim:
            vec = np.pad(vec, (0, dim-vec.shape[0]))
        return vec

    def search(self, query_text: str, k: int = 10) -> List[ConceptSearchResult]:
        if not self._index:
            raise RuntimeError("Index not fitted")
        qv = self._build_query_vector(query_text)
        hits = self._index.search(qv, k=k)
        return [ConceptSearchResult(concept=h[0], score=h[1]) for h in hits]

    # ---- Graphex construction ----
    def build_graphex(
        self,
        dim: int = 2,
        grid_size: int = 64,
        similarity_threshold: float = 0.5,
        smoothing_sigma: float = 1.0,
    ) -> GraphexRepresentation:
        """Construct a global graphex over all fitted concepts.

        Returns a GraphexRepresentation capturing:
          - latent coordinates (PCA) of concepts
          - discretized W(x,y) over latent space
          - hubness signal S(x)
          - prominent edge list I
        """
        if not self._concept_vectors:
            raise RuntimeError("Indexer not fitted or no concept vectors available")
        return build_graphex_from_concepts(
            self._concept_vectors,
            dim=dim,
            grid_size=grid_size,
            similarity_threshold=similarity_threshold,
            smoothing_sigma=smoothing_sigma,
        )
