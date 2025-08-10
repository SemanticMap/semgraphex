"""Basic demonstration of building the concept graphon index and querying it.

Run after installing dependencies and spaCy model:
    python -m spacy download en_core_web_sm
    python examples/demo_basic.py
"""
from semgraphex import ConceptGraphonIndexer

CORPUS = [
    "Graph neural networks (GNNs) are powerful for learning over graph structured data.",
    "Attention mechanisms improved natural language processing.",
    "Transformers leverage self-attention for sequence modeling. The Transformer architecture has enabled large language models.",
    "Spectral graph theory analyzes eigenvalues of matrices derived from graphs.",
]

def main():
    indexer = ConceptGraphonIndexer(grid_size=32, descriptor_blocks=9, max_concepts=100)
    indexer.fit(CORPUS)
    query = "graph neural networks"
    results = indexer.search(query, k=5)
    print(f"Query: {query}\nTop results:")
    for r in results:
        print(f"  {r.concept:30s} score={r.score:.3f}")

if __name__ == '__main__':
    main()
