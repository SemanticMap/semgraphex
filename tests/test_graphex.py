from semgraphex import ConceptGraphonIndexer

def test_build_graphex():
    docs = [
        "Neural networks learn representations of data.",
        "Graph neural networks operate on graph structures.",
        "Convolutional networks are a type of neural network.",
        "Transformers use attention mechanisms.",
    ]
    idx = ConceptGraphonIndexer(grid_size=16, descriptor_blocks=4, max_concepts=50)
    idx.fit(docs)
    grx = idx.build_graphex(similarity_threshold=0.2, grid_size=16)
    assert grx.W_grid.shape == (16,16)
    assert len(grx.concepts) > 0
    assert grx.S.shape[0] == len(grx.concepts)
    if grx.I:
        assert all(0 <= w <= 1 for _,_,w in grx.I)
