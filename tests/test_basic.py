from semgraphex import ConceptGraphonIndexer

def test_index_small_corpus():
    docs = [
        "Apple releases new iPhone in California event.",
        "Microsoft announces partnership with OpenAI in Seattle.",
        "OpenAI develops advanced models for natural language understanding.",
    ]
    indexer = ConceptGraphonIndexer(grid_size=16, descriptor_blocks=4, max_concepts=50)
    indexer.fit(docs)
    results = indexer.search("OpenAI models")
    assert len(results) > 0
    assert any('OpenAI' in r.concept for r in results)
