"""Frozen legacy prototype retained for import compatibility during migration.

Optional NLP/indexing dependencies are loaded only when a legacy public symbol is
requested so installing the active package does not require the legacy stack.
"""

__all__ = ["ConceptGraphonIndexer", "ConceptSearchResult", "GraphexRepresentation"]


def __getattr__(name: str):
    if name in {"ConceptGraphonIndexer", "ConceptSearchResult"}:
        from .pipeline import ConceptGraphonIndexer, ConceptSearchResult

        return {"ConceptGraphonIndexer": ConceptGraphonIndexer, "ConceptSearchResult": ConceptSearchResult}[name]
    if name == "GraphexRepresentation":
        from .graphex import GraphexRepresentation

        return GraphexRepresentation
    raise AttributeError(name)
