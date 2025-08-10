from typing import List, Dict
from collections import Counter, defaultdict
import itertools
import networkx as nx
from .preprocess import tokenize


def build_cooccurrence_graph(context_blocks: List[str], window: int = 5, min_weight: int = 2) -> nx.Graph:
    G = nx.Graph()
    for block in context_blocks:
        toks = tokenize(block)
        for i in range(len(toks)):
            w = toks[i]
            if not G.has_node(w):
                G.add_node(w)
            # window-based pairs
            for j in range(i+1, min(len(toks), i+window)):
                u = toks[j]
                if w == u:
                    continue
                if G.has_edge(w, u):
                    G[w][u]['weight'] += 1
                else:
                    G.add_edge(w, u, weight=1)
    # prune
    to_remove = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < min_weight]
    G.remove_edges_from(to_remove)
    # remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G
