"""
Graph Construction and Comparison Utilities

Builds NetworkX graphs from LLM extractions and computes
similarity metrics for U score computation.
"""

import torch
import numpy as np
import networkx as nx
from itertools import combinations


def build_graph(extraction: dict) -> nx.DiGraph:
    """Build a directed graph from LLM extraction output."""
    G = nx.DiGraph()

    for node in extraction["nodes"]:
        G.add_node(node.lower().strip())

    for edge in extraction["edges"]:
        src = edge["source"].lower().strip()
        tgt = edge["target"].lower().strip()
        rel = edge.get("relation", "causes")
        G.add_edge(src, tgt, relation=rel)

    return G


def node_jaccard(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """Jaccard similarity between node sets."""
    n1 = set(G1.nodes())
    n2 = set(G2.nodes())
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)


def edge_jaccard(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """Jaccard similarity between edge sets (ignoring relation types)."""
    e1 = set(G1.edges())
    e2 = set(G2.edges())
    if not e1 and not e2:
        return 1.0
    if not e1 or not e2:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)


def graph_similarity(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """Combined node + edge similarity."""
    return 0.5 * node_jaccard(G1, G2) + 0.5 * edge_jaccard(G1, G2)


def extraction_to_tensors(extraction: dict, embedder) -> tuple:
    """
    Convert LLM extraction to PyTorch tensors for the GAT.

    Uses ONLY LLM-extracted edges (no fully connected, no KNN).
    The brain refines existing causal edges, not discovers new ones.

    Returns:
        node_features: (N, 384) from embedder
        edge_index: (2, E) LLM causal edges only
        edge_mask: (E,) all 1.0 (all edges are LLM-flagged)
        node_names: list of node name strings
    """
    nodes = [n.lower().strip() for n in extraction["nodes"]]
    edges = extraction["edges"]

    # Add nodes mentioned in edges but not in node list
    edge_nodes = set()
    for e in edges:
        edge_nodes.add(e["source"].lower().strip())
        edge_nodes.add(e["target"].lower().strip())
    for en in edge_nodes:
        if en not in nodes:
            nodes.append(en)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # Embed nodes
    node_features_np = embedder.embed_nodes(nodes)
    node_features = torch.tensor(node_features_np, dtype=torch.float32)

    # Build edge index from LLM edges ONLY
    src_list = []
    tgt_list = []
    seen = set()
    for e in edges:
        s = e["source"].lower().strip()
        t = e["target"].lower().strip()
        if s in node_to_idx and t in node_to_idx:
            si, ti = node_to_idx[s], node_to_idx[t]
            if si != ti and (si, ti) not in seen:
                src_list.append(si)
                tgt_list.append(ti)
                # Also add reverse for message passing
                src_list.append(ti)
                tgt_list.append(si)
                seen.add((si, ti))
                seen.add((ti, si))

    if not src_list:
        # Fallback: connect first two nodes
        for i in range(min(N, 2)):
            for j in range(min(N, 2)):
                if i != j:
                    src_list.append(i)
                    tgt_list.append(j)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float32)

    return node_features, edge_index, edge_mask, nodes
