"""
Understanding Score (U) — Graph Embedding Based

U = 0.4 * U_s + 0.4 * U_c + 0.2 * U_n

Computed via cosine similarity on graph-level embeddings, not edge weights.
The brain produces a single embedding per input graph. Similar inputs
(paraphrases) should produce similar embeddings. Contradictions should diverge.
"""

import torch
import torch.nn.functional as F
import networkx as nx
from itertools import combinations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scoring.graph_utils import build_graph, graph_similarity, extraction_to_tensors


def _cosine_sim_vec(v1, v2):
    """Differentiable cosine similarity between two vectors. Returns scalar in [-1, 1]."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze()


def _get_graph_embedding(brain, extraction, embedder, memory):
    """Run brain, return (final_embedding, alpha) — both with gradients."""
    nf, ei, em, nn_ = extraction_to_tensors(extraction, embedder)
    final_emb, alpha, _, _ = brain(nf, ei, em, memory)
    return final_emb, alpha


def _run_variants(brain, variant_list, embedder, memory):
    """Get graph embeddings for a list of extractions."""
    results = []
    for ext in variant_list:
        if "error" in ext:
            continue
        emb, alpha = _get_graph_embedding(brain, ext, embedder, memory)
        results.append((emb, alpha))
    return results


def compute_u_differentiable(brain, obj_data, embedder, memory):
    """
    Compute fully differentiable U on graph-level embeddings.
    """
    base = _run_variants(brain, obj_data.get("base", []), embedder, memory)
    para1 = _run_variants(brain, obj_data.get("paraphrase_1", []), embedder, memory)
    para2 = _run_variants(brain, obj_data.get("paraphrase_2", []), embedder, memory)
    swap1 = _run_variants(brain, obj_data.get("attribute_swap_1", []), embedder, memory)
    swap2 = _run_variants(brain, obj_data.get("attribute_swap_2", []), embedder, memory)
    contra1 = _run_variants(brain, obj_data.get("contradiction_1", []), embedder, memory)
    contra2 = _run_variants(brain, obj_data.get("contradiction_2", []), embedder, memory)

    # Collect alphas
    all_alphas = [a for results in [base, para1, para2, swap1, swap2, contra1, contra2]
                  for _, a in results]
    mean_alpha = torch.stack(all_alphas).mean() if all_alphas else torch.tensor(0.5)

    # ── U_s: consistency across base + paraphrases ──
    all_bp = base + para1 + para2
    if len(all_bp) >= 2:
        sims = []
        for i in range(len(all_bp)):
            for j in range(i + 1, len(all_bp)):
                sim = _cosine_sim_vec(all_bp[i][0], all_bp[j][0])
                sims.append((sim + 1) / 2)  # map [-1,1] to [0,1]
        u_s = torch.stack(sims).mean()
    else:
        u_s = torch.tensor(0.0)

    # ── U_c: causal invariance (base vs swaps) ──
    all_sw = swap1 + swap2
    if base and all_sw:
        sims = []
        for be, _ in base:
            for se, _ in all_sw:
                sim = _cosine_sim_vec(be, se)
                sims.append((sim + 1) / 2)
        u_c = torch.stack(sims).mean()
    else:
        u_c = torch.tensor(0.0)

    # ── U_n: contradiction detection ──
    all_co = contra1 + contra2
    if base and all_co:
        sims = []
        for be, _ in base:
            for ce, _ in all_co:
                sim = _cosine_sim_vec(be, ce)
                sims.append((sim + 1) / 2)
        u_n = 1.0 - torch.stack(sims).mean()
    else:
        u_n = torch.tensor(0.0)

    u_total = 0.4 * u_s + 0.4 * u_c + 0.2 * u_n

    return {"U_s": u_s, "U_c": u_c, "U_n": u_n, "U_total": u_total,
            "alpha": mean_alpha}


def compute_u_for_object(brain, obj_data, embedder, memory):
    """Non-differentiable eval wrapper."""
    with torch.no_grad():
        result = compute_u_differentiable(brain, obj_data, embedder, memory)
    return {k: v.item() if torch.is_tensor(v) else v for k, v in result.items()}


def compute_u_baseline_for_object(obj_data):
    """Baseline U from raw LLM graphs (no brain). Unchanged."""
    def _get_graphs(vl):
        return [build_graph(ext) for ext in vl if "error" not in ext]

    base_g = _get_graphs(obj_data.get("base", []))
    para1_g = _get_graphs(obj_data.get("paraphrase_1", []))
    para2_g = _get_graphs(obj_data.get("paraphrase_2", []))
    swap1_g = _get_graphs(obj_data.get("attribute_swap_1", []))
    swap2_g = _get_graphs(obj_data.get("attribute_swap_2", []))
    contra1_g = _get_graphs(obj_data.get("contradiction_1", []))
    contra2_g = _get_graphs(obj_data.get("contradiction_2", []))

    all_bp = base_g + para1_g + para2_g
    all_sw = swap1_g + swap2_g
    all_co = contra1_g + contra2_g

    if len(all_bp) >= 2:
        sims = [graph_similarity(g1, g2) for g1, g2 in combinations(all_bp, 2)]
        u_s = sum(sims) / len(sims)
    else:
        u_s = 0.0

    if base_g and all_sw:
        sims = []
        for bg in base_g:
            be = set(bg.edges())
            if not be: continue
            for sg in all_sw:
                sims.append(len(be & set(sg.edges())) / len(be))
        u_c = sum(sims) / len(sims) if sims else 0.0
    else:
        u_c = 0.0

    if base_g and all_co:
        sims = [graph_similarity(bg, cg) for bg in base_g for cg in all_co]
        u_n = 1.0 - sum(sims) / len(sims)
    else:
        u_n = 0.0

    u_total = 0.4 * u_s + 0.4 * u_c + 0.2 * u_n
    return {"U_s": round(u_s, 4), "U_c": round(u_c, 4),
            "U_n": round(u_n, 4), "U_total": round(u_total, 4)}
