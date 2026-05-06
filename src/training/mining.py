"""
Triplet Mining Strategies

Semi-hard: keep triplets where negative is farther than positive but within margin.
Batch-hard: for each anchor, select hardest positive and hardest cross-concept negative.
"""

import torch


def compute_active_fraction(distances_ap, distances_an, margin=1.0):
    """Fraction of triplets with loss > 0 (before any filtering)."""
    raw_loss = distances_ap - distances_an + margin
    active = (raw_loss > 0).float()
    return active.mean().item() if active.numel() > 0 else 0.0


def filter_semihard(distances_ap, distances_an, margin=1.0):
    """
    Returns boolean mask of semi-hard triplets.
    Semi-hard: negative farther than positive but within margin.
    i.e., 0 < raw_loss < margin
    Equivalently: distances_an > distances_ap AND distances_an < distances_ap + margin
    """
    raw_loss = distances_ap - distances_an + margin
    mask = (raw_loss > 0) & (raw_loss < margin)
    return mask, raw_loss


def mine_batch_hard(anchor_embs, anchor_concepts, pos_embs, pos_concepts,
                    all_neg_embs, all_neg_concepts, margin=1.0):
    """
    Batch-hard mining: for each anchor, find hardest positive (farthest same-concept)
    and hardest negative (closest different-concept).

    Args:
        anchor_embs: (A, D) anchor embeddings
        anchor_concepts: list of A concept names
        pos_embs: list of lists — pos_embs[i] = list of positive embeddings for anchor i
        pos_concepts: list of A concept names (same as anchor_concepts)
        all_neg_embs: (N, D) all potential negative embeddings
        all_neg_concepts: list of N concept names

    Returns:
        hard_anchors: (K, D)
        hard_positives: (K, D)
        hard_negatives: (K, D)
    """
    hard_a, hard_p, hard_n = [], [], []

    for i in range(len(anchor_embs)):
        anchor = anchor_embs[i]
        concept = anchor_concepts[i]

        # Hardest positive: farthest same-concept variant
        if not pos_embs[i]:
            continue
        pos_stack = torch.stack(pos_embs[i])  # (P, D)
        pos_dists = (anchor.unsqueeze(0) - pos_stack).pow(2).sum(dim=-1)  # (P,)
        hardest_pos_idx = pos_dists.argmax()
        hardest_pos = pos_stack[hardest_pos_idx]

        # Hardest negative: closest different-concept embedding
        neg_mask = torch.tensor([c != concept for c in all_neg_concepts], dtype=torch.bool)
        if neg_mask.sum() == 0:
            continue
        valid_negs = all_neg_embs[neg_mask]  # (M, D)
        neg_dists = (anchor.unsqueeze(0) - valid_negs).pow(2).sum(dim=-1)  # (M,)
        hardest_neg_idx = neg_dists.argmin()
        hardest_neg = valid_negs[hardest_neg_idx]

        hard_a.append(anchor)
        hard_p.append(hardest_pos)
        hard_n.append(hardest_neg)

    if not hard_a:
        return None, None, None

    return torch.stack(hard_a), torch.stack(hard_p), torch.stack(hard_n)
