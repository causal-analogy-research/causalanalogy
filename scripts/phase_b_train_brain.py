"""
Phase B Brain training entry point.

Single-stage-per-invocation script for the Phase B sweep:
  5 seeds (42, 123, 456, 789, 1024) x 3 stages x 200K-param Brain.
  Architecture matches the Phase 5A 200K checkpoint
  (CausalGraphBrain hidden=128, num_layers=4, n_heads=8, alpha_mlp_hidden=64,
   CausalMemory n_slots=256).

5-seed runs may exhibit small numeric variance from non-deterministic GAT
operations on CPU; warn_only=True logs violations without halting.

For exact reproducibility set PYTHONHASHSEED in the shell wrapper:
  PYTHONHASHSEED=42 python scripts/phase_b_train_brain.py --seed 42 --stage 1

Usage:
  python scripts/phase_b_train_brain.py --seed 42 --stage 1 --epochs 500
  python scripts/phase_b_train_brain.py --seed 42 --stage 2 --epochs 500
  python scripts/phase_b_train_brain.py --seed 42 --stage 3 --epochs 500

Pilot (Day 3 evening):
  python scripts/phase_b_train_brain.py --seed 42 --stage 1 --epochs 50
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────
# Determinism setup MUST run before any training-related imports that
# may instantiate CUDA generators or load CUDA kernels.
# ────────────────────────────────────────────────────────────────────────
def setup_determinism(seed):
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
WARMUP_EPOCHS = 10
COLLAPSE_U_THRESHOLD = 0.5
OVERSMOOTH_SIM_THRESHOLD = 0.95


# ────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--alpha_res", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────
def is_valid_obj(obj_data, min_successful=5):
    """Match run_phase5b.py filter: >=5 non-error extractions across all variants."""
    if not isinstance(obj_data, dict):
        return False
    n = 0
    for v in obj_data.values():
        if not isinstance(v, list):
            continue
        for r in v:
            if isinstance(r, dict) and "error" not in r:
                n += 1
    return n >= min_successful


def pick_valid(variant_list):
    if not variant_list:
        return None
    for ext in variant_list:
        if isinstance(ext, dict) and "error" not in ext:
            return ext
    return None


def filter_pool(concepts_at_level, all_extractions):
    """Keep only concepts with cached extractions and >=5 successful runs."""
    out = []
    missing = []
    for name in concepts_at_level:
        od = all_extractions.get(name)
        if od is None:
            missing.append((name, "missing from extraction cache"))
        elif not is_valid_obj(od):
            missing.append((name, "fewer than 5 successful extractions"))
        else:
            out.append(name)
    return out, missing


# ────────────────────────────────────────────────────────────────────────
# Bridge config (Stage 3 only) — pure function, deterministic
# ────────────────────────────────────────────────────────────────────────
def build_bridge_triplets(level3_abstract_path, training_concept_names,
                          l1_pool, all_extractions):
    """
    Build bridge_config from level3_abstract.json + training pool.

    Returns:
      bridge_config: dict[abstract_name -> {ground, unrelated_physicals, unrelated_abstracts}]
      usable_pair_count: int
      dropped_pairs: list of (abstract_name, ground_name, reason)

    A pair is usable iff abstract concept AND ground concept are both
    in training_concept_names AND have valid cached extractions.
    unrelated_physicals = first 4 from l1_pool, excluding ground.
    unrelated_abstracts = first 4 from pure abstracts (l3 without ground field), excluding self.
    """
    with open(level3_abstract_path) as f:
        l3_cfg = json.load(f)

    # All abstract concepts and their ground (if any) from level3 config
    ground_map = {}
    all_abstracts_in_l3 = []
    for gn, gd in l3_cfg["groups"].items():
        for on, od in gd["objects"].items():
            all_abstracts_in_l3.append(on)
            if "ground" in od:
                ground_map[on] = od["ground"]

    pure_abstracts = [o for o in all_abstracts_in_l3 if o not in ground_map]
    training_set = set(training_concept_names)

    bridge_config = {}
    dropped_pairs = []
    for abstract_name, ground_name in ground_map.items():
        reasons = []
        if abstract_name not in training_set:
            reasons.append(f"abstract '{abstract_name}' not in training pool")
        elif not is_valid_obj(all_extractions.get(abstract_name)):
            reasons.append(f"abstract '{abstract_name}' missing valid extraction")
        if ground_name not in training_set:
            reasons.append(f"ground '{ground_name}' not in training pool")
        elif not is_valid_obj(all_extractions.get(ground_name)):
            reasons.append(f"ground '{ground_name}' missing valid extraction")
        if reasons:
            dropped_pairs.append((abstract_name, ground_name, "; ".join(reasons)))
            continue

        unrelated_phys = [o for o in l1_pool if o != ground_name][:4]
        unrelated_abs = [o for o in pure_abstracts
                         if o != abstract_name
                         and o in training_set
                         and is_valid_obj(all_extractions.get(o))][:4]
        bridge_config[abstract_name] = {
            "ground": ground_name,
            "unrelated_physicals": unrelated_phys,
            "unrelated_abstracts": unrelated_abs,
        }

    return bridge_config, len(bridge_config), dropped_pairs


# ────────────────────────────────────────────────────────────────────────
# Brain evaluation helpers
# ────────────────────────────────────────────────────────────────────────
def make_get_embedding(brain, memory, extraction_to_tensors, emb_module):
    def _get_embedding(extraction):
        nf, ei, em, _ = extraction_to_tensors(extraction, emb_module)
        emb, alpha, _, _ = brain(nf, ei, em, memory)
        return emb, alpha
    return _get_embedding


def compute_u_perceptual(brain, obj_data, extraction_to_tensors, emb_module):
    """Perceptual baseline using brain.perceptual_proj only (no GAT)."""
    def _ge(vl):
        out = []
        for e in vl:
            if isinstance(e, dict) and "error" not in e:
                nf, _, _, _ = extraction_to_tensors(e, emb_module)
                with torch.no_grad():
                    proj = F.normalize(
                        brain.perceptual_proj(nf.mean(dim=0)), p=2, dim=-1
                    )
                out.append(proj)
        return out

    def _c(a, b):
        return (F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item() + 1) / 2

    bp = (_ge(obj_data.get("base", []))
          + _ge(obj_data.get("paraphrase_1", []))
          + _ge(obj_data.get("paraphrase_2", [])))
    sw = _ge(obj_data.get("attribute_swap_1", [])) + _ge(obj_data.get("attribute_swap_2", []))
    co = _ge(obj_data.get("contradiction_1", [])) + _ge(obj_data.get("contradiction_2", []))
    b = _ge(obj_data.get("base", []))
    u_s = (np.mean([_c(bp[i], bp[j]) for i in range(len(bp)) for j in range(i + 1, len(bp))])
           if len(bp) >= 2 else 0.0)
    u_c = np.mean([_c(x, y) for x in b for y in sw]) if b and sw else 0.0
    u_n = 1 - np.mean([_c(x, y) for x in b for y in co]) if b and co else 0.0
    return float(0.4 * u_s + 0.4 * u_c + 0.2 * u_n)


def eval_heldout(brain, memory, ho_objs, all_extractions,
                 extraction_to_tensors, emb_module, compute_u_for_object):
    brain.eval()
    u_vals, uc_vals, perc_vals = [], [], []
    for o in ho_objs:
        od = all_extractions.get(o)
        if not od:
            continue
        with torch.no_grad():
            s = compute_u_for_object(brain, od, emb_module, memory)
            p = compute_u_perceptual(brain, od, extraction_to_tensors, emb_module)
        u_vals.append(s["U_total"])
        uc_vals.append(s["U_c"])
        perc_vals.append(p)
    if not u_vals:
        return 0.0, 0.0, 0.0
    ho_u = float(np.mean(u_vals))
    ho_uc = float(np.mean(uc_vals))
    delta = ho_u - float(np.mean(perc_vals))
    return ho_u, ho_uc, delta


def compute_mean_inter_concept_sim(brain, memory, l1_pool, all_extractions,
                                    extraction_to_tensors, emb_module, n_sample=20):
    """Stage 1 over-smoothing diagnostic — replicates run_phase5b.py lines 277-289."""
    brain.eval()
    embs = []
    for o in random.sample(l1_pool, min(n_sample, len(l1_pool))):
        e = pick_valid(all_extractions.get(o, {}).get("base", []))
        if e is None:
            continue
        nf, ei, _, _ = extraction_to_tensors(e, emb_module)
        with torch.no_grad():
            emb, _, _, _ = brain(nf, ei, None, memory)
        embs.append(emb.detach())
    if len(embs) < 2:
        return 0.0
    stack = torch.stack(embs)
    sims = F.cosine_similarity(stack.unsqueeze(0), stack.unsqueeze(1), dim=-1)
    mask = ~torch.eye(len(embs), dtype=torch.bool)
    return float(sims[mask].mean().item())


# ────────────────────────────────────────────────────────────────────────
# Per-level metric breakdown
# ────────────────────────────────────────────────────────────────────────
def per_level_metrics(brain, memory, level_pools, all_extractions,
                       compute_u_for_object, emb_module, sample_size=10):
    """Sample sample_size per level, return per-level U / Uc / alpha (mean/min/max)."""
    brain.eval()
    out = {}
    for level, pool in level_pools.items():
        if not pool:
            continue
        sample = random.sample(pool, min(sample_size, len(pool)))
        u_vals, uc_vals, alpha_vals = [], [], []
        for o in sample:
            od = all_extractions.get(o)
            if not od:
                continue
            with torch.no_grad():
                s = compute_u_for_object(brain, od, emb_module, memory)
            u_vals.append(s["U_total"])
            uc_vals.append(s["U_c"])
            alpha_vals.append(s.get("alpha", 0.5))
        if u_vals:
            out[level] = {
                "U": float(np.mean(u_vals)),
                "Uc": float(np.mean(uc_vals)),
                "alpha_mean": float(np.mean(alpha_vals)),
                "alpha_min": float(np.min(alpha_vals)),
                "alpha_max": float(np.max(alpha_vals)),
                "alpha_std": float(np.std(alpha_vals)),
            }
    return out


# ────────────────────────────────────────────────────────────────────────
# Stage configuration
# ────────────────────────────────────────────────────────────────────────
def stage_config(stage, l1_pool, l2_pool, l3_pool, out_dir):
    if stage == 1:
        return {
            "lr": 1e-3,
            "batch_split": [(l1_pool, 16)],
            "load_ckpt": None,
            "use_bridge": False,
        }
    if stage == 2:
        return {
            "lr": 1e-3,
            "batch_split": [(l2_pool, 8), (l1_pool, 8)],
            "load_ckpt": f"{out_dir}/stage_1_best.pt",
            "use_bridge": False,
        }
    if stage == 3:
        return {
            "lr": 5e-4,
            "batch_split": [(l3_pool, 7), (l2_pool, 3), (l1_pool, 6)],
            "load_ckpt": f"{out_dir}/stage_2_best.pt",
            "use_bridge": True,
        }
    raise ValueError(f"unknown stage {stage}")


# ────────────────────────────────────────────────────────────────────────
# Single-batch step
# ────────────────────────────────────────────────────────────────────────
def build_ordered_batch(batch_split, l1_pool, l2_pool, l3_pool):
    """
    Sample per (pool, n) then reorder physical -> compound -> abstract for
    memory priming (matches run_stage3.py reordering, NOT run_phase5b.py order).
    """
    raw = []
    for pool, n in batch_split:
        if pool:
            raw.extend(random.sample(pool, min(n, len(pool))))
    l1_set, l2_set, l3_set = set(l1_pool), set(l2_pool), set(l3_pool)
    phys = [o for o in raw if o in l1_set]
    comp = [o for o in raw if o in l2_set]
    abst = [o for o in raw if o in l3_set]
    return phys + comp + abst


def collect_anchor_pos_neg(ordered_batch, all_extractions, get_embedding,
                            bridge_config):
    """
    For each concept in ordered_batch, build:
      anchor (base), positives (paraphrase + swap),
      negatives pool (contradictions, plus bridge unrelated_physicals if Stage 3).
    Bridge ground concepts are ALSO appended as extra positives for the abstract anchor.
    """
    anchor_embs, anchor_concepts, pos_per_anchor = [], [], []
    neg_pool, neg_pool_concepts = [], []
    skipped = 0

    for obj_name in ordered_batch:
        od = all_extractions.get(obj_name)
        if not od:
            skipped += 1
            print(f"    WARN: extraction missing for {obj_name!r}, skipping")
            continue
        base = pick_valid(od.get("base", []))
        if base is None:
            skipped += 1
            continue
        a_emb, _ = get_embedding(base)
        anchor_embs.append(a_emb)
        anchor_concepts.append(obj_name)

        plist = []
        for k in ("paraphrase_1", "paraphrase_2", "attribute_swap_1", "attribute_swap_2"):
            e = pick_valid(od.get(k, []))
            if e is not None:
                plist.append(get_embedding(e)[0])
        for k in ("contradiction_1", "contradiction_2"):
            e = pick_valid(od.get(k, []))
            if e is not None:
                neg_pool.append(get_embedding(e)[0])
                neg_pool_concepts.append(obj_name)

        # Bridge: ground -> extra positive; unrelated_physicals -> negatives.
        # Bridge triplets themselves are constructed separately below in the
        # forward+reverse explicit path; here we just enrich the mining pools.
        if bridge_config and obj_name in bridge_config:
            bc = bridge_config[obj_name]
            ge = pick_valid(all_extractions.get(bc["ground"], {}).get("base", []))
            if ge is not None:
                plist.append(get_embedding(ge)[0])
            for ur in bc.get("unrelated_physicals", [])[:2]:
                ue = pick_valid(all_extractions.get(ur, {}).get("base", []))
                if ue is not None:
                    neg_pool.append(get_embedding(ue)[0])
                    neg_pool_concepts.append(ur)

        pos_per_anchor.append(plist)

    return (anchor_embs, anchor_concepts, pos_per_anchor,
            neg_pool, neg_pool_concepts, skipped)


def bridge_loss_explicit(ordered_batch, bridge_config, all_extractions,
                          get_embedding, margin):
    """
    Forward+reverse bridge triplets, mirroring src/brain/optimizer.py lines 137-157.
    Returns (mean_bridge_loss tensor or None, n_triplets).

    Forward: anchor=abstract, pos=ground,    neg=unrelated_physical (up to 2)
    Reverse: anchor=ground,   pos=abstract,  neg=unrelated_abstract (up to 2)
    => 4 triplets per usable bridge pair in the batch.
    """
    losses = []
    for obj_name in ordered_batch:
        bc = bridge_config.get(obj_name)
        if bc is None:
            continue
        abs_ext = pick_valid(all_extractions.get(obj_name, {}).get("base", []))
        gnd_ext = pick_valid(all_extractions.get(bc["ground"], {}).get("base", []))
        if abs_ext is None or gnd_ext is None:
            continue
        a_emb, _ = get_embedding(abs_ext)
        g_emb, _ = get_embedding(gnd_ext)

        # Forward: abstract -> ground, away from unrelated_physical
        for ur in bc.get("unrelated_physicals", [])[:2]:
            ue = pick_valid(all_extractions.get(ur, {}).get("base", []))
            if ue is None:
                continue
            u_emb, _ = get_embedding(ue)
            d_p = (a_emb - g_emb).pow(2).sum()
            d_n = (a_emb - u_emb).pow(2).sum()
            losses.append(torch.clamp(d_p - d_n + margin, min=0.0))

        # Reverse: ground -> abstract, away from unrelated_abstract
        for ur in bc.get("unrelated_abstracts", [])[:2]:
            ue = pick_valid(all_extractions.get(ur, {}).get("base", []))
            if ue is None:
                continue
            u_emb, _ = get_embedding(ue)
            d_p = (g_emb - a_emb).pow(2).sum()
            d_n = (g_emb - u_emb).pow(2).sum()
            losses.append(torch.clamp(d_p - d_n + margin, min=0.0))

    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


# ────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────
def train(args, out_dir):
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    # Imports are deferred until after determinism is configured
    from src.brain.gat import CausalGraphBrain
    from src.brain.memory import CausalMemory
    from src.sensors import embedder as emb_module
    from src.scoring.u_score import compute_u_for_object
    from src.scoring.graph_utils import extraction_to_tensors
    from src.training.mining import mine_batch_hard

    # ─── Load data ───────────────────────────────────────────────
    with open(repo_root / "data/heldout.json") as f:
        split = json.load(f)
    with open(repo_root / "cache/all_extractions_expanded.json") as f:
        all_extractions = json.load(f)

    training_concepts = split["training_concepts"]
    held_out_concepts = split["held_out_concepts"]
    training_names = [c["name"] for c in training_concepts]

    raw_l1 = [c["name"] for c in training_concepts if c["level"] == "physical"]
    raw_l2 = [c["name"] for c in training_concepts if c["level"] == "compound"]
    raw_l3 = [c["name"] for c in training_concepts if c["level"] == "abstract"]
    raw_ho = [c["name"] for c in held_out_concepts]

    l1_pool, l1_drop = filter_pool(raw_l1, all_extractions)
    l2_pool, l2_drop = filter_pool(raw_l2, all_extractions)
    l3_pool, l3_drop = filter_pool(raw_l3, all_extractions)
    ho_pool, ho_drop = filter_pool(raw_ho, all_extractions)

    print("=" * 72)
    print(f"Phase B Brain training — seed={args.seed} stage={args.stage} "
          f"epochs={args.epochs} alpha_res={args.alpha_res} margin={args.margin}")
    print("=" * 72)
    print(f"Training pool (filtered):")
    print(f"  physical  {len(l1_pool):>3} (raw {len(raw_l1)}, dropped {len(l1_drop)})")
    print(f"  compound  {len(l2_pool):>3} (raw {len(raw_l2)}, dropped {len(l2_drop)})")
    print(f"  abstract  {len(l3_pool):>3} (raw {len(raw_l3)}, dropped {len(l3_drop)})")
    print(f"  held-out  {len(ho_pool):>3} (raw {len(raw_ho)}, dropped {len(ho_drop)})")
    for name, dropped in [("L1", l1_drop), ("L2", l2_drop), ("L3", l3_drop), ("HO", ho_drop)]:
        for n, reason in dropped:
            print(f"    [{name}] dropped {n!r}: {reason}")

    # ─── Build bridge config (Stage 3) ───────────────────────────
    cfg = stage_config(args.stage, l1_pool, l2_pool, l3_pool, out_dir)
    bridge_config = {}
    bridge_dropped = []
    if cfg["use_bridge"]:
        bridge_config, n_usable, bridge_dropped = build_bridge_triplets(
            repo_root / "config/objects/level3_abstract.json",
            training_names, l1_pool, all_extractions
        )
        total_pairs = n_usable + len(bridge_dropped)
        print(f"\nBridge construction:")
        print(f"  Total bridge pairs in level3_abstract.json: {total_pairs}")
        print(f"  Usable pairs (anchor + ground both in training pool with valid extractions): {n_usable}")
        if bridge_dropped:
            print(f"  Dropped pairs:")
            for a, g, reason in bridge_dropped:
                print(f"    - ({a}, {g}): {reason}")
        print(f"  Bridge triplets per Stage 3 epoch (max): {4 * n_usable} "
              f"(forward 2 + reverse 2 per pair, gated by per-batch sampling)")

    # ─── Architecture (locked: matches Phase 5A 200K checkpoint) ──
    brain = CausalGraphBrain(
        embed_dim=384, hidden_dim=128, n_heads=8,
        n_iterations=3, alpha_res=args.alpha_res,
        alpha_mlp_hidden=64,
    )
    memory = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
    n_params = sum(p.numel() for p in brain.parameters()) + sum(p.numel() for p in memory.parameters())
    print(f"\nModel: {n_params:,} parameters "
          f"(brain {sum(p.numel() for p in brain.parameters()):,} + "
          f"memory {sum(p.numel() for p in memory.parameters()):,})")

    # ─── Load prior-stage checkpoint if needed ───────────────────
    if cfg["load_ckpt"] is not None:
        ckpt_path = Path(cfg["load_ckpt"])
        if not ckpt_path.exists():
            print(f"\nERROR: Stage {args.stage} requires {ckpt_path} but it does not exist.")
            print(f"Run earlier stages first.")
            sys.exit(1)
        print(f"\nLoading prior-stage checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if hasattr(brain, "load_checkpoint_compat"):
            brain.load_checkpoint_compat(ckpt["brain"])
        else:
            brain.load_state_dict(ckpt["brain"])
        memory.load_state_dict(ckpt["memory"])

    # ─── Optimizer + scheduler ──────────────────────────────────
    base_lr = cfg["lr"]
    optimizer = torch.optim.Adam(
        list(brain.parameters()) + list(memory.parameters()), lr=base_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    get_embedding = make_get_embedding(brain, memory, extraction_to_tensors, emb_module)
    level_pools = {"physical": l1_pool, "compound": l2_pool, "abstract": l3_pool}

    # ─── Training loop ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"TRAINING — stage {args.stage}, base_lr={base_lr}, epochs={args.epochs}")
    print(f"{'='*72}")

    best_heldout_u = -1.0
    best_state = copy.deepcopy(brain.state_dict())
    best_mem = copy.deepcopy(memory.state_dict())
    log = []
    t_start = time.time()

    for epoch in range(args.epochs):
        brain.train()

        # Warmup
        if epoch < WARMUP_EPOCHS:
            warm_lr = base_lr / 50 + (base_lr - base_lr / 50) * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        # Build batch (sampled per pool, then reordered phys -> comp -> abs)
        ordered_batch = build_ordered_batch(
            cfg["batch_split"], l1_pool, l2_pool, l3_pool
        )

        # Collect anchors / positives / negatives
        anchor_embs, anchor_concepts, pos_per_anchor, neg_pool, neg_pool_concepts, skipped = (
            collect_anchor_pos_neg(
                ordered_batch, all_extractions, get_embedding,
                bridge_config if cfg["use_bridge"] else None,
            )
        )
        if len(anchor_embs) < 2 or len(neg_pool) < 2:
            log.append({"epoch": epoch, "status": "skipped", "skipped_concepts": skipped})
            continue

        # Batch-hard mining
        ha, hp, hn = mine_batch_hard(
            torch.stack(anchor_embs), anchor_concepts,
            pos_per_anchor, anchor_concepts,
            torch.stack(neg_pool), neg_pool_concepts, args.margin,
        )
        if ha is None:
            log.append({"epoch": epoch, "status": "no_mined_triplets", "skipped_concepts": skipped})
            continue

        d_ap = (ha - hp).pow(2).sum(dim=-1)
        d_an = (ha - hn).pow(2).sum(dim=-1)
        mining_loss = torch.clamp(d_ap - d_an + args.margin, min=0.0).mean()
        active = float((torch.clamp(d_ap - d_an + args.margin, min=0.0) > 0).float().mean().item())

        # Bridge loss (Stage 3)
        bridge_loss_val = None
        n_bridge = 0
        if cfg["use_bridge"]:
            bl, n_bridge = bridge_loss_explicit(
                ordered_batch, bridge_config, all_extractions, get_embedding, args.margin
            )
            if bl is not None:
                loss = mining_loss + 0.2 * bl
                bridge_loss_val = float(bl.item())
            else:
                loss = mining_loss
        else:
            loss = mining_loss

        # Backward + clip both brain+memory
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(brain.parameters()) + list(memory.parameters()), max_norm=1.0
        )
        optimizer.step()
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]

        # ─── Logging every 10 epochs ────────────────────────────
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            level_metrics = per_level_metrics(
                brain, memory, level_pools, all_extractions,
                compute_u_for_object, emb_module, sample_size=10,
            )
            train_u_sample = float(np.mean([m["U"] for m in level_metrics.values()])) \
                if level_metrics else 0.0
            row = {
                "epoch": epoch,
                "loss": round(float(loss.item()), 5),
                "mining_loss": round(float(mining_loss.item()), 5),
                "bridge_loss": round(bridge_loss_val, 5) if bridge_loss_val is not None else None,
                "n_bridge_triplets": n_bridge,
                "active_fraction": round(active, 4),
                "grad_norm": round(float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm), 4),
                "skipped_concepts": skipped,
                "lr": round(cur_lr, 6),
                "U_sample": round(train_u_sample, 4),
                "per_level": level_metrics,
            }
            log.append(row)
            level_str = " | ".join(
                f"{lv[:4]} U={m['U']:.3f} α={m['alpha_mean']:.2f}±{m['alpha_std']:.2f}"
                for lv, m in level_metrics.items()
            )
            print(f"  ep {epoch:>4d}  loss={loss.item():.4f}  active={active:.2f}  "
                  f"grad={float(grad_norm):.2f}  lr={cur_lr:.5f}  U={train_u_sample:.3f}  "
                  f"| {level_str}")

            # ─── Halt: collapse detection ───────────────────────
            if epoch > WARMUP_EPOCHS and train_u_sample < COLLAPSE_U_THRESHOLD:
                print(f"\nCOLLAPSE DETECTED at epoch {epoch}: "
                      f"train_U={train_u_sample:.3f} < {COLLAPSE_U_THRESHOLD}")
                col_path = f"{out_dir}/stage_{args.stage}_collapsed.pt"
                torch.save({"brain": brain.state_dict(), "memory": memory.state_dict(),
                            "epoch": epoch, "train_U": train_u_sample}, col_path)
                _save_log(log, args, out_dir, t_start, status="collapsed",
                          best_heldout_u=best_heldout_u)
                print(f"State saved to {col_path}")
                sys.exit(1)

        # ─── Held-out eval every 50 epochs ──────────────────────
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            ho_u, ho_uc, ho_delta = eval_heldout(
                brain, memory, ho_pool, all_extractions,
                extraction_to_tensors, emb_module, compute_u_for_object,
            )
            log.append({
                "epoch": epoch, "event": "heldout_eval",
                "heldout_U": round(ho_u, 4),
                "heldout_Uc": round(ho_uc, 4),
                "heldout_delta": round(ho_delta, 4),
            })
            print(f"      [held-out @ ep {epoch}]  U={ho_u:.4f}  Uc={ho_uc:.4f}  "
                  f"delta_brain={ho_delta:+.4f}")
            if ho_u > best_heldout_u:
                best_heldout_u = ho_u
                best_state = copy.deepcopy(brain.state_dict())
                best_mem = copy.deepcopy(memory.state_dict())
                torch.save(
                    {"brain": best_state, "memory": best_mem,
                     "epoch": epoch, "heldout_U": ho_u, "stage": args.stage,
                     "seed": args.seed},
                    f"{out_dir}/stage_{args.stage}_best.pt",
                )

    # ─── End of training ────────────────────────────────────────
    wall = time.time() - t_start

    # Final checkpoint
    torch.save(
        {"brain": brain.state_dict(), "memory": memory.state_dict(),
         "epoch": args.epochs - 1, "stage": args.stage, "seed": args.seed},
        f"{out_dir}/stage_{args.stage}_final.pt",
    )

    # Stage 1 over-smoothing diagnostic
    over_smooth_sim = None
    over_smooth_halt = False
    if args.stage == 1 and l1_pool:
        # Reload best for the diagnostic (matches run_phase5b.py pattern)
        brain.load_state_dict(best_state)
        memory.load_state_dict(best_mem)
        over_smooth_sim = compute_mean_inter_concept_sim(
            brain, memory, l1_pool, all_extractions,
            extraction_to_tensors, emb_module, n_sample=20,
        )
        print(f"\nOver-smoothing check: mean inter-concept sim = {over_smooth_sim:.4f}")
        if over_smooth_sim >= OVERSMOOTH_SIM_THRESHOLD:
            print(f"WARNING: over-smoothing detected (>= {OVERSMOOTH_SIM_THRESHOLD}). "
                  f"Relaunch with --alpha_res 0.35")
            over_smooth_halt = True

    # Final summary
    final_train_u = log[-1].get("U_sample", 0.0) if log else 0.0
    final_loss = log[-1].get("loss", 0.0) if log else 0.0
    print(f"\n{'='*72}")
    print(f"FINAL SUMMARY — stage {args.stage}, seed {args.seed}")
    print(f"{'='*72}")
    print(f"  total epochs:       {args.epochs}")
    print(f"  wall time:          {wall:.1f}s")
    print(f"  final loss:         {final_loss}")
    print(f"  final train U:      {final_train_u}")
    print(f"  best held-out U:    {best_heldout_u:.4f}")
    if over_smooth_sim is not None:
        print(f"  inter-concept sim:  {over_smooth_sim:.4f}")
    print(f"  best ckpt:          {out_dir}/stage_{args.stage}_best.pt")
    print(f"  final ckpt:         {out_dir}/stage_{args.stage}_final.pt")

    _save_log(log, args, out_dir, t_start,
              status="oversmooth_halt" if over_smooth_halt else "ok",
              best_heldout_u=best_heldout_u,
              over_smooth_sim=over_smooth_sim,
              bridge_config=bridge_config,
              bridge_dropped=bridge_dropped)

    if over_smooth_halt:
        sys.exit(2)


def _save_log(log, args, out_dir, t_start, status, best_heldout_u,
              over_smooth_sim=None, bridge_config=None, bridge_dropped=None):
    payload = {
        "seed": args.seed,
        "stage": args.stage,
        "epochs": args.epochs,
        "alpha_res": args.alpha_res,
        "margin": args.margin,
        "out_dir": out_dir,
        "wall_time_s": time.time() - t_start,
        "status": status,
        "best_heldout_U": best_heldout_u,
        "over_smooth_sim": over_smooth_sim,
        "bridge_config": bridge_config or {},
        "bridge_dropped": bridge_dropped or [],
        "log": log,
    }
    log_path = f"{out_dir}/stage_{args.stage}_log.json"
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  log saved:          {log_path}")


# ────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.out_dir or f"checkpoints/phase_b/brain_200k/seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    setup_determinism(args.seed)
    train(args, out_dir)


if __name__ == "__main__":
    main()
