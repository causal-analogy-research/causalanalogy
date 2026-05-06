"""
Phase B Projection-Only training entry point.

Brain-minus-GAT ablation: trains a single nn.Linear(384, 128) on the same
420-concept pool the Brain trains on, using:
  - the same node-feature input pipeline (LLM-extracted graph nodes, 384-dim
    MiniLM embeddings via cache/all_extractions_expanded.json)
  - the same triplet loss + batch-hard mining (src.training.mining)
  - the same curriculum staging (Stage 1 phys -> Stage 2 +comp -> Stage 3 +abs+bridge)
  - the same bridge-triplet construction (4 triplets per usable pair, weight 0.2)
  - the same determinism wiring as phase_b_train_brain.py

49,280 trainable parameters — matches CausalGraphBrain.self.perceptual_proj.
The ablation isolates the GAT's contribution by training the residual bypass alone.

Usage:
  PYTHONHASHSEED=42 python scripts/phase_b_train_projection.py --seed 42 --stage 1
  PYTHONHASHSEED=42 python scripts/phase_b_train_projection.py --seed 42 --stage all --epochs 500
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
# Determinism setup MUST run before training-related imports that may
# instantiate CUDA generators or load CUDA kernels.
# ────────────────────────────────────────────────────────────────────────
def setup_determinism(seed):
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────────────────
# Constants — mirror phase_b_train_brain.py
# ────────────────────────────────────────────────────────────────────────
MARGIN = 1.0
WARMUP_EPOCHS = 10
COLLAPSE_U_THRESHOLD = 0.5
DEFAULT_LR_PER_STAGE = {1: 1e-3, 2: 1e-3, 3: 5e-4}


# ────────────────────────────────────────────────────────────────────────
# Argparse
# ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", default="all",
                   help="1, 2, 3, or 'all' (run all three sequentially in one process)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Data helpers — copied from phase_b_train_brain.py for self-containment
# ────────────────────────────────────────────────────────────────────────
def is_valid_obj(obj_data, min_successful=5):
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
# Bridge config — mirrors phase_b_train_brain.py:build_bridge_triplets
# ────────────────────────────────────────────────────────────────────────
def build_bridge_triplets(level3_abstract_path, training_concept_names,
                          l1_pool, all_extractions):
    with open(level3_abstract_path) as f:
        l3_cfg = json.load(f)

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
    dropped = []
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
            dropped.append((abstract_name, ground_name, "; ".join(reasons)))
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

    return bridge_config, len(bridge_config), dropped


# ────────────────────────────────────────────────────────────────────────
# Embedding + U computation — projection-only specific
# ────────────────────────────────────────────────────────────────────────
def make_get_embedding(proj_model, extraction_to_tensors, emb_module):
    """Returns a closure (extraction -> (emb, None)) so it slots into the same
    triplet/mining/bridge pipeline as the Brain's get_embedding."""
    def _get_embedding(extraction):
        nf, _, _, _ = extraction_to_tensors(extraction, emb_module)
        emb = proj_model(nf)
        return emb, None
    return _get_embedding


def compute_u_proj(proj_model, obj_data, extraction_to_tensors, emb_module):
    """
    U score using the projection-only model. Matches the formula
    phase_b_train_brain.py:compute_u_perceptual uses on the Brain's residual
    bypass — same _ge / _c helpers, same 0.4/0.4/0.2 weighting.
    """
    def _ge(vl):
        out = []
        for e in vl:
            if isinstance(e, dict) and "error" not in e:
                nf, _, _, _ = extraction_to_tensors(e, emb_module)
                with torch.no_grad():
                    out.append(proj_model(nf))
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

    u_s = float(u_s); u_c = float(u_c); u_n = float(u_n)
    return {"U_total": 0.4 * u_s + 0.4 * u_c + 0.2 * u_n,
            "U_s": u_s, "U_c": u_c, "U_n": u_n}


def eval_heldout(proj_model, ho_objs, all_extractions,
                 extraction_to_tensors, emb_module):
    proj_model.eval()
    u_vals, uc_vals = [], []
    for o in ho_objs:
        od = all_extractions.get(o)
        if not od:
            continue
        s = compute_u_proj(proj_model, od, extraction_to_tensors, emb_module)
        u_vals.append(s["U_total"])
        uc_vals.append(s["U_c"])
    if not u_vals:
        return 0.0, 0.0
    return float(np.mean(u_vals)), float(np.mean(uc_vals))


def per_level_metrics(proj_model, level_pools, all_extractions,
                      extraction_to_tensors, emb_module, sample_size=10):
    proj_model.eval()
    out = {}
    for level, pool in level_pools.items():
        if not pool:
            continue
        sample = random.sample(pool, min(sample_size, len(pool)))
        u_vals, uc_vals = [], []
        for o in sample:
            od = all_extractions.get(o)
            if not od:
                continue
            s = compute_u_proj(proj_model, od, extraction_to_tensors, emb_module)
            u_vals.append(s["U_total"])
            uc_vals.append(s["U_c"])
        if u_vals:
            out[level] = {"U": float(np.mean(u_vals)), "Uc": float(np.mean(uc_vals))}
    return out


# ────────────────────────────────────────────────────────────────────────
# Stage configuration / batch building / mining / bridge — mirror Brain
# ────────────────────────────────────────────────────────────────────────
def stage_config(stage_num, l1_pool, l2_pool, l3_pool):
    if stage_num == 1:
        return {"lr": DEFAULT_LR_PER_STAGE[1],
                "batch_split": [(l1_pool, 16)],
                "use_bridge": False}
    if stage_num == 2:
        return {"lr": DEFAULT_LR_PER_STAGE[2],
                "batch_split": [(l2_pool, 8), (l1_pool, 8)],
                "use_bridge": False}
    if stage_num == 3:
        return {"lr": DEFAULT_LR_PER_STAGE[3],
                "batch_split": [(l3_pool, 7), (l2_pool, 3), (l1_pool, 6)],
                "use_bridge": True}
    raise ValueError(f"unknown stage {stage_num}")


def build_ordered_batch(batch_split, l1_pool, l2_pool, l3_pool):
    raw = []
    for pool, n in batch_split:
        if pool:
            raw.extend(random.sample(pool, min(n, len(pool))))
    l1_set, l2_set, l3_set = set(l1_pool), set(l2_pool), set(l3_pool)
    phys = [o for o in raw if o in l1_set]
    comp = [o for o in raw if o in l2_set]
    abst = [o for o in raw if o in l3_set]
    return phys + comp + abst


def collect_anchor_pos_neg(ordered_batch, all_extractions, get_embedding, bridge_config):
    anchor_embs, anchor_concepts, pos_per_anchor = [], [], []
    neg_pool, neg_pool_concepts = [], []
    skipped = 0

    for obj_name in ordered_batch:
        od = all_extractions.get(obj_name)
        if not od:
            skipped += 1
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

        for ur in bc.get("unrelated_physicals", [])[:2]:
            ue = pick_valid(all_extractions.get(ur, {}).get("base", []))
            if ue is None:
                continue
            u_emb, _ = get_embedding(ue)
            d_p = (a_emb - g_emb).pow(2).sum()
            d_n = (a_emb - u_emb).pow(2).sum()
            losses.append(torch.clamp(d_p - d_n + margin, min=0.0))

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
# Stage runner
# ────────────────────────────────────────────────────────────────────────
def run_stage(stage_num, proj_model, args, pools, ho_pool, all_extractions,
              bridge_config, extraction_to_tensors, emb_module, mine_batch_hard,
              out_dir):
    """Train one stage. Returns (best_state, best_heldout_u, log, stage_wall_s)."""
    l1_pool, l2_pool, l3_pool = pools
    cfg = stage_config(stage_num, l1_pool, l2_pool, l3_pool)
    base_lr = cfg["lr"]

    optimizer = torch.optim.Adam(proj_model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    get_embedding = make_get_embedding(proj_model, extraction_to_tensors, emb_module)
    level_pools = {"physical": l1_pool, "compound": l2_pool, "abstract": l3_pool}

    batch_str = ", ".join(f"n={n}/{len(pool)}" for pool, n in cfg["batch_split"])
    print(f"\n{'='*72}")
    print(f"STAGE {stage_num}: epochs={args.epochs}, lr={base_lr}, "
          f"batch=[{batch_str}], bridge={'yes' if cfg['use_bridge'] else 'no'}")
    print(f"{'='*72}")

    best_heldout_u = -1.0
    best_state = copy.deepcopy(proj_model.state_dict())
    log = []
    stage_t0 = time.time()
    stage_dir = Path(out_dir) / f"stage_{stage_num}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        proj_model.train()

        if epoch < WARMUP_EPOCHS:
            warm_lr = base_lr / 50 + (base_lr - base_lr / 50) * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        ordered_batch = build_ordered_batch(cfg["batch_split"], l1_pool, l2_pool, l3_pool)

        anchor_embs, anchor_concepts, pos_per_anchor, neg_pool, neg_pool_concepts, skipped = (
            collect_anchor_pos_neg(
                ordered_batch, all_extractions, get_embedding,
                bridge_config if cfg["use_bridge"] else None,
            )
        )
        if len(anchor_embs) < 2 or len(neg_pool) < 2:
            log.append({"epoch": epoch, "status": "skipped", "skipped_concepts": skipped})
            continue

        ha, hp, hn = mine_batch_hard(
            torch.stack(anchor_embs), anchor_concepts,
            pos_per_anchor, anchor_concepts,
            torch.stack(neg_pool), neg_pool_concepts, MARGIN,
        )
        if ha is None:
            log.append({"epoch": epoch, "status": "no_mined_triplets", "skipped_concepts": skipped})
            continue

        d_ap = (ha - hp).pow(2).sum(dim=-1)
        d_an = (ha - hn).pow(2).sum(dim=-1)
        mining_loss = torch.clamp(d_ap - d_an + MARGIN, min=0.0).mean()
        active = float((torch.clamp(d_ap - d_an + MARGIN, min=0.0) > 0).float().mean().item())

        bridge_loss_val = None
        n_bridge = 0
        if cfg["use_bridge"]:
            bl, n_bridge = bridge_loss_explicit(
                ordered_batch, bridge_config, all_extractions, get_embedding, MARGIN,
            )
            if bl is not None:
                loss = mining_loss + 0.2 * bl
                bridge_loss_val = float(bl.item())
            else:
                loss = mining_loss
        else:
            loss = mining_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(proj_model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]

        # ── Per-eval logging every 10 epochs (and at last epoch) ────
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            level_metrics = per_level_metrics(
                proj_model, level_pools, all_extractions,
                extraction_to_tensors, emb_module, sample_size=10,
            )
            train_u_sample = float(np.mean([m["U"] for m in level_metrics.values()])) \
                if level_metrics else 0.0
            log.append({
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
            })
            level_str = " | ".join(f"{lv[:4]} U={m['U']:.3f}" for lv, m in level_metrics.items())
            print(f"  ep {epoch:>4d}  loss={loss.item():.4f}  active={active:.2f}  "
                  f"grad={float(grad_norm):.2f}  lr={cur_lr:.5f}  U={train_u_sample:.3f}  "
                  f"| {level_str}")

            if epoch > WARMUP_EPOCHS and train_u_sample < COLLAPSE_U_THRESHOLD:
                print(f"\nCOLLAPSE DETECTED at epoch {epoch}: train_U={train_u_sample:.3f}")
                col_path = stage_dir / "collapsed.pt"
                torch.save({"proj_model": proj_model.state_dict(),
                            "epoch": epoch, "train_U": train_u_sample}, col_path)
                _save_log(log, args, stage_num, str(stage_dir),
                          time.time() - stage_t0, status="collapsed",
                          best_heldout_u=best_heldout_u)
                sys.exit(1)

        # ── Held-out eval every 50 epochs (and at last epoch) ───────
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            ho_u, ho_uc = eval_heldout(
                proj_model, ho_pool, all_extractions,
                extraction_to_tensors, emb_module,
            )
            log.append({"epoch": epoch, "event": "heldout_eval",
                        "heldout_U": round(ho_u, 4), "heldout_Uc": round(ho_uc, 4)})
            print(f"      [held-out @ ep {epoch}]  U={ho_u:.4f}  Uc={ho_uc:.4f}")
            if ho_u > best_heldout_u:
                best_heldout_u = ho_u
                best_state = copy.deepcopy(proj_model.state_dict())
                torch.save({"proj_model": best_state,
                            "epoch": epoch, "heldout_U": ho_u, "stage": stage_num,
                            "seed": args.seed},
                           stage_dir / "best.pt")

    # Final ckpt
    torch.save({"proj_model": proj_model.state_dict(),
                "epoch": args.epochs - 1, "stage": stage_num, "seed": args.seed},
               stage_dir / "final.pt")

    # Reload best (so downstream stages and final eval see the best state)
    proj_model.load_state_dict(best_state)

    return best_state, best_heldout_u, log, time.time() - stage_t0


def _save_log(log, args, stage_num, stage_dir, wall_s, status, best_heldout_u):
    payload = {
        "seed": args.seed, "stage": stage_num, "epochs": args.epochs,
        "wall_time_s": wall_s, "status": status,
        "best_heldout_U": best_heldout_u, "log": log,
    }
    log_path = Path(stage_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def get_stages_to_run(stage_arg):
    if stage_arg == "all":
        return [1, 2, 3]
    try:
        s = int(stage_arg)
        if s in (1, 2, 3):
            return [s]
    except ValueError:
        pass
    raise ValueError(f"--stage must be 1, 2, 3, or 'all'; got {stage_arg!r}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.out_dir or f"checkpoints/phase_b/projection_only/seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    setup_determinism(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from src.baselines.projection_only import ProjectionOnly
    from src.sensors import embedder as emb_module
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
    print(f"Phase B Projection-Only — seed={args.seed} stage={args.stage} epochs={args.epochs}")
    print("=" * 72)
    print("Training pool (filtered):")
    print(f"  physical  {len(l1_pool):>3} (raw {len(raw_l1)}, dropped {len(l1_drop)})")
    print(f"  compound  {len(l2_pool):>3} (raw {len(raw_l2)}, dropped {len(l2_drop)})")
    print(f"  abstract  {len(l3_pool):>3} (raw {len(raw_l3)}, dropped {len(l3_drop)})")
    print(f"  held-out  {len(ho_pool):>3} (raw {len(raw_ho)}, dropped {len(ho_drop)})")

    # ─── Bridge config (used only when Stage 3 fires) ────────────
    bridge_config, n_usable, bridge_dropped = build_bridge_triplets(
        repo_root / "config/objects/level3_abstract.json",
        training_names, l1_pool, all_extractions,
    )
    print("\nBridge construction:")
    print(f"  Usable pairs (abstract + ground both in training pool): {n_usable}")
    if bridge_dropped:
        print(f"  Dropped pairs: {len(bridge_dropped)}")
        for a, g, reason in bridge_dropped[:5]:
            print(f"    - ({a}, {g}): {reason}")
    print(f"  Bridge triplets per Stage 3 epoch (max): {4 * n_usable}")

    # ─── Init model ─────────────────────────────────────────────
    proj_model = ProjectionOnly(embed_dim=384, hidden_dim=128)
    n_params = sum(p.numel() for p in proj_model.parameters())
    print(f"\nModel: ProjectionOnly, params={n_params:,}")

    pools = (l1_pool, l2_pool, l3_pool)
    stages = get_stages_to_run(args.stage)
    t_start = time.time()

    # ─── Stage loop ─────────────────────────────────────────────
    for stage_num in stages:
        # Load prior-stage best.pt if needed (works for both --stage all and standalone --stage N)
        if stage_num > 1:
            prev_ckpt = Path(out_dir) / f"stage_{stage_num - 1}" / "best.pt"
            if not prev_ckpt.exists():
                print(f"\nERROR: Stage {stage_num} requires {prev_ckpt} but it is missing.")
                print("Run earlier stages first.")
                sys.exit(1)
            print(f"\nLoading prior-stage checkpoint: {prev_ckpt}")
            ckpt = torch.load(prev_ckpt, weights_only=False, map_location="cpu")
            proj_model.load_state_dict(ckpt["proj_model"])

        best_state, best_u, log, stage_wall = run_stage(
            stage_num, proj_model, args, pools, ho_pool, all_extractions,
            bridge_config, extraction_to_tensors, emb_module, mine_batch_hard,
            out_dir,
        )

        stage_dir = Path(out_dir) / f"stage_{stage_num}"
        _save_log(log, args, stage_num, str(stage_dir), stage_wall,
                  status="ok", best_heldout_u=best_u)

        # ─── Final eval against best (already loaded) ──────────
        ho_u, ho_uc = eval_heldout(
            proj_model, ho_pool, all_extractions,
            extraction_to_tensors, emb_module,
        )
        all_pool_metrics = per_level_metrics(
            proj_model, {"all": l1_pool + l2_pool + l3_pool},
            all_extractions, extraction_to_tensors, emb_module, sample_size=20,
        )
        final_train_u = float(all_pool_metrics.get("all", {}).get("U", 0.0))

        with open(stage_dir / "final_metrics.json", "w") as f:
            json.dump({
                "stage": stage_num, "seed": args.seed,
                "best_heldout_U": float(best_u),
                "final_heldout_U": float(ho_u),
                "final_heldout_Uc": float(ho_uc),
                "final_train_U_sample": final_train_u,
                "wall_time_s": stage_wall,
                "ckpt": str(stage_dir / "best.pt"),
            }, f, indent=2)

        print(f"\n  Stage {stage_num} done: best_heldout_U={best_u:.4f}, "
              f"final_heldout_U={ho_u:.4f}, wall={stage_wall:.1f}s")

    print(f"\nDone. Total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
