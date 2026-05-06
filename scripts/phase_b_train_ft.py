"""
Phase B FT (fine-tune) training entry point.

Drives STFineTuner against the 420-concept training pool from data/heldout.json.
Supports MiniLM and mpnet, curriculum and no-curriculum variants, single-stage
or all-stages-at-once.

Curriculum stages mirror the 99-concept pilot:
  Stage 1 — physicals only (lr 2e-5, 500 ep)
  Stage 2 — compounds + 50% physical replay (lr 2e-5, 500 ep)
  Stage 3 — abstracts + compounds + physicals at 40/20/40 with bridge triplets
            (forward + reverse, 4/pair, weight 0.2; lr 1e-5, 500 ep)

No-curriculum: 1500 ep single-stage on the full 420-concept pool, NO bridge triplets.

Determinism: this script wires PYTHONHASHSEED-aware torch/numpy/random seeding plus
cudnn.deterministic=True / cudnn.benchmark=False. Some CUDA ops remain
non-deterministic (scatter_add, reduction kernels, etc.); expect small numeric
variance across CUDA hosts/drivers when not running on CPU.

Outputs per stage (curriculum):  {out_dir}/stage_{N}/{best.pt, training_log.json, final_metrics.json}
Outputs (nocurriculum):          {out_dir}/{best.pt, training_log.json, final_metrics.json}

Usage:
  PYTHONHASHSEED=42 python scripts/phase_b_train_ft.py \\
      --model minilm --variant curriculum --seed 42 --stage all
  PYTHONHASHSEED=42 python scripts/phase_b_train_ft.py \\
      --model mpnet --variant nocurriculum --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


MODEL_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_EPOCHS_PER_STAGE = {1: 500, 2: 500, 3: 500}
DEFAULT_LR_PER_STAGE     = {1: 2e-5, 2: 2e-5, 3: 1e-5}
DEFAULT_NOCURR_EPOCHS    = 1500
DEFAULT_NOCURR_LR        = 2e-5
WARMUP_EPOCHS            = 10
BRIDGE_WEIGHT            = 0.2  # documented for clarity; trainer hardcodes the same constant


# ────────────────────────────────────────────────────────────────────────
# Argparse
# ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=["minilm", "mpnet"], required=True)
    p.add_argument("--variant", choices=["curriculum", "nocurriculum"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", default="all",
                   help="1, 2, 3, or 'all' (curriculum only; ignored for nocurriculum)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Overrides per-stage default (500 curriculum / 1500 nocurriculum)")
    p.add_argument("--lr", type=float, default=None,
                   help="Overrides per-stage default (2e-5 stages 1-2, 1e-5 stage 3, "
                        "2e-5 nocurriculum)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", type=str, default=None,
                   help="Default checkpoints/phase_b/ft_{model}_{variant}/seed_{seed}/")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Determinism
# ────────────────────────────────────────────────────────────────────────
def setup_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Determinism: PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'NOT SET')}, "
          f"seed={seed}, cudnn.deterministic=True, cudnn.benchmark=False")
    print("Note: some CUDA ops (scatter_add, certain reductions) remain non-deterministic; "
          "expect small numeric variance across CUDA hosts/drivers when not on CPU.")


# ────────────────────────────────────────────────────────────────────────
# Data adapters
# ────────────────────────────────────────────────────────────────────────
def build_level_config(concepts):
    """
    Convert a list of {name, level, group, variants} dicts into the trainer-expected shape:
      {"groups": {group_name: {"objects": {obj_name: {variant_name: text, ...}}}}}
    The trainer iterates groups -> objects -> variants and skips a "ground" key, so
    bridge ground links are NOT embedded here — they live in the standalone bridge_config.
    """
    groups = {}
    for c in concepts:
        gn = c.get("group", "default")
        if gn not in groups:
            groups[gn] = {"objects": {}}
        groups[gn]["objects"][c["name"]] = dict(c["variants"])
    return {"groups": groups}


def build_bridge_config(level3_abstract_path, training_concept_names, l1_pool_names):
    """
    Mirror phase_b_train_brain.py:build_bridge_triplets — read level3_abstract.json
    ground_map, intersect with training pool, emit forward+reverse bridge_config.

    Validity here is membership-only: heldout.json training_concepts have full variants by
    construction, so unlike the Brain script we do not gate on "extraction validity".
    """
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
        if ground_name not in training_set:
            reasons.append(f"ground '{ground_name}' not in training pool")
        if reasons:
            dropped.append((abstract_name, ground_name, "; ".join(reasons)))
            continue

        unrelated_phys = [o for o in l1_pool_names if o != ground_name][:4]
        unrelated_abs = [o for o in pure_abstracts
                         if o != abstract_name and o in training_set][:4]
        bridge_config[abstract_name] = {
            "ground": ground_name,
            "unrelated_physicals": unrelated_phys,
            "unrelated_abstracts": unrelated_abs,
        }

    return bridge_config, len(bridge_config), dropped


# ────────────────────────────────────────────────────────────────────────
# Stage dispatch
# ────────────────────────────────────────────────────────────────────────
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


def stage_inputs(stage_num, cfgs, pools, args, bridge_config):
    cfg_l1, cfg_l2, cfg_l3 = cfgs
    l1, l2, l3 = pools
    if stage_num == 1:
        return {
            "configs": [cfg_l1],
            "concept_list": list(l1),
            "bridge": None,
            "all_configs": None,
            "epochs": args.epochs if args.epochs is not None else DEFAULT_EPOCHS_PER_STAGE[1],
            "lr": args.lr if args.lr is not None else DEFAULT_LR_PER_STAGE[1],
        }
    if stage_num == 2:
        return {
            "configs": [cfg_l1, cfg_l2],
            "concept_list": list(l2) + list(l1),
            "bridge": None,
            "all_configs": None,
            "epochs": args.epochs if args.epochs is not None else DEFAULT_EPOCHS_PER_STAGE[2],
            "lr": args.lr if args.lr is not None else DEFAULT_LR_PER_STAGE[2],
        }
    if stage_num == 3:
        return {
            "configs": [cfg_l1, cfg_l2, cfg_l3],
            "concept_list": list(l3) + list(l2) + list(l1),
            "bridge": bridge_config,
            "all_configs": [cfg_l1, cfg_l2, cfg_l3],
            "epochs": args.epochs if args.epochs is not None else DEFAULT_EPOCHS_PER_STAGE[3],
            "lr": args.lr if args.lr is not None else DEFAULT_LR_PER_STAGE[3],
        }
    raise ValueError(f"unknown stage {stage_num}")


# ────────────────────────────────────────────────────────────────────────
# Artifact saving
# ────────────────────────────────────────────────────────────────────────
def save_artifacts(stage_dir, log, ft, ho_cfg, ho_pool, configs, concept_list,
                   best_epoch, best_heldout_u, args, stage_label, n_bridge_in_pool):
    """Persist training_log.json + final_metrics.json. Returns (final_ho_u, final_train_u)."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    with open(stage_dir / "training_log.json", "w") as f:
        json.dump({
            "stage": stage_label,
            "model": MODEL_MAP[args.model],
            "variant": args.variant,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_heldout_U": float(best_heldout_u),
            "n_bridge_pairs_in_pool": n_bridge_in_pool,
            "log": log,
        }, f, indent=2, default=str)

    # Final eval against the just-loaded best state
    ho_scores = ft.compute_u_text(ho_cfg, ho_pool)
    final_ho_u = float(np.mean([s["U_total"] for s in ho_scores.values()])) if ho_scores else 0.0

    train_scores = {}
    for cfg in configs:
        train_scores.update(ft.compute_u_text(cfg, concept_list))
    final_train_u = float(np.mean([s["U_total"] for s in train_scores.values()])) \
        if train_scores else 0.0

    with open(stage_dir / "final_metrics.json", "w") as f:
        json.dump({
            "stage": stage_label,
            "model": MODEL_MAP[args.model],
            "variant": args.variant,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_heldout_U": float(best_heldout_u),
            "final_heldout_U": final_ho_u,
            "final_train_U": final_train_u,
            "ckpt": str(stage_dir / "best.pt"),
        }, f, indent=2)

    return final_ho_u, final_train_u


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    setup_determinism(args.seed)

    # Defer trainer import until after determinism is configured
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from src.baselines.finetune_st import STFineTuner

    out_dir = args.out_dir or \
        f"checkpoints/phase_b/ft_{args.model}_{args.variant}/seed_{args.seed}"
    out_dir = str(Path(out_dir))

    # ─── Load split + per-level configs ──────────────────────────
    with open(repo_root / "data/heldout.json") as f:
        split = json.load(f)
    train_concepts = split["training_concepts"]
    held_out_concepts = split["held_out_concepts"]

    l1_concepts = [c for c in train_concepts if c["level"] == "physical"]
    l2_concepts = [c for c in train_concepts if c["level"] == "compound"]
    l3_concepts = [c for c in train_concepts if c["level"] == "abstract"]

    cfg_l1 = build_level_config(l1_concepts)
    cfg_l2 = build_level_config(l2_concepts)
    cfg_l3 = build_level_config(l3_concepts)
    cfg_ho = build_level_config(held_out_concepts)

    l1_names = [c["name"] for c in l1_concepts]
    l2_names = [c["name"] for c in l2_concepts]
    l3_names = [c["name"] for c in l3_concepts]
    ho_names = [c["name"] for c in held_out_concepts]
    training_names = l1_names + l2_names + l3_names

    print("=" * 72)
    print(f"Phase B FT training — model={args.model} ({MODEL_MAP[args.model]})")
    print(f"  variant={args.variant}  seed={args.seed}  stage={args.stage}")
    print(f"  out_dir={out_dir}")
    print("=" * 72)
    print("Training pool:")
    print(f"  physical  {len(l1_names):>3}")
    print(f"  compound  {len(l2_names):>3}")
    print(f"  abstract  {len(l3_names):>3}")
    print(f"  held-out  {len(ho_names):>3}")
    print(f"  total train: {len(training_names)}, held-out {len(ho_names)}")

    # ─── Bridge config ───────────────────────────────────────────
    bridge_config, n_usable, bridge_dropped = build_bridge_config(
        repo_root / "config/objects/level3_abstract.json",
        training_names, l1_names,
    )
    print("\nBridge construction:")
    print(f"  Usable pairs (abstract + ground both in training pool): {n_usable}")
    if bridge_dropped:
        print(f"  Dropped pairs: {len(bridge_dropped)}")
        for a, g, reason in bridge_dropped[:5]:
            print(f"    - ({a}, {g}): {reason}")
        if len(bridge_dropped) > 5:
            print(f"    ... and {len(bridge_dropped)-5} more")
    print(f"  Bridge triplets per Stage 3 epoch (max): {4 * n_usable}")

    # ─── Init trainer ────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft = STFineTuner(model_name=MODEL_MAP[args.model], seed=args.seed, device=device)
    n_params = sum(p.numel() for p in ft.model.parameters())
    print(f"\nModel: {MODEL_MAP[args.model]}, embed_dim={ft.embed_dim}, "
          f"params={n_params:,}, device={device}")

    cfgs = (cfg_l1, cfg_l2, cfg_l3)
    pools = (l1_names, l2_names, l3_names)

    t_start = time.time()

    # ─── No-curriculum branch ────────────────────────────────────
    if args.variant == "nocurriculum":
        if args.stage != "all":
            print(f"\nNOTE: --stage {args.stage} ignored for nocurriculum (single-stage).")

        epochs = args.epochs if args.epochs is not None else DEFAULT_NOCURR_EPOCHS
        lr = args.lr if args.lr is not None else DEFAULT_NOCURR_LR
        all_train = list(l1_names) + list(l2_names) + list(l3_names)

        print(f"\n{'='*72}")
        print(f"NO-CURRICULUM: epochs={epochs}, lr={lr}, batch={args.batch_size}, "
              f"{len(all_train)} concepts, NO bridge triplets")
        print(f"{'='*72}")

        best_ep, best_u, log = ft.train_stage(
            stage_name=f"NoCurriculum (seed {args.seed})",
            configs=[cfg_l1, cfg_l2, cfg_l3],
            concept_list=all_train,
            epochs=epochs, lr=lr,
            warmup_epochs=WARMUP_EPOCHS, batch_size=args.batch_size,
            bridge_config=None,
            all_configs=None,
            heldout_config=cfg_ho, heldout_concepts=ho_names,
            ckpt_dir=out_dir,
        )
        ho_u, train_u = save_artifacts(
            out_dir, log, ft, cfg_ho, ho_names,
            [cfg_l1, cfg_l2, cfg_l3], all_train,
            best_ep, best_u, args, "nocurriculum", n_usable,
        )
        print(f"  Final eval: heldout_U={ho_u:.4f}, train_U={train_u:.4f}")
        print(f"\nDone. wall_time={time.time() - t_start:.1f}s")
        return

    # ─── Curriculum branch ───────────────────────────────────────
    stages = get_stages_to_run(args.stage)

    for stage_num in stages:
        # Load prior-stage best.pt if requested mid-curriculum (and not stage 1)
        if stage_num > 1:
            prev_ckpt = Path(out_dir) / f"stage_{stage_num - 1}" / "best.pt"
            if not prev_ckpt.exists():
                print(f"\nERROR: Stage {stage_num} requires {prev_ckpt} but it is missing.")
                print(f"Run earlier stages first.")
                sys.exit(1)
            print(f"\nLoading prior-stage checkpoint: {prev_ckpt}")
            ft.load_checkpoint(str(prev_ckpt))

        si = stage_inputs(stage_num, cfgs, pools, args, bridge_config)
        stage_dir = Path(out_dir) / f"stage_{stage_num}"

        print(f"\n{'='*72}")
        print(f"STAGE {stage_num}: epochs={si['epochs']}, lr={si['lr']}, "
              f"batch={args.batch_size}, "
              f"{len(si['concept_list'])} concepts, "
              f"bridge={'yes' if si['bridge'] else 'no'}")
        print(f"{'='*72}")

        best_ep, best_u, log = ft.train_stage(
            stage_name=f"Stage {stage_num} (seed {args.seed})",
            configs=si["configs"],
            concept_list=si["concept_list"],
            epochs=si["epochs"], lr=si["lr"],
            warmup_epochs=WARMUP_EPOCHS, batch_size=args.batch_size,
            bridge_config=si["bridge"],
            all_configs=si["all_configs"],
            heldout_config=cfg_ho, heldout_concepts=ho_names,
            ckpt_dir=str(stage_dir),
        )
        ho_u, train_u = save_artifacts(
            stage_dir, log, ft, cfg_ho, ho_names,
            si["configs"], si["concept_list"],
            best_ep, best_u, args, f"stage_{stage_num}", n_usable,
        )
        print(f"  Stage {stage_num} final eval: heldout_U={ho_u:.4f}, train_U={train_u:.4f}")

    print(f"\nDone. wall_time={time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
