"""
Phase 5A: Training Signal Quality Experiment
Two training runs from Stage 3 checkpoint: semi-hard and batch-hard mining.
"""
import sys, json, copy, random, torch, time, os, math
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, '.')

from src.sensors.extractor import run_extraction
from src.brain.gat import CausalGraphBrain
from src.brain.memory import CausalMemory
from src.sensors import embedder as emb_module
from src.scoring.u_score import compute_u_for_object, compute_u_differentiable
from src.scoring.graph_utils import extraction_to_tensors
from src.training.mining import compute_active_fraction, filter_semihard, mine_batch_hard

os.makedirs('results/phase5a', exist_ok=True)
os.makedirs('checkpoints/phase5a_semihard', exist_ok=True)
os.makedirs('checkpoints/phase5a_batchhard', exist_ok=True)

MARGIN = 1.0

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def load_stage3():
    brain = CausalGraphBrain(embed_dim=384, hidden_dim=128, n_heads=8, n_iterations=1, alpha_res=0.2)
    mem = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
    ckpt = torch.load('checkpoints/stage1/stage3_best.pt', weights_only=False)
    brain.load_state_dict(ckpt['brain'])
    mem.load_state_dict(ckpt['memory'])
    return brain, mem

def pick_valid(vlist):
    for e in vlist:
        if "error" not in e: return e
    return None

def get_embedding(brain, memory, ext):
    nf, ei, em, _ = extraction_to_tensors(ext, emb_module)
    emb, alpha, _, _ = brain(nf, ei, em, memory)
    return emb, alpha

def get_perceptual_embedding(brain, ext):
    nf, ei, em, _ = extraction_to_tensors(ext, emb_module)
    with torch.no_grad():
        raw = nf.mean(dim=0)
        proj = brain.perceptual_proj(raw)
        return F.normalize(proj, p=2, dim=-1)

def compute_u_perceptual(brain, obj_data):
    def _ge(vl): return [get_perceptual_embedding(brain, e) for e in vl if "error" not in e]
    def _c(a,b): return ((F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)).item()+1)/2)
    bp = _ge(obj_data.get("base",[])) + _ge(obj_data.get("paraphrase_1",[])) + _ge(obj_data.get("paraphrase_2",[]))
    sw = _ge(obj_data.get("attribute_swap_1",[])) + _ge(obj_data.get("attribute_swap_2",[]))
    co = _ge(obj_data.get("contradiction_1",[])) + _ge(obj_data.get("contradiction_2",[]))
    b = _ge(obj_data.get("base",[]))
    u_s = np.mean([_c(bp[i],bp[j]) for i in range(len(bp)) for j in range(i+1,len(bp))]) if len(bp)>=2 else 0
    u_c = np.mean([_c(x,y) for x in b for y in sw]) if b and sw else 0
    u_n = 1-np.mean([_c(x,y) for x in b for y in co]) if b and co else 0
    return {"U_s":round(u_s,4),"U_c":round(u_c,4),"U_n":round(u_n,4),"U_total":round(0.4*u_s+0.4*u_c+0.2*u_n,4)}

def eval_all(brain, mem, all_ext, level_map):
    brain.eval()
    scores = {}
    for name, od in all_ext.items():
        scores[name] = compute_u_for_object(brain, od, emb_module, mem)
    by_level = {}
    for name, s in scores.items():
        lv = level_map.get(name, "?")
        by_level.setdefault(lv, []).append(s)
    summary = {}
    for lv, sl in by_level.items():
        summary[lv] = {"U": np.mean([s["U_total"] for s in sl]),
                        "U_c": np.mean([s["U_c"] for s in sl]),
                        "U_n": np.mean([s["U_n"] for s in sl]),
                        "alpha": np.mean([s["alpha"] for s in sl])}
    overall_u = np.mean([s["U_total"] for s in scores.values()])
    return overall_u, summary, scores

def eval_heldout(brain, mem, ho_ext, t_brain_for_perc):
    brain.eval()
    results = {}
    for name, od in ho_ext.items():
        t = compute_u_for_object(brain, od, emb_module, mem)
        with torch.no_grad():
            p = compute_u_perceptual(t_brain_for_perc, od)
        delta = round(t["U_total"] - p["U_total"], 4)
        results[name] = {"trained": t, "perceptual": p, "delta_brain": delta}
    mean_u = np.mean([v["trained"]["U_total"] for v in results.values()])
    mean_uc = np.mean([v["trained"]["U_c"] for v in results.values()])
    mean_delta = np.mean([v["delta_brain"] for v in results.values()])
    return mean_u, mean_uc, mean_delta, results


# ════════════════════════════════════════════════════════════════
# TRAINING LOOP (parameterized by mining strategy)
# ════════════════════════════════════════════════════════════════

def train_run(strategy, brain, memory, all_ext, level_map, ho_ext, t_brain_perc,
              l3_objs, l2_objs, l1_objs, bridge_config, n_epochs=500):
    """
    Run one training strategy. Returns (logs, best_state, best_mem_state).
    strategy: "semihard" or "batchhard"
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {strategy.upper()} MINING ({n_epochs} epochs)")
    print(f"{'='*70}")

    brain.train()
    lr = 5e-4
    optimizer = torch.optim.Adam(list(brain.parameters()) + list(memory.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Batch-hard warmup
    warmup_epochs = 10 if strategy == "batchhard" else 0
    base_lr = lr

    logs = []
    best_ho_u = -1
    best_state = copy.deepcopy(brain.state_dict())
    best_mem_state = copy.deepcopy(memory.state_dict())
    loss_zero_streak = 0
    loss_zero_epoch = None
    skipped_batches = 0
    total_batches = 0
    all_objects = l3_objs + l2_objs + l1_objs

    for epoch in range(n_epochs):
        brain.train()

        # LR warmup for batch-hard
        if strategy == "batchhard" and epoch < warmup_epochs:
            warmup_lr = 1e-5 + (base_lr - 1e-5) * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Batch: 7 abstract + 3 compound + 6 physical = 16
        n_abs = min(7, len(l3_objs))
        n_comp = min(3, len(l2_objs))
        n_phys = 16 - n_abs - n_comp
        batch = (random.sample(l3_objs, n_abs) +
                 random.sample(l2_objs, n_comp) +
                 random.sample(l1_objs, min(n_phys, len(l1_objs))))

        # Order: physical, compound, abstract (for memory priming)
        batch_phys = [b for b in batch if b in set(l1_objs)]
        batch_comp = [b for b in batch if b in set(l2_objs)]
        batch_abs = [b for b in batch if b in set(l3_objs)]
        ordered_batch = batch_phys + batch_comp + batch_abs

        # ── Collect all embeddings and build triplets ──
        all_anchors = []
        all_positives = []
        all_negatives = []
        all_anchor_concepts = []
        all_pos_per_anchor = []  # for batch-hard: list of lists
        all_neg_pool = []
        all_neg_pool_concepts = []

        for obj_name in ordered_batch:
            od = all_ext.get(obj_name)
            if not od: continue
            base = pick_valid(od.get("base", []))
            if not base: continue

            anchor_emb, _ = get_embedding(brain, memory, base)
            pos_exts = []
            for k in ["paraphrase_1","paraphrase_2","attribute_swap_1","attribute_swap_2"]:
                e = pick_valid(od.get(k, []))
                if e: pos_exts.append(e)
            neg_exts = []
            for k in ["contradiction_1","contradiction_2"]:
                e = pick_valid(od.get(k, []))
                if e: neg_exts.append(e)

            pos_embs = [get_embedding(brain, memory, e)[0] for e in pos_exts]
            neg_embs = [get_embedding(brain, memory, e)[0] for e in neg_exts]

            # Standard triplets
            for pe in pos_embs:
                for ne in neg_embs:
                    all_anchors.append(anchor_emb)
                    all_positives.append(pe)
                    all_negatives.append(ne)
                    all_anchor_concepts.append(obj_name)

            # For batch-hard mining
            all_pos_per_anchor.append(pos_embs)
            all_neg_pool.extend(neg_embs)
            all_neg_pool_concepts.extend([obj_name] * len(neg_embs))

            # Bridge triplets for grounded abstracts
            if obj_name in bridge_config:
                bc = bridge_config[obj_name]
                gnd = bc["ground"]
                if gnd in all_ext:
                    ge = pick_valid(all_ext[gnd].get("base", []))
                    if ge:
                        gnd_emb, _ = get_embedding(brain, memory, ge)
                        for ur_name in bc.get("unrelated_physicals", [])[:2]:
                            if ur_name in all_ext:
                                ue = pick_valid(all_ext[ur_name].get("base", []))
                                if ue:
                                    ur_emb, _ = get_embedding(brain, memory, ue)
                                    all_anchors.append(anchor_emb)
                                    all_positives.append(gnd_emb)
                                    all_negatives.append(ur_emb)
                                    all_anchor_concepts.append(obj_name)
                        for ur_name in bc.get("unrelated_abstracts", [])[:2]:
                            if ur_name in all_ext:
                                ue = pick_valid(all_ext[ur_name].get("base", []))
                                if ue:
                                    ur_emb, _ = get_embedding(brain, memory, ue)
                                    all_anchors.append(gnd_emb)
                                    all_positives.append(anchor_emb)
                                    all_negatives.append(ur_emb)
                                    all_anchor_concepts.append(gnd)

        if not all_anchors:
            total_batches += 1
            continue

        a_stack = torch.stack(all_anchors)
        p_stack = torch.stack(all_positives)
        n_stack = torch.stack(all_negatives)

        d_ap = (a_stack - p_stack).pow(2).sum(dim=-1)
        d_an = (a_stack - n_stack).pow(2).sum(dim=-1)

        # Metric L1: active fraction (before filtering)
        active_frac = compute_active_fraction(d_ap, d_an, MARGIN)

        # ── Apply mining strategy ──
        total_batches += 1

        if strategy == "semihard":
            mask, raw_loss = filter_semihard(d_ap, d_an, MARGIN)
            if mask.sum() == 0:
                skipped_batches += 1
                continue
            loss = raw_loss[mask].mean()

        elif strategy == "batchhard":
            # Build per-anchor data for batch-hard mining
            anchor_embs_bh = []
            anchor_concepts_bh = []
            pos_per_anchor_bh = []
            seen = set()
            idx = 0
            for obj_name in ordered_batch:
                od = all_ext.get(obj_name)
                if not od or not pick_valid(od.get("base",[])): continue
                if obj_name in seen: continue
                seen.add(obj_name)
                base = pick_valid(od.get("base",[]))
                a_emb, _ = get_embedding(brain, memory, base)
                anchor_embs_bh.append(a_emb)
                anchor_concepts_bh.append(obj_name)
                plist = []
                for k in ["paraphrase_1","paraphrase_2","attribute_swap_1","attribute_swap_2"]:
                    e = pick_valid(od.get(k,[]))
                    if e: plist.append(get_embedding(brain, memory, e)[0])
                pos_per_anchor_bh.append(plist)

            if not anchor_embs_bh or not all_neg_pool:
                skipped_batches += 1
                continue

            neg_pool_t = torch.stack(all_neg_pool)
            ha, hp, hn = mine_batch_hard(
                torch.stack(anchor_embs_bh), anchor_concepts_bh,
                pos_per_anchor_bh, anchor_concepts_bh,
                neg_pool_t, all_neg_pool_concepts, MARGIN
            )
            if ha is None:
                skipped_batches += 1
                continue
            d_ap_h = (ha - hp).pow(2).sum(dim=-1)
            d_an_h = (ha - hn).pow(2).sum(dim=-1)
            raw = d_ap_h - d_an_h + MARGIN
            loss = torch.clamp(raw, min=0).mean()

        # Gradient norm (before clipping)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(brain.parameters()) + list(memory.parameters()), max_norm=float('inf'))
        # Now actually clip
        torch.nn.utils.clip_grad_norm_(
            list(brain.parameters()) + list(memory.parameters()), max_norm=1.0)
        optimizer.step()
        if epoch >= warmup_epochs:
            scheduler.step()

        # Track loss zero
        if loss.item() < 0.001:
            loss_zero_streak += 1
            if loss_zero_streak >= 5 and loss_zero_epoch is None:
                loss_zero_epoch = epoch - 4
        else:
            loss_zero_streak = 0

        # ── Logging ──
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            brain.eval()
            train_u, train_summary, _ = eval_all(brain, memory, all_ext, level_map)
            log_entry = {
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "active_fraction": round(active_frac, 4),
                "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 6),
                "training_U": round(train_u, 4),
                "skipped_batches": skipped_batches,
            }
            for lv in ["physical", "compound", "abstract"]:
                if lv in train_summary:
                    log_entry[f"U_{lv}"] = round(train_summary[lv]["U"], 4)
                    log_entry[f"Uc_{lv}"] = round(train_summary[lv]["U_c"], 4)
                    log_entry[f"alpha_{lv}"] = round(train_summary[lv]["alpha"], 4)

            # Held-out eval every 50 epochs
            if epoch % 50 == 0 or epoch == n_epochs - 1:
                ho_u, ho_uc, ho_delta, _ = eval_heldout(brain, memory, ho_ext, t_brain_perc)
                log_entry["heldout_U"] = round(ho_u, 4)
                log_entry["heldout_Uc"] = round(ho_uc, 4)
                log_entry["heldout_delta"] = round(ho_delta, 4)
                if ho_u > best_ho_u:
                    best_ho_u = ho_u
                    best_state = copy.deepcopy(brain.state_dict())
                    best_mem_state = copy.deepcopy(memory.state_dict())

            logs.append(log_entry)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Ep {epoch:4d}: loss={loss.item():.5f} active={active_frac:.3f} "
                  f"gnorm={log_entry['grad_norm']:.4f} U={train_u:.3f} "
                  f"a_phys={log_entry.get('alpha_physical','?'):.3f} "
                  f"a_abs={log_entry.get('alpha_abstract','?'):.3f} "
                  f"{'ho_U='+str(log_entry.get('heldout_U',''))+'  ' if 'heldout_U' in log_entry else ''}"
                  f"skip={skipped_batches} lr={lr_now:.6f}")

            brain.train()

            # Batch-hard collapse check
            if strategy == "batchhard" and train_u < 0.5:
                print(f"  COLLAPSE DETECTED at epoch {epoch}. Halting.")
                break

    log_entry_final = {"loss_zero_epoch": loss_zero_epoch, "total_skipped": skipped_batches,
                       "total_batches": total_batches}
    logs.append(log_entry_final)

    # Save checkpoints
    ckpt_dir = f'checkpoints/phase5a_{strategy}'
    torch.save({'brain': best_state, 'memory': best_mem_state}, f'{ckpt_dir}/best.pt')
    torch.save({'brain': brain.state_dict(), 'memory': memory.state_dict()}, f'{ckpt_dir}/final.pt')

    return logs, best_state, best_mem_state


# ════════════════════════════════════════════════════════════════
print("="*70)
print("PHASE 5A: TRAINING SIGNAL QUALITY EXPERIMENT")
print("="*70)

# Load all data
print("\nLoading data...")
ext_l1 = run_extraction(json.load(open('config/objects/level1_physical.json')), num_runs=3)
ext_l2 = run_extraction(json.load(open('config/objects/level2_compound.json')), num_runs=3, compound=True)
ext_l3 = run_extraction(json.load(open('config/objects/level3_abstract.json')), num_runs=3, abstract=True)
all_ext = {**ext_l1, **ext_l2, **ext_l3}

l1_objs = [k for k in ext_l1 if sum(1 for v in ext_l1[k].values() for r in v if 'error' not in r) >= 5]
l2_objs = [k for k in ext_l2 if sum(1 for v in ext_l2[k].values() for r in v if 'error' not in r) >= 5]
l3_objs = [k for k in ext_l3 if sum(1 for v in ext_l3[k].values() for r in v if 'error' not in r) >= 5]
print(f"  {len(l1_objs)} physical, {len(l2_objs)} compound, {len(l3_objs)} abstract")

level_map = {}
for o in l1_objs: level_map[o] = "physical"
for o in l2_objs: level_map[o] = "compound"
for o in l3_objs: level_map[o] = "abstract"

# Held-out concepts from Phase 4
with open('config/objects/phase4_test_a.json') as f:
    ho_cfg = json.load(f)
ho_ext = {}
for gn, gd in ho_cfg["groups"].items():
    compound = "compound" in gn
    abstract = "abstract" in gn
    fake = {"groups": {gn: gd}}
    ext = run_extraction(fake, num_runs=3, compound=compound, abstract=abstract)
    for k, v in ext.items():
        if sum(1 for vv in v.values() for r in vv if 'error' not in r) >= 3:
            ho_ext[k] = v
print(f"  {len(ho_ext)} held-out concepts")

# Bridge config (from Stage 3)
l3_cfg = json.load(open('config/objects/level3_abstract.json'))
bridge_config = {}
for gn, gd in l3_cfg["groups"].items():
    for on, od in gd["objects"].items():
        if "ground" in od and on in l3_objs:
            bridge_config[on] = {
                "ground": od["ground"],
                "unrelated_physicals": [o for o in l1_objs if o != od["ground"]][:4],
                "unrelated_abstracts": [o for o in l3_objs if o != on and o not in bridge_config][:4]
            }

# Verify Stage 3 checkpoint
print("\nVerifying Stage 3 checkpoint...")
s3_brain, s3_mem = load_stage3()
s3_brain.eval()
s3_u, s3_summary, _ = eval_all(s3_brain, s3_mem, all_ext, level_map)
print(f"  Stage 3 U: {s3_u:.4f} (expected ~0.819)")
for lv in ["physical","compound","abstract"]:
    if lv in s3_summary:
        print(f"    {lv}: U={s3_summary[lv]['U']:.3f} Uc={s3_summary[lv]['U_c']:.3f} alpha={s3_summary[lv]['alpha']:.3f}")

# For perceptual baseline
t_brain_perc = copy.deepcopy(s3_brain)

# ── Run 1: Semi-Hard ──
brain_sh, mem_sh = load_stage3()
sh_logs, sh_best, sh_best_mem = train_run(
    "semihard", brain_sh, mem_sh, all_ext, level_map, ho_ext, t_brain_perc,
    l3_objs, l2_objs, l1_objs, bridge_config, n_epochs=500)

with open('results/phase5a/semihard_training_log.json', 'w') as f:
    json.dump(sh_logs, f, indent=2, default=str)

# ── Run 2: Batch-Hard ──
brain_bh, mem_bh = load_stage3()
bh_logs, bh_best, bh_best_mem = train_run(
    "batchhard", brain_bh, mem_bh, all_ext, level_map, ho_ext, t_brain_perc,
    l3_objs, l2_objs, l1_objs, bridge_config, n_epochs=500)

with open('results/phase5a/batchhard_training_log.json', 'w') as f:
    json.dump(bh_logs, f, indent=2, default=str)

# ════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 5A DIAGNOSTICS")
print("="*70)

# Load best states for eval
brain_sh_eval = CausalGraphBrain(embed_dim=384, hidden_dim=128, n_heads=8, n_iterations=1, alpha_res=0.2)
mem_sh_eval = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
brain_sh_eval.load_state_dict(sh_best)
mem_sh_eval.load_state_dict(sh_best_mem)

brain_bh_eval = CausalGraphBrain(embed_dim=384, hidden_dim=128, n_heads=8, n_iterations=1, alpha_res=0.2)
mem_bh_eval = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
brain_bh_eval.load_state_dict(bh_best)
mem_bh_eval.load_state_dict(bh_best_mem)

# Diagnostic 5A-1: Head-to-head
print("\nDIAGNOSTIC 5A-1: Head-to-head comparison")

_, sh_sum, _ = eval_all(brain_sh_eval, mem_sh_eval, all_ext, level_map)
_, bh_sum, _ = eval_all(brain_bh_eval, mem_bh_eval, all_ext, level_map)

sh_train_u = np.mean([sh_sum[l]["U"] for l in sh_sum])
bh_train_u = np.mean([bh_sum[l]["U"] for l in bh_sum])

sh_ho = eval_heldout(brain_sh_eval, mem_sh_eval, ho_ext, t_brain_perc)
bh_ho = eval_heldout(brain_bh_eval, mem_bh_eval, ho_ext, t_brain_perc)

# Get saturation epochs
sh_sat = [l for l in sh_logs if isinstance(l, dict) and "loss_zero_epoch" in l]
bh_sat = [l for l in bh_logs if isinstance(l, dict) and "loss_zero_epoch" in l]
sh_zero = sh_sat[0]["loss_zero_epoch"] if sh_sat else "never"
bh_zero = bh_sat[0]["loss_zero_epoch"] if bh_sat else "never"

# Get grad norms and active fractions at specific epochs
def get_log_at(logs, ep):
    for l in logs:
        if isinstance(l, dict) and l.get("epoch") == ep: return l
    return {}

sh_50 = get_log_at(sh_logs, 50)
bh_50 = get_log_at(bh_logs, 50)
sh_500 = get_log_at(sh_logs, 499) or get_log_at(sh_logs, 490)
bh_500 = get_log_at(bh_logs, 499) or get_log_at(bh_logs, 490)

print(f"\n{'Metric':<30} {'Easy (S3)':>12} {'Semi-Hard':>12} {'Batch-Hard':>12}")
print("─"*70)
print(f"{'Training U':<30} {'0.819':>12} {sh_train_u:>12.3f} {bh_train_u:>12.3f}")
for lv in ["physical","compound","abstract"]:
    print(f"{'  U_c ('+lv+')':<30} {'—':>12} {sh_sum.get(lv,{}).get('U_c',0):>12.3f} {bh_sum.get(lv,{}).get('U_c',0):>12.3f}")
    print(f"{'  alpha ('+lv+')':<30} {'—':>12} {sh_sum.get(lv,{}).get('alpha',0):>12.3f} {bh_sum.get(lv,{}).get('alpha',0):>12.3f}")
print(f"{'Loss zero epoch':<30} {'~50-90':>12} {str(sh_zero):>12} {str(bh_zero):>12}")
print(f"{'Active frac (ep 50)':<30} {'~0%':>12} {sh_50.get('active_fraction','?'):>12} {bh_50.get('active_fraction','?'):>12}")
print(f"{'Grad norm (ep 50)':<30} {'~0':>12} {sh_50.get('grad_norm','?'):>12} {bh_50.get('grad_norm','?'):>12}")
print(f"{'Grad norm (ep 500)':<30} {'~0':>12} {sh_500.get('grad_norm','?'):>12} {bh_500.get('grad_norm','?'):>12}")
print(f"{'Held-out U':<30} {'0.758':>12} {sh_ho[0]:>12.3f} {bh_ho[0]:>12.3f}")
print(f"{'Held-out U_c':<30} {'0.902':>12} {sh_ho[1]:>12.3f} {bh_ho[1]:>12.3f}")
print(f"{'Held-out delta_brain':<30} {'-0.041':>12} {sh_ho[2]:>+12.3f} {bh_ho[2]:>+12.3f}")

# Diagnostic 5A-2: Per-concept held-out delta
print(f"\nDIAGNOSTIC 5A-2: Per-concept held-out delta_brain")
print(f"{'Concept':<22} {'Easy':>8} {'Semi-Hard':>10} {'Batch-Hard':>11}")
print("─"*55)
easy_deltas = {
    "pendulum": -0.088, "soap_bubble": -0.015, "anvil": -0.025,
    "snowball": -0.043, "spinning_top": -0.054, "mousetrap": -0.029,
    "pressure_cooker": -0.022, "turbine": 0.024, "cultural_erosion": -0.091,
    "institutional_momentum": -0.071
}
for name in sorted(ho_ext):
    ed = easy_deltas.get(name, 0)
    sd = sh_ho[3].get(name, {}).get("delta_brain", 0)
    bd = bh_ho[3].get(name, {}).get("delta_brain", 0)
    print(f"  {name:<20} {ed:>+8.3f} {sd:>+10.3f} {bd:>+11.3f}")

# Diagnostic 5A-3: Alpha shift
print(f"\nDIAGNOSTIC 5A-3: Alpha shift")
print(f"{'Strategy':<15} {'α Phys':>8} {'α Comp':>8} {'α Abs':>8} {'Gradient':>10}")
print("─"*55)
print(f"{'Easy':<15} {'0.708':>8} {'0.707':>8} {'0.669':>8} {'0.039':>10}")
sh_ap = sh_sum.get("physical",{}).get("alpha",0); sh_ac = sh_sum.get("compound",{}).get("alpha",0); sh_aa = sh_sum.get("abstract",{}).get("alpha",0)
bh_ap = bh_sum.get("physical",{}).get("alpha",0); bh_ac = bh_sum.get("compound",{}).get("alpha",0); bh_aa = bh_sum.get("abstract",{}).get("alpha",0)
print(f"{'Semi-Hard':<15} {sh_ap:>8.3f} {sh_ac:>8.3f} {sh_aa:>8.3f} {sh_ap-sh_aa:>10.3f}")
print(f"{'Batch-Hard':<15} {bh_ap:>8.3f} {bh_ac:>8.3f} {bh_aa:>8.3f} {bh_ap-bh_aa:>10.3f}")

# Diagnostic 5A-6: Forgetting
print(f"\nDIAGNOSTIC 5A-6: Forgetting check")
print(f"{'Level':<12} {'Stage3':>8} {'Semi-Hard':>10} {'Batch-Hard':>11}")
print("─"*45)
for lv in ["physical","compound","abstract"]:
    s3v = s3_summary.get(lv,{}).get("U",0)
    shv = sh_sum.get(lv,{}).get("U",0)
    bhv = bh_sum.get(lv,{}).get("U",0)
    print(f"  {lv:<10} {s3v:>8.3f} {shv:>10.3f} {bhv:>11.3f}")

# Diagnostic 5A-7: Alpha at epoch 50 vs 500
print(f"\nDIAGNOSTIC 5A-7: Alpha abstract at epoch 50 vs 500")
sh_a50 = sh_50.get("alpha_abstract", "?")
sh_a500 = sh_500.get("alpha_abstract", "?")
bh_a50 = bh_50.get("alpha_abstract", "?")
bh_a500 = bh_500.get("alpha_abstract", "?")
print(f"  Semi-Hard:  ep50={sh_a50}  ep500={sh_a500}  {'Still changing' if isinstance(sh_a50,float) and isinstance(sh_a500,float) and abs(sh_a50-sh_a500)>0.01 else 'Plateaued'}")
print(f"  Batch-Hard: ep50={bh_a50}  ep500={bh_a500}  {'Still changing' if isinstance(bh_a50,float) and isinstance(bh_a500,float) and abs(bh_a50-bh_a500)>0.01 else 'Plateaued'}")

# ── Success criteria ──
print(f"\n{'='*70}")
print("SUCCESS CRITERIA")
print("─"*70)
c1 = sh_ho[2] > 0 or bh_ho[2] > 0
c2_val = min(sh_aa, bh_aa)
c2 = (0.669 - c2_val) >= 0.03
sh_sat_ep = sh_zero if sh_zero != "never" else 500
bh_sat_ep = bh_zero if bh_zero != "never" else 500
easy_sat = 70
c3 = (isinstance(sh_sat_ep,int) and sh_sat_ep > easy_sat + 100) or (isinstance(bh_sat_ep,int) and bh_sat_ep > easy_sat + 100) or sh_zero == "never" or bh_zero == "never"
s3_phys = s3_summary.get("physical",{}).get("U",0)
s3_comp = s3_summary.get("compound",{}).get("U",0)
s3_abs = s3_summary.get("abstract",{}).get("U",0)
sh_forg = max(abs(sh_sum.get("physical",{}).get("U",0)-s3_phys), abs(sh_sum.get("compound",{}).get("U",0)-s3_comp), abs(sh_sum.get("abstract",{}).get("U",0)-s3_abs))
bh_forg = max(abs(bh_sum.get("physical",{}).get("U",0)-s3_phys), abs(bh_sum.get("compound",{}).get("U",0)-s3_comp), abs(bh_sum.get("abstract",{}).get("U",0)-s3_abs))
c4 = sh_forg < 0.05 and bh_forg < 0.05
sh_af200 = get_log_at(sh_logs, 200).get("active_fraction", 0)
bh_af200 = get_log_at(bh_logs, 200).get("active_fraction", 0)
c5 = sh_af200 > 0.10 or bh_af200 > 0.10

print(f"  [{'PASS' if c1 else 'FAIL'}] delta_brain > 0: SH={sh_ho[2]:+.3f} BH={bh_ho[2]:+.3f}")
print(f"  [{'PASS' if c2 else 'FAIL'}] α abstract decrease >= 0.03: SH={sh_aa:.3f} BH={bh_aa:.3f} (easy=0.669)")
print(f"  [{'PASS' if c3 else 'FAIL'}] Saturation +100 epochs: SH={sh_zero} BH={bh_zero} (easy=~70)")
print(f"  [{'PASS' if c4 else 'FAIL'}] No forgetting: SH={sh_forg:.3f} BH={bh_forg:.3f}")
print(f"  [{'PASS' if c5 else 'FAIL'}] Active fraction >10% at ep200: SH={sh_af200:.3f} BH={bh_af200:.3f}")

total = sum([c1,c2,c3,c4,c5])
if total >= 4:
    verdict = "STRONG SUCCESS — training signal was the bottleneck"
elif total >= 2:
    verdict = "PARTIAL — signal helps but capacity also matters"
else:
    verdict = "FAILURE — model capacity is the bottleneck"

print(f"\n  {total}/5 criteria met")
print(f"  VERDICT: {verdict}")
print("="*70)

# Save summary
summary = {
    "verdict": verdict, "criteria_met": total,
    "semihard": {"train_U": round(sh_train_u,4), "heldout_U": round(sh_ho[0],4),
                 "heldout_delta": round(sh_ho[2],4), "loss_zero": sh_zero},
    "batchhard": {"train_U": round(bh_train_u,4), "heldout_U": round(bh_ho[0],4),
                  "heldout_delta": round(bh_ho[2],4), "loss_zero": bh_zero}
}
with open('results/phase5a/phase5a_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nResults saved to results/phase5a/")
