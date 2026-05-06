"""
Phase D evaluation pipeline.

Three tiers per system:
- Tier 1 measures DIRECTIONAL SIMILARITY INFLATION against pre-training baseline.
  This is NOT analogy detection — it asks "did training move sim up." Matches Phase 5A's
  6/13 protocol for comparability to the project's empirical foundation.
- Tier 2 measures SAME-MECHANISM DISCRIMINATION: can the model rank a rubric-validated
  analogy higher than a same-mechanism rubric-rejected analogy?
- Tier 3 measures CROSS-MECHANISM DISCRIMINATION: can the model rank the claimed mechanism
  higher than a different valid mechanism (Pool A)?

Tiers measure different things and are reported alongside each other.

PER-SYSTEM TIER 1 BASELINE ROUTING (each system compared against the right zero-point
for its own input pipeline):

- frozen_st (raw_minilm, raw_mpnet)  → Tier 1 N/A. The system IS the baseline; recorded
                                       in output as null with note.
- ft_st on minilm  (curr × 5, nocurr × 3) → live frozen MiniLM cosine on the same
                                       description text used at eval time. Win iff the
                                       fine-tuned MiniLM moved sim above its own frozen base.
- ft_st on mpnet   (curr × 3)        → same idea, with frozen mpnet on the description.
- projection_only  (× 10)            → recorded raw_minilm_sim from benchmark_v2 metadata.
                                       This baseline is itself the mean-of-Haiku-node MiniLM
                                       cosine, which exactly matches projection-only's input
                                       (it consumes the same node-mean before projecting).
- brain_phase5a                      → recorded raw_minilm_sim. Matches the Phase 5A
                                       brain_a0 > raw_minilm 6/13 protocol used during
                                       training.

The recorded raw_minilm_sim and live FT-encode-of-description sims are NOT
interchangeable — they come from different input pipelines (node-mean vs sentence-level)
and produce different cosine distributions. Routing per system above keeps each Tier 1
comparison on a like-for-like axis.

Inputs:
- results/phase_a_prime/benchmark_v2.json
- results/phase_d/distractors_tier2.json
- results/phase_d/distractors_tier3.json
- cache/all_extractions_phase_d.json

Output:
- results/phase_d/eval_results.json

Determinism: torch + numpy + random seeded to 42 at startup. Each system loaded,
evaluated, and freed sequentially to keep peak memory under one transformer.
"""

import argparse
import copy
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


# ───────────────────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_PATH         = REPO_ROOT / "results/phase_a_prime/benchmark_v2.json"
DISTRACTORS_TIER2_PATH = REPO_ROOT / "results/phase_d/distractors_tier2.json"
DISTRACTORS_TIER3_PATH = REPO_ROOT / "results/phase_d/distractors_tier3.json"
EXTRACTIONS_PATH       = REPO_ROOT / "cache/all_extractions_phase_d.json"
OUTPUT_PATH            = REPO_ROOT / "results/phase_d/eval_results.json"

EXPECTED_EXTRACTIONS = 142
EXPECTED_BENCHMARK_PAIRS = 28


# ───────────────────────────────────────────────────────────────────────────
# System configuration
# ───────────────────────────────────────────────────────────────────────────
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MPNET  = "sentence-transformers/all-mpnet-base-v2"

FT_MINILM_CURR_SEEDS    = [42, 123, 456, 789, 1024]
FT_MINILM_NOCURR_SEEDS  = [42, 123, 456]
FT_MPNET_CURR_SEEDS     = [42, 123, 456]
PROJECTION_SEEDS        = [42, 123, 456, 789, 1024]


def _ft_minilm_curr(seed):
    return {
        "name": f"ft_minilm_curriculum_seed_{seed}",
        "type": "ft_st",
        "model": MINILM,
        "checkpoint": f"checkpoints/phase_b/ft_minilm_curriculum/seed_{seed}/stage_3/best.pt",
    }


def _ft_minilm_nocurr(seed):
    return {
        "name": f"ft_minilm_nocurriculum_seed_{seed}",
        "type": "ft_st",
        "model": MINILM,
        "checkpoint": f"checkpoints/phase_b/ft_minilm_nocurriculum/seed_{seed}/best.pt",
    }


def _ft_mpnet_curr(seed):
    return {
        "name": f"ft_mpnet_curriculum_seed_{seed}",
        "type": "ft_st",
        "model": MPNET,
        "checkpoint": f"checkpoints/phase_b/ft_mpnet_curriculum/seed_{seed}/stage_3/best.pt",
    }


def _projection(seed, which):
    return {
        "name": f"projection_only_{which}_seed_{seed}",
        "type": "projection_only",
        "checkpoint": f"checkpoints/phase_b/projection_only/seed_{seed}/stage_3/{which}.pt",
    }


SYSTEMS = (
    [
        {"name": "raw_minilm", "type": "frozen_st", "model": MINILM, "checkpoint": None},
        {"name": "raw_mpnet",  "type": "frozen_st", "model": MPNET,  "checkpoint": None},
    ]
    + [_ft_minilm_curr(s)   for s in FT_MINILM_CURR_SEEDS]
    + [_ft_minilm_nocurr(s) for s in FT_MINILM_NOCURR_SEEDS]
    + [_ft_mpnet_curr(s)    for s in FT_MPNET_CURR_SEEDS]
    + [_projection(s, w)    for s in PROJECTION_SEEDS for w in ("best", "final")]
    + [{"name": "brain_phase5a", "type": "brain",
        "checkpoint": "checkpoints/phase5a_batchhard/best.pt"}]
)
# Total: 2 + 5 + 3 + 3 + 10 + 1 = 24 systems


# ───────────────────────────────────────────────────────────────────────────
# Determinism
# ───────────────────────────────────────────────────────────────────────────
def setup_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ───────────────────────────────────────────────────────────────────────────
# Baseline routing for Tier 1
# ───────────────────────────────────────────────────────────────────────────
# baseline_source values, used in per-pair output rows + diagnostics
BASELINE_SOURCE_NA              = "n/a"
BASELINE_SOURCE_LIVE_MINILM     = "live_frozen_minilm"
BASELINE_SOURCE_LIVE_MPNET      = "live_frozen_mpnet"
BASELINE_SOURCE_RECORDED_MINILM = "recorded_raw_minilm_sim"


def tier1_baseline_for(system, pair, live_baselines):
    """
    Returns (baseline_value, baseline_source) for this system + pair.

    baseline_value is None for frozen baselines (Tier 1 undefined).
    """
    t = system["type"]
    if t == "frozen_st":
        return None, BASELINE_SOURCE_NA
    if t == "ft_st" and system.get("model") == MINILM:
        return live_baselines[pair["pair_id"]]["live_frozen_minilm_sim"], \
               BASELINE_SOURCE_LIVE_MINILM
    if t == "ft_st" and system.get("model") == MPNET:
        return live_baselines[pair["pair_id"]]["live_frozen_mpnet_sim"], \
               BASELINE_SOURCE_LIVE_MPNET
    if t in ("projection_only", "brain"):
        return float(pair["raw_minilm_sim"]), BASELINE_SOURCE_RECORDED_MINILM
    raise ValueError(f"unknown system type: {t}")


# ───────────────────────────────────────────────────────────────────────────
# Encoders — one per system type
# ───────────────────────────────────────────────────────────────────────────
class STEncoder:
    """Wraps a SentenceTransformer (frozen or fine-tuned). Encodes raw text."""

    def __init__(self, model_name, ckpt_path=None, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, text):
        emb = self.model.encode([text], convert_to_tensor=True,
                                show_progress_bar=False, device=self.device)[0]
        return F.normalize(emb, p=2, dim=-1)


class ProjectionOnlyEncoder:
    """Wraps a trained ProjectionOnly. Consumes graph extraction (not raw text)."""

    def __init__(self, ckpt_path, extractions, embedder_module):
        from src.baselines.projection_only import ProjectionOnly
        from src.scoring.graph_utils import extraction_to_tensors
        self._ProjectionOnly = ProjectionOnly
        self._ext_to_tensors = extraction_to_tensors
        self.extractions = extractions
        self.embedder_module = embedder_module
        self.model = ProjectionOnly(embed_dim=384, hidden_dim=128)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        self.model.load_state_dict(ckpt["proj_model"])
        self.model.eval()
        self.embed_dim = 128

    @torch.no_grad()
    def encode(self, concept_name):
        ext = self.extractions[concept_name]
        nf, _, _, _ = self._ext_to_tensors(ext, self.embedder_module)
        return self.model(nf)  # already L2-normalized in ProjectionOnly.forward


class BrainEncoder:
    """
    Wraps the Phase 5A canonical Brain. Restores memory state before each forward
    so concept order does not affect outputs (memory.write is in-place).
    """

    def __init__(self, ckpt_path, extractions, embedder_module):
        from src.brain.gat import CausalGraphBrain
        from src.brain.memory import CausalMemory
        from src.scoring.graph_utils import extraction_to_tensors
        self._ext_to_tensors = extraction_to_tensors
        self.extractions = extractions
        self.embedder_module = embedder_module

        self.brain = CausalGraphBrain(
            embed_dim=384, hidden_dim=128, n_heads=8,
            num_layers=4, n_iterations=1, alpha_res=0.2,
            alpha_mlp_hidden=64,
        )
        self.memory = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        # Phase 5A checkpoints use legacy gat1/gat2/... keys
        self.brain.load_checkpoint_compat(ckpt["brain"])
        self.memory.load_state_dict(ckpt["memory"])
        self.brain.eval()
        self.memory.eval()
        # Snapshot memory state to restore between encodings (memory.write mutates in-place)
        self._mem_snapshot = copy.deepcopy(self.memory.state_dict())
        self.embed_dim = 128

    @torch.no_grad()
    def encode(self, concept_name):
        # Restore memory to checkpoint state so per-concept encoding is order-independent
        self.memory.load_state_dict(self._mem_snapshot)
        ext = self.extractions[concept_name]
        nf, ei, em, _ = self._ext_to_tensors(ext, self.embedder_module)
        emb, _alpha, _x, _attn = self.brain(nf, ei, em, self.memory)
        return emb  # already L2-normalized in CausalGraphBrain.forward


# ───────────────────────────────────────────────────────────────────────────
# Live frozen baselines (used by FT systems for Tier 1)
# ───────────────────────────────────────────────────────────────────────────
def compute_live_frozen_baselines(benchmark_pairs, device="cpu"):
    """
    For each benchmark pair, compute cos(frozen_minilm.encode(desc_a),
    frozen_minilm.encode(desc_b)) and the same for frozen mpnet. These match
    the encode-the-description pipeline that FT systems use, so they're the
    correct Tier 1 baselines for FT-vs-frozen comparisons.

    Returns {pair_id: {"live_frozen_minilm_sim": float, "live_frozen_mpnet_sim": float}}.
    Loads each frozen model, computes 28 cosines, frees.
    """
    print("\nComputing live frozen-base Tier 1 baselines (MiniLM + mpnet)...",
          file=sys.stderr)

    # MiniLM pass
    minilm = STEncoder(MINILM, ckpt_path=None, device=device)
    minilm_per_pair = {}
    for p in benchmark_pairs:
        ea = minilm.encode(p["concept_a_description"])
        eb = minilm.encode(p["concept_b_description"])
        minilm_per_pair[p["pair_id"]] = cosine(ea, eb)
    del minilm
    gc.collect()

    # mpnet pass
    mpnet = STEncoder(MPNET, ckpt_path=None, device=device)
    mpnet_per_pair = {}
    for p in benchmark_pairs:
        ea = mpnet.encode(p["concept_a_description"])
        eb = mpnet.encode(p["concept_b_description"])
        mpnet_per_pair[p["pair_id"]] = cosine(ea, eb)
    del mpnet
    gc.collect()

    out = {}
    for p in benchmark_pairs:
        pid = p["pair_id"]
        out[pid] = {
            "live_frozen_minilm_sim": minilm_per_pair[pid],
            "live_frozen_mpnet_sim":  mpnet_per_pair[pid],
        }
    print(f"  Computed live baselines for {len(out)} pairs.", file=sys.stderr)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Encoding façade — uniform interface across systems
# ───────────────────────────────────────────────────────────────────────────
def make_encoder(system, extractions, embedder_module, device="cpu"):
    t = system["type"]
    if t == "frozen_st":
        return STEncoder(system["model"], ckpt_path=None, device=device)
    if t == "ft_st":
        return STEncoder(system["model"], ckpt_path=system["checkpoint"], device=device)
    if t == "projection_only":
        return ProjectionOnlyEncoder(system["checkpoint"], extractions, embedder_module)
    if t == "brain":
        return BrainEncoder(system["checkpoint"], extractions, embedder_module)
    raise ValueError(f"unknown system type: {t}")


def encode_concept(encoder, system_type, concept_name, concept_description, cache):
    """
    Returns L2-normalized embedding (1D tensor in system's native dim).
    Caches per-(system, concept_name) to avoid re-encoding.
    """
    if concept_name in cache:
        return cache[concept_name]
    if system_type in ("frozen_st", "ft_st"):
        emb = encoder.encode(concept_description)
    else:  # projection_only, brain
        emb = encoder.encode(concept_name)
    cache[concept_name] = emb
    return emb


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ───────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ───────────────────────────────────────────────────────────────────────────
def preflight(systems, extractions, benchmark_pairs, t2_map, t3_map):
    """Verify all inputs. Fail loudly. Return a serializable summary dict."""
    log = []
    def emit(msg):
        log.append(msg)
        print(msg, file=sys.stderr)

    emit("=" * 78)
    emit("PRE-FLIGHT CHECKS")
    emit("=" * 78)

    # 1) Checkpoint paths
    missing = []
    for s in systems:
        ckpt = s.get("checkpoint")
        if ckpt is None:
            continue
        p = REPO_ROOT / ckpt
        if not p.exists():
            missing.append((s["name"], str(p)))
    if missing:
        emit(f"FAIL: {len(missing)} missing checkpoint(s):")
        for name, path in missing:
            emit(f"  {name}: {path}")
        sys.exit(1)
    emit(f"OK: all {sum(1 for s in systems if s.get('checkpoint'))} checkpoint paths exist.")

    # 2) Extractions
    if len(extractions) != EXPECTED_EXTRACTIONS:
        emit(f"FAIL: extractions has {len(extractions)} entries, expected {EXPECTED_EXTRACTIONS}.")
        sys.exit(1)
    emit(f"OK: extractions has {len(extractions)} entries.")

    # 3) Benchmark / distractor mappings
    if len(benchmark_pairs) != EXPECTED_BENCHMARK_PAIRS:
        emit(f"FAIL: benchmark has {len(benchmark_pairs)} pairs, expected {EXPECTED_BENCHMARK_PAIRS}.")
        sys.exit(1)
    pair_ids = [p["pair_id"] for p in benchmark_pairs]
    if len(set(pair_ids)) != len(pair_ids):
        emit("FAIL: benchmark contains duplicate pair_ids.")
        sys.exit(1)
    bad = []
    for pid in pair_ids:
        if pid not in t2_map:
            bad.append(("tier2", pid))
        if pid not in t3_map:
            bad.append(("tier3", pid))
    if bad:
        emit(f"FAIL: {len(bad)} benchmark pairs lack a distractor:")
        for tier, pid in bad:
            emit(f"  {tier}: {pid}")
        sys.exit(1)
    emit(f"OK: all {len(pair_ids)} benchmark pair_ids have distractors in both tiers.")

    # 4) Print 28 distractor mappings
    emit("\nDistractor mappings (benchmark → tier2 distractor / tier3 distractor):")
    emit(f"  {'pair_id':<28} {'mech':<18} {'tier2 distractor B':<32} {'tier3 distractor B':<32}")
    emit("  " + "─" * 110)
    for p in benchmark_pairs:
        pid = p["pair_id"]
        d2b = t2_map[pid]["distractor_concept_b"]["name"]
        d3b = t3_map[pid]["distractor_concept_b"]["name"]
        emit(f"  {pid:<28} {p['claimed_mechanism']:<18} {d2b:<32} {d3b:<32}")

    # 5) Extraction coverage for Brain/projection systems
    needed_concept_names = set()
    for p in benchmark_pairs:
        needed_concept_names.add(p["concept_a_name"])
        needed_concept_names.add(p["concept_b_name"])
    for pid in pair_ids:
        needed_concept_names.add(t2_map[pid]["distractor_concept_b"]["name"])
        needed_concept_names.add(t3_map[pid]["distractor_concept_b"]["name"])
    missing_ext = sorted(n for n in needed_concept_names if n not in extractions)
    if missing_ext:
        emit(f"\nFAIL: {len(missing_ext)} concepts needed by Brain/projection lack extractions:")
        for n in missing_ext[:20]:
            emit(f"  {n}")
        sys.exit(1)
    emit(f"\nOK: all {len(needed_concept_names)} concepts referenced by benchmark+distractors "
         f"have extractions.")

    return {
        "n_systems": len(systems),
        "n_checkpoints_verified": sum(1 for s in systems if s.get("checkpoint")),
        "n_extractions": len(extractions),
        "n_benchmark_pairs": len(benchmark_pairs),
        "n_concepts_needed": len(needed_concept_names),
    }


def smoke_test_encoder(system, encoder, extractions, benchmark_pairs, t2_map, t3_map):
    """
    For Brain: smoke-test 3 concepts (1 benchmark, 1 tier2 distractor, 1 tier3 distractor),
    verify shape (128,) and L2 norm ≈ 1.0.
    For non-Brain: encode 1 sample concept and report shape, L2 norm, and first 3 dims.
    Returns a row dict for the summary table; raises on shape/norm failure.
    """
    # Pick concept(s)
    bm0 = benchmark_pairs[0]
    pid0 = bm0["pair_id"]
    bm_name = bm0["concept_a_name"]
    bm_desc = bm0["concept_a_description"]
    t2_name = t2_map[pid0]["distractor_concept_b"]["name"]
    t2_desc = t2_map[pid0]["distractor_concept_b"]["description"]
    t3_name = t3_map[pid0]["distractor_concept_b"]["name"]
    t3_desc = t3_map[pid0]["distractor_concept_b"]["description"]

    cache = {}

    def _enc(name, desc):
        return encode_concept(encoder, system["type"], name, desc, cache)

    if system["type"] == "brain":
        results = []
        for label, name, desc in [
            ("benchmark", bm_name, bm_desc),
            ("tier2_distractor", t2_name, t2_desc),
            ("tier3_distractor", t3_name, t3_desc),
        ]:
            emb = _enc(name, desc)
            shape = tuple(emb.shape)
            l2 = float(emb.norm().item())
            if shape != (128,):
                raise RuntimeError(f"Brain {label}: shape {shape} != (128,)")
            if abs(l2 - 1.0) > 1e-3:
                raise RuntimeError(f"Brain {label}: L2 norm {l2:.4f} != 1.0")
            results.append((label, name, shape, l2))
        # Summary row uses the first concept
        first = results[0]
        return {
            "system": system["name"],
            "embed_dim": encoder.embed_dim,
            "sample_concept": first[1],
            "sample_l2_norm": round(first[3], 5),
            "sample_shape": str(first[2]),
            "first_3_dims": [round(float(x), 5) for x in cache[first[1]][:3].tolist()],
            "brain_extra_smoke": [
                {"label": lab, "concept": nm, "shape": str(sh), "l2_norm": round(l2, 5)}
                for lab, nm, sh, l2 in results
            ],
        }
    else:
        emb = _enc(bm_name, bm_desc)
        shape = tuple(emb.shape)
        l2 = float(emb.norm().item())
        expected = encoder.embed_dim
        if shape != (expected,):
            raise RuntimeError(f"{system['name']}: shape {shape} != ({expected},)")
        if abs(l2 - 1.0) > 1e-3:
            raise RuntimeError(f"{system['name']}: L2 norm {l2:.4f} != 1.0")
        return {
            "system": system["name"],
            "embed_dim": encoder.embed_dim,
            "sample_concept": bm_name,
            "sample_l2_norm": round(l2, 5),
            "sample_shape": str(shape),
            "first_3_dims": [round(float(x), 5) for x in emb[:3].tolist()],
        }


# ───────────────────────────────────────────────────────────────────────────
# Per-system tier evaluation
# ───────────────────────────────────────────────────────────────────────────
def evaluate_system(system, encoder, benchmark_pairs, t2_map, t3_map, live_baselines):
    """
    Returns (per_pair_rows, aggregate_dict). per_pair_rows is a list of dicts
    suitable for dropping straight into the JSON output. aggregate is the
    per-system summary including binomial p-values for tiers 2 and 3.

    Tier 1 baseline is routed per system via tier1_baseline_for(); for frozen
    baselines (raw_minilm, raw_mpnet) the tier1 fields are emitted as null with
    a tier1_note explaining why.
    """
    cache = {}
    rows = []
    is_frozen = system["type"] == "frozen_st"
    tier1_note = ("frozen baseline; Tier 1 undefined (system is its own baseline)"
                  if is_frozen else None)

    for pair in benchmark_pairs:
        pid = pair["pair_id"]
        a_name = pair["concept_a_name"]; a_desc = pair["concept_a_description"]
        b_name = pair["concept_b_name"]; b_desc = pair["concept_b_description"]
        d2 = t2_map[pid]
        d3 = t3_map[pid]
        d2b_name = d2["distractor_concept_b"]["name"]
        d2b_desc = d2["distractor_concept_b"]["description"]
        d3b_name = d3["distractor_concept_b"]["name"]
        d3b_desc = d3["distractor_concept_b"]["description"]

        emb_a       = encode_concept(encoder, system["type"], a_name,   a_desc,   cache)
        emb_b_kept  = encode_concept(encoder, system["type"], b_name,   b_desc,   cache)
        emb_b_rej   = encode_concept(encoder, system["type"], d2b_name, d2b_desc, cache)
        emb_b_poola = encode_concept(encoder, system["type"], d3b_name, d3b_desc, cache)

        sim_kept  = cosine(emb_a, emb_b_kept)
        sim_rej   = cosine(emb_a, emb_b_rej)
        sim_poola = cosine(emb_a, emb_b_poola)

        baseline_value, baseline_source = tier1_baseline_for(system, pair, live_baselines)
        if baseline_value is None:
            t1_win = None
            t1_model_sim = None
            t1_baseline_out = None
        else:
            t1_win = sim_kept > baseline_value
            t1_model_sim = round(sim_kept, 6)
            t1_baseline_out = round(float(baseline_value), 6)

        row = {
            "system": system["name"],
            "pair_id": pid,
            "mechanism": pair["claimed_mechanism"],
            "baseline_source": baseline_source,
            "tier1_win": t1_win,
            "tier1_model_sim": t1_model_sim,
            "tier1_raw_baseline": t1_baseline_out,
            "tier2_win": sim_kept > sim_rej,
            "tier2_sim_kept": round(sim_kept, 6),
            "tier2_sim_rejected": round(sim_rej, 6),
            "tier2_distractor_pair_id": d2["distractor_pair_id"],
            "tier2_distractor_concept_b": d2b_name,
            "tier3_win": sim_kept > sim_poola,
            "tier3_sim_kept": round(sim_kept, 6),
            "tier3_sim_pool_a": round(sim_poola, 6),
            "tier3_distractor_pair_id": d3["distractor_pair_id"],
            "tier3_distractor_concept_b": d3b_name,
        }
        if tier1_note is not None:
            row["tier1_note"] = tier1_note
        rows.append(row)

    aggregate = aggregate_rows(system["name"], rows, tier1_undefined=is_frozen,
                               tier1_note=tier1_note)
    return rows, aggregate


def _wins_block(wins, n, with_binomial=False):
    block = {"wins": int(wins), "n": int(n),
             "rate": round(wins / n, 6) if n else 0.0}
    if with_binomial and n > 0:
        res = stats.binomtest(int(wins), n, p=0.5, alternative="two-sided")
        block["binomial_p_vs_chance"] = float(res.pvalue)
    return block


def _null_wins_block(note=None):
    block = {"wins": None, "n": None, "rate": None}
    if note is not None:
        block["note"] = note
    return block


def aggregate_rows(system_name, rows, tier1_undefined=False, tier1_note=None):
    n = len(rows)
    t2 = sum(1 for r in rows if r["tier2_win"])
    t3 = sum(1 for r in rows if r["tier3_win"])

    by_mech = {}
    for r in rows:
        m = r["mechanism"]
        by_mech.setdefault(m, {"t1": 0, "t2": 0, "t3": 0, "n": 0})
        by_mech[m]["n"] += 1
        if not tier1_undefined and r["tier1_win"] is True:
            by_mech[m]["t1"] += 1
        by_mech[m]["t2"] += int(r["tier2_win"])
        by_mech[m]["t3"] += int(r["tier3_win"])

    if tier1_undefined:
        t1_block = _null_wins_block(note=tier1_note)
        t1_by_mech = {m: _null_wins_block(note=tier1_note) for m in by_mech}
    else:
        t1 = sum(1 for r in rows if r["tier1_win"])
        t1_block = _wins_block(t1, n)
        t1_by_mech = {m: _wins_block(d["t1"], d["n"]) for m, d in by_mech.items()}

    return {
        "system": system_name,
        "tier1": t1_block,
        "tier2": _wins_block(t2, n, with_binomial=True),
        "tier3": _wins_block(t3, n, with_binomial=True),
        "tier1_by_mechanism": t1_by_mech,
        "tier2_by_mechanism": {m: _wins_block(d["t2"], d["n"], with_binomial=True)
                               for m, d in by_mech.items()},
        "tier3_by_mechanism": {m: _wins_block(d["t3"], d["n"], with_binomial=True)
                               for m, d in by_mech.items()},
    }


# ───────────────────────────────────────────────────────────────────────────
# Paired Wilcoxon: curriculum vs nocurriculum on FT MiniLM
# ───────────────────────────────────────────────────────────────────────────
def per_pair_rate_table(per_pair_rows, system_names, tier_key):
    """
    Build an n_pairs × n_systems matrix of binary tier outcomes (1/0), aggregated
    later into per-pair rates by averaging across the supplied systems.
    Returns ordered list of pair_ids and dict {pair_id: rate}.
    """
    by_pair = {}
    for r in per_pair_rows:
        if r["system"] not in system_names:
            continue
        by_pair.setdefault(r["pair_id"], []).append(int(r[tier_key]))
    pair_ids = sorted(by_pair)
    rates = {pid: float(np.mean(by_pair[pid])) for pid in pair_ids}
    return pair_ids, rates


def paired_wilcoxon_curr_vs_nocurr(per_pair_rows):
    """
    On Tier 2 outcomes: compute per-pair win rates separately for curriculum and
    nocurriculum FT MiniLM systems, then run a paired Wilcoxon signed-rank over
    the 28 paired (curr_rate, nocurr_rate) values.
      - paired:        seeds {42, 123, 456} on each side (3 vs 3)
      - supplementary: all 5 curriculum seeds vs all 3 nocurriculum seeds (still
                       paired by benchmark pair_id)
    """
    curr_paired_names   = [f"ft_minilm_curriculum_seed_{s}"  for s in (42, 123, 456)]
    nocurr_paired_names = [f"ft_minilm_nocurriculum_seed_{s}" for s in (42, 123, 456)]
    curr_all_names      = [f"ft_minilm_curriculum_seed_{s}"  for s in FT_MINILM_CURR_SEEDS]
    nocurr_all_names    = [f"ft_minilm_nocurriculum_seed_{s}" for s in FT_MINILM_NOCURR_SEEDS]

    out = {}
    for label, curr_names, nocurr_names in [
        ("paired_3v3",   curr_paired_names, nocurr_paired_names),
        ("supplementary_5v3", curr_all_names,    nocurr_all_names),
    ]:
        c_pairs, c_rates = per_pair_rate_table(per_pair_rows, curr_names,   "tier2_win")
        n_pairs, n_rates = per_pair_rate_table(per_pair_rows, nocurr_names, "tier2_win")
        common = sorted(set(c_pairs) & set(n_pairs))
        c_arr = np.array([c_rates[p] for p in common], dtype=float)
        n_arr = np.array([n_rates[p] for p in common], dtype=float)
        diffs = c_arr - n_arr

        block = {
            "curr_seeds": curr_names,
            "nocurr_seeds": nocurr_names,
            "n_pairs": int(len(common)),
            "mean_curr_rate":   float(c_arr.mean())   if len(c_arr) else None,
            "mean_nocurr_rate": float(n_arr.mean())   if len(n_arr) else None,
            "mean_diff":        float(diffs.mean())   if len(diffs) else None,
            "median_diff":      float(np.median(diffs)) if len(diffs) else None,
        }
        if np.all(diffs == 0):
            block["wilcoxon_stat"] = None
            block["p_value"] = None
            block["wilcoxon_note"] = "all per-pair differences are zero; Wilcoxon undefined"
        else:
            try:
                wres = stats.wilcoxon(c_arr, n_arr, zero_method="wilcox",
                                      alternative="two-sided")
                block["wilcoxon_stat"] = float(wres.statistic)
                block["p_value"] = float(wres.pvalue)
            except ValueError as e:
                block["wilcoxon_stat"] = None
                block["p_value"] = None
                block["wilcoxon_note"] = f"scipy raised: {e}"
        out[label] = block
    return out


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu",
                   help="Device for SentenceTransformer encoders (default cpu).")
    p.add_argument("--preflight_only", action="store_true",
                   help="Run pre-flight + per-system smoke test, then exit (no full eval).")
    p.add_argument("--output", default=str(OUTPUT_PATH),
                   help=f"Output JSON path (default {OUTPUT_PATH}).")
    return p.parse_args()


def main():
    args = parse_args()
    setup_determinism(42)

    print(f"Loading inputs...", file=sys.stderr)
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    benchmark_pairs = benchmark["pairs"]
    with open(DISTRACTORS_TIER2_PATH) as f:
        d2 = json.load(f)
    with open(DISTRACTORS_TIER3_PATH) as f:
        d3 = json.load(f)
    with open(EXTRACTIONS_PATH) as f:
        extractions = json.load(f)

    t2_map = {e["benchmark_pair_id"]: e for e in d2["entries"]}
    t3_map = {e["benchmark_pair_id"]: e for e in d3["entries"]}

    preflight_summary = preflight(SYSTEMS, extractions, benchmark_pairs, t2_map, t3_map)

    # Live frozen-base Tier 1 baselines for FT systems
    live_baselines = compute_live_frozen_baselines(benchmark_pairs, device=args.device)

    # Lazy embedder import (used by Brain + projection encoders only)
    from src.sensors import embedder as embedder_module

    # ── Single-pass per-system: load → smoke test → evaluate → free ────
    # Single pass keeps peak memory at one encoder. With mpnet (~420MB) being
    # the largest, holding all 24 simultaneously would exceed typical system memory.
    print("\n" + "=" * 78, file=sys.stderr)
    print("PER-SYSTEM SMOKE TEST + EVALUATION (single pass)", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"{'system':<40} {'dim':>5} {'l2':>8} {'sample_concept':<32} first_3", file=sys.stderr)
    print("─" * 110, file=sys.stderr)

    smoke_rows = []
    per_pair_rows = []
    per_system_aggregate = {}

    for system in SYSTEMS:
        name = system["name"]
        t_load0 = time.time()
        encoder = make_encoder(system, extractions, embedder_module, device=args.device)
        load_s = time.time() - t_load0

        # Smoke test
        smoke_row = smoke_test_encoder(system, encoder, extractions,
                                        benchmark_pairs, t2_map, t3_map)
        smoke_rows.append(smoke_row)
        print(f"{smoke_row['system']:<40} {smoke_row['embed_dim']:>5} "
              f"{smoke_row['sample_l2_norm']:>8.5f} "
              f"{smoke_row['sample_concept']:<32} {smoke_row['first_3_dims']}",
              file=sys.stderr)

        if args.preflight_only:
            del encoder
            gc.collect()
            continue

        # Tier evaluation
        t_eval0 = time.time()
        rows, agg = evaluate_system(system, encoder, benchmark_pairs, t2_map, t3_map,
                                     live_baselines)
        eval_s = time.time() - t_eval0

        per_pair_rows.extend(rows)
        per_system_aggregate[name] = agg

        t1_block = agg["tier1"]; t2_block = agg["tier2"]; t3_block = agg["tier3"]
        t1_str = (f"{t1_block['wins']}/{t1_block['n']}"
                  if t1_block["wins"] is not None else "N/A")
        print(f"  → {name}: load={load_s:.1f}s eval={eval_s:.1f}s "
              f"T1={t1_str} "
              f"T2={t2_block['wins']}/{t2_block['n']} (p={t2_block['binomial_p_vs_chance']:.4f}) "
              f"T3={t3_block['wins']}/{t3_block['n']} (p={t3_block['binomial_p_vs_chance']:.4f})",
              file=sys.stderr)

        # Free the encoder before loading the next system
        del encoder
        gc.collect()

    if args.preflight_only:
        print("\n--preflight_only set; exiting after smoke test.", file=sys.stderr)
        return

    # ── Paired comparisons ──────────────────────────────────────────────
    print("\nPaired comparisons (Tier 2, FT MiniLM curriculum vs nocurriculum):",
          file=sys.stderr)
    paired = paired_wilcoxon_curr_vs_nocurr(per_pair_rows)
    for label, block in paired.items():
        wstat = block.get("wilcoxon_stat"); pv = block.get("p_value")
        print(f"  {label}: mean_curr={block['mean_curr_rate']:.4f} "
              f"mean_nocurr={block['mean_nocurr_rate']:.4f} "
              f"mean_diff={block['mean_diff']:+.4f} "
              f"W={wstat} p={pv}", file=sys.stderr)

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 78, file=sys.stderr)
    print("FINAL SUMMARY", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"{'system':<40} {'T1':>14} {'T2':>14} {'T2_p':>10} {'T3':>14} {'T3_p':>10}",
          file=sys.stderr)
    print("─" * 110, file=sys.stderr)
    for name, agg in per_system_aggregate.items():
        t1 = agg["tier1"]; t2 = agg["tier2"]; t3 = agg["tier3"]
        if t1["wins"] is None:
            t1_str = "N/A"
        else:
            t1_str = f"{t1['wins']:>2}/{t1['n']:<2} ({t1['rate']:.2f})"
        print(f"{name:<40} "
              f"{t1_str:>14} "
              f"{t2['wins']:>2}/{t2['n']:<2} ({t2['rate']:.2f}) "
              f"{t2['binomial_p_vs_chance']:>10.4g} "
              f"{t3['wins']:>2}/{t3['n']:<2} ({t3['rate']:.2f}) "
              f"{t3['binomial_p_vs_chance']:>10.4g}", file=sys.stderr)

    # ── Output JSON ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "build_date": time.strftime("%Y-%m-%d"),
            "n_systems": len(SYSTEMS),
            "n_benchmark_pairs": len(benchmark_pairs),
            "n_extractions": len(extractions),
            "preflight_summary": preflight_summary,
            "smoke_test": smoke_rows,
            "tier_protocol": {
                "tier1": ("directional inflation: model_sim(A,B) > per-system Tier 1 baseline. "
                          "Baseline routing: frozen_st → null (system IS its baseline); "
                          "ft_st minilm → live_frozen_minilm cos(encode(desc_a), encode(desc_b)); "
                          "ft_st mpnet → live_frozen_mpnet cos(encode(desc_a), encode(desc_b)); "
                          "projection_only → recorded raw_minilm_sim (matches node-mean input); "
                          "brain → recorded raw_minilm_sim (matches Phase 5A 6/13 protocol)."),
                "tier2": "same-mechanism discrimination, k=1",
                "tier3": "Pool A different-mechanism discrimination, k=1",
            },
            "tier1_baselines": {
                "live_frozen_minilm_per_pair": {pid: round(v["live_frozen_minilm_sim"], 6)
                                                 for pid, v in live_baselines.items()},
                "live_frozen_mpnet_per_pair":  {pid: round(v["live_frozen_mpnet_sim"], 6)
                                                 for pid, v in live_baselines.items()},
            },
            "stats_notes": (
                "Tier 1: count only, no inferential test (the per-pair baseline is not a "
                "chance comparison). Frozen baselines (raw_minilm, raw_mpnet) report null T1 "
                "because they ARE the baseline. Tiers 2 & 3: scipy.stats.binomtest two-sided "
                "vs p=0.5. Curriculum vs nocurriculum: scipy.stats.wilcoxon two-sided on "
                "per-pair rates. All p-values are uncorrected — apply Bonferroni "
                "(0.05 / n_tests) for any specific significance claim."
            ),
        },
        "per_system_per_pair": per_pair_rows,
        "per_system_aggregate": per_system_aggregate,
        "paired_comparisons": {
            "ft_minilm_curriculum_vs_nocurriculum": paired,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
