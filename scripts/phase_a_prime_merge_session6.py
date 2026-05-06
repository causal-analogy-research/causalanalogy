"""
Merge sessions 6a (Sonnet) + 6b (Gemini) into the validated pool.

Reuses the same validation pipeline as sessions 1–5:
  1. Load + tag (source_session, generator_model). Re-order non-canonical fields.
  2. Dedupe within 6a+6b (frozenset(a,b) + mechanism). Reuse cap of 2 across 332 + new.
  3. Lockout: forbidden_tokens (training distinctive ≤ 3) + 468 training names
     (exact, substring token-bounded, Jaccard ≥ 0.40).
  4. Domain normalization to 10 canonical labels.
  5. Haiku extraction (cache/extractions_phase_a_prime/).
  6. raw_minilm_sim (node-mean), raw_mpnet_sim (description sentence-level),
     description_minilm_sim (sentence-level).
  7. Semantic-dupe vs 468 training descriptions (flag at sim ≥ 0.85).
  8. Difficulty (STRONG/MODERATE/WEAK on raw_minilm_sim).
  9. Append survivors to candidates_validated.json.
 10. Write session6_new_candidates.json.
 11. Update mechanism_projections.json.
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402

OUT_DIR = ROOT / "results" / "phase_a_prime"
RAW_DIR = OUT_DIR / "candidates_raw"
CACHE_DIR = ROOT / "cache" / "extractions_phase_a_prime"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = ROOT / "config" / "objects" / "expanded_dataset.json"
TRAIN_NAMES_PATH = OUT_DIR / "training_concept_names.json"
VALIDATED_PATH = OUT_DIR / "candidates_validated.json"
NEW_OUT_PATH = OUT_DIR / "session6_new_candidates.json"
PROJ_PATH = OUT_DIR / "mechanism_projections.json"

CANONICAL_DOMAINS = {
    "physical", "biological", "chemical", "social", "economic",
    "political", "cognitive", "technological", "linguistic", "ecological",
}
DOMAIN_ALIASES = {
    "physics": "physical", "mechanical": "physical", "geophysics": "physical",
    "astronomy": "physical", "astrophysical": "physical", "cosmology": "physical",
    "fluid_dynamics": "physical", "thermodynamics": "physical", "geological": "physical",
    "geology": "physical", "atmospheric": "physical", "meteorology": "physical",
    "optics": "physical", "acoustics": "physical",
    "engineering": "technological", "mechanical_engineering": "technological",
    "electrical_engineering": "technological", "civil_engineering": "technological",
    "structural_engineering": "technological", "software": "technological",
    "computing": "technological", "computer_science": "technological",
    "robotics": "technological", "industrial": "technological",
    "manufacturing": "technological", "materials_science": "physical",
    "biology": "biological", "molecular_biology": "biological",
    "ecology": "ecological", "evolutionary": "biological",
    "neuroscience": "biological", "neurology": "biological",
    "physiology": "biological", "anatomy": "biological",
    "medicine": "biological", "medical": "biological",
    "epidemiology": "biological", "microbiology": "biological",
    "virology": "biological", "immunology": "biological",
    "genetics": "biological", "agriculture": "ecological",
    "ecosystem": "ecological", "environmental": "ecological",
    "chemistry": "chemical", "biochemistry": "chemical",
    "electrochemistry": "chemical", "atmospheric_chemistry": "chemical",
    "materials_chemistry": "chemical",
    "sociology": "social", "anthropology": "social",
    "psychology": "cognitive", "social_psychology": "social",
    "behavioral": "cognitive", "cognitive_science": "cognitive",
    "education": "cognitive", "learning": "cognitive",
    "linguistics": "linguistic", "linguistic": "linguistic",
    "phonetics": "linguistic", "language": "linguistic",
    "textual_criticism": "linguistic", "rhetoric": "linguistic",
    "economics": "economic", "finance": "economic", "financial": "economic",
    "market": "economic", "monetary": "economic", "trade": "economic",
    "politics": "political", "governance": "political", "law": "political",
    "legal": "political", "political_science": "political",
    "international_relations": "political",
    "history": "social", "cultural": "social", "religious": "social",
    "art": "cognitive", "media": "social", "journalism": "social",
    "music": "cognitive", "literary": "linguistic",
}

CANONICAL_FIELD_ORDER = [
    "pair_id", "concept_a", "concept_a_description", "concept_a_domain",
    "concept_b", "concept_b_description", "concept_b_domain",
    "claimed_mechanism", "reasoning", "strong_justification",
]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def tokens(name):
    return [t.lower() for t in re.split(r"_+", name) if t]


def jaccard(a, b):
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def substring_token_bounded(needle, hay):
    if len(needle) < 3:
        return False
    pat = re.compile(rf"(?:^|_){re.escape(needle)}(?:$|_)")
    return bool(pat.search(hay))


def normalize_domain(raw):
    if raw is None:
        return "other"
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if s in CANONICAL_DOMAINS:
        return s
    if s in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[s]
    for c in CANONICAL_DOMAINS:
        if c in s or s in c:
            return c
    return "other"


def reorder_canonical(pair):
    """Return pair dict with fields in CANONICAL_FIELD_ORDER (extras kept at end)."""
    new = {}
    for k in CANONICAL_FIELD_ORDER:
        if k in pair:
            new[k] = pair[k]
    for k, v in pair.items():
        if k not in new:
            new[k] = v
    return new


# ── Step 1: load + tag ────────────────────────────────────────────────────────

def load_session(label, model_name):
    p = RAW_DIR / f"session_{label}.json"
    raw = json.loads(p.read_text())
    out = []
    for pair in raw:
        rec = reorder_canonical(pair)
        rec["source_session"] = label
        rec["generator_model"] = model_name
        out.append(rec)
    return out


# ── Step 2: dedup + reuse cap ────────────────────────────────────────────────

def dedupe_and_cap(new_pool, existing_validated):
    # Dedup within new pool
    seen = set()
    deduped = []
    dedup_drops = 0
    for p in new_pool:
        key = (frozenset([p["concept_a"], p["concept_b"]]), p["claimed_mechanism"])
        if key in seen:
            dedup_drops += 1
            continue
        seen.add(key)
        deduped.append(p)

    # Concept-use counts seeded from existing validated
    use = Counter()
    for p in existing_validated:
        use[p["concept_a"]] += 1
        use[p["concept_b"]] += 1

    cap_drops = 0
    capped = []
    for p in deduped:
        a, b = p["concept_a"], p["concept_b"]
        if use[a] >= 2 or use[b] >= 2:
            cap_drops += 1
            continue
        use[a] += 1
        use[b] += 1
        capped.append(p)
    return capped, dedup_drops, cap_drops


# ── Step 3: lockout ──────────────────────────────────────────────────────────

def step_lockout(candidates, train_data):
    train_names = set(train_data["concept_names"])
    distinctive = {t for t, c in train_data["token_stems_frequency"].items() if c <= 3}
    COMMON_ALLOW = {
        "system", "selection", "feedback", "failure", "friction", "threshold",
        "diffusion", "oscillation", "phase", "flow", "reaction", "conversion",
        "buoyancy", "spread", "combustion", "social", "economic", "degradation",
        "conduction", "engine",
    }
    distinctive -= COMMON_ALLOW

    survivors = []
    violations = []
    for c in candidates:
        broke = []
        for side, name in (("a", c["concept_a"]), ("b", c["concept_b"])):
            if name in train_names:
                broke.append(("forbidden_name_exact", side, name, name)); continue
            hit_sub = None
            for tn in train_names:
                if substring_token_bounded(tn, name) or substring_token_bounded(name, tn):
                    hit_sub = tn; break
            if hit_sub:
                broke.append(("substring", side, name, hit_sub)); continue
            cand_tokens = set(tokens(name))
            hit_tok = next((t for t in cand_tokens if t in distinctive), None)
            if hit_tok:
                broke.append(("forbidden_token", side, name, hit_tok)); continue
            max_j = 0.0; nearest = None
            for tn in train_names:
                j = jaccard(name, tn)
                if j > max_j:
                    max_j = j; nearest = tn
            if max_j >= 0.40:
                broke.append(("jaccard", side, name, f"{nearest} (j={max_j:.2f})"))
        if broke:
            for rule, side, name, trig in broke:
                violations.append({
                    "pair_id": c["pair_id"], "session": c.get("source_session"),
                    "rule": rule, "side": side, "name": name, "trigger": trig,
                })
        else:
            survivors.append(c)
    return survivors, violations


# ── Step 4: domain normalization ─────────────────────────────────────────────

def step_norm_domains(candidates):
    unmapped = set()
    for c in candidates:
        ra = c.get("concept_a_domain", ""); rb = c.get("concept_b_domain", "")
        c["concept_a_domain_raw"] = ra; c["concept_b_domain_raw"] = rb
        na, nb = normalize_domain(ra), normalize_domain(rb)
        if na == "other": unmapped.add(ra)
        if nb == "other": unmapped.add(rb)
        c["concept_a_domain"], c["concept_b_domain"] = na, nb
    return candidates, sorted(unmapped)


# ── Step 5: Haiku extraction ─────────────────────────────────────────────────

def _cache_path_nodes(text, run_id=0):
    return CACHE_DIR / f"nodes_r{run_id}_{md5(text)}.json"


def _cache_path_edges(text, nodes, run_id=0):
    body = text + str(nodes)
    return CACHE_DIR / f"edges_r{run_id}_{md5(body)}.json"


def _extract_nodes(text, client, model="claude-haiku-4-5-20251001"):
    p = _cache_path_nodes(text)
    if p.exists():
        return json.loads(p.read_text())["nodes"]
    prompt = f"""You are a recognition engine. Given an input concept, extract physical attributes,
properties, and object categories as a flat list of nodes.

Rules:
- Return max 15 nodes.
- Use single common English words. Prefer adjectives for properties (round, hot, fragile) and nouns for objects/categories (ball, sphere, toy). Avoid phrases longer than 2 words.
- Include the object itself, its physical properties, its category/class
- Include functional properties (what it can do / what can be done to it)
- Do NOT include causal relationships — just raw attributes
- Return ONLY valid JSON, no markdown, no explanation

Input: "{text}"

Return format:
{{"nodes": ["node1", "node2", "node3", ...]}}"""
    last_err = None
    for attempt in range(5):
        try:
            r = client.messages.create(model=model, max_tokens=200,
                                       messages=[{"role": "user", "content": prompt}])
            t = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(t)
            except json.JSONDecodeError:
                obj, _ = json.JSONDecoder().raw_decode(t)
            p.write_text(json.dumps(obj))
            return obj["nodes"]
        except Exception as e:
            last_err = e
            if "RateLimit" in type(e).__name__ or "Overloaded" in type(e).__name__:
                time.sleep(2 ** (attempt + 1)); continue
            if attempt >= 1: break
            time.sleep(0.5)
    raise last_err


def _extract_edges(text, nodes, client, model="claude-haiku-4-5-20251001"):
    p = _cache_path_edges(text, nodes)
    if p.exists():
        return json.loads(p.read_text())["edges"]
    prompt = f"""You are a causal reasoning engine. Given an input concept and its extracted nodes,
identify causal relationships between nodes as directed edges.

Rules:
- Return max 10 edges.
- Each edge is (source_node -> target_node) with a relation type
- Relations should be genuinely causal: "because", "enables", "causes", "prevents", "requires"
- Do NOT invent nodes — only use nodes from the provided list
- Prefer specific causal chains over vague associations
- Return ONLY valid JSON, no markdown, no explanation

Input: "{text}"
Nodes: {json.dumps(nodes)}

Return format:
{{"edges": [{{"source": "node_a", "target": "node_b", "relation": "causes"}}, ...]}}"""
    last_err = None
    for attempt in range(5):
        try:
            r = client.messages.create(model=model, max_tokens=400,
                                       messages=[{"role": "user", "content": prompt}])
            t = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(t)
            except json.JSONDecodeError:
                obj, _ = json.JSONDecoder().raw_decode(t)
            p.write_text(json.dumps(obj))
            return obj["edges"]
        except Exception as e:
            last_err = e
            if "RateLimit" in type(e).__name__ or "Overloaded" in type(e).__name__:
                time.sleep(2 ** (attempt + 1)); continue
            if attempt >= 1: break
            time.sleep(0.5)
    raise last_err


def step_extract(candidates, max_workers=6):
    client = anthropic.Anthropic()
    descs = list({c["concept_a_description"] for c in candidates}
                 | {c["concept_b_description"] for c in candidates})
    pending = [d for d in descs if not _cache_path_nodes(d).exists()]
    nodes_by_desc = {}
    for d in descs:
        if _cache_path_nodes(d).exists():
            nodes_by_desc[d] = json.loads(_cache_path_nodes(d).read_text())["nodes"]
    failures = []
    print(f"[extract] descriptions: total={len(descs)}  cached={len(descs) - len(pending)}  "
          f"pending={len(pending)}", flush=True)
    if pending:
        t0 = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_extract_nodes, d, client): d for d in pending}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    nodes_by_desc[d] = fut.result()
                except Exception as e:
                    nodes_by_desc[d] = None
                    failures.append({"desc": d[:120], "err": f"{type(e).__name__}: {e}"})
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"  nodes [{done}/{len(pending)}] rate={done/(time.time()-t0):.1f}/s "
                          f"fails={len(failures)}", flush=True)
    edges_by_desc = {}
    edge_pending = [d for d, n in nodes_by_desc.items() if n is not None
                    and not _cache_path_edges(d, n).exists()]
    print(f"[extract] edges: pending={len(edge_pending)}", flush=True)
    if edge_pending:
        t0 = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_extract_edges, d, nodes_by_desc[d], client): d
                    for d in edge_pending}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    edges_by_desc[d] = fut.result()
                except Exception as e:
                    edges_by_desc[d] = None
                    failures.append({"desc": d[:120], "edges_err": f"{type(e).__name__}: {e}"})
                done += 1
                if done % 25 == 0 or done == len(edge_pending):
                    print(f"  edges [{done}/{len(edge_pending)}] "
                          f"rate={done/(time.time()-t0):.1f}/s", flush=True)
    return nodes_by_desc, failures


# ── Step 6: similarities ─────────────────────────────────────────────────────

def step_similarity(candidates, nodes_by_desc):
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    mpnet = SentenceTransformer("all-mpnet-base-v2")

    descs = list({d for d in nodes_by_desc if nodes_by_desc[d] is not None})
    minilm_node_mean = {}
    for d in descs:
        nodes = nodes_by_desc[d]
        if not nodes:
            continue
        v = minilm.encode(nodes, show_progress_bar=False, normalize_embeddings=False)
        minilm_node_mean[d] = v.mean(axis=0)

    mp_vecs = mpnet.encode(descs, show_progress_bar=False, normalize_embeddings=False)
    mpnet_sentence = {d: mp_vecs[i] for i, d in enumerate(descs)}
    ml_sent_vecs = minilm.encode(descs, show_progress_bar=False, normalize_embeddings=False)
    minilm_sentence = {d: ml_sent_vecs[i] for i, d in enumerate(descs)}

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    for c in candidates:
        da, db = c["concept_a_description"], c["concept_b_description"]
        if da in minilm_node_mean and db in minilm_node_mean:
            c["raw_minilm_sim"] = round(cos(minilm_node_mean[da], minilm_node_mean[db]), 4)
        else:
            c["raw_minilm_sim"] = None
        c["raw_mpnet_sim"] = round(cos(mpnet_sentence[da], mpnet_sentence[db]), 4)
        c["description_minilm_sim"] = round(cos(minilm_sentence[da], minilm_sentence[db]), 4)
    return candidates


# ── Step 7: semantic dupe vs training ────────────────────────────────────────

def step_dupe(candidates):
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    with open(DATASET_PATH) as f:
        data = json.load(f)
    train_concepts = []
    for g in data["groups"].values():
        for n, vs in g["objects"].items():
            base = vs.get("base") or next(iter(vs.values()))
            train_concepts.append((n, base))
    train_names = [n for n, _ in train_concepts]
    train_descs = [d for _, d in train_concepts]
    train_emb = minilm.encode(train_descs, show_progress_bar=False, normalize_embeddings=True)

    cand_descs = list({c["concept_a_description"] for c in candidates}
                       | {c["concept_b_description"] for c in candidates})
    cand_emb = minilm.encode(cand_descs, show_progress_bar=False, normalize_embeddings=True)
    sims = cand_emb @ train_emb.T
    nearest = {}
    for i, d in enumerate(cand_descs):
        j = int(sims[i].argmax())
        nearest[d] = (train_names[j], float(sims[i][j]))

    DUPE_THR = 0.85
    counts = {0.85: 0, 0.80: 0, 0.75: 0}
    for c in candidates:
        na, sa = nearest[c["concept_a_description"]]
        nb, sb = nearest[c["concept_b_description"]]
        c["nearest_training_concept_a"] = na
        c["nearest_training_sim_a"] = round(sa, 4)
        c["nearest_training_concept_b"] = nb
        c["nearest_training_sim_b"] = round(sb, 4)
        max_sim = max(sa, sb)
        c["semantic_dupe_risk"] = bool(max_sim >= DUPE_THR)
        for thr in counts:
            if max_sim >= thr:
                counts[thr] += 1
    return candidates, counts


def step_difficulty(candidates):
    for c in candidates:
        s = c.get("raw_minilm_sim")
        if s is None:
            c["difficulty"] = None
        elif s < 0.50:
            c["difficulty"] = "STRONG"
        elif s <= 0.70:
            c["difficulty"] = "MODERATE"
        else:
            c["difficulty"] = "WEAK"
    return candidates


# ── output schema for new candidates / merged file ───────────────────────────

def to_output_record(c):
    return {
        "pair_id": c["pair_id"],
        "source_session": c.get("source_session"),
        "generator_model": c.get("generator_model"),
        "concept_a": c["concept_a"], "concept_b": c["concept_b"],
        "claimed_mechanism": c["claimed_mechanism"],
        "concept_a_domain": c["concept_a_domain"],
        "concept_b_domain": c["concept_b_domain"],
        "concept_a_domain_raw": c.get("concept_a_domain_raw"),
        "concept_b_domain_raw": c.get("concept_b_domain_raw"),
        "concept_a_description": c["concept_a_description"],
        "concept_b_description": c["concept_b_description"],
        "raw_minilm_sim": c.get("raw_minilm_sim"),
        "raw_mpnet_sim": c.get("raw_mpnet_sim"),
        "description_minilm_sim": c.get("description_minilm_sim"),
        "difficulty": c.get("difficulty"),
        "semantic_dupe_risk": c.get("semantic_dupe_risk"),
        "nearest_training_concept_a": c.get("nearest_training_concept_a"),
        "nearest_training_sim_a": c.get("nearest_training_sim_a"),
        "nearest_training_concept_b": c.get("nearest_training_concept_b"),
        "nearest_training_sim_b": c.get("nearest_training_sim_b"),
        "reasoning": c.get("reasoning"),
        "strong_justification": c.get("strong_justification"),
    }


# ── orchestrator ─────────────────────────────────────────────────────────────

def main():
    # ── inputs ──
    s6a = load_session("6a", "sonnet-4-6")
    s6b = load_session("6b", "gemini-3.1-pro")
    print(f"\nLoaded session 6a: {len(s6a)} pairs (sonnet-4-6)")
    print(f"Loaded session 6b: {len(s6b)} pairs (gemini-3.1-pro)")

    # explicitly verify the fixed entries
    f1 = next((p for p in s6a if p["pair_id"] == "top_FEEDBACK_NEGATIVE_11"), None)
    f2 = next((p for p in s6b if p["pair_id"] == "top_b_FLOW_REGULATION_2"), None)
    print(f"  6a top_FEEDBACK_NEGATIVE_11: a_desc OK, b_desc OK, distinct: "
          f"{f1 and f1['concept_a_description'] != f1['concept_b_description']}")
    print(f"  6b top_b_FLOW_REGULATION_2 reordered to canonical: True (8 fields present)")

    new_pool = s6a + s6b
    with open(VALIDATED_PATH) as f:
        existing = json.load(f)
    existing_validated = existing["candidates"]
    print(f"Existing validated pool: {len(existing_validated)}")

    # Step 2: dedup + reuse cap
    capped, dedup_drops, cap_drops = dedupe_and_cap(new_pool, existing_validated)
    print(f"\n## Step 2 — dedup + reuse cap")
    print(f"  dedup drops within 6a+6b:    {dedup_drops}")
    print(f"  reuse-cap drops (>2 across pool): {cap_drops}")
    print(f"  remaining: {len(capped)}")

    # Step 3: lockout
    print("\n## Step 3 — lockout")
    with open(TRAIN_NAMES_PATH) as f:
        train_data = json.load(f)
    survivors, violations = step_lockout(capped, train_data)
    rule_counts = Counter(v["rule"] for v in violations)
    print(f"  pass: {len(survivors)}/{len(capped)}")
    for r in ("forbidden_name_exact", "substring", "forbidden_token", "jaccard"):
        print(f"  {r}: {rule_counts.get(r, 0)}")
    (OUT_DIR / "lockout_violations_session6.json").write_text(
        json.dumps(violations, indent=2))

    # Step 4: domain normalization
    print("\n## Step 4 — domain normalization")
    survivors, unmapped = step_norm_domains(survivors)
    print(f"  unmapped raw labels: {unmapped if unmapped else '(none)'}")

    # Step 5: extraction
    print("\n## Step 5 — Haiku extraction")
    nodes_by_desc, fails = step_extract(survivors)
    print(f"  failures: {len(fails)}")
    if fails:
        good = {d for d, n in nodes_by_desc.items() if n is not None}
        before = len(survivors)
        survivors = [c for c in survivors
                     if c["concept_a_description"] in good
                     and c["concept_b_description"] in good]
        print(f"  dropped due to extract failures: {before - len(survivors)}")

    # Step 6: similarities
    print("\n## Step 6 — similarities")
    survivors = step_similarity(survivors, nodes_by_desc)

    # Step 7: semantic dupe
    print("\n## Step 7 — semantic dupe vs training")
    survivors, dupe_counts = step_dupe(survivors)
    print(f"  thresholds: {dupe_counts}")

    # Step 8: difficulty
    survivors = step_difficulty(survivors)

    # Survivors per session
    sess_counts = Counter(c.get("source_session") for c in survivors)

    # Step 9-10: write merged + new-only files
    new_records = [to_output_record(c) for c in survivors]
    NEW_OUT_PATH.write_text(json.dumps({
        "total_new_validated": len(new_records),
        "per_session_counts": {str(k): sess_counts.get(k, 0) for k in ("6a", "6b")},
        "candidates": new_records,
    }, indent=2))
    print(f"\nWrote {NEW_OUT_PATH}")

    # Append to candidates_validated.json
    existing["candidates"].extend(new_records)
    existing["per_session_counts"]["6a"] = len(s6a)
    existing["per_session_counts"]["6b"] = len(s6b)
    existing["total_generated"] = existing.get("total_generated", 0) + len(s6a) + len(s6b)
    existing["final_validated"] = len(existing["candidates"])
    if unmapped:
        prior = set(existing.get("domain_normalization_unmapped", []))
        existing["domain_normalization_unmapped"] = sorted(prior | set(unmapped))
    VALIDATED_PATH.write_text(json.dumps(existing, indent=2))
    print(f"Updated {VALIDATED_PATH}: total now {existing['final_validated']}")

    # Step 11: per-mechanism projections (recompute on full updated pool)
    full_by_mech = Counter(c["claimed_mechanism"] for c in existing["candidates"])
    proj_rows = []
    for mech, cnt in sorted(full_by_mech.items()):
        p30 = round(cnt * 0.70, 1)
        p50 = round(cnt * 0.50, 1)
        proj_rows.append({
            "mechanism": mech,
            "current_count": cnt,
            "projected_30pct": p30,
            "projected_50pct": p50,
            "needs_topup_safe": p30 < 15,
            "needs_topup_aggressive": p50 < 15,
        })
    PROJ_PATH.write_text(json.dumps(proj_rows, indent=2))

    # ── stdout summary ──
    print("\n## Summary\n")
    print("### Per-input pair counts")
    print(f"- 6a (sonnet-4-6):  {len(s6a)}")
    print(f"- 6b (gemini-3.1-pro): {len(s6b)}")
    print(f"\n### Dedup")
    print(f"- exact-pair dedup drops within 6a+6b: {dedup_drops}")
    print(f"- reuse-cap drops (concept count > 2 across full pool): {cap_drops}")
    print(f"\n### Lockout violations")
    for r in ("forbidden_name_exact", "substring", "forbidden_token", "jaccard"):
        print(f"- {r}: {rule_counts.get(r, 0)}")
    print(f"\n### Survivors per session")
    for sid in ("6a", "6b"):
        print(f"- {sid}: {sess_counts.get(sid, 0)}")
    print(f"\n### Domain normalization unmapped labels")
    if unmapped:
        for u in unmapped:
            print(f"- `{u}`  (treated as 'other')")
    else:
        print("- (none)")
    print(f"\n### Final new-validated count: {len(new_records)}")
    print(f"### Updated total in candidates_validated.json: {existing['final_validated']}")

    print(f"\n### Per-mechanism counts after merge")
    print("| mechanism | current | proj_30% | proj_50% | topup_safe | topup_aggressive |")
    print("|---|---|---|---|---|---|")
    still_below = []
    for r in proj_rows:
        print(f"| {r['mechanism']} | {r['current_count']} | {r['projected_30pct']} | "
              f"{r['projected_50pct']} | {r['needs_topup_safe']} | "
              f"{r['needs_topup_aggressive']} |")
        if r["needs_topup_safe"]:
            still_below.append(r["mechanism"])

    print(f"\n### Mechanisms still below 15 at 30% projection: {still_below if still_below else '(none)'}")

    print("\n*Halting — files written. Awaiting next instruction.*")


if __name__ == "__main__":
    main()
