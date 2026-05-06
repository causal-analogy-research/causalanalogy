"""
Phase A-prime — validate ~500 candidate adversarial pairs across 5 generation
sessions, prepare a manual-review queue.

Pipeline:
  1. Merge & dedupe (exact triple, then concept-reuse cap of 2)
  2. Lockout: name match, substring (token-bounded), distinctive-token, Jaccard >= 0.40
  3. Domain normalization to 10 canonical labels
  4. Haiku node+edge extraction (cache/extractions_phase_a_prime/)
  5. raw_minilm_sim (mean of node embeddings); raw_mpnet_sim (sentence-level on description)
  6. Semantic-dupe check vs training descriptions (MiniLM sentence-level)
  7. Difficulty class
  8. candidates_validated.json
  9. manual_review_queue.json
 10. Stdout markdown summary
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
CACHE_DIR = ROOT / "cache" / "extractions_phase_a_prime"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = ROOT / "config" / "objects" / "expanded_dataset.json"
TRAIN_NAMES_PATH = OUT_DIR / "training_concept_names.json"

CANONICAL_DOMAINS = {
    "physical", "biological", "chemical", "social", "economic",
    "political", "cognitive", "technological", "linguistic", "ecological",
}

# Domain alias mapping (fine-grained -> canonical). Items unmatched -> "other".
DOMAIN_ALIASES = {
    "physics": "physical", "mechanical": "physical", "geophysics": "physical",
    "astronomy": "physical", "astrophysical": "physical",
    "cosmology": "physical", "fluid_dynamics": "physical",
    "thermodynamics": "physical", "geological": "physical",
    "geology": "physical", "atmospheric": "physical",
    "meteorology": "physical", "optics": "physical", "acoustics": "physical",
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


# ── helpers ──────────────────────────────────────────────────────────────────

def tokens(name):
    return [t.lower() for t in re.split(r"_+", name) if t]


def tokens_set(name):
    return {t for t in tokens(name) if len(t) >= 3}


def jaccard(a, b):
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def substring_token_bounded(needle, hay):
    """Token-bounded substring: needle appears as a sequence of full tokens in hay."""
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
    # Last-chance: try substring against canonical
    for c in CANONICAL_DOMAINS:
        if c in s or s in c:
            return c
    return "other"


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


# ── Step 1: merge & dedupe ───────────────────────────────────────────────────

def step1_merge():
    sessions = {}
    for i in range(1, 6):
        p = OUT_DIR / "candidates_raw" / f"session_{i}.json"
        with open(p) as f:
            sessions[i] = json.load(f)

    merged = []
    for sid, lst in sessions.items():
        for c in lst:
            merged.append({**c, "source_session": sid})

    # exact dedupe by (concept_a, concept_b, claimed_mechanism). Order-insensitive on a/b.
    seen = set()
    deduped = []
    dedupe_drops = 0
    for c in merged:
        key = (frozenset([c["concept_a"], c["concept_b"]]), c["claimed_mechanism"])
        if key in seen:
            dedupe_drops += 1
            continue
        seen.add(key)
        deduped.append(c)

    # concept-reuse cap of 2
    concept_use = Counter()
    capped = []
    over_reuse_drops = 0
    for c in deduped:
        a, b = c["concept_a"], c["concept_b"]
        if concept_use[a] >= 2 or concept_use[b] >= 2:
            over_reuse_drops += 1
            continue
        concept_use[a] += 1
        concept_use[b] += 1
        capped.append(c)

    per_session = {str(i): len(sessions[i]) for i in range(1, 6)}
    return {
        "sessions": sessions,
        "per_session_counts": per_session,
        "merged_count": len(merged),
        "deduped_drops": dedupe_drops,
        "deduped_count": len(deduped),
        "over_reuse_drops": over_reuse_drops,
        "candidates": capped,
    }


# ── Step 2: lockout validation ───────────────────────────────────────────────

def step2_lockout(candidates, train_names_data):
    train_names = set(train_names_data["concept_names"])
    distinctive_tokens = {t for t, c in train_names_data["token_stems_frequency"].items()
                            if c <= 3}
    # Common-token allowlist (must remain allowed even if they could be distinctive)
    COMMON_ALLOW = {
        "system", "selection", "feedback", "failure", "friction", "threshold",
        "diffusion", "oscillation", "phase", "flow", "reaction", "conversion",
        "buoyancy", "spread", "combustion", "social", "economic", "degradation",
        "conduction", "engine",
    }
    distinctive_tokens -= COMMON_ALLOW

    survivors = []
    violations = []  # list of dicts: {pair_id, rule, detail}
    for c in candidates:
        a, b = c["concept_a"], c["concept_b"]
        broke = []
        for label, name in [("a", a), ("b", b)]:
            # rule 1: exact name match
            if name in train_names:
                broke.append(("exact_name_match", label, name, name))
                continue
            # rule 2: substring containment with any training concept (token-bounded)
            for tn in train_names:
                if substring_token_bounded(tn, name) or substring_token_bounded(name, tn):
                    broke.append(("substring_match", label, name, tn))
                    break
            else:
                # rule 3: distinctive-token presence
                cand_tokens = set(tokens(name))
                for t in cand_tokens:
                    if t in distinctive_tokens:
                        broke.append(("distinctive_token", label, name, t))
                        break
                else:
                    # rule 4: max Jaccard >= 0.40 vs any training concept
                    max_j = 0.0
                    nearest = None
                    for tn in train_names:
                        j = jaccard(name, tn)
                        if j > max_j:
                            max_j = j; nearest = tn
                    if max_j >= 0.40:
                        broke.append(("jaccard_ge_0.40", label, name,
                                      f"{nearest} (j={max_j:.2f})"))

        if broke:
            for rule, side, name, trig in broke:
                violations.append({
                    "pair_id": c.get("pair_id"),
                    "session": c.get("source_session"),
                    "rule": rule,
                    "side": side,
                    "name": name,
                    "trigger": trig,
                })
        else:
            survivors.append(c)
    return survivors, violations


# ── Step 3: domain normalization ─────────────────────────────────────────────

def step3_normalize_domains(candidates):
    unmapped = set()
    for c in candidates:
        ra = c.get("concept_a_domain", "")
        rb = c.get("concept_b_domain", "")
        c["concept_a_domain_raw"] = ra
        c["concept_b_domain_raw"] = rb
        norm_a = normalize_domain(ra)
        norm_b = normalize_domain(rb)
        if norm_a == "other":
            unmapped.add(ra)
        if norm_b == "other":
            unmapped.add(rb)
        c["concept_a_domain"] = norm_a
        c["concept_b_domain"] = norm_b
    return candidates, sorted(unmapped)


# ── Step 4: Haiku extraction (nodes + edges per concept description) ─────────

def _cache_paths(input_text, run_id=0):
    h = md5(input_text)
    return (CACHE_DIR / f"nodes_r{run_id}_{h}.json",
            CACHE_DIR / f"edges_r{run_id}_{h}.json")


def _extract_one(input_text, client, model="claude-haiku-4-5-20251001"):
    npath, _epath = _cache_paths(input_text)
    if npath.exists():
        cached = json.loads(npath.read_text())
        return cached["nodes"]

    prompt = f"""You are a recognition engine. Given an input concept, extract physical attributes,
properties, and object categories as a flat list of nodes.

Rules:
- Return max 15 nodes.
- Use single common English words. Prefer adjectives for properties (round, hot, fragile) and nouns for objects/categories (ball, sphere, toy). Avoid phrases longer than 2 words.
- Include the object itself, its physical properties, its category/class
- Include functional properties (what it can do / what can be done to it)
- Do NOT include causal relationships — just raw attributes
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"

Return format:
{{"nodes": ["node1", "node2", "node3", ...]}}"""
    last_err = None
    for attempt in range(5):
        try:
            resp = client.messages.create(model=model, max_tokens=200,
                                          messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj, _ = json.JSONDecoder().raw_decode(text)
            npath.write_text(json.dumps(obj))
            return obj["nodes"]
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            if "RateLimit" in etype or "Overloaded" in etype:
                time.sleep(2 ** (attempt + 1))
                continue
            if attempt >= 1:
                break
            time.sleep(0.5)
    raise last_err


def _edges_one(input_text, nodes, client, model="claude-haiku-4-5-20251001"):
    body = input_text + str(nodes)
    h = md5(body)
    epath = CACHE_DIR / f"edges_r0_{h}.json"
    if epath.exists():
        return json.loads(epath.read_text())["edges"]
    prompt = f"""You are a causal reasoning engine. Given an input concept and its extracted nodes,
identify causal relationships between nodes as directed edges.

Rules:
- Return max 10 edges.
- Each edge is (source_node -> target_node) with a relation type
- Relations should be genuinely causal: "because", "enables", "causes", "prevents", "requires"
- Do NOT invent nodes — only use nodes from the provided list
- Prefer specific causal chains over vague associations
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"
Nodes: {json.dumps(nodes)}

Return format:
{{"edges": [{{"source": "node_a", "target": "node_b", "relation": "causes"}}, ...]}}"""
    last_err = None
    for attempt in range(5):
        try:
            resp = client.messages.create(model=model, max_tokens=400,
                                          messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj, _ = json.JSONDecoder().raw_decode(text)
            epath.write_text(json.dumps(obj))
            return obj["edges"]
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            if "RateLimit" in etype or "Overloaded" in etype:
                time.sleep(2 ** (attempt + 1))
                continue
            if attempt >= 1:
                break
            time.sleep(0.5)
    raise last_err


def step4_extract(candidates, max_workers=6):
    client = anthropic.Anthropic()
    # Build (concept_name -> description) map (allowing same name with multiple descriptions)
    concept_desc = {}
    for c in candidates:
        for k_n, k_d in [("concept_a", "concept_a_description"),
                          ("concept_b", "concept_b_description")]:
            concept_desc.setdefault(c[k_n], set()).add(c[k_d])

    # Use UNIQUE descriptions as the cache key
    descriptions = set()
    for v in concept_desc.values():
        descriptions.update(v)
    descriptions = list(descriptions)

    nodes_by_desc = {}
    failures = []
    cached_hits = 0
    api_hits = 0

    # First, populate from cache
    pending = []
    for d in descriptions:
        npath, _ = _cache_paths(d)
        if npath.exists():
            nodes_by_desc[d] = json.loads(npath.read_text())["nodes"]
            cached_hits += 1
        else:
            pending.append(d)

    # API for the rest, parallel
    print(f"[step4] descriptions: total={len(descriptions)}  cached={cached_hits}  pending={len(pending)}",
          flush=True)
    if pending:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_extract_one, d, client): d for d in pending}
            for f in as_completed(futs):
                d = futs[f]
                try:
                    nodes_by_desc[d] = f.result()
                    api_hits += 1
                except Exception as e:
                    nodes_by_desc[d] = None
                    failures.append({"desc": d[:120], "err": f"{type(e).__name__}: {e}"})
                done += 1
                if done % 25 == 0 or done == len(pending):
                    rate = done / (time.time() - t0)
                    print(f"  [{done}/{len(pending)}] rate={rate:.1f}/s fails={len(failures)}",
                          flush=True)

    # Edges per description (also needed by some callers; cache for completeness)
    edges_by_desc = {}
    edge_pending = [d for d, n in nodes_by_desc.items() if n is not None]
    print(f"[step4] computing edges for {len(edge_pending)} successful descriptions",
          flush=True)
    if edge_pending:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_edges_one, d, nodes_by_desc[d], client): d for d in edge_pending}
            for f in as_completed(futs):
                d = futs[f]
                try:
                    edges_by_desc[d] = f.result()
                except Exception as e:
                    edges_by_desc[d] = None
                    failures.append({"desc": d[:120], "err_edges": f"{type(e).__name__}: {e}"})
                done += 1
                if done % 50 == 0 or done == len(edge_pending):
                    print(f"  [edges {done}/{len(edge_pending)}]", flush=True)

    return {
        "nodes_by_desc": nodes_by_desc,
        "edges_by_desc": edges_by_desc,
        "cached_hits": cached_hits,
        "api_hits": api_hits,
        "failures": failures,
    }


# ── Step 5: similarity computation ───────────────────────────────────────────

def step5_similarity(candidates, nodes_by_desc):
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    mpnet = SentenceTransformer("all-mpnet-base-v2")

    # Pre-compute mean-of-node MiniLM per unique description
    descriptions = list({d for d in nodes_by_desc if nodes_by_desc[d] is not None})
    minilm_node_mean = {}
    for d in descriptions:
        nodes = nodes_by_desc[d]
        if not nodes:
            continue
        v = minilm.encode(nodes, show_progress_bar=False, normalize_embeddings=False)
        minilm_node_mean[d] = v.mean(axis=0)

    # Sentence-level mpnet on description text
    mp_vecs = mpnet.encode(descriptions, show_progress_bar=False, normalize_embeddings=False)
    mpnet_sentence = {d: mp_vecs[i] for i, d in enumerate(descriptions)}

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    for c in candidates:
        da = c["concept_a_description"]; db = c["concept_b_description"]
        if da in minilm_node_mean and db in minilm_node_mean:
            c["raw_minilm_sim"] = round(cos(minilm_node_mean[da], minilm_node_mean[db]), 4)
        else:
            c["raw_minilm_sim"] = None
        c["raw_mpnet_sim"] = round(cos(mpnet_sentence[da], mpnet_sentence[db]), 4)
    return candidates, minilm_node_mean


# ── Step 6: semantic-dupe check vs training descriptions ─────────────────────

def step6_dupe(candidates):
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("all-MiniLM-L6-v2")

    # Training description text: per-concept "base" variant from expanded_dataset.json
    with open(DATASET_PATH) as f:
        data = json.load(f)
    train_concepts = []
    for gk, g in data["groups"].items():
        for n, vs in g["objects"].items():
            base = vs.get("base") or next(iter(vs.values()))
            train_concepts.append((n, base))
    train_names = [n for n, _ in train_concepts]
    train_descs = [d for _, d in train_concepts]

    train_emb = minilm.encode(train_descs, show_progress_bar=False, normalize_embeddings=True)

    # Encode candidate descriptions
    cand_descs = list({c["concept_a_description"] for c in candidates} |
                       {c["concept_b_description"] for c in candidates})
    cand_emb = minilm.encode(cand_descs, show_progress_bar=False, normalize_embeddings=True)
    desc_idx = {d: i for i, d in enumerate(cand_descs)}

    # nearest training per candidate description
    sims = cand_emb @ train_emb.T  # (N, M) cosine since both normalized
    nearest = {}
    for i, d in enumerate(cand_descs):
        j = int(sims[i].argmax())
        nearest[d] = (train_names[j], float(sims[i][j]))

    DUPE_THR = 0.85
    counts_at = {0.85: 0, 0.80: 0, 0.75: 0}
    for c in candidates:
        na, sa = nearest[c["concept_a_description"]]
        nb, sb = nearest[c["concept_b_description"]]
        c["nearest_training_concept_a"] = na
        c["nearest_training_sim_a"] = round(sa, 4)
        c["nearest_training_concept_b"] = nb
        c["nearest_training_sim_b"] = round(sb, 4)
        max_sim = max(sa, sb)
        c["semantic_dupe_risk"] = bool(max_sim >= DUPE_THR)
        for thr in counts_at:
            if max_sim >= thr:
                counts_at[thr] += 1
    return candidates, counts_at


# ── Step 7: difficulty ───────────────────────────────────────────────────────

def step7_difficulty(candidates):
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


# ── Step 8: write validated ──────────────────────────────────────────────────

def step8_write(candidates, meta, unmapped):
    out = {
        "total_generated": meta["merged_count"],
        "per_session_counts": meta["per_session_counts"],
        "deduped_drops": meta["deduped_drops"],
        "over_reuse_drops": meta["over_reuse_drops"],
        "lockout_failures": meta["lockout_failures"],
        "final_validated": len(candidates),
        "domain_normalization_unmapped": unmapped,
        "candidates": [
            {
                "pair_id": c.get("pair_id"),
                "source_session": c.get("source_session"),
                "generator_model": "opus-4-7",
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
                "difficulty": c.get("difficulty"),
                "semantic_dupe_risk": c.get("semantic_dupe_risk"),
                "nearest_training_concept_a": c.get("nearest_training_concept_a"),
                "nearest_training_sim_a": c.get("nearest_training_sim_a"),
                "nearest_training_concept_b": c.get("nearest_training_concept_b"),
                "nearest_training_sim_b": c.get("nearest_training_sim_b"),
                "reasoning": c.get("reasoning"),
                "strong_justification": c.get("strong_justification"),
            }
            for c in candidates
        ],
    }
    p = OUT_DIR / "candidates_validated.json"
    p.write_text(json.dumps(out, indent=2))
    return p, out


# ── Step 9: review queue ─────────────────────────────────────────────────────

def step9_queue(candidates):
    # Selection rules
    s5 = [c for c in candidates if c.get("source_session") == 5]
    dupe = [c for c in candidates if c.get("source_session") != 5
            and c.get("semantic_dupe_risk")]
    lowsim = [c for c in candidates if c.get("source_session") != 5
              and not c.get("semantic_dupe_risk")
              and c.get("raw_minilm_sim") is not None
              and c["raw_minilm_sim"] < 0.45]

    seen = set()
    queue = []
    for c in s5 + dupe + lowsim:
        pid = c.get("pair_id")
        if pid in seen:
            continue
        seen.add(pid)
        queue.append(c)

    queue.sort(key=lambda c: (c.get("raw_minilm_sim") if c.get("raw_minilm_sim") is not None else 1.0))
    queue = queue[:100]

    out = {
        "total_to_review": len(queue),
        "instructions": ("For each pair, mark keep=true if the pair shares a genuine deep "
                          "causal mechanism AND is not semantically duplicative of the nearest "
                          "training concepts. Mark keep=false otherwise. Optional notes field."),
        "queue": [
            {
                "pair_id": c.get("pair_id"),
                "source_session": c.get("source_session"),
                "concept_a": c["concept_a"],
                "concept_a_description": c["concept_a_description"],
                "concept_a_domain": c["concept_a_domain"],
                "concept_b": c["concept_b"],
                "concept_b_description": c["concept_b_description"],
                "concept_b_domain": c["concept_b_domain"],
                "claimed_mechanism": c["claimed_mechanism"],
                "reasoning": c.get("reasoning"),
                "strong_justification": c.get("strong_justification"),
                "raw_minilm_sim": c.get("raw_minilm_sim"),
                "nearest_training_a": c.get("nearest_training_concept_a"),
                "nearest_sim_a": c.get("nearest_training_sim_a"),
                "nearest_training_b": c.get("nearest_training_concept_b"),
                "nearest_sim_b": c.get("nearest_training_sim_b"),
                "semantic_dupe_risk": c.get("semantic_dupe_risk"),
            }
            for c in queue
        ],
    }
    p = OUT_DIR / "manual_review_queue.json"
    p.write_text(json.dumps(out, indent=2))
    bucket_counts = {
        "session_5": sum(1 for c in queue if c.get("source_session") == 5),
        "semantic_dupe_session_1_4": sum(1 for c in queue
            if c.get("source_session") != 5 and c.get("semantic_dupe_risk")),
        "low_sim_session_1_4": sum(1 for c in queue
            if c.get("source_session") != 5
            and not c.get("semantic_dupe_risk")
            and (c.get("raw_minilm_sim") or 1.0) < 0.45),
    }
    return p, out, bucket_counts


# ── orchestrator ─────────────────────────────────────────────────────────────

def main():
    print("Step 1: merge & dedupe", flush=True)
    s1 = step1_merge()
    print(f"  per-session: {s1['per_session_counts']}")
    print(f"  merged: {s1['merged_count']}  dedupe_drops: {s1['deduped_drops']}  "
          f"over_reuse_drops: {s1['over_reuse_drops']}  remaining: {len(s1['candidates'])}")

    print("\nStep 2: lockout validation")
    with open(TRAIN_NAMES_PATH) as f:
        train_data = json.load(f)
    survivors, violations = step2_lockout(s1["candidates"], train_data)
    print(f"  survivors: {len(survivors)}  violations: {len(violations)}")
    rule_counts = Counter(v["rule"] for v in violations)
    print(f"  rule counts: {dict(rule_counts.most_common())}")
    (OUT_DIR / "lockout_violations.json").write_text(json.dumps(violations, indent=2))

    print("\nStep 3: domain normalization")
    survivors, unmapped = step3_normalize_domains(survivors)
    print(f"  unmapped raw labels: {unmapped}")

    print("\nStep 4: Haiku extraction (nodes + edges)")
    ext = step4_extract(survivors)
    print(f"  cached_hits: {ext['cached_hits']}  api_hits: {ext['api_hits']}  "
          f"failures: {len(ext['failures'])}")
    if ext["failures"]:
        # Drop candidates referencing failed descriptions
        good_descs = {d for d, n in ext["nodes_by_desc"].items() if n is not None}
        before = len(survivors)
        survivors = [c for c in survivors
                     if c["concept_a_description"] in good_descs
                     and c["concept_b_description"] in good_descs]
        print(f"  dropped candidates with extraction failures: {before - len(survivors)}")

    print("\nStep 5: similarity")
    survivors, _ = step5_similarity(survivors, ext["nodes_by_desc"])
    if any(c.get("raw_minilm_sim") is None for c in survivors):
        n = sum(1 for c in survivors if c.get("raw_minilm_sim") is None)
        print(f"  WARN: {n} candidates lack raw_minilm_sim")

    print("\nStep 6: semantic-dupe check")
    survivors, dupe_counts = step6_dupe(survivors)
    print(f"  thresholds: {dupe_counts}")

    print("\nStep 7: difficulty classification")
    survivors = step7_difficulty(survivors)

    print("\nStep 8: write candidates_validated.json")
    meta = {
        "merged_count": s1["merged_count"],
        "per_session_counts": s1["per_session_counts"],
        "deduped_drops": s1["deduped_drops"],
        "over_reuse_drops": s1["over_reuse_drops"],
        "lockout_failures": len(violations),
    }
    vp, val = step8_write(survivors, meta, unmapped)
    print(f"  wrote {vp}")

    print("\nStep 9: build manual review queue")
    qp, q, bucket_counts = step9_queue(survivors)
    print(f"  wrote {qp}  size={q['total_to_review']}  buckets={bucket_counts}")

    # ── Step 10: stdout summary (markdown) ──
    print("\n## Phase A-prime validation summary\n")

    print("### Per-session generation & dedup")
    print("| session | generated | survived |")
    print("|---|---|---|")
    survived_by_sess = Counter(c["source_session"] for c in survivors)
    for sid in range(1, 6):
        gen = s1["per_session_counts"][str(sid)]
        sv = survived_by_sess.get(sid, 0)
        print(f"| {sid} | {gen} | {sv} |")
    print()
    print(f"- dedupe drops: {s1['deduped_drops']}")
    print(f"- over-reuse drops: {s1['over_reuse_drops']}")

    print("\n### Lockout validation (top 5 violations)")
    for rule, cnt in rule_counts.most_common(5):
        print(f"- {rule}: {cnt}")
    print(f"- pass: {len(survivors)}/{len(s1['candidates'])}")

    print("\n### Domain normalization unmapped labels")
    if unmapped:
        for u in unmapped:
            print(f"- `{u}`  (treated as 'other')")
    else:
        print("- (none)")

    print("\n### Final validated by mechanism")
    by_mech = Counter(c["claimed_mechanism"] for c in survivors)
    print("| mechanism | count |")
    print("|---|---|")
    for m, n in sorted(by_mech.items()):
        print(f"| {m} | {n} |")

    print("\n### Final validated by difficulty")
    by_diff = Counter(c["difficulty"] for c in survivors if c["difficulty"])
    print("| difficulty | count |")
    print("|---|---|")
    for d in ("STRONG", "MODERATE", "WEAK"):
        print(f"| {d} | {by_diff.get(d, 0)} |")

    print("\n### Semantic-dupe risk")
    print(f"- max_sim >= 0.85: {dupe_counts[0.85]}")
    print(f"- max_sim >= 0.80: {dupe_counts[0.80]}")
    print(f"- max_sim >= 0.75: {dupe_counts[0.75]}")

    print("\n### Manual review queue")
    print(f"- size: {q['total_to_review']}")
    print(f"- session_5 entries: {bucket_counts['session_5']}")
    print(f"- semantic-dupe (sessions 1-4): {bucket_counts['semantic_dupe_session_1_4']}")
    print(f"- low-sim (sessions 1-4, sim < 0.45): {bucket_counts['low_sim_session_1_4']}")

    print(f"\n*Halting per instruction. Awaiting "
          f"`results/phase_a_prime/manual_review_decisions.json`.*")


if __name__ == "__main__":
    main()
