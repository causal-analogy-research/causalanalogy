"""
Phase D extraction — Haiku graph extraction for 142 benchmark+distractor concepts.

Mirrors scripts/extract_expanded.py and src/sensors/extractor.py byte-for-byte:
  - Same model: claude-haiku-4-5-20251001
  - Same two-stage prompts (extract_nodes + extract_edges)
  - Same API params (max_tokens=200/400, no temperature/top_p/seed → API defaults =
    temperature 1.0, matching training-time stochasticity)
  - Reuses the live extract_full() — the cached MD5 entries under cache/extractions/
    are also reused, so re-runs are free if the input text repeats.

Sources (142 unique concepts after dedup, verified earlier; zero overlap across sources):
  results/phase_a_prime/benchmark_v2.json (56 concepts)
  results/phase_d/distractors_tier2.json   (56 concepts)
  results/phase_d/distractors_tier3.json   (30 concepts)

Per-concept level routing (rationale: match each concept's mechanism class to the
training-time level so the extractor's prompt-style — concrete physical attribute
extraction vs abstract metaphorical extraction — matches what the Brain saw at
training time):

  benchmark_v2 (all ELASTIC_STORAGE / STRUCTURAL_LOAD)            → physical
  tier2 distractors (rejected ELASTIC_STORAGE / STRUCTURAL_LOAD)  → physical
  tier3 distractors (Pool A from dropped mechanisms):
    ENERGY_CONVERSION, FLOW_REGULATION, THRESHOLD                 → physical
    FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE, SELECTION               → abstract

  No `compound=True` is used in Phase D — benchmark_v2 contains no compound-system
  concepts; ELASTIC_STORAGE / STRUCTURAL_LOAD pairs describe single-mechanism dynamics.

Output cache (NOT merged into cache/all_extractions_expanded.json):
  cache/all_extractions_phase_d.json
    {"concept_name": {
       "input": "<description text>",
       "nodes": [...],
       "edges": [...],
       "level": "physical" | "abstract",
       "source": "benchmark" | "tier2" | "tier3",
       "run_id": "phase_d_eval_v1"
    }}

  Failed concepts get an "error" key instead of "nodes"/"edges"; the failure log is
  also written to cache/extraction_phase_d_failures.json for easy review.

Prompts are written verbatim to prompts/phase_d_extraction_physical.txt and
prompts/phase_d_extraction_abstract.txt BEFORE any API call. Those files are a
documentation snapshot; the live prompts come from src/sensors/extractor.py via
extract_full(). Keep both in sync if extractor.py ever changes.

Schema validation per extraction (failures logged, NOT auto-retried):
  - len(nodes) >= 2
  - All edge source/target reference existing node names (after lowercase+strip)
  - No duplicate edges (same source+target+relation, after normalization)
  - nodes is non-null and >=2; edges may be empty list

API failure handling: 3 retries with exponential backoff (1s, 2s, 4s); after that
the concept is skipped and logged. Validation failures are NOT retried.

Usage:
  python scripts/extract_phase_d.py                # default = --spot-check
  python scripts/extract_phase_d.py --spot-check   # extract 3 sample concepts, halt
  python scripts/extract_phase_d.py --full         # extract all 142, skip cached non-errors

Requires ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sensors.extractor import extract_full, MODEL  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
BENCHMARK_PATH = ROOT / "results" / "phase_a_prime" / "benchmark_v2.json"
TIER2_PATH     = ROOT / "results" / "phase_d" / "distractors_tier2.json"
TIER3_PATH     = ROOT / "results" / "phase_d" / "distractors_tier3.json"
CANDIDATES_PATH = ROOT / "results" / "phase_a_prime" / "candidates_validated.json"
OUT_PATH       = ROOT / "cache" / "all_extractions_phase_d.json"
FAIL_PATH      = ROOT / "cache" / "extraction_phase_d_failures.json"
PROMPTS_DIR    = ROOT / "prompts"

# ── Constants ─────────────────────────────────────────────────────────
RUN_ID_TAG = "phase_d_eval_v1"
MAX_WORKERS = 2
RETRY_DELAYS = [5, 15, 45]  # 3 retries → 4 total attempts; longer backoff for 50 req/min Haiku cap

# tier3-only mechanism→level routing
TIER3_LEVEL_MAP = {
    "ENERGY_CONVERSION":  "physical",
    "FLOW_REGULATION":    "physical",
    "THRESHOLD":          "physical",
    "FEEDBACK_POSITIVE":  "abstract",
    "FEEDBACK_NEGATIVE":  "abstract",
    "SELECTION":          "abstract",
}

# Spot-check selections (deterministic):
#   - archer_longbow_draw   = first concept_a in benchmark_v2 ELASTIC_STORAGE pair
#   - roman_arch_keystone   = first concept_a in benchmark_v2 STRUCTURAL_LOAD pair
#   - viral_adoption        = explicit user request, FEEDBACK_POSITIVE → abstract
SPOT_CHECK_NAMES = {
    "archer_longbow_draw",
    "roman_arch_keystone",
    "viral_adoption",
}


# ──────────────────────────────────────────────────────────────────────
# Concept gathering
# ──────────────────────────────────────────────────────────────────────
def gather_concepts():
    """Load 3 sources, deduplicate by name (zero overlap expected),
    return list of {name, description, source, level, [claimed_mechanism]}."""
    concepts = {}

    # Benchmark — all physical
    bm = json.load(open(BENCHMARK_PATH))
    for p in bm["pairs"]:
        for side in ("a", "b"):
            n = p[f"concept_{side}_name"]
            d = p[f"concept_{side}_description"]
            if n not in concepts:
                concepts[n] = {
                    "name": n, "description": d,
                    "source": "benchmark", "level": "physical",
                    "claimed_mechanism": p["claimed_mechanism"],
                }

    # Tier 2 — all physical (same mechanisms as benchmark, just rejected)
    t2 = json.load(open(TIER2_PATH))
    for e in t2["entries"]:
        for side in ("a", "b"):
            n = e[f"distractor_concept_{side}"]["name"]
            d = e[f"distractor_concept_{side}"]["description"]
            if n not in concepts:
                concepts[n] = {
                    "name": n, "description": d,
                    "source": "tier2", "level": "physical",
                }

    # Tier 3 — level depends on the distractor's claimed_mechanism
    cands = json.load(open(CANDIDATES_PATH))["candidates"]
    cand_by_id = {c["pair_id"]: c for c in cands}

    t3 = json.load(open(TIER3_PATH))
    for e in t3["entries"]:
        dpid = e["distractor_pair_id"]
        dmech = cand_by_id[dpid]["claimed_mechanism"]
        level = TIER3_LEVEL_MAP[dmech]
        for side in ("a", "b"):
            n = e[f"distractor_concept_{side}"]["name"]
            d = e[f"distractor_concept_{side}"]["description"]
            if n not in concepts:
                concepts[n] = {
                    "name": n, "description": d,
                    "source": "tier3", "level": level,
                    "claimed_mechanism": dmech,
                }

    return list(concepts.values())


# ──────────────────────────────────────────────────────────────────────
# Prompt snapshot — verbatim from src/sensors/extractor.py
# ──────────────────────────────────────────────────────────────────────
_PHYSICAL_NODES_PROMPT = """You are a recognition engine. Given an input concept, extract physical attributes,
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
{"nodes": ["node1", "node2", "node3", ...]}"""

_ABSTRACT_NODES_PROMPT = """You are a recognition engine. Given an input concept, extract physical attributes,
properties, and object categories as a flat list of nodes.

Rules:
- Return max 15 nodes.
- Use single common English words. Prefer adjectives for properties (round, hot, fragile) and nouns for objects/categories (ball, sphere, toy). Avoid phrases longer than 2 words.
- Include the object itself, its physical properties, its category/class
- Include functional properties (what it can do / what can be done to it)
- Extract conceptual attributes and functional properties. Include structural metaphors — if the concept involves balance, fairness, flow, pressure, cycles, or other physical-metaphorical properties, include those as nodes. Use abstract nouns and adjectives (fair, balanced, legitimate, coercive).
- Do NOT include causal relationships — just raw attributes
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"

Return format:
{"nodes": ["node1", "node2", "node3", ...]}"""

_PHYSICAL_EDGES_PROMPT = """You are a causal reasoning engine. Given an input concept and its extracted nodes,
identify causal relationships between nodes as directed edges.

Rules:
- Return max 10 edges.
- Each edge is (source_node -> target_node) with a relation type
- Relations should be genuinely causal: "because", "enables", "causes", "prevents", "requires"
- Do NOT invent nodes — only use nodes from the provided list
- Prefer specific causal chains over vague associations
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"
Nodes: <node list as JSON>

Return format:
{"edges": [
  {"source": "node_a", "target": "node_b", "relation": "causes"},
  ...
]}"""

_ABSTRACT_EDGES_PROMPT = """You are a causal reasoning engine. Given an input concept and its extracted nodes,
identify causal relationships between nodes as directed edges.

Rules:
- Return max 10 edges.
- Each edge is (source_node -> target_node) with a relation type
- Relations include: "enables", "requires", "prevents", "undermines", "legitimizes", "destabilizes", "causes"
- Do NOT invent nodes — only use nodes from the provided list
- Prefer specific causal chains over vague associations
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"
Nodes: <node list as JSON>

Return format:
{"edges": [
  {"source": "node_a", "target": "node_b", "relation": "causes"},
  ...
]}"""


def save_prompts():
    """Write the two extraction prompts verbatim to prompts/ before any API calls.

    These files are a documentation snapshot. Live prompts originate in
    src/sensors/extractor.py via extract_full(). If extractor.py changes,
    update this snapshot too.
    """
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    header_common = (
        "# Phase D Haiku extraction prompts (snapshot from src/sensors/extractor.py)\n"
        f"# Model: {MODEL}\n"
        "# API params: messages.create(model=MODEL, max_tokens=200 [nodes] / 400 [edges],\n"
        "#                              messages=[{role:'user', content: <prompt>}])\n"
        "# Temperature, top_p, top_k, seed: NOT set → API default (temperature=1.0).\n"
        "# Output parsing: strip whitespace + markdown fences → json.loads.\n"
        "# Saved before any API call. Live prompts come from extractor.py — keep in sync.\n"
        "\n"
    )

    physical = (
        header_common
        + "=== NODES PROMPT (compound=False, abstract=False) ===\n\n"
        + _PHYSICAL_NODES_PROMPT
        + "\n\n=== EDGES PROMPT (abstract=False) ===\n\n"
        + _PHYSICAL_EDGES_PROMPT
        + "\n"
    )
    abstract = (
        header_common
        + "=== NODES PROMPT (compound=False, abstract=True) ===\n\n"
        + _ABSTRACT_NODES_PROMPT
        + "\n\n=== EDGES PROMPT (abstract=True) ===\n\n"
        + _ABSTRACT_EDGES_PROMPT
        + "\n"
    )

    (PROMPTS_DIR / "phase_d_extraction_physical.txt").write_text(physical)
    (PROMPTS_DIR / "phase_d_extraction_abstract.txt").write_text(abstract)
    print(f"Saved prompts → {PROMPTS_DIR}/phase_d_extraction_physical.txt")
    print(f"Saved prompts → {PROMPTS_DIR}/phase_d_extraction_abstract.txt")


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
def validate_extraction(result):
    """Schema validation. Returns (is_valid, error_msg or None).

    Rules (matching user spec):
      - len(nodes) >= 2
      - All edge source/target reference existing node names (after lowercase+strip)
      - No duplicate edges (same source+target+relation, after lowercase+strip)
      - nodes non-null and non-empty (>=2); edges may be empty list
    """
    if not isinstance(result, dict):
        return False, "result is not a dict"
    nodes = result.get("nodes")
    edges = result.get("edges")

    if not isinstance(nodes, list):
        return False, f"nodes is not a list (got {type(nodes).__name__})"
    if len(nodes) < 2:
        return False, f"nodes has fewer than 2 entries ({len(nodes)})"
    if not isinstance(edges, list):
        return False, f"edges is not a list (got {type(edges).__name__})"

    # Normalize the node set for endpoint lookup (matches consumption-time
    # normalization in src/scoring/graph_utils.py:70).
    node_set = {str(n).lower().strip() for n in nodes}

    seen_edges = set()
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            return False, f"edge[{i}] is not a dict"
        src = str(e.get("source", "")).lower().strip()
        tgt = str(e.get("target", "")).lower().strip()
        rel = str(e.get("relation", "")).lower().strip()
        if not src:
            return False, f"edge[{i}] missing source"
        if not tgt:
            return False, f"edge[{i}] missing target"
        if src not in node_set:
            return False, f"edge[{i}] source '{src}' not in nodes"
        if tgt not in node_set:
            return False, f"edge[{i}] target '{tgt}' not in nodes"
        edge_key = (src, tgt, rel)
        if edge_key in seen_edges:
            return False, f"edge[{i}] duplicate ({src}, {tgt}, {rel})"
        seen_edges.add(edge_key)

    return True, None


# ──────────────────────────────────────────────────────────────────────
# Per-concept extraction with retry
# ──────────────────────────────────────────────────────────────────────
def extract_one(concept):
    """Extract one concept with up to 3 retries on API failure. Returns
    (name, record). Validation failures are recorded but NOT retried."""
    name = concept["name"]
    text = concept["description"]
    level = concept["level"]
    source = concept["source"]
    is_abstract = (level == "abstract")

    last_err = None
    for attempt in range(len(RETRY_DELAYS) + 1):  # 4 total attempts
        try:
            data = extract_full(text, run_id=0,
                                compound=False, abstract=is_abstract)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return name, {
                "error": f"retry_fail: {last_err}",
                "input": text, "level": level, "source": source,
                "run_id": RUN_ID_TAG,
            }

        # Validate before accepting
        ok, vmsg = validate_extraction(data)
        if not ok:
            # Validation failure: NOT retried per spec
            return name, {
                "error": f"validation_fail: {vmsg}",
                "input": text, "raw": data,
                "level": level, "source": source,
                "run_id": RUN_ID_TAG,
            }

        return name, {
            "input": text,
            "nodes": data["nodes"],
            "edges": data["edges"],
            "level": level,
            "source": source,
            "run_id": RUN_ID_TAG,
        }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--spot-check", action="store_true",
                   help="Extract only the 3 spot-check concepts and halt (default behavior)")
    p.add_argument("--full", action="store_true",
                   help="Extract all 142 concepts; skips already-cached non-error entries")
    args = p.parse_args()
    if not args.full:
        args.spot_check = True

    # Save prompts BEFORE any API call
    save_prompts()

    all_concepts = gather_concepts()
    by_source, by_level = {}, {}
    for c in all_concepts:
        by_source.setdefault(c["source"], 0)
        by_source[c["source"]] += 1
        by_level.setdefault(c["level"], 0)
        by_level[c["level"]] += 1
    print(f"\nGathered {len(all_concepts)} unique concepts")
    print(f"  by source: {by_source}")
    print(f"  by level:  {by_level}")

    # Resume from existing cache
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            cache = json.load(f)
        print(f"  loaded existing cache: {len(cache)} entries")
    else:
        cache = {}
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Filter to-run set
    if args.spot_check:
        to_run = [c for c in all_concepts if c["name"] in SPOT_CHECK_NAMES]
        print(f"\n[SPOT-CHECK MODE] {len(to_run)} concepts queued:")
        for c in to_run:
            print(f"  - {c['name']:<35s}  source={c['source']:<10s} level={c['level']}")
    else:
        to_run = [c for c in all_concepts
                  if c["name"] not in cache or "error" in cache[c["name"]]]
        skipped = len(all_concepts) - len(to_run)
        print(f"\n[FULL MODE] {len(to_run)} concepts queued (skipping {skipped} already cached)")

    if not to_run:
        print("Nothing to do.")
        return

    # Extract
    successes, validation_fails, retry_fails = 0, [], []
    t0 = time.time()
    total = len(to_run)

    print(f"\nExtracting with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(extract_one, c): c for c in to_run}
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            name, record = fut.result()
            cache[name] = record
            elapsed = time.time() - t0

            if "error" in record:
                err = record["error"]
                if err.startswith("validation_fail"):
                    validation_fails.append((name, err))
                    status = "FAIL[V]"
                else:
                    retry_fails.append((name, err))
                    status = "FAIL[R]"
                detail = err
            else:
                successes += 1
                status = "ok"
                detail = f"{len(record['nodes'])} nodes / {len(record['edges'])} edges"

            print(f"  [{i:3d}/{total}] {status:<8s} {name:<38s} "
                  f"({c['level']:<8s}/{c['source']:<9s}) {detail}  t={elapsed:.1f}s")

            if i % 10 == 0:
                with open(OUT_PATH, "w") as f:
                    json.dump(cache, f, indent=2)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    wall = time.time() - t0
    print(f"\n{'='*72}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*72}")
    print(f"  successes:        {successes}")
    print(f"  validation fails: {len(validation_fails)}")
    print(f"  retry fails:      {len(retry_fails)}")
    print(f"  wall time:        {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  cache:            {OUT_PATH}")

    if validation_fails:
        print(f"\nValidation failures:")
        for n, e in validation_fails:
            print(f"  {n}: {e}")
    if retry_fails:
        print(f"\nRetry failures:")
        for n, e in retry_fails:
            print(f"  {n}: {e}")

    if validation_fails or retry_fails:
        with open(FAIL_PATH, "w") as f:
            json.dump({
                "validation_fails": [{"name": n, "error": e} for n, e in validation_fails],
                "retry_fails": [{"name": n, "error": e} for n, e in retry_fails],
            }, f, indent=2)
        print(f"  failure log:      {FAIL_PATH}")

    # Spot-check: print full extracted records for review
    if args.spot_check:
        print(f"\n{'='*72}")
        print("SPOT-CHECK FULL OUTPUT (review before --full)")
        print(f"{'='*72}")
        for c in to_run:
            n = c["name"]
            r = cache.get(n, {})
            print(f"\n--- {n} ({c['source']} / {c['level']}) ---")
            print(json.dumps(r, indent=2))
        print(f"\n{'='*72}")
        print("Spot-check complete. Confirm before running --full for remaining 139.")
        print(f"{'='*72}")


if __name__ == "__main__":
    main()
