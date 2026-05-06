"""
0d follow-up: detailed overlap analysis between the 467 (post-glass-rename: 468)
training concepts and the 180-pair benchmark.

For each of the 3 exact overlaps, report (pair count, distinct mechanism types).
Then run a broader scan: substring containment + token-overlap >0.7.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "config" / "objects" / "expanded_dataset.json"
BENCHMARK = ROOT / "results" / "adversarial_phase_a" / "benchmark_180.json"


def tokenize(name):
    return set(t for t in re.split(r"[_\s]+", name.lower()) if t)


def token_overlap(a, b):
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def substring_match(short, long_):
    """True iff `short` occurs as a full-token-bounded substring of `long_`."""
    if len(short) < 4:
        return False
    # token-bounded: ensure it's at start/end or surrounded by _
    pat = re.compile(rf"(?:^|_){re.escape(short)}(?:$|_)")
    return bool(pat.search(long_))


def main():
    with open(DATASET) as f:
        data = json.load(f)
    train_concepts = set()
    for g in data["groups"].values():
        for n in g["objects"]:
            train_concepts.add(n)

    with open(BENCHMARK) as f:
        bench = json.load(f)

    bench_pairs = bench  # list of 180 pair dicts
    bench_concepts = set()
    pair_counts = defaultdict(list)   # concept -> list of (idx, mechanism_type, other)
    for i, p in enumerate(bench_pairs):
        ca, cb, mt = p["concept_a"], p["concept_b"], p["mechanism_type"]
        bench_concepts.add(ca); bench_concepts.add(cb)
        pair_counts[ca].append((i, mt, cb))
        pair_counts[cb].append((i, mt, ca))

    # ── 1. Exact overlaps: per-concept pair-count + mechanism breakdown ──
    exact = sorted(train_concepts & bench_concepts)
    print(f"Exact overlaps: {len(exact)}")
    print(f"{'concept':<30} {'pairs':>5} {'mechanisms':<40}")
    print("-" * 80)
    for c in exact:
        pair_refs = pair_counts[c]
        mechs = sorted({m for (_, m, _) in pair_refs})
        print(f"{c:<30} {len(pair_refs):>5}  {', '.join(mechs)}")
        for (idx, m, other) in pair_refs:
            print(f"    pair {idx:>3}  [{m:<20}]  paired with {other}")

    # ── 2. Broader scan ──
    print(f"\n{'='*70}")
    print("Broader scan (non-exact overlaps)")
    print(f"{'='*70}")

    substring_hits = []
    token_hits = []
    for tc in train_concepts:
        for bc in bench_concepts:
            if tc == bc:
                continue
            if substring_match(tc, bc) or substring_match(bc, tc):
                substring_hits.append((tc, bc))
            ov = token_overlap(tc, bc)
            if ov >= 0.7:
                token_hits.append((tc, bc, ov))

    # Dedup substring hits (a→b and b→a)
    dedup_sub = set()
    for tc, bc in substring_hits:
        dedup_sub.add((tc, bc))
    print(f"\nSubstring (token-bounded) matches: {len(dedup_sub)}")
    for tc, bc in sorted(dedup_sub):
        # show which mechanism types the benchmark concept appears in
        mechs = sorted({m for (_, m, _) in pair_counts.get(bc, [])})
        print(f"  train={tc:<30} bench={bc:<35} bench_mechs=[{', '.join(mechs)}]")

    # Dedup token-overlap hits by sorted pair
    dedup_tok = {}
    for tc, bc, ov in token_hits:
        key = tuple(sorted([tc, bc]))
        if key not in dedup_tok or dedup_tok[key][2] < ov:
            dedup_tok[key] = (tc, bc, ov)
    unique_tok = [v for v in dedup_tok.values()]
    unique_tok.sort(key=lambda x: -x[2])

    # Filter out tok hits already covered by substring/exact
    exact_pair_set = {tuple(sorted([t, b])) for t in exact for b in [t]}
    sub_pair_set = {tuple(sorted([t, b])) for (t, b) in dedup_sub}
    new_tok_hits = [h for h in unique_tok if tuple(sorted([h[0], h[1]])) not in sub_pair_set]

    print(f"\nToken-overlap >= 0.70 (not also substring): {len(new_tok_hits)}")
    for tc, bc, ov in new_tok_hits:
        mechs_b = sorted({m for (_, m, _) in pair_counts.get(bc, [])})
        print(f"  overlap={ov:.2f}  train={tc:<30} bench={bc:<35} bench_mechs=[{', '.join(mechs_b)}]")

    # Summary JSON
    out = {
        "exact_overlaps": exact,
        "exact_overlap_detail": {
            c: [{"pair_id": i, "mechanism_type": m, "paired_with": o}
                for (i, m, o) in pair_counts[c]]
            for c in exact
        },
        "substring_matches": [{"train": tc, "bench": bc} for tc, bc in sorted(dedup_sub)],
        "token_overlap_matches": [
            {"train": tc, "bench": bc, "overlap": round(ov, 3)}
            for (tc, bc, ov) in new_tok_hits
        ],
    }
    out_path = ROOT / "results" / "phase_b" / "overlap_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
