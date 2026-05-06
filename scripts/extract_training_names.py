"""
Extract training concept names for benchmark regeneration lockout list.
Writes results/phase_a_prime/training_concept_names.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "config" / "objects" / "expanded_dataset.json"
OUT = ROOT / "results" / "phase_a_prime" / "training_concept_names.json"


def main():
    with open(DATASET) as f:
        data = json.load(f)

    # Enumerate all (name, level) tuples
    tuples = []
    for gk, g in data["groups"].items():
        for obj in g["objects"]:
            tuples.append((obj, g["level"]))

    total_entries = len(tuples)
    unique_names = sorted({t[0] for t in tuples})
    unique_name_level = {t for t in tuples}

    by_level = {"physical": [], "compound": [], "abstract": []}
    seen_name_level = set()
    for name, level in tuples:
        if (name, level) in seen_name_level:
            continue
        seen_name_level.add((name, level))
        by_level[level].append(name)
    for k in by_level:
        by_level[k] = sorted(set(by_level[k]))

    # Token extraction: split on underscore, lowercased, length >= 3
    tokens_per_concept = {}
    for name in unique_names:
        toks = [t.lower() for t in re.split(r"_+", name) if len(t) >= 3]
        tokens_per_concept[name] = set(toks)

    token_freq = Counter()
    for toks in tokens_per_concept.values():
        for t in toks:
            token_freq[t] += 1

    token_stems = sorted(token_freq.keys())
    distinctive_count = sum(1 for t, c in token_freq.items() if c == 1)

    out = {
        "total_concepts": len(unique_names),
        "total_entries": total_entries,
        "unique_name_level_tuples": len(unique_name_level),
        "concept_names": unique_names,
        "by_level": by_level,
        "token_stems": token_stems,
        "token_stems_frequency": dict(token_freq.most_common()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    # Stdout summary
    print(f"Total concept entries (group-memberships): {total_entries}")
    print(f"Unique (name, level) tuples:               {len(unique_name_level)}")
    print(f"Unique concept names:                      {len(unique_names)}")
    print()
    print(f"Count by level:")
    for k, v in by_level.items():
        print(f"  {k:>10}: {len(v)}")
    print()
    print(f"Total distinct token stems (>=3 chars): {len(token_stems)}")
    print(f"Tokens appearing in exactly 1 concept (distinctive): {distinctive_count}")
    print()
    print(f"Top 20 most frequent token stems:")
    print(f"  {'token':<24} {'count':>5}")
    print(f"  {'-'*24} {'-'*5}")
    for tok, cnt in token_freq.most_common(20):
        print(f"  {tok:<24} {cnt:>5}")
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
