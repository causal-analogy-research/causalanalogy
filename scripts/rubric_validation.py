"""
Rubric validation by side-by-side comparison.
Combines all 4 batches of GPT-5.2 reviews (120 pairs) and produces:
  Section 1: All keeps (full pair info)
  Section 2: 10 stratified rejects with Q2 = false
  Section 3: 10 stratified rejects with Q1=true Q2=true Q5=false
  Section 4: All Q2-false pairs (first sentence of notes)
  Section 5: 12-mechanism × 5-question failure-rate table
  Section 6: Calibration anchor input from corrected_evaluation.json
  Plus the rubric prompt text (reconstructed).

Writes results/phase_a_prime/gpt_review/calibration_anchor_input.json.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase_a_prime"
GPT = OUT / "gpt_review"
VAL = OUT / "candidates_validated.json"
CORRECTED = ROOT / "results" / "adversarial" / "corrected_evaluation.json"
CALIB_OUT = GPT / "calibration_anchor_input.json"

QS = ["q1_mechanism_reality", "q2_causal_structure", "q3_cross_domain",
      "q4_groundedness", "q5_reasoning_soundness"]
QS_LABEL = {"q1_mechanism_reality": "Q1", "q2_causal_structure": "Q2",
            "q3_cross_domain": "Q3", "q4_groundedness": "Q4",
            "q5_reasoning_soundness": "Q5"}

CANDIDATE_MECHANISMS = ["PROPAGATION", "DEGRADATION", "THRESHOLD",
                        "FEEDBACK_POSITIVE", "FEEDBACK_NEGATIVE", "OSCILLATION",
                        "DIFFUSION", "SELECTION", "ENERGY_CONVERSION",
                        "ELASTIC_STORAGE", "STRUCTURAL_LOAD", "FLOW_REGULATION"]


def load():
    decs = []
    for i in range(1, 5):
        decs.extend(json.loads((GPT / f"batch_{i}_decisions.json").read_text()))
    val = json.loads(VAL.read_text())["candidates"]
    val_by = {c["pair_id"]: c for c in val}
    return decs, val_by


def fmt_pair(e, val_by):
    pid = e["pair_id"]
    c = val_by.get(pid, {})
    failed = [QS_LABEL[q] for q in QS if e[q] is False]
    print(f"--- {pid} ---")
    print(f"mechanism: {c.get('claimed_mechanism')}")
    print(f"source_session: {c.get('source_session')}")
    print(f"concept_a: {c.get('concept_a')}")
    print(f"concept_b: {c.get('concept_b')}")
    print(f"original justification (full): {c.get('reasoning')}")
    if c.get("strong_justification"):
        print(f"strong_justification: {c['strong_justification']}")
    print(f"score: {e['score']}")
    qline = "  ".join(f"{QS_LABEL[q]}={e[q]}" for q in QS)
    print(f"q-fields: {qline}")
    print(f"failed: {', '.join(failed) if failed else '(none)'}")
    print(f"notes: {e['notes']}")
    print()


def first_sentence(s, max_chars=200):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    end = s.find(". ")
    if end == -1:
        end = s.find(".")
    if end != -1 and end + 1 <= max_chars:
        return s[:end + 1]
    return s[:max_chars]


def section_1_keeps(decs, val_by):
    print("=" * 78)
    print("SECTION 1: All keeps (verdict='keep') across batches 1-4")
    print("=" * 78)
    keeps = [e for e in decs if e["verdict"] == "keep"]
    print(f"Total keeps: {len(keeps)}\n")
    for e in keeps:
        fmt_pair(e, val_by)


def section_2_rejects_q2(decs, val_by):
    print("=" * 78)
    print("SECTION 2: 10 stratified rejects with Q2 = false (seed=42, span ≥5 mechanisms)")
    print("=" * 78)
    rng = random.Random(42)
    rejs = [e for e in decs
            if e["verdict"] == "reject" and e.get("q2_causal_structure") is False]
    print(f"Eligible: {len(rejs)} rejects with Q2=false\n")

    by_m = defaultdict(list)
    for e in rejs:
        m = val_by.get(e["pair_id"], {}).get("claimed_mechanism", "?")
        by_m[m].append(e)
    for m in by_m:
        rng.shuffle(by_m[m])

    target = 10
    sampled = []
    seen = set()
    # round 1: one per mechanism (cap span >=5)
    for m in sorted(by_m):
        if by_m[m] and len(sampled) < target:
            e = by_m[m][0]
            if e["pair_id"] not in seen:
                sampled.append(e); seen.add(e["pair_id"])
    # round 2: fill remaining slots round-robin
    rotation = [m for m in by_m if by_m[m][1:]]
    idx = 0
    while len(sampled) < target and rotation:
        m = rotation[idx % len(rotation)]
        pool = [e for e in by_m[m] if e["pair_id"] not in seen]
        if pool:
            sampled.append(pool[0]); seen.add(pool[0]["pair_id"])
            if len(pool) == 1:
                rotation.remove(m); continue
        else:
            rotation.remove(m); continue
        idx += 1

    span = sorted({val_by[e["pair_id"]]["claimed_mechanism"] for e in sampled})
    print(f"Sampled {len(sampled)} pairs spanning mechanisms: {span}\n")
    for e in sampled:
        fmt_pair(e, val_by)


def section_3_q5_only(decs, val_by):
    print("=" * 78)
    print("SECTION 3: 10 stratified rejects with Q1=true, Q2=true, Q5=false")
    print("=" * 78)
    rng = random.Random(42)
    eligible = [e for e in decs
                if e["verdict"] == "reject"
                and e.get("q1_mechanism_reality") is True
                and e.get("q2_causal_structure") is True
                and e.get("q5_reasoning_soundness") is False]
    print(f"Eligible: {len(eligible)} pairs\n")

    if len(eligible) < 10:
        print(f"NOTE: fewer than 10 qualifying pairs available — printing all {len(eligible)}.\n")
        sampled = eligible
    else:
        by_m = defaultdict(list)
        for e in eligible:
            m = val_by.get(e["pair_id"], {}).get("claimed_mechanism", "?")
            by_m[m].append(e)
        for m in by_m:
            rng.shuffle(by_m[m])
        sampled = []; seen = set()
        for m in sorted(by_m):
            if by_m[m] and len(sampled) < 10:
                e = by_m[m][0]; sampled.append(e); seen.add(e["pair_id"])
        rotation = [m for m in by_m if by_m[m][1:]]
        idx = 0
        while len(sampled) < 10 and rotation:
            m = rotation[idx % len(rotation)]
            pool = [e for e in by_m[m] if e["pair_id"] not in seen]
            if pool:
                sampled.append(pool[0]); seen.add(pool[0]["pair_id"])
                if len(pool) == 1: rotation.remove(m); continue
            else:
                rotation.remove(m); continue
            idx += 1

    span = sorted({val_by[e["pair_id"]]["claimed_mechanism"] for e in sampled})
    print(f"Printed {len(sampled)} pairs spanning mechanisms: {span}\n")
    for e in sampled:
        fmt_pair(e, val_by)


def section_4_q2_pattern(decs, val_by):
    print("=" * 78)
    print("SECTION 4: All Q2-false pairs — first-sentence of notes (raw)")
    print("=" * 78)
    q2_false = [e for e in decs if e.get("q2_causal_structure") is False]
    print(f"Total Q2=false: {len(q2_false)}\n")
    for e in q2_false:
        pid = e["pair_id"]
        c = val_by.get(pid, {})
        snippet = first_sentence(e.get("notes", ""), max_chars=200)
        print(f"- {pid} | {c.get('claimed_mechanism')} | "
              f"{c.get('concept_a')} / {c.get('concept_b')}")
        print(f"    notes[1st]: {snippet}")


def section_5_mech_q_table(decs, val_by):
    print("\n" + "=" * 78)
    print("SECTION 5: Mechanism × question failure-rate table")
    print("=" * 78)
    by_m = defaultdict(list)
    for e in decs:
        m = val_by.get(e["pair_id"], {}).get("claimed_mechanism", "?")
        by_m[m].append(e)

    mechs = sorted(by_m, key=lambda m: -len(by_m[m]))
    print(f"\n| mechanism | n | Q1_fail% | Q2_fail% | Q3_fail% | Q4_fail% | Q5_fail% | flag |")
    print(f"|---|---|---|---|---|---|---|---|")
    for m in mechs:
        lst = by_m[m]; n = len(lst)
        rates = []
        for q in QS:
            f_n = sum(1 for e in lst if e.get(q) is False)
            rates.append(100 * f_n / n if n else 0)
        flag = "n<5 unreliable" if n < 5 else ""
        print(f"| {m} | {n} | {rates[0]:.0f}% | {rates[1]:.0f}% | {rates[2]:.0f}% | "
              f"{rates[3]:.0f}% | {rates[4]:.0f}% | {flag} |")


def section_6_calibration(decs):
    print("\n" + "=" * 78)
    print("SECTION 6: Calibration anchor input (13-pair adversarial benchmark)")
    print("=" * 78)
    pairs = json.loads(CORRECTED.read_text())

    # Map original mechanism descriptions -> 12 candidate mechanisms.
    # Mechanism descriptions in corrected_evaluation.json (lowercase, free text):
    MAP = {
        "gradual internal degradation": "DEGRADATION",
        "flow splits into branches": "UNCLEAR",          # could be DIFFUSION or PROPAGATION
        "exponential spread": "PROPAGATION",
        "structural load distribution": "STRUCTURAL_LOAD",
        "competitive selection": "SELECTION",
        "layers accumulate over time": "UNCLEAR",        # accumulation, not in 12-mechanism list
        "frequency match amplifies signal": "UNCLEAR",   # resonance — between OSCILLATION & FEEDBACK_POSITIVE
        "shed old layer replace with new": "UNCLEAR",    # cyclical replacement, not in list
        "protective inoculation against future threat": "UNCLEAR",  # not in 12-list
        "dispersed becomes concentrated": "UNCLEAR",     # aggregation/condensation, not in list
        "threshold-triggered release": "THRESHOLD",
        "exponential spread through contact": "PROPAGATION",
    }

    out_pairs = []
    unclear = []
    for i, p in enumerate(pairs):
        mech_orig = p.get("mechanism", "")
        mech_canonical = MAP.get(mech_orig, "UNCLEAR")
        if mech_canonical == "UNCLEAR":
            unclear.append({"pair_id": f"calib_{i:02d}",
                            "concept_a": p["concept_a"], "concept_b": p["concept_b"],
                            "original_mechanism": mech_orig})
        out_pairs.append({
            "pair_id": f"calib_{i:02d}",
            "concept_a": p["concept_a"],
            "concept_b": p["concept_b"],
            "claimed_mechanism": mech_canonical,
            "original_mechanism_description": mech_orig,
            "concept_a_description": p["concept_a"].replace("_", " "),
            "concept_b_description": p["concept_b"].replace("_", " "),
            "reasoning": mech_orig,  # use the brief mechanism description as justification
            "trained_status": p.get("trained_status"),
            "round": p.get("round"),
        })

    print(f"\nUNCLEAR mechanism mappings — assign manually before running calibration:")
    for u in unclear:
        print(f"  - {u['pair_id']}: {u['concept_a']} / {u['concept_b']} "
              f"(orig: '{u['original_mechanism']}')")

    payload = {
        "n_pairs": len(out_pairs),
        "n_unclear": len(unclear),
        "unclear_pair_ids": [u["pair_id"] for u in unclear],
        "candidates": out_pairs,
    }
    CALIB_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {CALIB_OUT}: {len(out_pairs)} pairs, {len(unclear)} flagged UNCLEAR")


def section_7_rubric_prompt():
    print("\n" + "=" * 78)
    print("RUBRIC PROMPT TEXT (RECONSTRUCTED — NOT LITERAL)")
    print("=" * 78)
    print()
    print("Source: not stored on disk in this repository. Original prompt was "
          "applied via the VS Code Agent in an external chat session and was not "
          "checkpointed. The text below is RECONSTRUCTED from:")
    print("  - the schema of batch_*_decisions.json (q1-q5 boolean fields, score, verdict, notes)")
    print("  - the failure-mode descriptions in results/decisions/paper_phrases.md")
    print("  - the patterns visible in the actual notes output across 120 pairs")
    print("Treat as a faithful reconstruction; verify against the original chat history "
          "before relying on byte-equivalence.")
    print()
    print("--- BEGIN RECONSTRUCTED PROMPT ---")
    print(r'''
You are reviewing candidate adversarial concept pairs for a cross-domain causal-analogy
benchmark. Each pair claims that two concepts from different domains share a deep causal
mechanism. Apply the following 5-question rubric strictly. The bar is structural parity,
not surface similarity.

For each pair, answer each question true or false:

Q1 — Mechanism reality:
   Are BOTH concept_a and concept_b genuine instances of the claimed mechanism in their
   respective domains? Mark false if one side is a real mechanism and the other is a
   loose narrative or a static structural description that does not actually instantiate
   the mechanism's dynamics.

Q2 — Causal structure parity:
   Do the two concepts share the SAME causal structure under the claimed mechanism, with
   matching variables and the same form of dynamic relationship (e.g., both are closed
   feedback loops with sensed error and actuator; both have the same critical-parameter
   threshold; both follow the same diffusion law)? Mark false if the structures are
   merely metaphorically similar (e.g., one is dynamic, the other is static; one is a
   continuous controller, the other is a discretionary intervention).

Q3 — Cross-domain:
   Are the two concepts from genuinely different domains (e.g., physical vs social,
   biological vs technological)? Mark false only if they are from the same narrow domain
   such that the analogy is trivial.

Q4 — Groundedness:
   Are both concepts grounded in real-world phenomena that a domain expert would
   recognize, rather than invented or LLM-confabulated entities? Mark false if either
   side is fabricated, mislabeled, or describes a non-canonical phenomenon presented
   as canonical.

Q5 — Reasoning soundness:
   Is the supplied justification (the "reasoning" field) a sound argument for shared
   causal structure, demonstrating parity at the level of dynamic rules, not just
   shared vocabulary? Mark false if the reasoning is generic or templated language
   that would equally fit unrelated mechanisms ("both are periodic", "both have
   feedback", "both regulate flow") without committing to specific shared dynamics.

After answering Q1-Q5, compute:
  score = number of true answers (integer 0-5)
  verdict:
    5 -> "keep"
    4 -> "second_pass"
    <=3 -> "reject"

Provide a notes field with concise reasoning citing which questions failed and why.

Output a JSON array, one object per pair, with these exact keys:
  pair_id (str)
  q1_mechanism_reality (bool)
  q2_causal_structure (bool)
  q3_cross_domain (bool)
  q4_groundedness (bool)
  q5_reasoning_soundness (bool)
  score (int)
  verdict (str: "keep" | "second_pass" | "reject")
  notes (non-empty str)
''')
    print("--- END RECONSTRUCTED PROMPT ---")


def main():
    decs, val_by = load()
    assert len(decs) == 120, f"expected 120, got {len(decs)}"
    section_1_keeps(decs, val_by)
    section_2_rejects_q2(decs, val_by)
    section_3_q5_only(decs, val_by)
    section_4_q2_pattern(decs, val_by)
    section_5_mech_q_table(decs, val_by)
    section_6_calibration(decs)
    section_7_rubric_prompt()


if __name__ == "__main__":
    main()
