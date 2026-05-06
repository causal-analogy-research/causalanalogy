"""
Multi-judge inter-rater agreement on the 50-pair sample.

Inputs:
  results/phase_a_prime/multi_judge/sample_judge_{gemini,sonnet,gpt}.json
Output:
  results/phase_a_prime/multi_judge/agreement_results.json

Sonnet file is wrapped in prose+code-fence; extracted via regex.
GPT may be partial (e.g., missing 8 of 50); pairwise/Fleiss kappa computed on
the intersection of pair_ids covered by the relevant judges.
"""
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

MJ = Path("results/phase_a_prime/multi_judge")
QUESTIONS = ["q1_mechanism_reality", "q2_causal_structure", "q3_cross_domain",
             "q4_groundedness", "q5_reasoning_soundness"]
VERDICT_KEY = "verdict"
VERDICT_CATS = ["keep", "second_pass", "reject"]


def load_judge(path):
    txt = path.read_text()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", txt, re.S)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"(\[.*\])", txt, re.S)
        if m:
            return json.loads(m.group(1))
        raise


def cohens_kappa(a, b):
    """Cohen's kappa for two parallel sequences of categorical labels."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((len(cats), len(cats)), dtype=int)
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    po = cm.trace() / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float((row_marg * col_marg).sum())
    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def fleiss_kappa(ratings, categories):
    """ratings: list of dicts {category: count} per item; n raters per item assumed equal."""
    if not ratings:
        return None
    N = len(ratings)
    n = sum(ratings[0].values())
    if n < 2:
        return None
    # check uniform n
    for r in ratings:
        if sum(r.values()) != n:
            raise ValueError("non-uniform raters per item")
    # category proportions overall
    totals = Counter()
    for r in ratings:
        for c, k in r.items():
            totals[c] += k
    p = {c: totals.get(c, 0) / (N * n) for c in categories}
    P_e = sum(v * v for v in p.values())
    # per-item agreement
    P_is = []
    for r in ratings:
        s = sum(r.get(c, 0) ** 2 for c in categories)
        P_i = (s - n) / (n * (n - 1))
        P_is.append(P_i)
    P_bar = sum(P_is) / N
    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def main():
    judges_raw = {
        "gemini": load_judge(MJ / "sample_judge_gemini.json"),
        "sonnet": load_judge(MJ / "sample_judge_sonnet.json"),
        "gpt":    load_judge(MJ / "sample_judge_gpt.json"),
    }
    # index by pair_id for each judge
    by_id = {j: {r["pair_id"]: r for r in arr} for j, arr in judges_raw.items()}
    coverage = {j: len(by_id[j]) for j in by_id}
    print(f"Coverage: {coverage}")

    judges = ["gemini", "sonnet", "gpt"]

    # ── Pairwise Cohen's kappa for q1-q5 and verdict ──
    pairwise_q = {}     # question -> {pair-name: kappa}
    pairwise_v = {}
    for j1, j2 in combinations(judges, 2):
        ids = sorted(set(by_id[j1].keys()) & set(by_id[j2].keys()))
        for q in QUESTIONS:
            a = [bool(by_id[j1][pid][q]) for pid in ids]
            b = [bool(by_id[j2][pid][q]) for pid in ids]
            pairwise_q.setdefault(q, {})[f"{j1}-{j2}"] = {
                "kappa": round(cohens_kappa(a, b), 4),
                "n_items": len(ids),
            }
        a = [by_id[j1][pid][VERDICT_KEY] for pid in ids]
        b = [by_id[j2][pid][VERDICT_KEY] for pid in ids]
        pairwise_v[f"{j1}-{j2}"] = {
            "kappa": round(cohens_kappa(a, b), 4),
            "n_items": len(ids),
        }

    # ── Fleiss' kappa: requires same n_raters per item; use intersection of all 3 ──
    intersect_all = sorted(set(by_id["gemini"]) & set(by_id["sonnet"]) & set(by_id["gpt"]))
    fleiss_q = {}
    for q in QUESTIONS:
        ratings = []
        for pid in intersect_all:
            c = Counter()
            for j in judges:
                c[bool(by_id[j][pid][q])] += 1
            ratings.append(c)
        fleiss_q[q] = {"kappa": round(fleiss_kappa(ratings, [True, False]), 4),
                        "n_items": len(intersect_all)}

    fleiss_verdict_ratings = []
    for pid in intersect_all:
        c = Counter()
        for j in judges:
            c[by_id[j][pid][VERDICT_KEY]] += 1
        fleiss_verdict_ratings.append(c)
    fleiss_verdict = {"kappa": round(fleiss_kappa(fleiss_verdict_ratings, VERDICT_CATS), 4),
                       "n_items": len(intersect_all)}

    # ── Per-pair disagreement breakdown (verdict) on intersection ──
    breakdown = Counter()  # 'all_agree' | 'majority' | 'split'
    per_pair_dis = []
    for pid in intersect_all:
        verdicts = {j: by_id[j][pid][VERDICT_KEY] for j in judges}
        vals = list(verdicts.values())
        c = Counter(vals)
        max_count = max(c.values())
        if max_count == 3:
            breakdown["all_agree"] += 1
            dis_score = 0
        elif max_count == 2:
            breakdown["majority"] += 1
            dis_score = 1
        else:
            breakdown["split"] += 1
            dis_score = 2
        # Refine score by also counting q1-q5 disagreement
        q_dis = 0
        for q in QUESTIONS:
            qvals = [bool(by_id[j][pid][q]) for j in judges]
            if not (qvals[0] == qvals[1] == qvals[2]):
                q_dis += 1
        per_pair_dis.append({
            "pair_id": pid, "verdicts": verdicts, "verdict_dis": dis_score,
            "q_dis_count": q_dis, "total_dis": dis_score * 5 + q_dis,
        })

    per_pair_dis.sort(key=lambda x: -x["total_dis"])
    top5 = per_pair_dis[:5]

    # ── Per-judge verdict distribution (over their full coverage) ──
    per_judge_verdict = {}
    for j in judges:
        c = Counter(r[VERDICT_KEY] for r in judges_raw[j])
        per_judge_verdict[j] = {k: c.get(k, 0) for k in VERDICT_CATS}

    # ── Output JSON ──
    out = {
        "coverage": coverage,
        "intersection_all_three": len(intersect_all),
        "pairwise_question_kappa": pairwise_q,
        "fleiss_question_kappa": fleiss_q,
        "pairwise_verdict_kappa": pairwise_v,
        "fleiss_verdict_kappa": fleiss_verdict,
        "verdict_agreement_breakdown": dict(breakdown),
        "top_disagreement_pairs": top5,
        "per_judge_verdict_distribution": per_judge_verdict,
    }
    out_path = MJ / "agreement_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ── Stdout ──
    print("\n## Per-question kappa (5 questions × 4 columns)")
    print(f"| question | gemini-sonnet | gemini-gpt | sonnet-gpt | fleiss |")
    print(f"|---|---|---|---|---|")
    for q in QUESTIONS:
        gs = pairwise_q[q]["gemini-sonnet"]["kappa"]
        gg = pairwise_q[q]["gemini-gpt"]["kappa"]
        sg = pairwise_q[q]["sonnet-gpt"]["kappa"]
        fl = fleiss_q[q]["kappa"]
        print(f"| {q} | {gs:.3f} | {gg:.3f} | {sg:.3f} | {fl:.3f} |")

    print("\n## Verdict-level kappa")
    print(f"| comparison | kappa | n |")
    print(f"|---|---|---|")
    for k, v in pairwise_v.items():
        print(f"| {k} | {v['kappa']:.3f} | {v['n_items']} |")
    print(f"| fleiss (3 judges) | {fleiss_verdict['kappa']:.3f} | {fleiss_verdict['n_items']} |")

    # Threshold check
    all_kappas = []
    for q in QUESTIONS:
        for k, v in pairwise_q[q].items():
            all_kappas.append((f"{q}/{k}", v["kappa"]))
        all_kappas.append((f"{q}/fleiss", fleiss_q[q]["kappa"]))
    for k, v in pairwise_v.items():
        all_kappas.append((f"verdict/{k}", v["kappa"]))
    all_kappas.append(("verdict/fleiss", fleiss_verdict["kappa"]))

    problem = [(n, k) for n, k in all_kappas if k < 0.4]
    moderate = [(n, k) for n, k in all_kappas if 0.4 <= k < 0.6]
    good = [(n, k) for n, k in all_kappas if k >= 0.6]

    print("\n## Threshold counts (out of {} kappa values)".format(len(all_kappas)))
    print(f"- problem (<0.4):  {len(problem)}")
    if problem:
        for n, k in problem: print(f"    {n}: {k:.3f}")
    print(f"- moderate (0.4-0.6): {len(moderate)}")
    if moderate:
        for n, k in moderate: print(f"    {n}: {k:.3f}")
    print(f"- good (>=0.6): {len(good)}")

    # Per-pair agreement breakdown
    print("\n## Verdict agreement breakdown (over {} pairs all 3 rated)".format(len(intersect_all)))
    print(f"- all 3 agree:        {breakdown.get('all_agree', 0)}")
    print(f"- 2 agree + 1 dissent: {breakdown.get('majority', 0)}")
    print(f"- all 3 disagree:     {breakdown.get('split', 0)}")

    # Per-judge verdict distribution
    print("\n## Per-judge verdict distribution (full coverage)")
    print(f"| judge | n_rated | keep | second_pass | reject |")
    print(f"|---|---|---|---|---|")
    for j in judges:
        v = per_judge_verdict[j]
        n = coverage[j]
        print(f"| {j} | {n} | {v['keep']} | {v['second_pass']} | {v['reject']} |")

    # Top-5 disagreement
    print("\n## Top 5 most-disagreed pairs")
    for r in top5:
        print(f"- {r['pair_id']}  verdicts={r['verdicts']}  q_dis={r['q_dis_count']}/5")

    # Final overall decision (per spec)
    # Use the Fleiss verdict kappa as the headline number
    headline = fleiss_verdict["kappa"]
    if headline >= 0.6:
        label = "good"
    elif headline >= 0.4:
        label = "moderate"
    else:
        label = "problem"
    print(f"\n## Overall paper-reportable agreement: **{label}** "
          f"(Fleiss verdict κ = {headline:.3f})")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
