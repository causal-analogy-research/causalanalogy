# CausalAnalogy

Code release for *CausalAnalogy: A Strict-Rubric Benchmark for Cross-Domain Causal Analogy with Negative Results Across Four Architecture Families*.

This repository contains the evaluation pipeline, training scripts, and benchmark construction code accompanying the paper. The benchmark dataset, distractor pools, and per-system evaluation results are released separately on HuggingFace (https://huggingface.co/datasets/causal-analogy-dataset/causalanalogy-bench).

## What this repository contains

```
scripts/
  phase_d_eval.py              24-system evaluation pipeline (Pillar 2)
  phase_b_train_brain.py       Brain Phase 5A training
  phase_b_train_ft.py          Sentence-transformer fine-tuning
  phase_b_train_projection.py  Projection-only ablation training
  phase_a_prime_validate.py    Phase A-prime candidate validation pipeline
  phase_a_prime_merge_session6.py  Session-6 candidate merge into main pool
  phase_b_overlap_scan.py      Three-tier contamination audit
  extract_phase_d.py           Haiku extraction over benchmark concepts
  extract_training_names.py    Training-pool name extraction (lockout source)
  multi_judge_agreement.py     Fleiss kappa across three LLM judges
  rubric_validation.py         Strict five-question rubric implementation
src/
  baselines/
    finetune_st.py             Sentence-transformer fine-tuning module
    projection_only.py         Projection-only ablation module
  brain/
    gat.py                     Graph-attention causal-graph module
    memory.py                  Causal memory module
  scoring/
    graph_utils.py             Causal-graph tensor conversions
    u_score.py                 Brain U-score computation
  sensors/
    embedder.py                Sentence-transformer wrapper
    extractor.py               Haiku extraction wrapper
  training/
    mining.py                  Triplet mining (semihard, batchhard)
run_phase5a.py                 Brain Phase 5A training driver
results/phase_a_prime/
  forbidden_names.txt          468 forbidden concept names (lockout corpus)
  forbidden_tokens.txt         597 forbidden tokens (lockout corpus)
  training_concept_names.json  Upstream JSON for the lockout corpus
```

## Pipeline overview

Construction methodology (Pillar 1):
- `phase_a_prime_validate.py` produces the 384-candidate pool after lockout filtering
- `rubric_validation.py` applies the strict five-question rubric (87.9% reject rate across 404 reviewed pairs)
- `multi_judge_agreement.py` computes the Fleiss kappa = 0.649 multi-judge agreement
- `phase_b_overlap_scan.py` runs the three-tier contamination audit (0/0/0 hits on Phase A-prime)

Evaluation methodology (Pillar 2):
- `extract_phase_d.py` runs Haiku extraction over the 28-pair benchmark and distractor concepts (cached output is the canonical input for evaluation)
- `phase_d_eval.py` runs the 24-system three-tier evaluation against the cached extractions

## Reproducing Phase D evaluation results

Phase D evaluation runs against the cached Haiku extractions (`cache/all_extractions_phase_d.json`, released as part of the HuggingFace dataset). No API key is required for headline reproduction.

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset and cache from HuggingFace (https://huggingface.co/datasets/causal-analogy-dataset/causalanalogy-bench)
# Place under ./dataset_release/ and ./cache/

# Run evaluation
python scripts/phase_d_eval.py
```

Output: per-system results in `results/phase_d/eval_results.json`, including T1, T2, and T3 scores for all 24 systems.

## Re-running LLM-dependent stages

Re-running the Haiku extraction or strict rubric review requires `ANTHROPIC_API_KEY`. The construction pipeline additionally requires API access to multiple frontier providers (Anthropic, Google, OpenAI). Re-running these stages produces different specific extractions and verdicts due to API non-determinism. Aggregate findings (87.9% reject rate, Fleiss kappa = 0.649, mechanism-class admissibility gradient) reflect a single sample and are not directly re-derivable; the qualitative patterns are expected to persist across re-runs. The released benchmark JSON, decision files, and Haiku extraction cache are the auditable artifacts. Full construction reproduction is API-cost-prohibitive and not the intended verification path.

## Compute requirements

Evaluation pipeline runs CPU-only with under 16GB RAM. Training scripts run CPU-only; runtime varies by hardware. Tested on Python 3.13.

## Provenance note

The lockout artifacts `forbidden_names.txt` and `forbidden_tokens.txt` were generated from `training_concept_names.json` (produced by `extract_training_names.py`) via an ad-hoc CSV-flattening transformation. The released `.txt` files are the canonical lockout corpus used in the paper; counts of 468 names and 597 tokens are directly verifiable from the file contents.

## License

MIT (see LICENSE).

## Citation

This work is under double-blind review. Citation information will be added upon publication.
