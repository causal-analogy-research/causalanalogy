"""
LLM Extraction Pipeline — Sensory Organs

LLM-R: Extracts physical attributes and object class as nodes
LLM-C: Extracts causal relationships as directed edges

Both are FROZEN. They are the outside world feeding raw signal to the brain.
"""

import json
import os
import hashlib
from pathlib import Path

import anthropic

CACHE_DIR = Path("cache/extractions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def _cache_key(prefix: str, input_text: str) -> str:
    h = hashlib.md5(input_text.encode()).hexdigest()
    return f"{prefix}_{h}"


def _load_cache(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data):
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def extract_nodes(input_text: str, run_id: int = 0, compound: bool = False,
                   abstract: bool = False) -> dict:
    """
    LLM-R: Recognition Engine.
    Extracts attributes, properties, and categories as nodes.
    compound=True: sub-component properties for compound systems.
    abstract=True: conceptual attributes with physical metaphors.
    """
    level_tag = '_a' if abstract else ('_c' if compound else '')
    ck = _cache_key(f"nodes_r{run_id}{level_tag}", input_text)
    cached = _load_cache(ck)
    if cached is not None:
        return cached

    extra_rule = ""
    if compound:
        extra_rule = ("- Include properties of the sub-components, not just the system as a whole. "
                      "For example, for 'steam engine', include properties of fire (hot, combustion), "
                      "water (liquid, boiling), and wheel (circular, rotating).\n")
    elif abstract:
        extra_rule = ("- Extract conceptual attributes and functional properties. "
                      "Include structural metaphors — if the concept involves balance, fairness, flow, "
                      "pressure, cycles, or other physical-metaphorical properties, include those as nodes. "
                      "Use abstract nouns and adjectives (fair, balanced, legitimate, coercive).\n")

    prompt = f"""You are a recognition engine. Given an input concept, extract physical attributes,
properties, and object categories as a flat list of nodes.

Rules:
- Return max 15 nodes.
- Use single common English words. Prefer adjectives for properties (round, hot, fragile) and nouns for objects/categories (ball, sphere, toy). Avoid phrases longer than 2 words.
- Include the object itself, its physical properties, its category/class
- Include functional properties (what it can do / what can be done to it)
{extra_rule}- Do NOT include causal relationships — just raw attributes
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"

Return format:
{{"nodes": ["node1", "node2", "node3", ...]}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    result = json.loads(text)
    _save_cache(ck, result)
    return result


def extract_edges(input_text: str, nodes: list, run_id: int = 0,
                   abstract: bool = False) -> dict:
    """
    LLM-C: Causal Engine.
    Given nodes, extracts causal relationships as directed edges.
    """
    level_tag = '_a' if abstract else ''
    ck = _cache_key(f"edges_r{run_id}{level_tag}", input_text + str(nodes))
    cached = _load_cache(ck)
    if cached is not None:
        return cached

    if abstract:
        relation_guidance = ('- Relations include: "enables", "requires", "prevents", '
                             '"undermines", "legitimizes", "destabilizes", "causes"')
    else:
        relation_guidance = ('- Relations should be genuinely causal: "because", "enables", '
                             '"causes", "prevents", "requires"')

    prompt = f"""You are a causal reasoning engine. Given an input concept and its extracted nodes,
identify causal relationships between nodes as directed edges.

Rules:
- Return max 10 edges.
- Each edge is (source_node -> target_node) with a relation type
{relation_guidance}
- Do NOT invent nodes — only use nodes from the provided list
- Prefer specific causal chains over vague associations
- Return ONLY valid JSON, no markdown, no explanation

Input: "{input_text}"
Nodes: {json.dumps(nodes)}

Return format:
{{"edges": [
  {{"source": "node_a", "target": "node_b", "relation": "causes"}},
  ...
]}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    result = json.loads(text)
    _save_cache(ck, result)
    return result


def extract_full(input_text: str, run_id: int = 0, compound: bool = False,
                  abstract: bool = False) -> dict:
    """Full extraction pipeline: nodes then edges."""
    nodes_data = extract_nodes(input_text, run_id, compound=compound, abstract=abstract)
    edges_data = extract_edges(input_text, nodes_data["nodes"], run_id, abstract=abstract)
    return {
        "input": input_text,
        "nodes": nodes_data["nodes"],
        "edges": edges_data["edges"],
        "run_id": run_id
    }


def run_extraction(config: dict, num_runs: int = 3, compound: bool = False,
                    abstract: bool = False) -> dict:
    """
    Run full extraction on all objects, all variants, multiple runs.

    Args:
        config: loaded level1_physical.json or level2_compound.json
        num_runs: number of extraction runs per variant
        compound: if True, use compound-aware node extraction prompt

    Returns:
        dict mapping object_name -> variant_name -> list of extraction dicts
    """
    results = {}
    total = 0
    success = 0
    failed = 0

    for group_name, group_data in config["groups"].items():
        objects = group_data["objects"]
        print(f"\n{'='*60}")
        print(f"Group: {group_name} ({len(objects)} objects)")
        print(f"{'='*60}")

        for obj_name, variants in objects.items():
            results[obj_name] = {}

            for variant_name, variant_text in variants.items():
                results[obj_name][variant_name] = []

                for run_id in range(num_runs):
                    total += 1
                    print(f"  {obj_name}/{variant_name} (run {run_id})... ", end="", flush=True)
                    try:
                        data = extract_full(variant_text, run_id, compound=compound,
                                            abstract=abstract)
                        results[obj_name][variant_name].append(data)
                        success += 1
                        print(f"ok ({len(data['nodes'])} nodes, {len(data['edges'])} edges)")
                    except Exception as e:
                        failed += 1
                        results[obj_name][variant_name].append({"error": str(e)})
                        print(f"FAIL ({e})")

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE: {success}/{total} succeeded, {failed} failed")
    print(f"{'='*60}")

    # Save all results
    all_path = Path("cache/all_extractions.json")
    with open(all_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Quick test on 2 objects
    test_config = {
        "groups": {
            "test": {
                "causal_property": "test",
                "objects": {
                    "ball": {
                        "base": "red ball",
                        "paraphrase_1": "spherical red object used for throwing"
                    }
                }
            }
        }
    }
    results = run_extraction(test_config, num_runs=1)
    print(json.dumps(results, indent=2))
