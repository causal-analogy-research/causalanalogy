"""
Sentence-transformer embeddings for graph nodes.

Replaces one-hot embeddings AND string normalization from the PoC.
"round" and "circular" get similar vectors automatically.
Uses all-MiniLM-L6-v2 (384 dimensions), runs on CPU.
"""

import hashlib
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cache_key(name: str) -> str:
    return hashlib.md5(name.lower().strip().encode()).hexdigest()


def embed_node(name: str) -> np.ndarray:
    """Embed a single node name. Returns (384,) array. Cached."""
    key = _cache_key(name)
    cache_path = CACHE_DIR / f"{key}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    model = _get_model()
    vec = model.encode(name.lower().strip(), show_progress_bar=False)
    vec = vec.astype(np.float32)
    np.save(cache_path, vec)
    return vec


def embed_nodes(names: list) -> np.ndarray:
    """
    Embed a list of node names. Returns (N, 384) array.
    Uses batch encoding for uncached nodes, individual cache for cached ones.
    """
    results = {}
    uncached = []
    uncached_indices = []

    for i, name in enumerate(names):
        key = _cache_key(name)
        cache_path = CACHE_DIR / f"{key}.npy"
        if cache_path.exists():
            results[i] = np.load(cache_path)
        else:
            uncached.append(name.lower().strip())
            uncached_indices.append(i)

    # Batch-encode uncached nodes
    if uncached:
        model = _get_model()
        vecs = model.encode(uncached, show_progress_bar=False, batch_size=64)
        for j, idx in enumerate(uncached_indices):
            vec = vecs[j].astype(np.float32)
            results[idx] = vec
            # Cache it
            key = _cache_key(names[idx])
            np.save(CACHE_DIR / f"{key}.npy", vec)

    # Assemble in order
    return np.stack([results[i] for i in range(len(names))], axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    # Quick sanity check
    print("Loading model...")
    v1 = embed_node("round")
    v2 = embed_node("circular")
    v3 = embed_node("hot")
    v4 = embed_node("temperature")
    print(f"round vs circular: {cosine_similarity(v1, v2):.3f}")
    print(f"round vs hot: {cosine_similarity(v1, v3):.3f}")
    print(f"hot vs temperature: {cosine_similarity(v3, v4):.3f}")

    batch = embed_nodes(["ball", "sphere", "wheel", "fire", "ice"])
    print(f"\nBatch shape: {batch.shape}")
    print(f"ball vs sphere: {cosine_similarity(batch[0], batch[1]):.3f}")
    print(f"ball vs fire: {cosine_similarity(batch[0], batch[3]):.3f}")
