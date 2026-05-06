"""
Persistent Causal Memory Module

Key-value store where keys are embedding vectors and values are
causal primitive representations. This is how "round->rolls" learned
from ball becomes available when processing wheel.

256 slots x 128 dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMemory(nn.Module):
    """
    Differentiable key-value memory for causal primitives.

    - read(): cosine similarity lookup, returns weighted sum of top-k
    - write(): EMA update of matching slots
    - Slots are learned parameters that persist across objects within an episode
    """

    def __init__(self, n_slots: int = 256, key_dim: int = 128, value_dim: int = 128, top_k: int = 8):
        super().__init__()
        self.n_slots = n_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.top_k = top_k

        # Learnable memory slots
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_slots, value_dim) * 0.02)

        # EMA decay for writes
        self.ema_alpha = 0.1

    def read(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using cosine similarity.

        Args:
            queries: (N, key_dim) query vectors

        Returns:
            (N, value_dim) retrieved memory values
        """
        # Normalize for cosine similarity
        q_norm = F.normalize(queries, dim=-1)          # (N, key_dim)
        k_norm = F.normalize(self.keys, dim=-1)        # (S, key_dim)

        # Similarity scores
        sim = torch.mm(q_norm, k_norm.t())             # (N, S)

        # Top-k selection
        k = min(self.top_k, self.n_slots)
        topk_sim, topk_idx = sim.topk(k, dim=-1)      # (N, k)

        # Softmax over top-k similarities for attention weights
        attn = F.softmax(topk_sim * 10.0, dim=-1)     # (N, k), temperature-scaled

        # Gather top-k values and compute weighted sum
        topk_values = self.values[topk_idx]             # (N, k, value_dim)
        result = (attn.unsqueeze(-1) * topk_values).sum(dim=1)  # (N, value_dim)

        return result

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Update memory slots via exponential moving average.
        Non-differentiable — runs under no_grad.

        Args:
            keys: (M, key_dim) new key vectors
            values: (M, value_dim) new value vectors
        """
        # Find closest existing slot for each new key
        k_norm = F.normalize(keys, dim=-1)
        mem_k_norm = F.normalize(self.keys.data, dim=-1)
        sim = torch.mm(k_norm, mem_k_norm.t())         # (M, S)
        closest = sim.argmax(dim=-1)                    # (M,)

        # EMA update of matched slots
        alpha = self.ema_alpha
        for i, slot_idx in enumerate(closest):
            self.keys.data[slot_idx] = (1 - alpha) * self.keys.data[slot_idx] + alpha * keys[i]
            self.values.data[slot_idx] = (1 - alpha) * self.values.data[slot_idx] + alpha * values[i]

    def reset(self):
        """Reset memory to small random values (for new episodes if needed)."""
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.values, std=0.02)


if __name__ == "__main__":
    mem = CausalMemory(n_slots=256, key_dim=128, value_dim=128)
    print(f"Memory parameters: {sum(p.numel() for p in mem.parameters()):,}")
    print(f"  Keys: {mem.keys.shape}")
    print(f"  Values: {mem.values.shape}")

    # Test read
    q = torch.randn(5, 128)
    out = mem.read(q)
    print(f"\nRead test: query {q.shape} -> output {out.shape}")

    # Test write
    k = torch.randn(3, 128)
    v = torch.randn(3, 128)
    mem.write(k, v)
    print(f"Write test: wrote 3 entries")

    # Test that reading after writing returns something different
    out2 = mem.read(q)
    diff = (out - out2).norm().item()
    print(f"Read changed after write: delta={diff:.4f}")
