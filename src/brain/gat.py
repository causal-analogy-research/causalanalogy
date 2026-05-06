"""
Configurable GATv2 Brain — Graph-Level Embedding Output

Variable hidden_dim, num_layers, n_heads, alpha_mlp_hidden.
Defaults match the original 200K architecture exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """GATv2: e_ij = a^T · LeakyReLU(W_l·h_i + W_r·h_j)"""

    def __init__(self, in_features, out_features, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        assert out_features % n_heads == 0

        self.W_l = nn.Linear(in_features, out_features, bias=False)
        self.W_r = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x, edge_index, edge_mask=None):
        N = x.size(0)
        h_l = self.W_l(x).view(N, self.n_heads, self.head_dim)
        h_r = self.W_r(x).view(N, self.n_heads, self.head_dim)

        src, tgt = edge_index
        combined = self.leaky_relu(h_l[src] + h_r[tgt])
        attn_scores = (combined * self.attn.unsqueeze(0)).sum(dim=-1)

        attn_weights = self._sparse_softmax(attn_scores, tgt, N)
        attn_weights = self.dropout(attn_weights)

        h_src = h_l[src]
        weighted = attn_weights.unsqueeze(-1) * h_src
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)

        out = self.layer_norm(out.reshape(N, -1))
        return out, attn_scores

    def _sparse_softmax(self, scores, indices, N):
        max_scores = torch.zeros(N, scores.size(-1), device=scores.device)
        max_scores.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(scores),
                                    scores, reduce='amax', include_self=False)
        scores = scores - max_scores[indices]
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(N, scores.size(-1), device=scores.device)
        sum_exp.scatter_add_(0, indices.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        return exp_scores / (sum_exp[indices] + 1e-8)


class CausalGraphBrain(nn.Module):
    """
    Configurable graph-level embedding brain.

    Args:
        embed_dim: input embedding dimension (384 from sentence-transformers)
        hidden_dim: GATv2 hidden dimension (default 128 for 200K config)
        n_heads: attention heads (default 8)
        num_layers: total GATv2 layers (default 4 for 200K config)
        n_iterations: iteration loops over the full layer stack (default 1)
        alpha_res: initial residual mixing ratio (default 0.2)
        alpha_mlp_hidden: hidden size for α MLP (default 64 for 200K config)
    """

    def __init__(self, embed_dim=384, hidden_dim=128, n_heads=8,
                 num_layers=4, n_iterations=1, alpha_res=0.2,
                 alpha_mlp_hidden=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_iterations = n_iterations
        self.alpha_res = alpha_res

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # Variable number of GATv2 layers
        self.gat_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, n_heads) for _ in range(num_layers)
        ])

        # Memory gate
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )

        # Perceptual projection: embed_dim → hidden_dim
        self.perceptual_proj = nn.Linear(embed_dim, hidden_dim)

        # Alpha MLP: input-dependent mixing (input always 384)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(embed_dim, alpha_mlp_hidden),
            nn.ReLU(),
            nn.Linear(alpha_mlp_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, edge_index, edge_mask=None, memory=None,
                return_attention=False):
        x = self.input_proj(node_features)
        x_initial = x.clone()
        attn_data = {}

        for t in range(self.n_iterations):
            prev_x = x.clone()

            # First half of layers
            half = self.num_layers // 2
            for i, gat in enumerate(self.gat_layers[:half]):
                x, a = gat(x, edge_index, edge_mask)
                x = F.relu(x)
                x = (1 - self.alpha_res) * x + self.alpha_res * x_initial
                if return_attention:
                    attn_data[f"gat{i}"] = a
            x = x + prev_x  # residual over first half

            # Memory integration (between halves)
            if memory is not None:
                mem_out = memory.read(x)
                gate = self.memory_gate(torch.cat([x, mem_out], dim=-1))
                x = x * (1 - gate) + mem_out * gate

            mid_x = x.clone()

            # Second half of layers
            for i, gat in enumerate(self.gat_layers[half:]):
                x, a = gat(x, edge_index, edge_mask)
                x = F.relu(x)
                x = (1 - self.alpha_res) * x + self.alpha_res * x_initial
                if return_attention:
                    attn_data[f"gat{half+i}"] = a
            x = x + mid_x  # residual over second half

            # Memory write
            if memory is not None:
                memory.write(x.detach(), x.detach())

        # Graph-level embedding via mean pooling
        graph_embedding = x.mean(dim=0)

        # Perceptual embedding
        perceptual_raw = node_features.mean(dim=0)
        perceptual_proj = self.perceptual_proj(perceptual_raw)

        # Alpha: input-dependent mixing, clamped to [0.1, 0.9]
        alpha = self.alpha_mlp(perceptual_raw).squeeze().clamp(0.1, 0.9)

        # Final embedding: mix and L2-normalize
        final_embedding = alpha * perceptual_proj + (1 - alpha) * graph_embedding
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding, alpha, x, attn_data

    def load_checkpoint_compat(self, state_dict):
        """Load checkpoint with backward compatibility for old gat1/gat2/... naming."""
        new_sd = {}
        for k, v in state_dict.items():
            # Remap gat1.xxx -> gat_layers.0.xxx, gat2 -> gat_layers.1, etc.
            import re
            m = re.match(r'^gat(\d+)\.(.+)$', k)
            if m:
                idx = int(m.group(1)) - 1  # gat1 -> 0, gat2 -> 1, etc.
                new_key = f'gat_layers.{idx}.{m.group(2)}'
                new_sd[new_key] = v
            else:
                new_sd[k] = v
        self.load_state_dict(new_sd)
