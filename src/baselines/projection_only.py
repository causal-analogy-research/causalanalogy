"""
Projection-only ablation — Brain minus GAT.

Single nn.Linear(384, 128) matching CausalGraphBrain.self.perceptual_proj exactly.
Consumes the same node_features Brain sees (LLM-extracted graph nodes embedded by
sentence-transformers all-MiniLM-L6-v2, shape (N, 384)), mean-pools, projects, and
L2-normalizes. No GAT, no memory, no alpha-MLP gate.

49,280 trainable parameters (= 384 * 128 + 128 bias). Isolates the perceptual
projection's contribution from the GAT path; the ablation tests whether the residual
bypass alone explains whatever signal the Brain produces.
"""
import torch.nn as nn
import torch.nn.functional as F


class ProjectionOnly(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=128):
        super().__init__()
        # Matches CausalGraphBrain.self.perceptual_proj exactly:
        # nn.Linear(embed_dim, hidden_dim), bias=True, no activation/norm/dropout.
        self.perceptual_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, node_features):
        x = node_features.mean(dim=0)        # (N, embed_dim) -> (embed_dim,)
        x = self.perceptual_proj(x)          # (embed_dim,) -> (hidden_dim,)
        return F.normalize(x, p=2, dim=-1)   # unit-sphere output, matching Brain
