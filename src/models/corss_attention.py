"""
cross_attention.py

Minimal multi-head cross-attention block used in the pipeline.
Design note: inputs are single-vector encodings (no explicit sequence length),
so attention degenerates to per-head scalar weighting across modality embeddings.
This mirrors your original implementation and preserves all logic/parameters.
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Cross-attention between two single-vector embeddings (query vs. key/value).
    - Projects query, key, value to the same hidden size
    - Computes per-head attention weights (scalar in this single-token setup)
    - Returns fused output projected back to hidden size
    """

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = key_dim  # use key dimension as hidden size (matches your code)

        # Ensure hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            self.hidden_dim = (self.hidden_dim // self.num_heads) * self.num_heads

        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, self.hidden_dim)
        self.k_proj = nn.Linear(key_dim, self.hidden_dim)
        self.v_proj = nn.Linear(key_dim, self.hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, query_dim)
            key_value: (B, key_dim)
        Returns:
            Tensor of shape (B, hidden_dim)
        """
        batch_size = query.size(0)

        # Linear projections to multi-head representations
        q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, self.num_heads, self.head_dim)

        # Reshape for attention scores: (B, H, 1, D) @ (B, H, D, 1) -> (B, H, 1, 1)
        q = q.unsqueeze(2)
        k = k.unsqueeze(2).transpose(2, 3)

        attn = (q @ k) * self.scale           # (B, H, 1, 1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply weights to values and merge heads
        v = v.unsqueeze(2)                    # (B, H, 1, D)
        out = (attn @ v).view(batch_size, self.hidden_dim)  # (B, H*D)

        # Final linear projection
        out = self.out_proj(out)
        return out
