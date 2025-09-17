"""
model.py

EnhancedCrossAttentionModel that:
- Encodes ESM embedding and 4 topology partitions (contact/interface/distance/topology)
- Applies an adaptive gating to fuse the 4 topo sub-encoders
- Stacks two cross-attention layers (topo->esm and esm->topo) with FFN+LayerNorm
- Predicts a single ΔG value via an MLP head
- Provides extract_features() to return fused representation for Lasso stage

All dimensions, layers, and logic exactly match your original code.
"""

import torch
import torch.nn as nn
from cross_attention import CrossAttention


class EnhancedCrossAttentionModel(nn.Module):
    def __init__(self, esm_dim, topo_dim,
                 contact_dim, interface_dim, distance_dim, topology_dim,
                 hidden_dim=256, num_heads=8, dropout=0.1):
        super(EnhancedCrossAttentionModel, self).__init__()
        self.contact_dim = contact_dim
        self.interface_dim = interface_dim
        self.distance_dim = distance_dim
        self.topology_dim = topology_dim

        # Keep hidden_dim multiple of heads (as in your original)
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads

        # ESM encoder
        self.esm_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Topology sub-encoders (4 branches)
        self.contact_encoder = nn.Sequential(
            nn.Linear(contact_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.interface_encoder = nn.Sequential(
            nn.Linear(interface_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.distance_encoder = nn.Sequential(
            nn.Linear(distance_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Adaptive gate to combine the four topo branches
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 4),
            nn.Softmax(dim=-1)
        )

        # Two stacked cross-attention layers + FFNs (ESM <-> TOPO)
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'esm2topo_attn': CrossAttention(hidden_dim, hidden_dim, num_heads, dropout),
                'topo2esm_attn': CrossAttention(hidden_dim, hidden_dim, num_heads, dropout),
                'esm_norm': nn.LayerNorm(hidden_dim),
                'topo_norm': nn.LayerNorm(hidden_dim),
                'esm_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'topo_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'esm_ffn_norm': nn.LayerNorm(hidden_dim),
                'topo_ffn_norm': nn.LayerNorm(hidden_dim)
            })
            for _ in range(2)
        ])

        # Final MLP head for regression
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _encode_topo(self, topo_features: torch.Tensor) -> torch.Tensor:
        """Split topo vector into 4 partitions and encode each branch."""
        split_sizes = [self.contact_dim, self.interface_dim, self.distance_dim, self.topology_dim]
        contact, interface, distance, topology = torch.split(topo_features, split_sizes, dim=1)

        c = self.contact_encoder(contact)
        i = self.interface_encoder(interface)
        d = self.distance_encoder(distance)
        t = self.topology_encoder(topology)

        concat = torch.cat([c, i, d, t], dim=1)
        weights = self.adaptive_gate(concat)
        fused_topo = weights[:, 0:1] * c + weights[:, 1:2] * i + weights[:, 2:3] * d + weights[:, 3:4] * t
        return fused_topo

    def _cross_fuse(self, esm: torch.Tensor, topo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Two blocks of cross-attention + FFN with residual + LayerNorm (kept as original)."""
        for layer in self.cross_layers:
            esm_attended  = layer['topo2esm_attn'](topo, esm)
            topo_attended = layer['esm2topo_attn'](esm, topo)

            esm  = layer['esm_norm'](esm  + esm_attended)
            topo = layer['topo_norm'](topo + topo_attended)

            esm_ffn  = layer['esm_ffn'](esm)
            topo_ffn = layer['topo_ffn'](topo)

            esm  = layer['esm_ffn_norm'](esm  + esm_ffn)
            topo = layer['topo_ffn_norm'](topo + topo_ffn)
        return esm, topo

    def forward(self, esm_features: torch.Tensor, topo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_features: (B, esm_dim)
            topo_features: (B, topo_dim)
        Returns:
            (B,) predicted ΔG values
        """
        esm  = self.esm_encoder(esm_features)
        topo = self._encode_topo(topo_features)
        esm, topo = self._cross_fuse(esm, topo)

        combined = torch.cat([esm, topo], dim=1)
        output = self.predictor(combined)
        return output.squeeze()

    def extract_features(self, esm_features: torch.Tensor, topo_features: torch.Tensor):
        """
        Returns the fused representation before regression head.
        Used for downstream Lasso regression stage (exactly as your code).
        """
        esm  = self.esm_encoder(esm_features)
        topo = self._encode_topo(topo_features)
        esm, topo = self._cross_fuse(esm, topo)

        combined = torch.cat([esm, topo], dim=1)
        return combined.detach().cpu().numpy()
