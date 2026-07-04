import torch
import torch.nn as nn
import math
from typing import Optional


class GraphEmbedding(nn.Module):
    def __init__(self, input_dim: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe(self.proj(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DistanceBias(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.dist_proj = nn.Linear(1, embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, node_emb: torch.Tensor, dist_matrix: torch.Tensor
    ) -> torch.Tensor:
        dist_feat = self.dist_proj(dist_matrix.unsqueeze(-1))
        n = node_emb.size(1)
        node_exp = node_emb.unsqueeze(2).expand(-1, -1, n, -1)
        g = self.gate(torch.cat([node_exp, dist_feat], dim=-1))
        return node_emb + (g * dist_feat).sum(dim=2)


class AdaptiveAttentionFusion(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 12,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = GraphEmbedding(input_dim=2, embedding_dim=embedding_dim)
        self.distance_bias = DistanceBias(embedding_dim)
        self.self_attn = AdaptiveAttentionFusion(embedding_dim, num_heads)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        coords: torch.Tensor,
        distance_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(coords)
        x = self.distance_bias(x, distance_matrix)
        x = self.self_attn(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.output_proj(x)
