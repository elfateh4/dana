import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RGCR(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        dist_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = node_emb.shape
        Q = (
            self.q_proj(node_emb)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(node_emb)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(node_emb)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return self.norm(node_emb + out)


class TSNR(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.gru = nn.GRUCell(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, node_emb: torch.Tensor, prev_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.gru(node_emb.mean(dim=1), prev_context)
        context = self.norm(context)
        return node_emb + context.unsqueeze(1), context


class DynamicContext(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.rgcr = RGCR(embedding_dim, num_heads)
        self.tsnr = TSNR(embedding_dim)
        self.disaster_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        dist_matrix: torch.Tensor,
        prev_context: torch.Tensor,
        disaster_occurred: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.rgcr(node_emb, dist_matrix, mask)
        x, context = self.tsnr(x, prev_context)
        return x, context
