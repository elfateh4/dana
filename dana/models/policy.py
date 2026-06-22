import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DecoderLayer(nn.Module):
    def __init__(
        self, embedding_dim: int = 128, num_heads: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, self_mask=None, cross_mask=None
    ):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + attn_out)
        attn_out, _ = self.cross_attn(x, memory, memory, attn_mask=cross_mask)
        x = self.norm2(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x


class RouteDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_proj = nn.Linear(embedding_dim * 3, embedding_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        # Project query to key-space for per-node compatibility scoring
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        context: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_starts: int = 1,
    ) -> torch.Tensor:
        B, N, D = node_embeddings.shape
        context_in = torch.cat(
            [context, vehicle_state, node_embeddings.mean(dim=1)], dim=-1
        )
        decoder_input = self.context_proj(context_in).unsqueeze(1)
        for layer in self.layers:
            decoder_input = layer(decoder_input, node_embeddings)
        query = decoder_input  # [B, 1, D]
        if num_starts > 1:
            query = query.expand(-1, num_starts, -1)
        # Per-node compatibility: query · node_embedding_i / sqrt(D)
        # This gives each node its own logit (unlike the old glimpse+logit_proj
        # which collapsed to a single scalar broadcast to all nodes)
        query = self.query_proj(query)  # [B, num_starts, D]
        logits = torch.matmul(
            query, node_embeddings.transpose(-2, -1)
        )  # [B, num_starts, N]
        logits = logits / math.sqrt(D)
        # Clip to prevent extreme logits (standard NCO trick)
        logits = 10 * torch.tanh(logits)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # [B, 1, N] for broadcasting
            logits = logits.masked_fill(mask, float("-inf"))
        if num_starts == 1:
            logits = logits.squeeze(1)  # [B, N]
        return logits


class DisasterPolicy(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        context_module: nn.Module,
        decoder: nn.Module,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.context_module = context_module
        self.decoder = decoder
        self.embedding_dim = embedding_dim
        self.vehicle_embedding = nn.Linear(3, embedding_dim)

    def forward(
        self,
        coords: torch.Tensor,
        distance_matrix: torch.Tensor,
        duration_matrix: torch.Tensor,
        depot_mask: torch.Tensor,
        demand: torch.Tensor,
        tw_start: torch.Tensor,
        tw_end: torch.Tensor,
        visited_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        node_emb = self.encoder(coords, distance_matrix)
        B, N, D = node_emb.shape
        context = torch.zeros(B, D, device=node_emb.device)
        node_emb, context = self.context_module(node_emb, distance_matrix, context)
        visit_frac = (
            visited_mask.float().mean(dim=-1)
            if visited_mask is not None
            else torch.zeros(B, device=node_emb.device)
        )
        avg_demand = demand.float().mean(dim=-1) if demand.dim() > 1 else demand.float()
        avg_tw_end = tw_end.float().mean(dim=-1) if tw_end.dim() > 1 else tw_end.float()
        vehicle_state = torch.stack([visit_frac, avg_demand, avg_tw_end], dim=-1)
        vehicle_feat = self.vehicle_embedding(vehicle_state)
        mask = visited_mask if visited_mask is not None else depot_mask
        logits = self.decoder(node_emb, context, vehicle_feat, mask=mask)
        if return_logits:
            return logits
        if action is not None:
            return F.cross_entropy(logits, action)
        probs = F.softmax(logits / 0.1, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
