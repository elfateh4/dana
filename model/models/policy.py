"""RouteFinder/RRNCO-style decoder and full DANA policy.

Decoder follows RRNCO Sec. 5.2 (which combines ReLD and MatNet):
- Context vector built from the LAST VISITED node's row embedding, the
  dynamic (GRU) context, and the vehicle-state features (Eq. 12 context)
- Single-query MHA over the column node embeddings (Eq. 12)
- ReLD identity residual: h' = h' + IDT(h_ctx) (Eq. 13)
- MLP residual: q = h' + MLP(h') (Eq. 14)
- Compatibility with logit clipping and a -log(dist) nearest-neighbor
  heuristic term (Eq. 15). DANA keeps its learnable clip/temperature
  instead of the paper's fixed C.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IdentityMap(nn.Module):
    """IDT(.) from ReLD (Eq. 13): reshape context to the query dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RouteDecoder(nn.Module):
    """Single-query attention decoder over dual node embeddings."""

    CTX_MULT = 3  # last-node row embedding + dynamic context + vehicle state

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,  # kept for config compatibility; paper uses one MHA
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        ctx_dim = embedding_dim * self.CTX_MULT

        self.ctx_proj = nn.Linear(ctx_dim, embedding_dim)
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.key_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.val_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.idt = IdentityMap(ctx_dim, embedding_dim)  # Eq. 13
        self.mlp = nn.Sequential(  # Eq. 14
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.logit_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Learnable clipping & temperature (DANA improvement over fixed C=10)
        self.logit_clip = nn.Parameter(torch.tensor(10.0))
        self.logit_temperature = nn.Parameter(torch.tensor(1.0))
        self.logit_clip_min = 0.5

    def forward(
        self,
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor,
        context: torch.Tensor,
        vehicle_state: torch.Tensor,
        last_node: torch.Tensor,
        dist_row: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        row_embeddings: [B, N, D]   encoder row (outgoing) embeddings
        col_embeddings: [B, N, D]   encoder column (incoming) embeddings
        context:        [B, D]      dynamic (GRU) context vector
        vehicle_state:  [B, D]      embedded vehicle-state features
        last_node:      [B]         index of the last visited node
        dist_row:       [B, N]      normalized distances from last node (Eq. 15)
        mask:           [B, N]      True = infeasible/visited
        """
        B, N, D = row_embeddings.shape

        # Context vector: [h_r(a_{t-1}); dynamic context; vehicle state]
        h_last = row_embeddings.gather(
            1, last_node.view(B, 1, 1).expand(-1, 1, D)
        ).squeeze(1)
        h_ctx = torch.cat([h_last, context, vehicle_state], dim=-1)  # [B, 3D]

        # Single-query MHA over column embeddings (Eq. 12)
        query = self.ctx_proj(h_ctx).unsqueeze(1)  # [B, 1, D]
        keys = self.key_proj(col_embeddings)
        vals = self.val_proj(col_embeddings)
        h, _ = self.mha(query, keys, vals)

        h = h + self.idt(h_ctx).unsqueeze(1)  # ReLD residual (Eq. 13)
        q = h + self.mlp(h)  # Eq. 14

        # Compatibility scores (Eq. 15)
        logits = torch.matmul(q, self.logit_proj(col_embeddings).transpose(-2, -1))
        logits = logits.squeeze(1) / math.sqrt(D)

        clip = F.softplus(self.logit_clip).clamp(min=self.logit_clip_min)
        temperature = F.softplus(self.logit_temperature).clamp(min=0.1)
        logits = clip * torch.tanh(logits / temperature)

        if dist_row is not None:
            # Nearest-neighbor heuristic: prioritize close nodes (Eq. 15)
            logits = logits - torch.log(dist_row.clamp(min=1e-3))

        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        return logits


class DisasterPolicy(nn.Module):
    """Full policy: dual-embedding encoder -> dynamic context -> decoder."""

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

        # Vehicle-state embedding (4 features -> D-dim)
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def build_node_feats(
        demand: torch.Tensor,
        tw_start: torch.Tensor,
        tw_end: torch.Tensor,
        depot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-node attribute features, normalized per instance."""
        d = demand.float() / demand.float().amax(dim=-1, keepdim=True).clamp(min=1e-9)
        horizon = tw_end.float().amax(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.stack(
            [d, tw_start.float() / horizon, tw_end.float() / horizon,
             depot_mask.float()],
            dim=-1,
        )

    def compute_vehicle_state(
        self,
        visited_mask: Optional[torch.Tensor],
        demand: torch.Tensor,
        tw_end: torch.Tensor,
        batch: int,
        num_nodes: int,
        device,
    ) -> torch.Tensor:
        if visited_mask is not None:
            visit_frac = visited_mask.float().mean(dim=-1)
        else:
            visit_frac = torch.zeros(batch, device=device)
        remaining_cap = 1.0 - (
            demand.float().mean(dim=-1) * visit_frac
            if demand.dim() > 1
            else demand.float() * visit_frac
        )
        if visited_mask is not None and tw_end.dim() > 1:
            unvisited_tw_end = tw_end.float().masked_fill(visited_mask, float("inf"))
            min_tw_end = unvisited_tw_end.min(dim=-1).values
            max_tw_end = unvisited_tw_end.max(dim=-1).values
            tw_urgency = 1.0 - (min_tw_end / (max_tw_end + 1e-8))
        else:
            tw_urgency = torch.zeros(batch, device=device)
        if visited_mask is not None:
            unvisited_count = (~visited_mask).float().sum(dim=-1) / num_nodes
        else:
            unvisited_count = torch.ones(batch, device=device)
        vs = torch.stack(
            [visit_frac, remaining_cap, tw_urgency, unvisited_count], dim=-1
        )
        return self.vehicle_embedding(vs)

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
        num_starts: int = 1,  # kept for API compatibility; batch is flat
        last_node: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device

        node_feats = self.build_node_feats(demand, tw_start, tw_end, depot_mask)
        row_emb, col_emb = self.encoder(
            coords, distance_matrix, duration_matrix, node_feats
        )

        context = torch.zeros(B, self.embedding_dim, device=device)
        col_emb, context = self.context_module(col_emb, distance_matrix, context)

        vehicle_feat = self.compute_vehicle_state(
            visited_mask, demand, tw_end, B, N, device
        )

        if last_node is None:
            # Default: start from the first depot
            last_node = depot_mask.float().argmax(dim=-1).long()

        dist_n = distance_matrix / distance_matrix.amax(
            dim=(-2, -1), keepdim=True
        ).clamp(min=1e-9)
        dist_row = dist_n[torch.arange(B, device=device), last_node]  # [B, N]

        mask = visited_mask if visited_mask is not None else depot_mask
        logits = self.decoder(
            row_emb, col_emb, context, vehicle_feat, last_node, dist_row, mask=mask
        )

        if return_logits:
            return logits
        if action is not None:
            return F.cross_entropy(logits, action)
        probs = F.softmax(logits / 0.1, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
