import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm for stable training.

    Uses pre-norm (norm -> attn/ff -> residual) instead of post-norm,
    which provides more stable gradients in deep decoders.
    """

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
        # Pre-LayerNorm (applied *before* each sublayer)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, self_mask=None, cross_mask=None
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=self_mask)
        x = x + attn_out
        # Cross-attention with pre-norm
        norm_x = self.norm2(x)
        attn_out, _ = self.cross_attn(norm_x, memory, memory, attn_mask=cross_mask)
        x = x + attn_out
        # FF with pre-norm
        norm_x = self.norm3(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x


class RouteDecoder(nn.Module):
    """Attention-based decoder with per-node compatibility scoring.

    Improvements over baseline:
    - Pre-LayerNorm transformer layers (stable deep training)
    - Learnable logit scale & clip (adaptive output distribution)
    - GELU activation in FF (smoother gradients)
    - Richer context construction
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Context projection: embeds the concatenation of context + vehicle state
        self.context_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.context_norm = nn.LayerNorm(embedding_dim)

        # Decoder transformer layers
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Per-node compatibility scoring
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

        # Learnable temperature & clipping (instead of hardcoded 10*tanh)
        # Initialized to produce similar behavior as 10*tanh at start
        # min_clip prevents entropy regularization from collapsing logits to zero
        self.logit_clip = nn.Parameter(torch.tensor(10.0))
        self.logit_temperature = nn.Parameter(torch.tensor(1.0))
        self.logit_clip_min = 0.5

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        context: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_starts: int = 1,
    ) -> torch.Tensor:
        B, N, D = node_embeddings.shape

        # Build decoder input from context + vehicle state
        # (removed mean node embedding — it leaked information about unvisited nodes
        #  and duplicated the context which already summarizes the graph)
        if num_starts > 1:
            # Per-start context: each start gets its own context by attending to
            # a different node embedding as the initial query
            context_in = torch.cat(
                [
                    context.unsqueeze(1).expand(-1, num_starts, -1),
                    vehicle_state.unsqueeze(1).expand(-1, num_starts, -1),
                ],
                dim=-1,
            )  # [B, num_starts, 2*D]
            decoder_input = self.context_proj(context_in)  # [B, num_starts, D]
        else:
            context_in = torch.cat([context, vehicle_state], dim=-1)  # [B, 2*D]
            decoder_input = self.context_proj(context_in).unsqueeze(1)  # [B, 1, D]

        decoder_input = self.context_norm(decoder_input)
        decoder_input = self.dropout(decoder_input)

        # Pass through decoder layers
        query = decoder_input
        for layer in self.layers:
            query = layer(query, node_embeddings)

        # Per-node compatibility: query · node_embedding_i / sqrt(D)
        query = self.query_proj(query)  # [B, num_starts or 1, D]
        logits = torch.matmul(query, node_embeddings.transpose(-2, -1))
        logits = logits / math.sqrt(D)

        # Learnable clipping with temperature scaling
        # logit_clip_min floor prevents entropy reg from collapsing output to zero
        clip = F.softplus(self.logit_clip).clamp(min=self.logit_clip_min)  # ~10 at init
        temperature = F.softplus(self.logit_temperature).clamp(min=0.1)  # min 0.1
        logits = clip * torch.tanh(logits / temperature)

        # Apply mask (visited nodes, capacity violations, etc.)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # [B, 1, N] for broadcasting
            logits = logits.masked_fill(mask, float("-inf"))

        if num_starts == 1:
            logits = logits.squeeze(1)  # [B, N]

        return logits


class DisasterPolicy(nn.Module):
    """Full policy: encoder -> context -> decoder with vehicle state tracking."""

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

        # Richer vehicle state embedding (4 features -> D-dim)
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

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
        num_starts: int = 1,
    ) -> torch.Tensor:
        # Encode graph
        node_emb = self.encoder(coords, distance_matrix)
        B, N, D = node_emb.shape

        # Context: initialized as zeros, updated by DynamicContext
        context = torch.zeros(B, D, device=node_emb.device)
        node_emb, context = self.context_module(node_emb, distance_matrix, context)

        # Richer vehicle state features (per-batch-element):
        visit_frac = (
            visited_mask.float().mean(dim=-1)
            if visited_mask is not None
            else torch.zeros(B, device=node_emb.device)
        )
        # Capacity remaining (normalized)
        remaining_cap = 1.0 - (
            demand.float().mean(dim=-1) * visit_frac
            if demand.dim() > 1
            else demand.float() * visit_frac
        )
        # Time window urgency: how tight are unvisited nodes' windows
        if visited_mask is not None and tw_end.dim() > 1:
            unvisited_tw_end = tw_end.float().masked_fill(visited_mask, float("inf"))
            min_tw_end = unvisited_tw_end.min(dim=-1).values  # [B]
            max_tw_end = unvisited_tw_end.max(dim=-1).values
            tw_urgency = 1.0 - (min_tw_end / (max_tw_end + 1e-8))
        else:
            tw_urgency = torch.zeros(B, device=node_emb.device)
        # Unvisited count (normalized)
        unvisited_count = (
            (~visited_mask).float().sum(dim=-1) / N
            if visited_mask is not None
            else torch.ones(B, device=node_emb.device)
        )

        vehicle_state = torch.stack(
            [visit_frac, remaining_cap, tw_urgency, unvisited_count], dim=-1
        )  # [B, 4]
        vehicle_feat = self.vehicle_embedding(vehicle_state)  # [B, D]

        # Mask: visited nodes + depots (depots stay always "visited" as return points)
        mask = visited_mask if visited_mask is not None else depot_mask

        # Decode
        logits = self.decoder(
            node_emb,
            context,
            vehicle_feat,
            mask=mask,
            num_starts=num_starts,
        )

        if return_logits:
            return logits
        if action is not None:
            return F.cross_entropy(logits, action)
        probs = F.softmax(logits / 0.1, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
