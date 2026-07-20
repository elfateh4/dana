"""RRNCO-style graph encoder (Son et al., 2025, arXiv:2503.16159).

Implements the paper's encoder:
- Inverse-distance selective sampling of the distance matrix (Eq. 4-5)
- Contextual Gating fusing coordinate and distance features (Eq. 7-8)
- Dual row/column embeddings (MatNet-style) for asymmetric matrices (Eq. 9-10)
- Neural Adaptive Bias (NAB) built from distance and angle matrices
- Adaptation Attention-Free Module (AAFM, Eq. 11) with instance normalization

DANA extension over the paper: the NAB gate fuses THREE edge channels
(distance, duration, angle) instead of two, so travel-time asymmetry
reaches the encoder alongside distance asymmetry.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InstanceNorm(nn.Module):
    """Instance normalization over the node dimension (per-feature)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> normalize each feature across nodes
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class InverseDistanceEmbedding(nn.Module):
    """Selective sampling of the k most relevant distances per node (Eq. 4-5).

    For each node i, k neighbors are sampled with probability proportional to
    1/d_ij; the sampled distances are projected into the embedding space.
    Sampling is stochastic during training and deterministic (top-k by
    probability) at eval time.
    """

    def __init__(self, k: int = 25, embedding_dim: int = 128):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k, embedding_dim)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        B, N, _ = dist.shape
        k = min(self.k, N - 1)
        eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        inv = 1.0 / (dist + 1e-6)
        inv = inv.masked_fill(eye, 0.0)  # exclude self-distances
        probs = inv / (inv.sum(dim=-1, keepdim=True) + 1e-12)
        if self.training:
            idx = torch.multinomial(probs.reshape(B * N, N), k).view(B, N, k)
        else:
            idx = probs.topk(k, dim=-1).indices
        sampled = dist.gather(-1, idx)  # [B, N, k]
        if k < self.k:
            sampled = F.pad(sampled, (0, self.k - k))
        return self.proj(sampled)


class ContextualGating(nn.Module):
    """h = g * f_coord + (1 - g) * f_dist, g = sigmoid(MLP([f_coord; f_dist])) (Eq. 7-8)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, f_coord: torch.Tensor, f_dist: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_coord, f_dist], dim=-1))
        return g * f_coord + (1.0 - g) * f_dist


class NeuralAdaptiveBias(nn.Module):
    """Learned edge bias A [B, N, N] from distance, duration and angle matrices.

    Per the paper: each edge channel is embedded (scalar -> edge_dim), a
    contextual gate blends the channels, and the fused embedding is projected
    to a scalar bias. The paper gates distance vs. angle; DANA adds duration
    as a third gated channel (softmax gate).
    """

    def __init__(self, edge_dim: int = 16):
        super().__init__()

        def channel():
            return nn.Sequential(
                nn.Linear(1, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim),
            )

        self.dist_emb = channel()
        self.dur_emb = channel()
        self.angle_emb = channel()
        self.gate = nn.Linear(edge_dim * 3, 3)
        self.out = nn.Linear(edge_dim, 1)

    def forward(
        self, dist: torch.Tensor, dur: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        d = self.dist_emb(dist.unsqueeze(-1))  # [B, N, N, E]
        t = self.dur_emb(dur.unsqueeze(-1))
        a = self.angle_emb(angle.unsqueeze(-1))
        w = F.softmax(self.gate(torch.cat([d, t, a], dim=-1)), dim=-1)  # [B,N,N,3]
        h = w[..., 0:1] * d + w[..., 1:2] * t + w[..., 2:3] * a
        return self.out(h).squeeze(-1)  # [B, N, N]


def aafm_op(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, A: torch.Tensor
) -> torch.Tensor:
    """Attention-free operation (Eq. 11), numerically stabilized.

    AAFM(Q,K,V,A) = sigma(Q) * [exp(A) @ (exp(K) * V)] / [exp(A) @ exp(K)]

    Row-wise max of A and per-feature max of K are subtracted before exp;
    both constants cancel between numerator and denominator.
    """
    A = A - A.amax(dim=-1, keepdim=True)
    K = K - K.amax(dim=1, keepdim=True)
    exp_a = A.exp()  # [B, N, N]
    exp_k = K.exp()  # [B, N, D]
    num = exp_a @ (exp_k * V)
    den = exp_a @ exp_k
    return torch.sigmoid(Q) * num / (den + 1e-9)


class AAFMBlock(nn.Module):
    """Dual-stream AAFM layer: row embeddings attend over column embeddings
    (bias A) and column embeddings attend over row embeddings (bias A^T),
    each followed by a feed-forward sublayer. Instance normalization per the
    paper's hyperparameter table.
    """

    def __init__(
        self, embedding_dim: int = 128, feedforward_dim: int = 512, dropout: float = 0.1
    ):
        super().__init__()

        def qkv():
            return nn.ModuleDict(
                {
                    "q": nn.Linear(embedding_dim, embedding_dim),
                    "k": nn.Linear(embedding_dim, embedding_dim),
                    "v": nn.Linear(embedding_dim, embedding_dim),
                    "out": nn.Linear(embedding_dim, embedding_dim),
                }
            )

        def ffn():
            return nn.Sequential(
                nn.Linear(embedding_dim, feedforward_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, embedding_dim),
                nn.Dropout(dropout),
            )

        self.row = qkv()
        self.col = qkv()
        self.row_ff = ffn()
        self.col_ff = ffn()
        self.norm_r1 = InstanceNorm(embedding_dim)
        self.norm_r2 = InstanceNorm(embedding_dim)
        self.norm_c1 = InstanceNorm(embedding_dim)
        self.norm_c2 = InstanceNorm(embedding_dim)

    def forward(
        self, h_r: torch.Tensor, h_c: torch.Tensor, A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r_out = self.row["out"](
            aafm_op(self.row["q"](h_r), self.row["k"](h_c), self.row["v"](h_c), A)
        )
        c_out = self.col["out"](
            aafm_op(
                self.col["q"](h_c),
                self.col["k"](h_r),
                self.col["v"](h_r),
                A.transpose(-2, -1),
            )
        )
        h_r = self.norm_r1(h_r + r_out)
        h_c = self.norm_c1(h_c + c_out)
        h_r = self.norm_r2(h_r + self.row_ff(h_r))
        h_c = self.norm_c2(h_c + self.col_ff(h_c))
        return h_r, h_c


class GraphEncoder(nn.Module):
    """RRNCO encoder producing dual row/column node embeddings.

    forward(coords, distance_matrix, duration_matrix, node_feats) ->
        (h_row [B,N,D], h_col [B,N,D])

    Row embeddings encode outgoing relations (rows of the matrices), column
    embeddings incoming relations, so asymmetric distance/duration matrices
    are representable. Matrices are normalized internally by their per-instance
    maximum, so raw meters/seconds can be passed directly.
    """

    NODE_FEAT_DIM = 4  # demand, tw_start, tw_end, is_depot

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,  # kept for config compatibility (AAFM is head-free)
        num_layers: int = 12,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        sample_k: int = 25,
        edge_dim: int = 16,
    ):
        super().__init__()
        self.coord_proj = nn.Linear(2, embedding_dim)
        self.row_dist = InverseDistanceEmbedding(sample_k, embedding_dim)
        self.col_dist = InverseDistanceEmbedding(sample_k, embedding_dim)
        self.row_gate = ContextualGating(embedding_dim)
        self.col_gate = ContextualGating(embedding_dim)
        self.node_feat_proj = nn.Linear(self.NODE_FEAT_DIM, embedding_dim)
        self.row_comb = nn.Linear(embedding_dim * 2, embedding_dim)  # Eq. 9
        self.col_comb = nn.Linear(embedding_dim * 2, embedding_dim)  # Eq. 10
        self.nab = NeuralAdaptiveBias(edge_dim)
        self.layers = nn.ModuleList(
            [
                AAFMBlock(embedding_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _normalize(matrix: torch.Tensor) -> torch.Tensor:
        scale = matrix.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-9)
        return matrix / scale

    @staticmethod
    def _angle_matrix(coords: torch.Tensor) -> torch.Tensor:
        # Angular relationship between every node pair, normalized to [-1, 1]
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
        return torch.atan2(delta[..., 1], delta[..., 0]) / math.pi

    def forward(
        self,
        coords: torch.Tensor,
        distance_matrix: torch.Tensor,
        duration_matrix: Optional[torch.Tensor] = None,
        node_feats: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = coords.shape
        if duration_matrix is None:
            duration_matrix = distance_matrix
        dist_n = self._normalize(distance_matrix)
        dur_n = self._normalize(duration_matrix)
        angle = self._angle_matrix(coords)

        # Initial embedding: coord/distance fusion via contextual gating
        f_coord = self.coord_proj(coords)
        h_r = self.row_gate(f_coord, self.row_dist(dist_n))
        h_c = self.col_gate(f_coord, self.col_dist(dist_n.transpose(-2, -1)))

        # Combine with node attributes (Eq. 9-10)
        if node_feats is None:
            node_feats = coords.new_zeros(B, N, self.NODE_FEAT_DIM)
        f_node = self.node_feat_proj(node_feats)
        h_r = self.row_comb(torch.cat([h_r, f_node], dim=-1))
        h_c = self.col_comb(torch.cat([h_c, f_node], dim=-1))

        # Neural Adaptive Bias shared across AAFM layers
        A = self.nab(dist_n, dur_n, angle)

        for layer in self.layers:
            h_r, h_c = layer(h_r, h_c, A)
        return h_r, h_c
