import math
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.encoder import GraphEncoder
from models.context import DynamicContext
from models.policy import RouteDecoder, DisasterPolicy
from data.osm_loader import CityRotation, city_to_tensor_dict


def synthetic_city_to_tensor_dict(
    num_locations: int, num_depots: int = 1, rng=None
) -> dict:
    """Random synthetic instance with ASYMMETRIC distance/duration matrices.

    Real road networks yield d_ij != d_ji (one-way streets, turn restrictions),
    so the synthetic fallback perturbs the Euclidean base with independent
    directional noise, and durations get their own noise (traffic patterns
    differ from geometry).
    """
    if rng is None:
        rng = np.random.default_rng()
    points = torch.tensor(rng.uniform(0, 1, (num_locations, 2)), dtype=torch.float)
    coords = points.numpy()
    diff = coords[:, None] - coords[None, :]
    base = np.sqrt((diff**2).sum(axis=-1))
    # Independent directional perturbations -> asymmetric matrices
    dist = base * (1.0 + rng.uniform(0.0, 0.2, base.shape))
    dur = base * (1.0 + rng.uniform(0.0, 0.5, base.shape))
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(dur, 0.0)
    dist_mat = torch.tensor(dist, dtype=torch.float)
    dur_mat = torch.tensor(dur, dtype=torch.float)
    depot_mask = torch.zeros(num_locations, dtype=torch.bool)
    depot_mask[:num_depots] = True
    demand = torch.tensor(rng.uniform(1, 10, (num_locations,)), dtype=torch.float)
    tw_start = torch.zeros(num_locations, dtype=torch.float)
    tw_end = torch.full((num_locations,), 480.0, dtype=torch.float)
    return {
        "coords": points,
        "distance_matrix": dist_mat,
        "duration_matrix": dur_mat,
        "depot_mask": depot_mask,
        "demand": demand,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "num_depots": torch.tensor(num_depots),
        "num_locations": torch.tensor(num_locations),
    }


def dihedral_augment(coords: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """One of the 8 dihedral symmetries of the unit square (POMO-style aug).

    Valid for real (matrix-based) instances too: distances/durations come from
    the matrices, which are invariant; only the coordinate features transform.
    """
    x, y = coords[..., 0], coords[..., 1]
    variants = [
        (x, y), (y, x), (1 - x, y), (y, 1 - x),
        (x, 1 - y), (1 - y, x), (1 - x, 1 - y), (1 - y, 1 - x),
    ]
    vx, vy = variants[aug_idx % 8]
    return torch.stack([vx, vy], dim=-1)


def load_config(config_path: str = "configs/dana.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_policy(cfg: dict) -> DisasterPolicy:
    encoder = GraphEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_encoder_layers"],
        feedforward_dim=cfg["model"]["feedforward_dim"],
        dropout=cfg["model"]["dropout"],
        sample_k=cfg["model"].get("sample_k", 25),
        edge_dim=cfg["model"].get("edge_dim", 16),
    )
    context_module = DynamicContext(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
    )
    decoder = RouteDecoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"].get("num_decoder_layers", 1),
    )
    return DisasterPolicy(
        encoder, context_module, decoder, embedding_dim=cfg["model"]["embedding_dim"]
    )


class POMOTrainer:
    def __init__(self, policy: DisasterPolicy, cfg: dict, device: str = "cpu"):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = optim.AdamW(
            policy.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        # Set up city rotation. Falls back to synthetic if no OSM data available.
        try:
            self.city_rotation = CityRotation(
                data_root=cfg["paths"].get("osm_cache", "data/osm_cache")
            )
            if self.city_rotation.available_count() > 0:
                print(
                    f"OSM city rotation: {self.city_rotation.available_count()} cities available"
                )
                self.synthetic_mode = False
            else:
                print("No OSM city data found — using synthetic instances")
                self.synthetic_mode = True
        except Exception as e:
            print(f"City rotation init failed — using synthetic instances: {e}")
            self.synthetic_mode = True
        self.num_starts = cfg["pomo"]["num_starts"]
        self.num_aug = max(1, min(cfg["pomo"]["num_augmentations"], 8))
        self.scheduler, self.scheduler_step_per = self._build_scheduler()

    def _build_scheduler(self):
        """LR scheduler per config: 'cosine' (with linear warmup), 'multistep'
        (paper's MultiStepLR, milestones [180,195], gamma 0.1), or 'none'."""
        tcfg = self.cfg["training"]
        name = tcfg.get("lr_scheduler", "none")
        steps_per_epoch = max(1, tcfg["instances_per_epoch"] // tcfg["batch_size"])
        if name == "cosine":
            warmup = tcfg.get("warmup_steps", 0)
            total = max(1, tcfg["num_epochs"] * steps_per_epoch)

            def lr_lambda(step):
                if warmup > 0 and step < warmup:
                    return (step + 1) / warmup
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda), "batch"
        if name == "multistep":
            return (
                optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=tcfg.get("lr_milestones", [180, 195]),
                    gamma=tcfg.get("lr_gamma", 0.1),
                ),
                "epoch",
            )
        return None, None

    def train_epoch(self) -> float:
        self.policy.train()
        total_loss = 0.0
        num_batches = (
            self.cfg["training"]["instances_per_epoch"]
            // self.cfg["training"]["batch_size"]
        )
        pbar = tqdm(range(num_batches), desc="Training")
        for batch_idx in pbar:
            batch_loss = self._train_batch(debug=(batch_idx == 0))
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)
        if self.scheduler is not None and self.scheduler_step_per == "epoch":
            self.scheduler.step()
        # Advance city rotation for next epoch (if using OSM data)
        if not self.synthetic_mode:
            self.city_rotation.next_epoch()
        return total_loss / num_batches

    def _load_instance(self) -> dict:
        num_loc = self.cfg["data"]["num_locations"]
        num_depots = self.cfg["environment"]["max_depots"]
        if self.synthetic_mode:
            return synthetic_city_to_tensor_dict(num_loc, num_depots)
        try:
            city = self.city_rotation.get_city()
            return city_to_tensor_dict(
                city,
                num_loc,
                num_depots,
                data_root=self.cfg["paths"].get("osm_cache", "data/osm_cache"),
            )
        except (FileNotFoundError, OSError):
            print("City data load failed — falling back to synthetic")
            self.synthetic_mode = True
            return synthetic_city_to_tensor_dict(num_loc, num_depots)

    def _train_batch(self, debug: bool = False) -> float:
        B = self.cfg["training"]["batch_size"]
        K = self.num_starts
        A = self.num_aug
        num_depots = self.cfg["environment"]["max_depots"]
        max_vehicles = self.cfg["environment"]["max_vehicles"]
        instance = self._load_instance()
        N = instance["coords"].size(0)
        S = B * K  # decoding streams

        coords0 = instance["coords"].to(self.device)  # [N, 2]
        dist0 = instance["distance_matrix"].to(self.device)  # [N, N]
        dur0 = instance["duration_matrix"].to(self.device)  # [N, N]
        depot_mask0 = instance["depot_mask"].to(self.device)  # [N]
        demand0 = instance["demand"].to(self.device)  # [N]
        tw_start0 = instance["tw_start"].to(self.device)  # [N]
        tw_end0 = instance["tw_end"].to(self.device)  # [N]
        dist_n0 = dist0 / dist0.amax().clamp(min=1e-9)  # normalized, for Eq. 15

        self.optimizer.zero_grad()

        # ------------------------------------------------------------
        # Encode once per augmentation variant, expand to decoding streams.
        # Dihedral transforms change coordinate features (and hence angle
        # matrices) but not the distance/duration matrices.
        # ------------------------------------------------------------
        coords_aug = torch.stack(
            [dihedral_augment(coords0, a) for a in range(A)], dim=0
        )  # [A, N, 2]
        dist_a = dist0.unsqueeze(0).expand(A, N, N)
        dur_a = dur0.unsqueeze(0).expand(A, N, N)
        node_feats = DisasterPolicy.build_node_feats(
            demand0.unsqueeze(0).expand(A, N),
            tw_start0.unsqueeze(0).expand(A, N),
            tw_end0.unsqueeze(0).expand(A, N),
            depot_mask0.unsqueeze(0).expand(A, N),
        )
        row_a, col_a = self.policy.encoder(coords_aug, dist_a, dur_a, node_feats)
        D = row_a.size(-1)
        col_a, ctx_a = self.policy.context_module(
            col_a, dist_a, torch.zeros(A, D, device=self.device)
        )

        aug_idx = torch.arange(S, device=self.device) % A
        row_emb = row_a[aug_idx]  # [S, N, D]
        col_emb = col_a[aug_idx]
        ctx = ctx_a[aug_idx]  # [S, D]

        depot_mask_flat = depot_mask0.unsqueeze(0).expand(S, N)
        demand = demand0.unsqueeze(0).expand(S, N)
        tw_end = tw_end0.unsqueeze(0).expand(S, N)

        # ------------------------------------------------------------
        # POMO multi-start rollout with last-node tracking
        # ------------------------------------------------------------
        visited = depot_mask_flat.clone()
        start_offset = num_depots
        start_nodes = (torch.arange(K, device=self.device).repeat(B) + start_offset) % N
        start_nodes = start_nodes.clamp(min=num_depots, max=N - 1)
        last_node = torch.zeros(S, dtype=torch.long, device=self.device)  # depot 0
        stream_arange = torch.arange(S, device=self.device)
        log_probs_list = []
        actions_list = []
        traj_done = torch.zeros(S, dtype=torch.bool, device=self.device)
        first_step = True
        for _ in range(max_vehicles):
            if traj_done.all():
                break
            for _ in range(N * 2):
                if traj_done.all():
                    break
                vf = self.policy.compute_vehicle_state(
                    visited, demand, tw_end, S, N, self.device
                )
                dist_row = dist_n0[last_node]  # [S, N]
                logits = self.policy.decoder(
                    row_emb, col_emb, ctx, vf, last_node, dist_row, mask=visited
                )
                m = torch.distributions.Categorical(logits=logits)
                if first_step:
                    actions = start_nodes
                    first_step = False
                else:
                    actions = m.sample()
                log_prob = m.log_prob(actions)
                actions = torch.where(traj_done, torch.zeros_like(actions), actions)
                log_prob = torch.where(traj_done, torch.zeros_like(log_prob), log_prob)
                log_probs_list.append(log_prob)
                actions_list.append(actions)
                last_node = actions
                step_mask = torch.zeros(S, N, dtype=torch.bool, device=self.device)
                step_mask.scatter_(1, actions.unsqueeze(-1), True)
                visited = visited | step_mask
                visited[:, :num_depots] = depot_mask_flat[:, :num_depots]
                traj_done = traj_done | visited.all(dim=-1)
        # ------------------------------------------------------------
        # Log-likelihood: sum of log_probs across all steps (per trajectory)
        # ------------------------------------------------------------
        log_likelihood = torch.stack(log_probs_list, dim=1).sum(dim=1)  # [S]
        # Cost per trajectory via the (asymmetric) distance matrix
        actions_seq = torch.stack(actions_list, dim=1)  # [S, T]
        prev = actions_seq[:, :-1]
        nxt = actions_seq[:, 1:]
        route_dist = dist0[prev, nxt].sum(dim=1)
        cost = route_dist.view(B, K)  # [B, K]
        # POMO shared baseline: advantage = baseline - cost
        with torch.no_grad():
            baseline = cost.mean(dim=1, keepdim=True)
            advantage = baseline - cost  # + means cheaper than average
        # REINFORCE loss (as in rl4co/POMO: no T scaling, no entropy reg)
        reinforce_loss = -(advantage * log_likelihood.view(B, K)).mean()
        reinforce_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.cfg["training"]["max_grad_norm"]
        )
        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_step_per == "batch":
            self.scheduler.step()
        loss_val = reinforce_loss.item()
        if debug:
            with torch.no_grad():
                cost_std = cost.std().item()
                adv_max = advantage.max().item()
                adv_min = advantage.min().item()
                grad_norm = (
                    sum(
                        p.grad.norm().item() ** 2
                        for p in self.policy.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                clip_val = self.policy.decoder.logit_clip.item()
                temp_val = self.policy.decoder.logit_temperature.item()
                lr_val = self.optimizer.param_groups[0]["lr"]
            print(
                f"  [debug] cost: mean={cost.mean().item():.4f} std={cost_std:.4f} "
                f"| adv: [{adv_min:.6f}, {adv_max:.6f}] "
                f"| ll={log_likelihood.mean().item():.2f} | loss={loss_val:.8f} "
                f"| grad_norm={grad_norm:.6f} | clip={clip_val:.2f} temp={temp_val:.2f} "
                f"| lr={lr_val:.2e}"
            )
        if not math.isfinite(loss_val):
            print(f"  WARNING: non-finite loss {loss_val}, skipping batch")
            loss_val = 0.0
        return loss_val

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    policy = build_policy(cfg)
    trainer = POMOTrainer(policy, cfg, device)
    num_epochs = cfg["training"]["num_epochs"]
    for epoch in range(num_epochs):
        loss = trainer.train_epoch()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/dana_epoch_{epoch + 1}.pt")
    trainer.save_checkpoint("checkpoints/dana_final.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
