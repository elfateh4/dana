import math
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.encoder import GraphEncoder
from models.context import DynamicContext
from models.policy import RouteDecoder, DisasterPolicy
from envs.mdvrptw_env import DisasterMDVRPTWEnv
from data.osm_loader import CityRotation, city_to_tensor_dict, get_city_lists


def synthetic_city_to_tensor_dict(
    num_locations: int, num_depots: int = 1, rng=None
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    points = torch.tensor(rng.uniform(0, 1, (num_locations, 2)), dtype=torch.float)
    coords = points.numpy()
    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    dist_mat = torch.tensor(dist, dtype=torch.float)
    dur_mat = dist_mat.clone()
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
    )
    context_module = DynamicContext(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
    )
    decoder = RouteDecoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"].get("num_decoder_layers", 3),
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
        self.env = DisasterMDVRPTWEnv()
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
        self.num_aug = cfg["pomo"]["num_augmentations"]

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
        # Advance city rotation for next epoch (if using OSM data)
        if not self.synthetic_mode:
            self.city_rotation.next_epoch()
        return total_loss / num_batches

    def _train_batch(self, debug: bool = False) -> float:
        B = self.cfg["training"]["batch_size"]
        K = self.num_starts
        num_loc = self.cfg["data"]["num_locations"]
        num_depots = self.cfg["environment"]["max_depots"]
        max_vehicles = self.cfg["environment"]["max_vehicles"]
        if self.synthetic_mode:
            instance = synthetic_city_to_tensor_dict(num_loc, num_depots)
        else:
            try:
                city = self.city_rotation.get_city()
                instance = city_to_tensor_dict(
                    city,
                    num_loc,
                    num_depots,
                    data_root=self.cfg["paths"].get("osm_cache", "data/osm_cache"),
                )
            except (FileNotFoundError, OSError):
                print("City data load failed — falling back to synthetic")
                self.synthetic_mode = True
                instance = synthetic_city_to_tensor_dict(num_loc, num_depots)
        N = instance["coords"].size(0)
        coords = (
            instance["coords"]
            .unsqueeze(0)
            .expand(B, K, -1, -1)
            .reshape(B * K, N, 2)
            .to(self.device)
        )
        dist_mat = (
            instance["distance_matrix"]
            .unsqueeze(0)
            .expand(B, K, -1, -1)
            .reshape(B * K, N, N)
            .to(self.device)
        )
        depot_mask = instance["depot_mask"].unsqueeze(0).expand(B, -1).to(self.device)
        depot_mask_flat = depot_mask.unsqueeze(1).expand(B, K, -1).reshape(B * K, N)
        demand = (
            instance["demand"]
            .unsqueeze(0)
            .expand(B, K, -1)
            .reshape(B * K, N)
            .to(self.device)
        )
        tw_end = (
            instance["tw_end"]
            .unsqueeze(0)
            .expand(B, K, -1)
            .reshape(B * K, N)
            .to(self.device)
        )
        D = self.cfg["model"]["embedding_dim"]
        # ------------------------------------------------------------
        # Single forward pass with grad: encoder once, decoder all steps
        # ------------------------------------------------------------
        self.optimizer.zero_grad()
        enc_out = self.policy.encoder(coords, dist_mat)  # [B*K, N, D]
        enc_out, ctx = self.policy.context_module(
            enc_out, dist_mat, torch.zeros(B * K, D, device=self.device)
        )
        visited = depot_mask_flat.clone()
        start_offset = num_depots
        num_customers = N - num_depots
        start_nodes = (torch.arange(K, device=self.device).repeat(B) + start_offset) % N
        start_nodes = start_nodes.clamp(min=num_depots, max=N - 1)
        log_probs_list = []
        actions_list = []
        traj_done = torch.zeros(B * K, dtype=torch.bool, device=self.device)
        first_step = True
        for _ in range(max_vehicles):
            if traj_done.all():
                break
            for _ in range(N * 2):
                if traj_done.all():
                    break
                visit_frac = visited.float().mean(dim=-1)
                remaining_cap = 1.0 - (demand.float().mean(dim=-1) * visit_frac)
                unvisited_tw_end = tw_end.float().masked_fill(visited, float("inf"))
                min_tw_end = unvisited_tw_end.min(dim=-1).values
                max_tw_end = unvisited_tw_end.max(dim=-1).values
                tw_urgency = 1.0 - (min_tw_end / (max_tw_end + 1e-8))
                unvisited_count = (~visited).float().sum(dim=-1) / N
                vs = torch.stack(
                    [visit_frac, remaining_cap, tw_urgency, unvisited_count], dim=-1
                )
                vf = self.policy.vehicle_embedding(vs)
                logits = self.policy.decoder(enc_out, ctx, vf, mask=visited)
                logits = logits.masked_fill(visited, float("-inf"))
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
                step_mask = torch.zeros(B * K, N, dtype=torch.bool, device=self.device)
                step_mask.scatter_(1, actions.unsqueeze(-1), True)
                visited = visited | step_mask
                visited[:, :num_depots] = depot_mask_flat[:, :num_depots]
                traj_done = traj_done | visited.all(dim=-1)
        # ------------------------------------------------------------
        # Log-likelihood: sum of log_probs across all steps (per trajectory)
        # ------------------------------------------------------------
        log_likelihood = torch.stack(log_probs_list, dim=1).sum(dim=1)  # [B*K]
        # Cost per trajectory
        actions_seq = torch.stack(actions_list, dim=1)  # [B*K, T]
        prev = actions_seq[:, :-1]
        nxt = actions_seq[:, 1:]
        row = dist_mat.gather(1, prev.unsqueeze(-1).expand(-1, -1, N))
        route_dist = row.gather(2, nxt.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        cost = route_dist.view(B, K)  # [B, K], raw distance (no normalization)
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
                ent_val = torch.stack(log_probs_list, dim=1).exp().mean().item()
            print(
                f"  [debug] cost: mean={cost.mean().item():.4f} std={cost_std:.4f} "
                f"| adv: [{adv_min:.6f}, {adv_max:.6f}] "
                f"| ll={log_likelihood.mean().item():.2f} | loss={loss_val:.8f} "
                f"| grad_norm={grad_norm:.6f} | clip={clip_val:.2f} temp={temp_val:.2f}"
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
        self.policy.load_state_dict(ckpt["policy_state_dict"])
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
