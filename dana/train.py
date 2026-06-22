import json
import math
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dana.models.encoder import GraphEncoder
from dana.models.context import DynamicContext
from dana.models.policy import RouteDecoder, DisasterPolicy
from dana.envs.mdvrptw_env import DisasterMDVRPTWEnv
from dana.data.osm_loader import get_city_lists, city_to_tensor_dict


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
        try:
            train_cities, _ = get_city_lists()
            self.train_cities = train_cities
            self.synthetic_mode = False
        except (FileNotFoundError, json.JSONDecodeError):
            print("City data not found — using synthetic instances")
            self.train_cities = None
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
        for _ in pbar:
            batch_loss = self._train_batch()
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)
        return total_loss / num_batches

    def _train_batch(self) -> float:
        B = self.cfg["training"]["batch_size"]
        num_loc = self.cfg["data"]["num_locations"]
        num_depots = self.cfg["environment"]["max_depots"]
        max_vehicles = self.cfg["environment"]["max_vehicles"]
        if self.synthetic_mode:
            instance = synthetic_city_to_tensor_dict(num_loc, num_depots)
        else:
            city = np.random.choice(self.train_cities)
            instance = city_to_tensor_dict(city, num_loc, num_depots)
        self.env.reset(instance)
        N = instance["coords"].size(0)
        coords = instance["coords"].unsqueeze(0).expand(B, -1, -1).to(self.device)
        dist_mat = (
            instance["distance_matrix"].unsqueeze(0).expand(B, -1, -1).to(self.device)
        )
        dur_mat = (
            instance["duration_matrix"].unsqueeze(0).expand(B, -1, -1).to(self.device)
        )
        depot_mask = instance["depot_mask"].unsqueeze(0).expand(B, -1).to(self.device)
        demand = instance["demand"].unsqueeze(0).expand(B, -1).to(self.device)
        tw_start = instance["tw_start"].unsqueeze(0).expand(B, -1).to(self.device)
        tw_end = instance["tw_end"].unsqueeze(0).expand(B, -1).to(self.device)
        visited = depot_mask.clone()
        log_probs_list = []
        actions_list = []
        # Track which trajectories in batch have finished all nodes
        traj_done = torch.zeros(B, dtype=torch.bool, device=self.device)
        for _ in range(max_vehicles):
            if traj_done.all():
                break
            for _ in range(N * 2):
                if traj_done.all():
                    break
                logits = self.policy(
                    coords,
                    dist_mat,
                    dur_mat,
                    depot_mask,
                    demand,
                    tw_start,
                    tw_end,
                    visited_mask=visited,
                    return_logits=True,
                )
                mask = visited
                logits = logits.masked_fill(mask, float("-inf"))
                # Prevent NaN: for done trajectories (all visited), use zero logits
                logits = torch.where(
                    traj_done.unsqueeze(-1),
                    torch.zeros_like(logits),
                    logits,
                )
                probs = torch.softmax(logits / 0.1, dim=-1)
                m = torch.distributions.Categorical(probs)
                actions = m.sample()
                log_probs = m.log_prob(actions)
                # Override: done trajectories take depot (no effect on visited)
                actions = torch.where(traj_done, torch.zeros_like(actions), actions)
                # Zero gradient contribution from done trajectories
                log_probs = torch.where(
                    traj_done, torch.zeros_like(log_probs), log_probs
                )
                log_probs_list.append(log_probs)
                actions_list.append(actions)
                step_mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
                step_mask.scatter_(1, actions.unsqueeze(-1), True)
                visited = visited | step_mask
                visited[:, :num_depots] = depot_mask[:, :num_depots]
                traj_done = traj_done | visited.all(dim=-1)
        log_probs = torch.stack(log_probs_list, dim=1)  # [B, T]
        actions_seq = torch.stack(actions_list, dim=1)  # [B, T]
        # Reward = negative normalized route distance (minimize route length)
        prev = actions_seq[:, :-1]
        nxt = actions_seq[:, 1:]
        # route_dist[b] = sum_t dist_mat[b, prev[b,t], nxt[b,t]]
        row = dist_mat.gather(1, prev.unsqueeze(-1).expand(-1, -1, N))
        route_dist = row.gather(2, nxt.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        max_dist = math.sqrt(2.0) * N
        cost = route_dist / max_dist  # normalized cost in [0, roughly 1]
        # REINFORCE: advantage = -(cost - mean_cost), i.e. lower cost = better
        baseline = cost.mean()
        advantage = -(cost - baseline)  # [B], positive for better-than-average
        # Expand advantage to match log_probs timesteps
        adv = advantage.unsqueeze(1).expand(-1, log_probs.size(1))
        loss = -(log_probs * adv.detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.cfg["training"]["max_grad_norm"]
        )
        self.optimizer.step()
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"  WARNING: non-finite loss {loss_val}, skipping batch")
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
