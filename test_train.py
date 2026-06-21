import torch
import numpy as np
import yaml
from dana.models.encoder import GraphEncoder
from dana.models.context import DynamicContext
from dana.models.policy import RouteDecoder, DisasterPolicy


def build_policy(cfg):
    encoder = GraphEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_encoder_layers"],
        feedforward_dim=cfg["model"]["feedforward_dim"],
        dropout=cfg["model"]["dropout"],
    )
    context_module = DynamicContext(embedding_dim=cfg["model"]["embedding_dim"])
    decoder = RouteDecoder(
        embedding_dim=cfg["model"]["embedding_dim"], num_heads=cfg["model"]["num_heads"]
    )
    return DisasterPolicy(
        encoder, context_module, decoder, embedding_dim=cfg["model"]["embedding_dim"]
    )


def gen_inst(num_depots=2, num_customers=10):
    n = num_depots + num_customers
    rng = np.random.default_rng()
    coords = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
    return {
        "coords": coords,
        "distance_matrix": dist,
        "duration_matrix": dist.copy(),
        "demand": rng.integers(1, 10, size=n).astype(np.float32),
        "tw_start": np.zeros(n, dtype=np.float32),
        "tw_end": np.concatenate(
            [
                np.full(num_depots, 480.0),
                rng.uniform(30, 120, size=num_customers).astype(np.float32),
            ]
        ),
        "depot_indices": list(range(num_depots)),
        "num_depots": num_depots,
        "num_locations": n,
    }


def main():
    with open("configs/test.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    policy = build_policy(cfg).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg["training"]["learning_rate"]
    )
    print(f"Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    num_depots = cfg["environment"]["max_depots"]
    num_cust = cfg["data"]["num_locations"]
    n = num_depots + num_cust
    B = cfg["training"]["batch_size"]
    num_epochs = cfg["training"]["num_epochs"]
    steps_per_epoch = cfg["training"]["instances_per_epoch"] // B

    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            inst = gen_inst(num_depots, num_cust)
            coords = (
                torch.tensor(inst["coords"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .to(device)
            )
            dist_mat = (
                torch.tensor(inst["distance_matrix"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .to(device)
            )
            dur_mat = (
                torch.tensor(inst["duration_matrix"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .to(device)
            )
            depot_mask = torch.zeros(B, n, dtype=torch.bool, device=device)
            depot_mask[:, :num_depots] = True
            demand = (
                torch.tensor(inst["demand"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1)
                .to(device)
            )
            tw_start = (
                torch.tensor(inst["tw_start"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1)
                .to(device)
            )
            tw_end = (
                torch.tensor(inst["tw_end"], dtype=torch.float)
                .unsqueeze(0)
                .expand(B, -1)
                .to(device)
            )

            visited = depot_mask.clone()
            log_probs = []
            rewards = []

            for _ in range(cfg["environment"]["max_vehicles"] * n):
                logits = policy(
                    coords=coords,
                    distance_matrix=dist_mat,
                    duration_matrix=dur_mat,
                    depot_mask=depot_mask,
                    demand=demand,
                    tw_start=tw_start,
                    tw_end=tw_end,
                    visited_mask=visited,
                    return_logits=True,
                )
                logits = logits.masked_fill(visited, float("-inf"))
                probs = torch.softmax(logits / 0.1, dim=-1)
                m = torch.distributions.Categorical(probs)
                actions = m.sample()
                log_probs.append(m.log_prob(actions))

                step_mask = torch.zeros(B, n, dtype=torch.bool, device=device)
                step_mask.scatter_(1, actions.unsqueeze(-1), True)
                visited = visited | step_mask
                visited[:, :num_depots] = depot_mask[:, :num_depots]

                action_tw_end = tw_end.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                r = (
                    -torch.clamp(
                        action_tw_end
                        - tw_start.gather(1, actions.unsqueeze(-1)).squeeze(-1),
                        min=0,
                    )
                    * 0.01
                )
                rewards.append(r)

                if visited.all():
                    break

            if len(log_probs) < 2:
                continue

            log_probs = torch.stack(log_probs, dim=1)
            rewards = torch.stack(rewards, dim=1)
            baseline = rewards.mean(dim=1, keepdim=True)
            advantage = rewards - baseline
            loss = -(log_probs * advantage.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg["training"]["max_grad_norm"]
            )
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        policy.eval()
        eval_r = 0.0
        with torch.no_grad():
            inst = gen_inst(num_depots, num_cust)
            coords = (
                torch.tensor(inst["coords"], dtype=torch.float).unsqueeze(0).to(device)
            )
            dist_mat = (
                torch.tensor(inst["distance_matrix"], dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            dur_mat = (
                torch.tensor(inst["duration_matrix"], dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            mask = torch.zeros(1, n, dtype=torch.bool, device=device)
            mask[:, :num_depots] = True
            demand = (
                torch.tensor(inst["demand"], dtype=torch.float).unsqueeze(0).to(device)
            )
            tw_s = (
                torch.tensor(inst["tw_start"], dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            tw_e = (
                torch.tensor(inst["tw_end"], dtype=torch.float).unsqueeze(0).to(device)
            )
            visited = mask.clone()
            route_len = 0.0
            for _ in range(n * 2):
                logits = policy(
                    coords=coords,
                    distance_matrix=dist_mat,
                    duration_matrix=dur_mat,
                    depot_mask=mask,
                    demand=demand,
                    tw_start=tw_s,
                    tw_end=tw_e,
                    visited_mask=visited,
                    return_logits=True,
                )
                logits = logits.masked_fill(visited, float("-inf"))
                action = torch.softmax(logits / 0.1, dim=-1).multinomial(1).squeeze(-1)
                visited.scatter_(1, action.unsqueeze(-1), True)
                if visited.all():
                    break

        avg_loss = total_loss / max(steps_per_epoch, 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.6f}")

    print("Test training complete.")
    torch.save(policy.state_dict(), "checkpoints/test_final.pt")
    print("Model saved to checkpoints/test_final.pt")


if __name__ == "__main__":
    main()
