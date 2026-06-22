import json, os, shutil, subprocess, sys, yaml

# Install official PyTorch CUDA 12.1 build (includes sm_60 for P100, sm_75 for T4)
# Kaggle's default PyTorch is a custom build that may lack sm_60 support
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
        "--force-reinstall",
    ],
    check=True,
)
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "scikit-learn",
        "pandas",
        "kagglehub",
        "networkx",
    ],
    check=True,
)

import torch

REPO = "https://github.com/elfateh4/dana.git"
REPO_DIR = "/kaggle/working/dana"
if not os.path.exists(REPO_DIR):
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO, REPO_DIR], check=True, env=env
    )
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["num_epochs"] = 10
cfg["training"]["instances_per_epoch"] = 6000
cfg["training"]["batch_size"] = 24
cfg["training"]["max_grad_norm"] = 5.0
cfg["pomo"]["num_starts"] = 45
cfg["model"]["num_encoder_layers"] = 4
cfg["model"]["num_decoder_layers"] = 2
cfg["model"]["feedforward_dim"] = 256
cfg["data"]["num_locations"] = 50

from dana.train import build_policy, POMOTrainer

device = "cuda"
print(f"Device: {device}")

policy = build_policy(cfg)
policy.to(device)
policy.train()

# Sanity check: ensure gradients flow through a forward+backward pass
print("Running gradient sanity check...")
with torch.no_grad():
    dummy_coords = torch.randn(2, 10, 2, device=device)
    dummy_dist = torch.rand(2, 10, 10, device=device).abs() * 2
    dummy_depot = torch.zeros(2, 10, dtype=torch.bool, device=device)
    dummy_depot[:, 0] = True
    dummy_demand = torch.rand(2, 10, device=device) * 10
    dummy_tw_s = torch.zeros(2, 10, device=device)
    dummy_tw_e = torch.full((2, 10), 480.0, device=device)
    dummy_vis = dummy_depot.clone()
    logits = policy(
        dummy_coords,
        dummy_dist,
        dummy_dist,
        dummy_depot,
        dummy_demand,
        dummy_tw_s,
        dummy_tw_e,
        visited_mask=dummy_vis,
        return_logits=True,
    )
    print(
        f"  Logits ok: shape={logits.shape}, min={logits.min().item():.2f}, max={logits.max().item():.2f}, has_nan={torch.isnan(logits).any().item()}"
    )

test_logits = torch.randn(2, 10, device=device, requires_grad=True)
m = torch.distributions.Categorical(logits=test_logits)
a = m.sample()
lp = m.log_prob(a)
lp.mean().backward()
print(
    f"  Categorical(logits=...) gradient ok: grad_norm={test_logits.grad.norm().item():.4f}"
)

del dummy_coords, dummy_dist, test_logits

trainer = POMOTrainer(policy, cfg, device)
num_epochs = cfg["training"]["num_epochs"]

for epoch in range(num_epochs):
    loss = trainer.train_epoch()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.6f}")
    # Log parameter change to verify learning
    total_norm = sum(p.norm().item() ** 2 for p in policy.parameters()) ** 0.5
    grad_norm = (
        sum(
            p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None
        )
        ** 0.5
        if epoch == 0
        else 0
    )
    if epoch == 0:
        print(f"  init: param_norm={total_norm:.2f}, grad_norm={grad_norm:.6f}")
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        trainer.save_checkpoint(
            f"/kaggle/working/checkpoints/dana_epoch_{epoch + 1}.pt"
        )

os.makedirs("/kaggle/working/dana-checkpoints", exist_ok=True)
shutil.copy(
    f"/kaggle/working/checkpoints/dana_epoch_{num_epochs}.pt",
    "/kaggle/working/dana-checkpoints/",
)
with open("/kaggle/working/dana-checkpoints/dataset-metadata.json", "w") as f:
    json.dump(
        {
            "title": "dana-checkpoints",
            "id": "elfateh/dana-checkpoints",
            "licenses": [{"name": "CC0-1.0"}],
        },
        f,
    )
subprocess.run(
    [
        "kaggle",
        "datasets",
        "version",
        "-p",
        "/kaggle/working/dana-checkpoints",
        "-m",
        f"Checkpoint after epoch {num_epochs}",
    ],
)
