import json, os, shutil, subprocess, sys, yaml

subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "torch==2.3.1",
        "torch-geometric",
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
cfg["training"]["instances_per_epoch"] = 800
cfg["training"]["batch_size"] = 16
cfg["model"]["num_encoder_layers"] = 4
cfg["model"]["feedforward_dim"] = 256
cfg["data"]["num_locations"] = 50

from dana.train import build_policy, POMOTrainer

device = "cuda"
print(f"Device: {device}")

policy = build_policy(cfg)
trainer = POMOTrainer(policy, cfg, device)
num_epochs = cfg["training"]["num_epochs"]

for epoch in range(num_epochs):
    loss = trainer.train_epoch()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
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
