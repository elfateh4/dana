import os
import sys
import subprocess
import yaml

subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "torch",
        "torch-geometric",
        "osmnx",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "scikit-learn",
        "pandas",
        "seaborn",
        "kagglehub",
        "networkx",
        "requests",
    ],
    check=True,
)

REPO_DIR = "/kaggle/working/dana"
if not os.path.exists(REPO_DIR):
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/elfateh4/dana.git",
            REPO_DIR,
        ],
        check=True,
    )
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

os.environ["KAGGLE_KEY"] = (
    "/kaggle/input/kaggle-credentials/kaggle.json"
    if os.path.exists("/kaggle/input/kaggle-credentials/kaggle.json")
    else ""
)

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

from dana.models.encoder import GraphEncoder
from dana.models.context import DynamicContext
from dana.models.policy import RouteDecoder, DisasterPolicy
from dana.train import build_policy, POMOTrainer, load_config

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Device: {device}")

policy = build_policy(cfg)
trainer = POMOTrainer(policy, cfg, device)
num_epochs = cfg["training"]["num_epochs"]
for epoch in range(num_epochs):
    loss = trainer.train_epoch()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    if (epoch + 1) % 10 == 0:
        trainer.save_checkpoint(
            f"/kaggle/working/checkpoints/dana_epoch_{epoch + 1}.pt"
        )
trainer.save_checkpoint("/kaggle/working/checkpoints/dana_final.pt")
print("Training complete.")

import kagglehub

kagglehub.upload_dataset(
    "/kaggle/working/checkpoints",
    "elfateh/dana-checkpoints",
    message=f"DANA checkpoints after {num_epochs} epochs",
)
