import glob, os, shutil, sys, subprocess, json, yaml
import numpy as np
import torch

# Install PyTorch with CUDA 11.8 (supports P100/Tesla sm_60 which Kaggle often allocates)
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
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
        "pyvrp",
        "ortools",
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

INSTANCE_DIR = "/kaggle/input/dana-benchmarks"
assert os.path.exists(INSTANCE_DIR), (
    f"Benchmark dataset not found at {INSTANCE_DIR}. "
    "Ensure elfateh/dana-benchmarks is in kernel-metadata.json dataset_sources."
)

CKPT_DIR = "/kaggle/input/dana-checkpoints"
assert os.path.exists(CKPT_DIR), (
    f"Checkpoint dataset not found at {CKPT_DIR}. "
    "Ensure elfateh/dana-checkpoints is in kernel-metadata.json dataset_sources."
)

os.makedirs("reports", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    print(f"Loading DANA checkpoint: {latest_ckpt}")
    from dana.train import build_policy

    dana_policy = build_policy(cfg).to(device)
    dana_policy.eval()
    ckpt = torch.load(latest_ckpt, map_location=device)
    dana_policy.load_state_dict(ckpt["policy_state_dict"])
else:
    print("No DANA checkpoint found — skipping DANA evaluation")
    dana_policy = None

HGS_BIN = "/usr/local/bin/HGS-CVRP"
if not os.path.exists(HGS_BIN):
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/vidalt/HGS-CVRP.git",
                "/tmp/HGS-CVRP",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "-S", "/tmp/HGS-CVRP", "-B", "/tmp/HGS-CVRP/build"],
            check=True,
        )
        subprocess.run(
            ["make", "-C", "/tmp/HGS-CVRP/build", "-j$(nproc)"],
            check=True,
            shell=True,
        )
        subprocess.run(["cp", "/tmp/HGS-CVRP/build/HGS-CVRP", HGS_BIN], check=True)
    except Exception as e:
        print(f"HGS-CVRP install failed (non-critical): {e}")

LKH_BIN = "/usr/local/bin/LKH-3"
if not os.path.exists(LKH_BIN):
    try:
        subprocess.run(
            [
                "wget",
                "-q",
                "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz",
                "-O",
                "/tmp/LKH-3.tgz",
            ],
            check=True,
        )
        subprocess.run(["tar", "-xzf", "/tmp/LKH-3.tgz", "-C", "/tmp"], check=True)
        subprocess.run(["make", "-C", "/tmp/LKH-3.0.9"], check=True)
        subprocess.run(["cp", "/tmp/LKH-3.0.9/LKH", LKH_BIN], check=True)
    except Exception as e:
        print(f"LKH-3 install failed (non-critical): {e}")

from dana.eval.baselines import BaselineRunner
from dana.eval.metrics import compute_gap, evaluate_solver_set, compute_summary
from dana.eval.plots import generate_report


def parse_vrp_coords(path: str) -> np.ndarray:
    coords, section = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("NODE_COORD_SECTION"):
                section = True
                continue
            if (
                s.upper().startswith("DEMAND_SECTION")
                or s.upper().startswith("DEPOT_SECTION")
                or s.upper().startswith("TIME_WINDOW_SECTION")
                or s.upper().startswith("EOF")
            ):
                section = False
                continue
            if section and s:
                parts = s.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords, dtype=np.float32)


def run_dana_on_instance(policy, vrp_path: str, device: str, cfg: dict) -> dict:
    try:
        coords_np = parse_vrp_coords(vrp_path)
        N = len(coords_np)
        B = 1
        coords = torch.tensor(coords_np, dtype=torch.float, device=device).unsqueeze(0)
        diff = coords_np[:, None] - coords_np[None, :]
        dist_np = np.sqrt((diff**2).sum(axis=-1))
        dist_mat = torch.tensor(dist_np, dtype=torch.float, device=device).unsqueeze(0)
        depot_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        depot_mask[:, 0] = True
        demand = torch.ones(B, N, dtype=torch.float, device=device)
        tw_start = torch.zeros(B, N, dtype=torch.float, device=device)
        tw_end = torch.full((B, N), 480.0, dtype=torch.float, device=device)
        visited = depot_mask.clone()
        actions = []
        with torch.no_grad():
            for _ in range(cfg["environment"]["max_vehicles"]):
                if visited.all():
                    break
                for _ in range(N * 2):
                    logits = policy(
                        coords,
                        dist_mat,
                        dist_mat,
                        depot_mask,
                        demand,
                        tw_start,
                        tw_end,
                        visited_mask=visited,
                        return_logits=True,
                    )
                    logits = logits.masked_fill(visited, float("-inf"))
                    a = logits.argmax(dim=-1)
                    actions.append(a)
                    step_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
                    step_mask.scatter_(1, a.unsqueeze(-1), True)
                    visited = visited | step_mask
                    visited[:, :1] = depot_mask[:, :1]
                    if visited.all():
                        break
        if len(actions) < 2:
            return {"cost": None, "status": "no_solution", "time": 0}
        acts = torch.cat(actions)
        prev = acts[:-1]
        nxt = acts[1:]
        route_dist = dist_mat[0, prev, nxt].sum().item()
        return {"cost": route_dist, "status": "success", "time": 0}
    except Exception as e:
        return {"cost": None, "status": "error", "error": str(e)}


runner = BaselineRunner(
    time_limit=cfg["evaluation"]["time_limit"], num_runs=cfg["evaluation"]["num_runs"]
)

benchmarks = {
    "cordeau": {"dir": "cordeau", "problem": "mdvrptw"},
    "solomon": {"dir": "solomon", "problem": "vrptw"},
    "gehring": {"dir": "gehring", "problem": "vrptw"},
    "x_instances": {"dir": "x_instances", "problem": "cvrp"},
}

all_results = {}
for bench_name, bench_info in benchmarks.items():
    print(f"\n=== {bench_name} ===")
    bench_dir = os.path.join(INSTANCE_DIR, bench_info["dir"])
    if not os.path.exists(bench_dir):
        print(f"  SKIP: {bench_dir} not found")
        continue
    instance_files = [
        os.path.join(bench_dir, f)
        for f in sorted(os.listdir(bench_dir))
        if f.endswith(".vrp")
    ][:3]

    results = {
        s: {"costs": [], "gaps": [], "times": []}
        for s in ["dana", "pyvrp", "hgs", "lkh3", "ortools"]
    }

    for inst_file in instance_files:
        print(f"  {os.path.basename(inst_file)}")
        baselines = runner.run_all(inst_file, bench_info["problem"])
        if dana_policy is not None:
            baselines["dana"] = run_dana_on_instance(
                dana_policy, inst_file, device, cfg
            )
        for solver, result in baselines.items():
            if result.get("cost") is not None:
                results[solver]["costs"].append(result["cost"])
                if result.get("time"):
                    results[solver]["times"].append(result["time"])

    best = min(min(v["costs"]) for v in results.values() if v["costs"]) or 1.0
    summary = {}
    for solver, data in results.items():
        if data["costs"]:
            s = compute_summary(data["costs"], best)
            s["mean_time"] = np.mean(data["times"]) if data["times"] else 0
            s["gaps"] = [compute_gap(c, best) for c in data["costs"]]
            s["times"] = data["times"]
            summary[solver] = s
    all_results[bench_name] = summary

    stats = evaluate_solver_set(
        {s: {"costs": d["costs"]} for s, d in results.items()}, reference_solver="hgs"
    )
    generate_report(summary, stats, bench_name, output_dir="/kaggle/working/results")

    with open(f"/kaggle/working/results/{bench_name}_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

print("\nEvaluation complete.")
os.makedirs("/kaggle/working/dana-results", exist_ok=True)
shutil.copytree(
    "/kaggle/working/results", "/kaggle/working/dana-results", dirs_exist_ok=True
)
with open("/kaggle/working/dana-results/dataset-metadata.json", "w") as f:
    json.dump(
        {
            "title": "dana-results",
            "id": "elfateh/dana-results",
            "licenses": [{"name": "CC0-1.0"}],
        },
        f,
    )
# Create or update results dataset
result = subprocess.run(
    [
        "kaggle",
        "datasets",
        "create",
        "-p",
        "/kaggle/working/dana-results",
        "--dir-mode",
        "zip",
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(f"Dataset create failed (may already exist): {result.stderr.strip()}")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            "/kaggle/working/dana-results",
            "-m",
            "Updated results",
            "--dir-mode",
            "zip",
        ],
        check=True,
    )
