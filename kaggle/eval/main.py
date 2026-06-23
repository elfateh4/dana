import glob, os, shutil, sys, subprocess, json, yaml
import numpy as np
import torch

# Install PyTorch with CUDA 11.8 build (supports sm_60 for P100 AND sm_75 for T4)
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
        "requests",
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
        "requests",
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

KAGGLE_CKPT_DIR = "/kaggle/input/dana-checkpoints"
INSTANCE_DIR = "/kaggle/working/instances"
if not os.path.exists(INSTANCE_DIR):
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    print("Downloading benchmark instances from PyVRP/Instances...")

    def _download_instances():
        import requests

        BASE_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main"
        sets = {
            "cordeau": (
                "MDVRPTW",
                [f"PR{i}{s}" for i in range(11, 25) for s in ("A", "B")],
            ),
            "solomon": (
                "VRPTW/Solomon",
                [
                    "C101",
                    "C102",
                    "C103",
                    "C104",
                    "C105",
                    "C106",
                    "C107",
                    "C108",
                    "C109",
                    "C201",
                    "C202",
                    "C203",
                    "C204",
                    "C205",
                    "C206",
                    "C207",
                    "C208",
                    "R101",
                    "R102",
                    "R103",
                    "R104",
                    "R105",
                    "R106",
                    "R107",
                    "R108",
                    "R109",
                    "R110",
                    "R111",
                    "R112",
                    "R201",
                    "R202",
                    "R203",
                    "R204",
                    "R205",
                    "R206",
                    "R207",
                    "R208",
                    "R209",
                    "R210",
                    "R211",
                    "RC101",
                    "RC102",
                    "RC103",
                    "RC104",
                    "RC105",
                    "RC106",
                    "RC107",
                    "RC108",
                    "RC201",
                    "RC202",
                    "RC203",
                    "RC204",
                    "RC205",
                    "RC206",
                    "RC207",
                    "RC208",
                ],
            ),
            "gehring": (
                "VRPTW",
                [
                    f"{t}{n}_10_{i}"
                    for t in ("C", "R", "RC")
                    for n in (1, 2)
                    for i in range(1, 11)
                ],
            ),
            "x_instances": (
                "CVRP",
                [
                    "X-n101-k25",
                    "X-n106-k14",
                    "X-n110-k13",
                    "X-n115-k10",
                    "X-n120-k6",
                    "X-n125-k30",
                    "X-n129-k18",
                    "X-n134-k13",
                    "X-n139-k10",
                    "X-n143-k7",
                    "X-n153-k22",
                    "X-n157-k13",
                    "X-n162-k11",
                    "X-n167-k10",
                    "X-n176-k26",
                    "X-n181-k23",
                    "X-n186-k15",
                    "X-n190-k8",
                    "X-n195-k51",
                    "X-n200-k36",
                    "X-n204-k19",
                    "X-n209-k16",
                    "X-n214-k11",
                    "X-n219-k73",
                    "X-n223-k34",
                    "X-n228-k23",
                    "X-n233-k16",
                    "X-n237-k14",
                    "X-n242-k48",
                    "X-n247-k50",
                ],
            ),
        }
        for name, (remote_dir, files) in sets.items():
            dest = os.path.join(INSTANCE_DIR, name)
            os.makedirs(dest, exist_ok=True)
            ok, fail = 0, 0
            for fname in files:
                url = f"{BASE_URL}/{remote_dir}/{fname}.vrp"
                path = os.path.join(dest, f"{fname}.vrp")
                if os.path.exists(path):
                    ok += 1
                    continue
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        with open(path, "w") as f:
                            f.write(r.text)
                        ok += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
            print(f"  {name}: {ok} ok, {fail} failed")

    _download_instances()
    print("Download complete.")

# Checkpoint: check Kaggle mount or download via kagglehub
CKPT_DIR = (
    KAGGLE_CKPT_DIR
    if os.path.exists(KAGGLE_CKPT_DIR)
    else "/kaggle/working/checkpoints_download"
)
if not os.path.exists(CKPT_DIR) or not glob.glob(os.path.join(CKPT_DIR, "*.pt")):
    os.makedirs(CKPT_DIR, exist_ok=True)
    try:
        import kagglehub

        path = kagglehub.dataset_download("elfateh/dana-checkpoints")
        print(f"Downloaded checkpoints via kagglehub to {path}")
        # kagglehub returns root dir; copy .pt files to our CKPT_DIR
        for f in glob.glob(os.path.join(path, "*.pt")):
            shutil.copy(f, CKPT_DIR)
    except Exception as e:
        print(f"Could not download checkpoints via kagglehub: {e}")
        CKPT_DIR = None

os.makedirs("reports", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))) if CKPT_DIR else []
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    print(f"Loading DANA checkpoint: {latest_ckpt}")
    from dana.train import build_policy

    # Match training config overrides (v32+)
    cfg["model"]["num_encoder_layers"] = 4
    cfg["model"]["num_decoder_layers"] = 2
    cfg["model"]["feedforward_dim"] = 256
    cfg["data"]["num_locations"] = 50

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

    valid = {s: {"costs": d["costs"]} for s, d in results.items() if d["costs"]}
    ref_solver = "hgs" if "hgs" in valid else next(iter(valid), None)
    stats = (
        evaluate_solver_set(valid, reference_solver=ref_solver)
        if ref_solver
        else {
            "reference_solver": None,
            "comparisons": {},
            "alpha": 0.05,
            "error": "No solver produced results",
        }
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
