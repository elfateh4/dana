import os, sys, subprocess, json, yaml
import numpy as np

subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "torch",
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
        "pyvrp",
        "ortools",
    ],
    check=True,
)

REPO = "https://github.com/elfateh4/dana.git"
REPO_DIR = "/kaggle/working/dana"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, "/kaggle/input/dana-checkpoints")

INSTANCE_DIR = "/kaggle/input/dana-benchmarks"
os.makedirs("reports", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

HGS_BIN = "/usr/local/bin/HGS-CVRP"
if not os.path.exists(HGS_BIN):
    subprocess.run(
        ["git", "clone", "https://github.com/vidalt/HGS-CVRP.git", "/tmp/HGS-CVRP"],
        check=True,
    )
    subprocess.run(["make", "-C", "/tmp/HGS-CVRP"], check=True)
    subprocess.run(["cp", "/tmp/HGS-CVRP/HGS-CVRP", HGS_BIN], check=True)

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
import kagglehub

kagglehub.upload_dataset(
    "/kaggle/working/results",
    "elfateh/dana-results",
    message="DANA vs HGS/LKH-3/PyVRP/OR-Tools evaluation",
)
