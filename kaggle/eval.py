import os
import sys
import subprocess
import json
import yaml
import numpy as np

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
        "pyvrp",
        "ortools",
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

INSTANCE_DIR = "/kaggle/input/dana-benchmarks"
if not os.path.exists(INSTANCE_DIR):
    import kagglehub

    INSTANCE_DIR = kagglehub.dataset_download("elfateh/dana-benchmarks")

os.makedirs("reports", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

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

from dana.eval.baselines import BaselineRunner
from dana.eval.metrics import (
    compute_passmark_factor,
    compute_gap,
    evaluate_solver_set,
    compute_summary,
)
from dana.eval.plots import generate_report

runner = BaselineRunner(time_limit=3600, num_runs=10)

benchmarks = {
    "cordeau": {"dir": "cordeau", "problem": "mdvrptw"},
    "solomon": {"dir": "solomon", "problem": "vrptw"},
    "gehring": {"dir": "gehring", "problem": "vrptw"},
    "x_instances": {"dir": "x_instances", "problem": "cvrp"},
}

all_results = {}
for bench_name, bench_info in benchmarks.items():
    print(f"\n=== Running {bench_name} ===")
    bench_dir = os.path.join(INSTANCE_DIR, bench_info["dir"])
    instance_files = [
        os.path.join(bench_dir, f) for f in os.listdir(bench_dir) if f.endswith(".txt")
    ][:3]
    bench_results = {
        solver: {"costs": [], "gaps": [], "times": []}
        for solver in ["dana", "pyvrp", "hgs", "lkh3", "ortools"]
    }
    for inst_file in instance_files:
        print(f"  Instance: {os.path.basename(inst_file)}")
        baselines = runner.run_all(inst_file, bench_info["problem"])
        for solver, result in baselines.items():
            if result.get("cost") is not None:
                bench_results[solver]["costs"].append(result["cost"])
                if result.get("time"):
                    bench_results[solver]["times"].append(result["time"])
        dana_cost = None
        if dana_cost is not None:
            bench_results["dana"]["costs"].append(dana_cost)
    best_known = (
        min(min(v["costs"]) for v in bench_results.values() if v["costs"])
        if any(v["costs"] for v in bench_results.values())
        else 1.0
    )
    summary_results = {}
    for solver, data in bench_results.items():
        if data["costs"]:
            summary = compute_summary(data["costs"], best_known)
            summary["mean_time"] = np.mean(data["times"]) if data["times"] else 0
            summary["gaps"] = [compute_gap(c, best_known) for c in data["costs"]]
            summary["times"] = data["times"]
            summary_results[solver] = summary
    all_results[bench_name] = summary_results
    stat_test = evaluate_solver_set(
        {s: {"costs": d["costs"]} for s, d in bench_results.items()},
        reference_solver="hgs",
    )
    generate_report(
        summary_results,
        stat_test,
        bench_name,
        output_dir="/kaggle/working/results",
    )
    with open(f"/kaggle/working/results/{bench_name}_results.json", "w") as f:
        json.dump(summary_results, f, indent=2, default=str)

print("\n=== Evaluation complete ===")
import kagglehub

kagglehub.upload_dataset(
    "/kaggle/working/results",
    "elfateh/dana-results",
    message="Full evaluation: DANA vs HGS/LKH-3/PyVRP/OR-Tools on all 5 benchmarks",
)
