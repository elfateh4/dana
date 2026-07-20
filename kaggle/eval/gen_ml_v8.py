"""Generate v8 — Solomon fallback + size guard for ML models + clean structure."""
import json, os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": [s]})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s]})

md("# ML Solver Inference v8\n\nRuns POMO (CVRP/VRPTW) and RouteFinder (rf-transformer) on suitable instances. "
   "Stubs for unsupported problem types or oversized instances.")

code(r'''import os, sys, json, csv, time, math, gc, shutil, subprocess, warnings, re
import numpy as np
import requests
from pathlib import Path
warnings.filterwarnings("ignore")

DEVICE = "cpu"
INSTANCE_DIR = "/kaggle/working/instances"
CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/ai4co/routefinder.git",
    "huggingface-hub", "tensordict", "kagglehub", "requests",
], check=True, timeout=600)

# ── Instance loading ──
print("Downloading instances...")
try:
    import kagglehub
    kpath = kagglehub.dataset_download("elfateh/dana-instances")
    for item in os.listdir(kpath):
        sp = os.path.join(kpath, item)
        if os.path.isdir(sp):
            shutil.copytree(sp, os.path.join(INSTANCE_DIR, item), dirs_exist_ok=True)
    print("Instances from Kaggle")
except Exception as e:
    print(f"Kaggle failed: {e}")

# Always ensure Solomon is present (missing from Kaggle dataset)
solomon_dir = os.path.join(INSTANCE_DIR, "solomon")
SOLOMON_FILES = [
    "C101","C102","C103","C104","C105","C106","C107","C108","C109",
    "C201","C202","C203","C204","C205","C206","C207","C208",
    "R101","R102","R103","R104","R105","R106","R107","R108","R109","R110","R111","R112",
    "R201","R202","R203","R204","R205","R206","R207","R208","R209","R210","R211",
    "RC101","RC102","RC103","RC104","RC105","RC106","RC107","RC108",
    "RC201","RC202","RC203","RC204","RC205","RC206","RC207","RC208",
]
if not os.path.isdir(solomon_dir) or len([f for f in os.listdir(solomon_dir) if f.endswith('.vrp')]) < 56:
    print("Downloading Solomon instances from GitHub...")
    os.makedirs(solomon_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/PyVRP/Instances/main/VRPTW/Solomon"
    for name in SOLOMON_FILES:
        fname = f"{name}.vrp"
        p = os.path.join(solomon_dir, fname)
        if os.path.exists(p): continue
        try:
            r = requests.get(f"{base}/{fname}", timeout=30)
            if r.status_code == 200:
                with open(p, "w") as fh: fh.write(r.text)
        except: pass
    print(f"Solomon: {len([f for f in os.listdir(solomon_dir) if f.endswith('.vrp')])} instances")
else:
    print(f"Solomon: already present ({len([f for f in os.listdir(solomon_dir) if f.endswith('.vrp')])} instances)")

# Also ensure any missing benchmark dirs from GitHub
BASE_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main"
MISSING_SETS = {
    "cordeau": ("MDVRPTW", [f"PR{i}{s}" for i in range(11, 25) for s in ("A","B")]),
    "gehring": ("VRPTW", [f"{t}{n}_10_{i}" for t in ("C","R","RC") for n in (1,2) for i in range(1,11)]),
    "x_instances": ("CVRP", ["X-n101-k25","X-n106-k14","X-n110-k13","X-n115-k10","X-n120-k6",
        "X-n125-k30","X-n129-k18","X-n134-k13","X-n139-k10","X-n143-k7","X-n153-k22","X-n157-k13",
        "X-n162-k11","X-n167-k10","X-n176-k26","X-n181-k23","X-n186-k15","X-n190-k8","X-n195-k51",
        "X-n200-k36","X-n204-k19","X-n209-k16","X-n214-k11","X-n219-k73","X-n223-k34","X-n228-k23",
        "X-n233-k16","X-n237-k14","X-n242-k48","X-n247-k50"]),
}
for bname, (rdir, names) in MISSING_SETS.items():
    bdir = os.path.join(INSTANCE_DIR, bname)
    os.makedirs(bdir, exist_ok=True)
    for name in names:
        fname = f"{name}.vrp"
        p = os.path.join(bdir, fname)
        if os.path.exists(p) and os.path.getsize(p) > 0: continue
        try:
            r = requests.get(f"{BASE_URL}/{rdir}/{fname}", timeout=30)
            if r.status_code == 200:
                with open(p, "w") as fh: fh.write(r.text)
        except: pass

# ── Parse & discover ──
def parse_vrp(path):
    coords, demands = [], []
    capacity = None
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif "NODE_COORD_SECTION" in line: section = "coords"
            elif "DEMAND_SECTION" in line: section = "demands"
            elif "DEPOT_SECTION" in line: section = "depot"
            elif section == "coords":
                parts = line.split()
                if len(parts) >= 3: coords.append((float(parts[1]), float(parts[2])))
            elif section == "demands":
                parts = line.split()
                if len(parts) >= 2: demands.append(int(parts[1]))
    return np.array(coords), np.array(demands), capacity

BENCHMARKS = {"cordeau": "mdvrptw", "solomon": "vrptw", "gehring": "vrptw", "x_instances": "cvrp"}
ALL_INSTANCES = []
for bname, ptype in BENCHMARKS.items():
    bdir = os.path.join(INSTANCE_DIR, bname)
    if not os.path.isdir(bdir):
        for d in os.listdir(INSTANCE_DIR):
            if d.lower() == bname.lower(): bdir = os.path.join(INSTANCE_DIR, d); break
    if not os.path.isdir(bdir): continue
    for fname in sorted(os.listdir(bdir)):
        if fname.lower().endswith(".vrp"):
            ALL_INSTANCES.append({
                "instance": fname.replace(".vrp",""),
                "benchmark": bname,
                "problem_type": ptype,
                "path": os.path.join(bdir, fname),
            })
print(f"Total instances: {len(ALL_INSTANCES)}")
for b in BENCHMARKS:
    print(f"  {b}: {sum(1 for i in ALL_INSTANCES if i['benchmark']==b)}")
''')

md("## RouteFinder Inference Helpers")
code(r'''def build_routefinder_td(coords, demands, capacity):
    NUM_CLIENTS = len(coords) - 1
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    range_xy = np.maximum(max_xy - min_xy, 1.0)
    locs_norm = (coords - min_xy) / range_xy
    cd = demands[1:] / capacity if (capacity and len(demands) > 1) else np.zeros(NUM_CLIENTS)
    td = {
        "num_depots": np.array([[1]], dtype=np.int32),
        "locs": locs_norm[np.newaxis, :, :].astype(np.float32),
        "demand_linehaul": cd[np.newaxis, :].astype(np.float32),
        "demand_backhaul": np.zeros((1, NUM_CLIENTS), dtype=np.float32),
        "backhaul_class": np.array([[1]], dtype=np.float32),
        "distance_limit": np.array([[1e8]], dtype=np.float32),
        "time_windows": np.zeros((1, 1+NUM_CLIENTS, 2), dtype=np.float32),
        "service_time": np.zeros((1, 1+NUM_CLIENTS), dtype=np.float32),
        "vehicle_capacity": np.array([[1.0]], dtype=np.float32),
        "capacity_original": np.array([[float(capacity or 100)]], dtype=np.float32),
        "open_route": np.array([[False]], dtype=bool),
        "speed": np.array([[1.0]], dtype=np.float32),
    }
    td["time_windows"][:, :, 1] = 1e8
    return td

def patch_torch_load():
    import torch
    import torchrl.data.tensor_specs as _ts
    for _old, _new in {
        "CompositeSpec": "Composite", "BoundedTensorSpec": "Bounded",
        "UnboundedContinuousTensorSpec": "UnboundedContinuous",
        "UnboundedDiscreteTensorSpec": "UnboundedDiscrete",
        "DiscreteTensorSpec": "Bounded", "BinaryTensorSpec": "Binary",
    }.items():
        if not hasattr(_ts, _old) and hasattr(_ts, _new):
            setattr(_ts, _old, getattr(_ts, _new))
    _orig = torch.load
    def _patch(f, *a, **kw):
        kw.pop("weights_only", None)
        return _orig(f, *a, weights_only=False, **kw)
    torch.load = _patch

def raw_cost_from_routes(routes, coords):
    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    total = 0.0
    for r in routes:
        if not r: continue
        total += dist[0, r[0]]
        for i in range(len(r)-1): total += dist[r[i], r[i+1]]
        total += dist[r[-1], 0]
    return total

def run_inference(td, model, coords, n_starts=8):
    import torch
    from tensordict.tensordict import TensorDict
    from routefinder.envs.mtvrp.env import MTVRPEnv
    env = MTVRPEnv(load_solutions=False)
    policy = model.policy.to(DEVICE).eval()
    td_t = TensorDict({k: torch.tensor(v) for k, v in td.items()}, batch_size=[1]).to(DEVICE)
    td_t = env.reset(td_t)
    with torch.inference_mode():
        out = policy(td_t, env, phase="test", num_starts=n_starts, return_actions=True)
        reward = out["reward"]
        mr = reward.max(dim=-1).values if n_starts > 1 else reward.squeeze(-1)
        model_cost = -mr.item()
        acts = out.get("actions")
        if acts is not None:
            acts_np = acts.cpu().numpy()[0].squeeze()
            routes, cur = [], []
            for a in acts_np:
                if a < 1:
                    if cur: routes.append(cur); cur = []
                else: cur.append(int(a))
            if cur: routes.append(cur)
            return {"cost": model_cost, "raw_euclidean_cost": raw_cost_from_routes(routes, coords),
                    "routes": routes, "status": "success"}
    return {"status": "no_actions"}

def load_model(ckpt):
    import torch
    from routefinder.models import RouteFinderBase
    patch_torch_load()
    return RouteFinderBase.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
''')

md("## Download HuggingFace Checkpoints")
code(r'''import torch
from huggingface_hub import hf_hub_download
ckpt_rf = hf_hub_download("ai4co/routefinder", "checkpoints/100/rf-transformer.ckpt")
ckpt_pomo_cvrp = hf_hub_download("ai4co/routefinder", "checkpoints/100/pomo/pomo-cvrp.ckpt")
ckpt_pomo_vrptw = hf_hub_download("ai4co/routefinder", "checkpoints/100/pomo/pomo-vrptw.ckpt")
print("Checkpoints downloaded")
''')

def solver_cell(name, label, ckpt_var, model_load_code, size_limit=300):
    """Generate solver cell with size guard."""
    limit_str = str(size_limit)
    code(rf'''print("\n=== {label.upper()} ===")
{name}_rows = []
try:
    {model_load_code}
    print("Model loaded")
    MAX_NODES = {limit_str}
    for inst in ALL_INSTANCES:
        try:
            c, _, _ = parse_vrp(inst["path"])
            n = len(c)
        except:
            n = 0
        if n > MAX_NODES or n < 10:
            {name}_rows.append({{**inst, "solver": "{label}", "cost": None,
                "raw_euclidean_cost": None, "cost_unit": "euclidean",
                "time_s": 0, "status": "stub",
                "error_msg": "Instance size mismatch (%d nodes)" % n}})
            continue
        try:
            coords, demands, cap = parse_vrp(inst["path"])
            t0 = time.time()
            td = build_routefinder_td(coords, demands, cap)
            res = run_inference(td, model, coords, n_starts=8)
            dt = time.time() - t0
            {name}_rows.append({{**inst, "solver": "{label}",
                "cost": res.get("cost"), "raw_euclidean_cost": res.get("raw_euclidean_cost"),
                "cost_unit": "euclidean", "time_s": round(dt, 2),
                "status": res.get("status","error"),
                "error_msg": "" if res["status"]=="success" else str(res)}})
            print(f"  {{inst['instance']}} ({{n}}): {{res.get('raw_euclidean_cost'):.2f}} ({{dt:.1f}}s)")
            gc.collect()
        except Exception as e:
            {name}_rows.append({{**inst, "solver": "{label}", "cost": None,
                "raw_euclidean_cost": None, "cost_unit": "euclidean",
                "time_s": 0, "status": "error", "error_msg": str(e)}})
            print(f"  {{inst['instance']}} ({{n}}): ERROR {{e}}")
except Exception as e:
    print(f"{label} failed: {{e}}")
    for inst in ALL_INSTANCES:
        {name}_rows.append({{**inst, "solver": "{label}", "cost": None,
            "raw_euclidean_cost": None, "cost_unit": "euclidean",
            "time_s": 0, "status": "stub", "error_msg": f"stub: {{e}}"}})
ok = sum(1 for r in {name}_rows if r["status"]=="success")
st = sum(1 for r in {name}_rows if r["status"]=="stub")
er = sum(1 for r in {name}_rows if r["status"]!="success" and r["status"]!="stub")
print(f"{label}: {{len({name}_rows)}} results ({{ok}} ok, {{st}} stub, {{er}} error)")
''')

md("## 1. POMO")
solver_cell("pomo", "pomo", "ckpt_pomo_cvrp",
    "m_cvrp = load_model(ckpt_pomo_cvrp)\n    m_vrptw = load_model(ckpt_pomo_vrptw)",
    size_limit=300)

md("## 2. RouteFinder (rf-transformer)")
solver_cell("rf", "routefinder", "ckpt_rf",
    "model = load_model(ckpt_rf)",
    size_limit=300)

md("## 3–8: Stub Solvers")
for solver, reason in [
    ("am", "no pretrained weights available"),
    ("deepaco", "no pretrained weights available"),
    ("parco", "only HCVRP checkpoint (incompatible)"),
    ("rrnco", "only RCVRP/RCVRPTW checkpoints"),
    ("bq-nco", "repo drakulic/bqnco returns 404"),
    ("goal", "repo kaist-silab/goal returns 404"),
]:
    var = solver.replace("-", "_")
    code(f'''print("\\n=== {solver.upper()} ===")
{var}_rows = [dict(i, solver="{solver}", cost=None, raw_euclidean_cost=None,
    cost_unit="euclidean", time_s=0, status="stub", error_msg="{reason}")
    for i in ALL_INSTANCES]
print(f"{solver}: {{len({var}_rows)}} stubs")
''')

md("## Upload Results")
code(r'''ALL_ROWS = {
    "pomo": pomo_rows,
    "routefinder": rf_rows,
    "am": am_rows, "deepaco": deepaco_rows, "parco": parco_rows,
    "rrnco": rrnco_rows, "bq-nco": bq_nco_rows, "goal": goal_rows,
}
FIELDS = ["instance","solver","benchmark","problem_type",
          "cost","raw_euclidean_cost","cost_unit","time_s","status","error_msg"]

total_ok = 0
for solver, rows in ALL_ROWS.items():
    total_ok += sum(1 for r in rows if r["status"]=="success")
    clean = [{k: r.get(k) for k in FIELDS} for r in rows]
    csv_path = f"/kaggle/working/{solver}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(clean)
    ddir = f"/kaggle/working/dana-results-{solver}"
    os.makedirs(ddir, exist_ok=True)
    shutil.copy(csv_path, os.path.join(ddir, f"{solver}_results.csv"))
    with open(os.path.join(ddir, "dataset-metadata.json"), "w") as f:
        json.dump({"title": f"dana-results-{solver}", "id": f"elfateh/dana-results-{solver}",
                    "licenses": [{"name": "CC0-1.0"}]}, f)
    r = subprocess.run(["kaggle","datasets","version","-p",ddir,
                        "-m",f"Updated {solver}","--dir-mode","zip"],
                       capture_output=True,text=True)
    if r.returncode != 0:
        r2 = subprocess.run(["kaggle","datasets","create","-p",ddir,"--dir-mode","zip"],
                           capture_output=True,text=True)
        print(f"{solver}: {'created' if r2.returncode==0 else 'FAIL: '+r2.stderr.strip()}")
    else:
        print(f"{solver}: version updated")
print(f"\nTotal: {sum(len(r) for r in ALL_ROWS.values())} rows, {total_ok} success")
''')

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True,
                    "language": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4, "nbformat_minor": 4,
}
with open("/home/elfateh/Projects/dana/kaggle/eval/06_solver_ml.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Notebook v8 generated!")
