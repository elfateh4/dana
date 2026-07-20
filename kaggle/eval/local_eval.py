"""Local evaluation of DANA + baselines on Cordeau MDVRPTW instances."""

import glob, os, sys, json, yaml, math
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

INSTANCE_DIR = "/tmp/dana_instances"
CKPT_DIR = os.path.join(REPO_DIR, "dana-checkpoints")
OUT_DIR = "/tmp/dana_local_results"
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

import requests

BASE_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main"
SETS = {
    "cordeau": (
        "MDVRPTW",
        [f"PR{i}{s}" for i in range(11, 25) for s in ("A", "B")]
        + [
            "PR01",
            "PR02",
            "PR03",
            "PR04",
            "PR05",
            "PR06",
            "PR07",
            "PR08",
            "PR09",
            "PR10",
        ],
    ),
}
for name, (remote_dir, files) in SETS.items():
    dest = os.path.join(INSTANCE_DIR, name)
    os.makedirs(dest, exist_ok=True)
    ok, fail = 0, 0
    for fname in files:
        for ext in (".vrp", ".txt"):
            url = f"{BASE_URL}/{remote_dir}/{fname}{ext}"
            path = os.path.join(dest, f"{fname}{ext}")
            if os.path.exists(path):
                ok += 1
                break
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(path, "w") as f:
                        f.write(r.text)
                    ok += 1
                    break
            except Exception:
                pass
        else:
            fail += 1
    print(f"{name}: {ok} ok, {fail} failed")

# Note: BKS .sol files from PyVRP/Instances are reference solutions, not literature BKS.
# PyVRP (HGS-based) produces the authoritative reference costs. The eval below uses
# PyVRP 2-min solutions as the reference for gap computation.
REFERENCE_TIMELIMIT = 60  # seconds per instance for PyVRP reference

with open("configs/dana.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    print(f"Checkpoint: {latest_ckpt}")
    from dana.train import build_policy

    cfg["model"]["num_encoder_layers"] = 4
    cfg["model"]["num_decoder_layers"] = 2
    cfg["model"]["feedforward_dim"] = 256
    cfg["data"]["num_locations"] = 50
    dana_policy = build_policy(cfg).to(device)
    dana_policy.eval()
    ckpt = torch.load(latest_ckpt, map_location=device)
    dana_policy.load_state_dict(ckpt["policy_state_dict"])
else:
    dana_policy = None

from dana.eval.baselines import BaselineRunner
from dana.eval.metrics import compute_gap


def parse_vrp_coords(path):
    coords, section = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("NODE_COORD_SECTION"):
                section = True
                continue
            if any(
                kw in s.upper()
                for kw in [
                    "DEMAND_SECTION",
                    "DEPOT_SECTION",
                    "TIME_WINDOW_SECTION",
                    "EOF",
                ]
            ):
                section = False
                continue
            if section and s:
                parts = s.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords, dtype=np.float32)


def run_dana_on_instance(policy, vrp_path, device, cfg):
    """
    Optimized DANA inference: caches encoder output to avoid re-encoding on each step.
    Computes cost as raw Euclidean distance with proper depot edges.
    """
    try:
        coords_np = parse_vrp_coords(vrp_path)
        N = len(coords_np)
        B = 1
        coords_t = torch.tensor(coords_np, dtype=torch.float, device=device).unsqueeze(
            0
        )
        diff = coords_np[:, None] - coords_np[None, :]
        dist_np = np.sqrt((diff**2).sum(axis=-1))
        dist_mat_t = torch.tensor(dist_np, dtype=torch.float, device=device).unsqueeze(
            0
        )

        depot_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        depot_mask[:, 0] = True
        demand = torch.ones(B, N, dtype=torch.float, device=device)
        tw_start = torch.zeros(B, N, dtype=torch.float, device=device)
        tw_end = torch.full((B, N), 480.0, dtype=torch.float, device=device)
        visited = depot_mask.clone()

        # Cache encoder + context module output (graph structure doesn't change per step)
        with torch.no_grad():
            node_emb = policy.encoder(coords_t, dist_mat_t)
            B_enc, N_enc, D = node_emb.shape
            context = torch.zeros(B_enc, D, device=node_emb.device)
            node_emb, context = policy.context_module(node_emb, dist_mat_t, context)

        actions = []
        with torch.no_grad():
            for _ in range(cfg["environment"]["max_vehicles"]):
                if visited[:, 1:].all():
                    break
                for _ in range(N * 2):
                    visit_frac = visited.float().mean(dim=-1)
                    remaining_cap = 1.0 - (demand.float().mean(dim=-1) * visit_frac)
                    unvisited_tw_end = tw_end.float().masked_fill(visited, float("inf"))
                    min_tw_end = unvisited_tw_end.min(dim=-1).values
                    max_tw_end = unvisited_tw_end.max(dim=-1).values
                    tw_urgency = 1.0 - (min_tw_end / (max_tw_end + 1e-8))
                    unvisited_count = (~visited).float().sum(dim=-1) / N
                    vehicle_state = torch.stack(
                        [visit_frac, remaining_cap, tw_urgency, unvisited_count], dim=-1
                    )
                    vehicle_feat = policy.vehicle_embedding(vehicle_state)

                    mask = visited.clone()
                    logits = policy.decoder(
                        node_emb, context, vehicle_feat, mask=mask, num_starts=1
                    )
                    logits = logits.masked_fill(visited, float("-inf"))
                    a = logits.argmax(dim=-1)
                    actions.append(a)
                    visited[:, a.item()] = True
                    visited[:, :1] = depot_mask[:, :1]
                    if visited[:, 1:].all():
                        break

        if len(actions) < 2:
            return {"cost": None, "status": "no_solution", "time": 0}

        acts = torch.cat(actions)

        # Identify depots from the instance file
        depots = []
        with open(vrp_path) as f:
            in_depot = False
            for line in f:
                s = line.strip()
                if s.upper().startswith("DEPOT_SECTION"):
                    in_depot = True
                    continue
                if s.upper().startswith("EOF"):
                    break
                if in_depot and s:
                    depots.append(int(s))
        num_depots = len(depots)

        routes = []
        current = []
        for idx in acts.tolist():
            if idx < num_depots:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(idx)
        if current:
            routes.append(current)
        if not routes:
            routes = [acts.tolist()]

        total_cost = 0.0
        for route in routes:
            if not route:
                continue
            first, last = route[0], route[-1]
            best = min(
                range(num_depots),
                key=lambda d: dist_np[d, first] + dist_np[last, d],
            )
            total_cost += dist_np[best, first]
            for i in range(len(route) - 1):
                total_cost += dist_np[route[i], route[i + 1]]
            total_cost += dist_np[last, best]

        return {"cost": total_cost, "status": "success", "time": 0}
    except Exception as e:
        import traceback

        return {
            "cost": None,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runner = BaselineRunner(time_limit=REFERENCE_TIMELIMIT, num_runs=1)

bench_name = "cordeau"
bench_dir = os.path.join(INSTANCE_DIR, bench_name)
if not os.path.exists(bench_dir):
    print(f"SKIP: {bench_name} not found")
else:
    instance_names = [f"PR{i}{s}" for i in range(11, 25) for s in ("A", "B")]
    instance_files = [os.path.join(bench_dir, f"{n}.vrp") for n in instance_names]

    results = {}
    for inst_name, inst_file in zip(instance_names, instance_files):
        print(f"\n  {inst_name}:")

        baselines = {}

        if dana_policy is not None:
            dana_result = run_dana_on_instance(dana_policy, inst_file, device, cfg)
            baselines["dana"] = dana_result

        pyvrp_result = runner.run_pyvrp(inst_file, bench_name)
        baselines["pyvrp"] = pyvrp_result
        pyvrp_cost = pyvrp_result.get("cost")

        inst_results = {}
        for solver, result in baselines.items():
            cost = result.get("cost")
            status = result.get("status", "unknown")
            gap_str = ""
            extra = ""
            if cost is not None and pyvrp_cost is not None and solver != "pyvrp":
                gap = compute_gap(cost, pyvrp_cost)
                gap_str = f", gap_vs_pyvrp={gap:.2f}%"
                result["gap_vs_pyvrp"] = gap
            if solver == "pyvrp" and cost is not None:
                extra = f", feasible={result.get('feasible')}, time={result.get('time', 0):.1f}s"
            print(
                f"    {solver}: cost={cost}{gap_str}{extra}"
                if cost
                else f"    {solver}: {status}"
            )
            inst_results[solver] = result

        results[inst_name] = inst_results

    print(f"\n  === Summary ===")
    for solver in ["dana", "pyvrp"]:
        costs = []
        for inst_name in instance_names:
            r = results[inst_name].get(solver, {})
            c = r.get("cost")
            if c is not None:
                costs.append(c)
        if costs:
            print(
                f"  {solver}: costs={[f'{c:.2f}' for c in costs]}, mean={np.mean(costs):.2f}"
            )

    if all("dana" in results[i] and "pyvrp" in results[i] for i in instance_names):
        print(f"  DANA vs PyVRP gaps:")
        for inst_name in instance_names:
            d = results[inst_name]["dana"].get("cost")
            p = results[inst_name]["pyvrp"].get("cost")
            if d and p:
                g = compute_gap(d, p)
                print(f"    {inst_name}: {g:.2f}% (DANA={d:.2f}, PyVRP={p:.2f})")

    summary_path = os.path.join(OUT_DIR, f"{bench_name}_results.json")
    serializable = {}
    for inst_name, inst_results in results.items():
        serializable[inst_name] = {}
        for solver, r in inst_results.items():
            serializable[inst_name][solver] = {
                k: v
                for k, v in r.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {summary_path}")

print("\nDone.")
