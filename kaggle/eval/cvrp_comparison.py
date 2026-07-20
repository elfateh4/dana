"""Compare PyVRP vs DANA vs RouteFinder on A-n32-k5 CVRP instance."""

import math, os, sys, json, warnings
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "external/routefinder"))

os.chdir(REPO_DIR)
warnings.filterwarnings("ignore")

INSTANCE_PATH = "/tmp/dana_instances/A-n32-k5.vrp"
CAPACITY = 100
NUM_DEPOTS = 1


# ── Shared parse ────────────────────────────────────────────────────
def parse_cvrp(path):
    with open(path) as f:
        text = f.read()
    lines = text.strip().split("\n")

    section = None
    coords, demands = [], []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        if s.upper().startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if s.upper().startswith("DEMAND_SECTION"):
            section = "demand"
            continue
        if s.upper().startswith("DEPOT_SECTION"):
            section = "depot"
            continue
        if s.upper().startswith("EOF"):
            break
        parts = s.split()
        if not parts:
            continue
        if section == "coords" and len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))
        elif section == "demand" and len(parts) >= 2:
            demands.append(float(parts[1]))

    coords = np.array(coords, dtype=np.float32)
    demands = np.array(demands[: len(coords)], dtype=np.float32)
    return coords, demands


def raw_cost(route, dist_mat):
    cost = 0.0
    if not route:
        return 0.0
    cost += dist_mat[0, route[0]]
    for i in range(len(route) - 1):
        cost += dist_mat[route[i], route[i + 1]]
    cost += dist_mat[route[-1], 0]
    return cost


# ── 1. PyVRP ────────────────────────────────────────────────────────
def run_pyvrp(path, time_limit=30):
    from pyvrp import read, Model
    from pyvrp.stop import MaxRuntime

    data = read(path)
    model = Model.from_data(data)
    result = model.solve(stop=MaxRuntime(time_limit), seed=42, display=False)

    depots = data.depots()
    clients = data.clients()
    loc_coords = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in clients]

    def loc_dist(i, j):
        dx = loc_coords[i][0] - loc_coords[j][0]
        dy = loc_coords[i][1] - loc_coords[j][1]
        return math.sqrt(dx * dx + dy * dy)

    routes = []
    total = 0.0
    for route in result.best.routes():
        visits = list(route.visits())
        if not visits:
            continue
        sd, ed = int(route.start_depot()), int(route.end_depot())
        cost = loc_dist(sd, visits[0])
        for i in range(len(visits) - 1):
            cost += loc_dist(visits[i], visits[i + 1])
        cost += loc_dist(visits[-1], ed)
        total += cost
        routes.append(visits)

    return {
        "cost": total,
        "cost_pyvrp": result.cost(),
        "feasible": result.is_feasible(),
        "time": result.runtime,
        "routes": routes,
        "num_vehicles_used": len(routes),
    }


# ── 2. DANA (capacity-aware inference) ────────────────────────────
def run_dana(coords, demands, policy, cfg, capacity=100, device="cpu"):
    import torch

    N = len(coords)
    B = 1
    coords_t = torch.tensor(coords, dtype=torch.float, device=device).unsqueeze(0)
    diff = coords[:, None] - coords[None, :]
    dist_np = np.sqrt((diff**2).sum(axis=-1))
    dist_mat_t = torch.tensor(dist_np, dtype=torch.float, device=device).unsqueeze(0)

    demand_t = torch.tensor(demands, dtype=torch.float, device=device).unsqueeze(0)

    with torch.no_grad():
        node_emb = policy.encoder(coords_t, dist_mat_t)
        B_enc, N_enc, D = node_emb.shape
        context = torch.zeros(B_enc, D, device=node_emb.device)
        node_emb, context = policy.context_module(node_emb, dist_mat_t, context)

    visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    visited[:, 0] = True  # depot starts visited (no need to visit it first)

    actions = []
    remaining_cap = torch.full((B, 1), float(capacity), device=device)
    at_depot = torch.ones(B, dtype=torch.bool, device=device)  # start at depot

    for vehicle_idx in range(cfg["environment"]["max_vehicles"]):
        if visited[:, 1:].all():
            break

        # Start a new route: reset capacity, add depot marker
        remaining_cap = torch.full((B, 1), float(capacity), device=device)
        route_started = False

        for _ in range(N * 2):
            if visited[:, 1:].all():
                break

            # Unvisited clients and their demands
            unvisited_mask = ~visited[:, 1:].squeeze(0)
            unvisited_demands = demand_t[:, 1:].squeeze(0)[unvisited_mask]

            if len(unvisited_demands) == 0:
                break

            # Check if any unvisited client fits in remaining capacity
            fits = unvisited_demands <= remaining_cap.squeeze()
            if not fits.any():
                # No client fits: end this route
                break

            # Build action mask: mask depot during route, mask clients exceeding capacity
            cap_mask = visited.clone()
            cap_mask[:, 0] = True  # never visit depot mid-route
            client_fits = (remaining_cap - demand_t[:, 1:]) >= 0
            cap_mask[:, 1:] |= ~client_fits

            visit_frac = visited.float().mean(dim=-1)
            norm_remaining = remaining_cap.squeeze(-1) / float(capacity)
            unvisited_count = (~visited).float().sum(dim=-1) / N
            tw_urgency = torch.zeros_like(visit_frac)
            vehicle_state = torch.stack(
                [visit_frac, norm_remaining, tw_urgency, unvisited_count], dim=-1
            )
            vehicle_feat = policy.vehicle_embedding(vehicle_state)

            logits = policy.decoder(
                node_emb, context, vehicle_feat, mask=cap_mask, num_starts=1
            )
            logits = logits.masked_fill(cap_mask, float("-inf"))
            a = logits.argmax(dim=-1)

            if not route_started:
                # Insert depot marker at start of each route (except first)
                if vehicle_idx > 0:
                    actions.append(torch.zeros(1, dtype=torch.long, device=device))
                route_started = True

            actions.append(a)
            chosen = a.item()
            visited[:, chosen] = True
            remaining_cap -= demand_t[:, chosen]

    if len(actions) < 2:
        return {"cost": None, "routes": [], "status": "no_solution"}

    acts = torch.cat(actions).tolist()

    # Split by depot visits
    routes = []
    current = []
    for idx in acts:
        if idx < NUM_DEPOTS:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(idx)
    if current:
        routes.append(current)
    if not routes:
        routes = [acts]

    total = sum(raw_cost(r, dist_np) for r in routes)
    return {
        "cost": total,
        "routes": routes,
        "status": "success",
        "num_routes": len(routes),
    }


# ── 3. RouteFinder (generic) ──────────────────────────────────────
def _build_routefinder_td(coords, demands, capacity):
    NUM_CLIENTS = len(coords) - 1
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1.0
    locs_norm = (coords - min_xy) / range_xy
    client_demands = demands[1:] / capacity

    td = {
        "num_depots": np.array([[1]], dtype=np.int32),
        "locs": locs_norm[np.newaxis, :, :].astype(np.float32),
        "demand_linehaul": client_demands[np.newaxis, :].astype(np.float32),
        "demand_backhaul": np.zeros((1, NUM_CLIENTS), dtype=np.float32),
        "backhaul_class": np.array([[1]], dtype=np.float32),
        "distance_limit": np.array([[1e8]], dtype=np.float32),
        "time_windows": np.zeros((1, 1 + NUM_CLIENTS, 2), dtype=np.float32),
        "service_time": np.zeros((1, 1 + NUM_CLIENTS), dtype=np.float32),
        "vehicle_capacity": np.array([[1.0]], dtype=np.float32),
        "capacity_original": np.array([[float(capacity)]], dtype=np.float32),
        "open_route": np.array([[False]], dtype=bool),
        "speed": np.array([[1.0]], dtype=np.float32),
    }
    td["time_windows"][:, :, 1] = 1e8
    return td


def _load_routefinder_model(ckpt_path, model_class):
    import torch
    import torchrl.data.tensor_specs as _ts

    for _old, _new in {
        "CompositeSpec": "Composite",
        "BoundedTensorSpec": "Bounded",
        "UnboundedContinuousTensorSpec": "UnboundedContinuous",
        "UnboundedDiscreteTensorSpec": "UnboundedDiscrete",
        "DiscreteTensorSpec": "Bounded",
        "BinaryTensorSpec": "Binary",
    }.items():
        if not hasattr(_ts, _old) and hasattr(_ts, _new):
            setattr(_ts, _old, getattr(_ts, _new))

    _orig_load = torch.load

    def _patch_load(f, *a, **kw):
        kw.pop("weights_only", None)
        return _orig_load(f, *a, weights_only=False, **kw)

    torch.load = _patch_load
    model = model_class.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    )
    torch.load = _orig_load
    return model


def _run_routefinder_inference(td, model, coords, n_starts=4):
    import torch
    from tensordict.tensordict import TensorDict
    from routefinder.envs.mtvrp.env import MTVRPEnv

    env = MTVRPEnv(load_solutions=False)
    policy = model.policy.to("cpu").eval()

    td_t = TensorDict({k: torch.tensor(v) for k, v in td.items()}, batch_size=[1])
    td_t = env.reset(td_t)

    with torch.inference_mode():
        out = policy(td_t, env, phase="test", num_starts=n_starts, return_actions=True)
        reward = out["reward"]
        max_reward = reward.max(dim=-1).values if n_starts > 1 else reward.squeeze(-1)
        model_cost = -max_reward.item()

        actions_np = out.get("actions")
        if actions_np is not None:
            actions_np = actions_np.cpu().numpy()
            act = actions_np[0].squeeze()
            routes = []
            current = []
            for a in act:
                if a < 1:
                    if current:
                        routes.append(current)
                        current = []
                else:
                    current.append(int(a))
            if current:
                routes.append(current)

            diff = coords[:, None] - coords[None, :]
            dist_np = np.sqrt((diff**2).sum(axis=-1))
            total = sum(raw_cost(r, dist_np) for r in routes)
            return {
                "cost": total,
                "model_cost": model_cost,
                "routes": routes,
                "num_routes": len(routes),
                "status": "success",
            }

    return {"cost": None, "status": "no_actions"}


def run_routefinder(coords, demands, capacity):
    td = _build_routefinder_td(coords, demands, capacity)
    ckpt_path = os.path.join(
        REPO_DIR, "external/routefinder/checkpoints/100/rf-transformer.ckpt"
    )
    from routefinder.models import RouteFinderBase

    model = _load_routefinder_model(ckpt_path, RouteFinderBase)
    return _run_routefinder_inference(td, model, coords, n_starts=4)


def run_pomo_cvrp(coords, demands, capacity):
    td = _build_routefinder_td(coords, demands, capacity)
    ckpt_path = os.path.join(
        REPO_DIR, "external/routefinder/checkpoints/100/pomo/pomo-cvrp.ckpt"
    )
    from routefinder.models import RouteFinderBase

    model = _load_routefinder_model(ckpt_path, RouteFinderBase)
    return _run_routefinder_inference(td, model, coords, n_starts=4)


def run_rf_transformer_50(coords, demands, capacity):
    td = _build_routefinder_td(coords, demands, capacity)
    ckpt_path = os.path.join(
        REPO_DIR, "external/routefinder/checkpoints/50/rf-transformer.ckpt"
    )
    from routefinder.models import RouteFinderBase

    model = _load_routefinder_model(ckpt_path, RouteFinderBase)
    return _run_routefinder_inference(td, model, coords, n_starts=4)


# ── 4. OR-Tools ────────────────────────────────────────────────────
def run_ortools(coords, demands, capacity, time_limit=30):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    import math

    N = len(coords)
    NUM_VEHICLES = 5

    # Distance callback
    def dist_cb(from_i, to_i):
        dx = coords[from_i, 0] - coords[to_i, 0]
        dy = coords[from_i, 1] - coords[to_i, 1]
        return int(math.sqrt(dx * dx + dy * dy) + 0.5)

    manager = pywrapcp.RoutingIndexManager(N, NUM_VEHICLES, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_cb = routing.RegisterTransitCallback(
        lambda f, t: dist_cb(manager.IndexToNode(f), manager.IndexToNode(t))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Capacity
    demand_cb = routing.RegisterUnaryTransitCallback(
        lambda i: int(demands[manager.IndexToNode(i)])
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [int(capacity)] * NUM_VEHICLES, True, "Capacity"
    )

    # Solve
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return {"cost": None, "routes": [], "status": "no_solution"}

    # Extract routes
    routes = []
    total = 0.0
    diff = coords[:, None] - coords[None, :]
    dist_np = np.sqrt((diff**2).sum(axis=-1))

    for v in range(NUM_VEHICLES):
        index = routing.Start(v)
        route = []
        prev = manager.IndexToNode(index)
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(index):
                break
            node = manager.IndexToNode(index)
            route.append(node)
        if route:
            routes.append(route)
            total += dist_np[0, route[0]]
            for i in range(len(route) - 1):
                total += dist_np[route[i], route[i + 1]]
            total += dist_np[route[-1], 0]

    return {
        "cost": total,
        "routes": routes,
        "status": "success",
        "num_routes": len(routes),
    }


# ═══════════════════════════════════════════════════════════════════
def main():
    import yaml
    import torch

    print("=" * 70)
    print("  CVRP Solver Comparison: A-n32-k5 (optimal = 784)")
    print("=" * 70)

    coords, demands = parse_cvrp(INSTANCE_PATH)
    print(
        f"\n  Instance: A-n32-k5 ({len(coords)} nodes, {len(coords) - 1} clients, cap={CAPACITY})"
    )

    # ── PyVRP ──
    print(f"\n  ┌─ PyVRP (30s time limit) ──────────────────────")
    pyvrp_r = run_pyvrp(INSTANCE_PATH, time_limit=30)
    print(f"  │ Cost: {pyvrp_r['cost']:.2f} (pyvrp internal: {pyvrp_r['cost_pyvrp']})")
    print(
        f"  │ Feasible: {pyvrp_r['feasible']}, Vehicles: {pyvrp_r['num_vehicles_used']}"
    )
    print(f"  │ Routes: {pyvrp_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── DANA ──
    with open("configs/dana.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["num_encoder_layers"] = 4
    cfg["model"]["num_decoder_layers"] = 2
    cfg["model"]["feedforward_dim"] = 256
    cfg["data"]["num_locations"] = 50

    from dana.train import build_policy

    policy = build_policy(cfg)
    policy.eval()
    ckpt = torch.load("dana-checkpoints/dana_epoch_50.pt", map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])

    print(f"\n  ┌─ DANA (capacity-aware inference) ─────────────")
    dana_r = run_dana(coords, demands, policy, cfg, capacity=CAPACITY, device="cpu")
    print(f"  │ Cost: {dana_r['cost']:.2f}")
    print(f"  │ Routes: {dana_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── RouteFinder ──
    print(f"\n  ┌─ RouteFinder RF-Transformer (4 starts) ───────")
    rf_r = run_routefinder(coords, demands, CAPACITY)
    print(f"  │ Cost: {rf_r['cost']:.2f}")
    print(f"  │ Routes: {rf_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── OR-Tools ──
    print(f"\n  ┌─ OR-Tools (30s GLS) ──────────────────────────")
    ortools_r = run_ortools(coords, demands, CAPACITY, time_limit=30)
    print(f"  │ Cost: {ortools_r['cost']:.2f}")
    print(f"  │ Vehicles: {ortools_r['num_routes']}")
    print(f"  │ Routes: {ortools_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── POMO CVRP ──
    print(f"\n  ┌─ POMO CVRP (100-ckpt, 4 starts) ─────────────")
    pomo_r = run_pomo_cvrp(coords, demands, CAPACITY)
    print(f"  │ Cost: {pomo_r['cost']:.2f}")
    print(f"  │ Routes: {pomo_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── RF-Transformer 50 ──
    print(f"\n  ┌─ RF-Transformer (50-ckpt, 4 starts) ─────────")
    rf50_r = run_rf_transformer_50(coords, demands, CAPACITY)
    print(f"  │ Cost: {rf50_r['cost']:.2f}")
    print(f"  │ Routes: {rf50_r['routes']}")
    print(f"  └──────────────────────────────────────────────")

    # ── Comparison ──
    print(f"\n  {'=' * 50}")
    print(f"  {'Solver':<20} {'Cost':<12} {'#Routes':<10}")
    print(f"  {'─' * 42}")
    print(
        f"  {'PyVRP (baseline)':<20} {pyvrp_r['cost']:<12.2f} {pyvrp_r['num_vehicles_used']:<10}"
    )
    print(
        f"  {'DANA':<20} {dana_r['cost'] if dana_r['cost'] else 0:<12.2f} {dana_r.get('num_routes', 0):<10}"
    )
    if rf_r["cost"]:
        print(f"  {'RouteFinder':<20} {rf_r['cost']:<12.2f} {rf_r['num_routes']:<10}")
    if ortools_r["cost"]:
        print(
            f"  {'OR-Tools':<20} {ortools_r['cost']:<12.2f} {ortools_r['num_routes']:<10}"
        )
    if pomo_r["cost"]:
        print(f"  {'POMO CVRP':<20} {pomo_r['cost']:<12.2f} {pomo_r['num_routes']:<10}")
    if rf50_r["cost"]:
        print(
            f"  {'RF-Transformer 50':<20} {rf50_r['cost']:<12.2f} {rf50_r['num_routes']:<10}"
        )
    print(f"  {'─' * 42}")
    if pyvrp_r["cost"] and dana_r["cost"]:
        dana_gap = (dana_r["cost"] - pyvrp_r["cost"]) / pyvrp_r["cost"] * 100
        print(f"  DANA gap vs PyVRP: {dana_gap:.1f}%")
    if pyvrp_r["cost"] and rf_r["cost"]:
        rf_gap = (rf_r["cost"] - pyvrp_r["cost"]) / pyvrp_r["cost"] * 100
        print(f"  RouteFinder-100 gap vs PyVRP: {rf_gap:.1f}%")
    if pyvrp_r["cost"] and ortools_r["cost"]:
        ortools_gap = (ortools_r["cost"] - pyvrp_r["cost"]) / pyvrp_r["cost"] * 100
        print(f"  OR-Tools gap vs PyVRP: {ortools_gap:.1f}%")
    if pyvrp_r["cost"] and pomo_r["cost"]:
        pomo_gap = (pomo_r["cost"] - pyvrp_r["cost"]) / pyvrp_r["cost"] * 100
        print(f"  POMO CVRP gap vs PyVRP: {pomo_gap:.1f}%")
    if pyvrp_r["cost"] and rf50_r["cost"]:
        rf50_gap = (rf50_r["cost"] - pyvrp_r["cost"]) / pyvrp_r["cost"] * 100
        print(f"  RF-Transformer 50 gap vs PyVRP: {rf50_gap:.1f}%")
    print(f"  {'=' * 50}")

    # Save
    def convert(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v

    results = {
        "pyvrp": {
            "cost": convert(pyvrp_r["cost"]),
            "num_vehicles": convert(pyvrp_r["num_vehicles_used"]),
        },
        "dana": {
            "cost": convert(dana_r["cost"]),
            "num_routes": convert(dana_r.get("num_routes", 0)),
        },
        "routefinder": {
            "cost": convert(rf_r["cost"]),
            "num_routes": convert(rf_r.get("num_routes", 0)),
        },
        "ortools": {
            "cost": convert(ortools_r["cost"]),
            "num_routes": convert(ortools_r.get("num_routes", 0)),
        },
        "pomo_cvrp": {
            "cost": convert(pomo_r["cost"]),
            "num_routes": convert(pomo_r.get("num_routes", 0)),
        },
        "rf_transformer_50": {
            "cost": convert(rf50_r["cost"]),
            "num_routes": convert(rf50_r.get("num_routes", 0)),
        },
    }
    out_dir = "/tmp/dana_local_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cvrp_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_dir}/cvrp_comparison.json")
    print()


if __name__ == "__main__":
    main()
