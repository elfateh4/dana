"""RouteFinder inference on Cordeau MDVRPTW instances.
Converts .vrp -> routefinder .npz format, runs model, computes costs in original coords.
"""

import math, os, sys, json, glob, warnings
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RF_DIR = os.path.join(REPO_DIR, "external", "routefinder")
sys.path.insert(0, RF_DIR)

os.chdir(RF_DIR)

INSTANCE_DIR = "/tmp/dana_instances/cordeau"
OUT_DIR = "/tmp/dana_local_results"
os.makedirs(OUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


def parse_cordeau_vrp(path):
    """Parse Cordeau MDVRPTW .vrp file. Returns dict of parsed sections."""
    coords = []
    demands = []
    service_times = []
    time_windows = []
    depots = []
    vehicles_per_depot = []
    num_vehicles = None
    capacity = None

    section = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("NODE_COORD_SECTION"):
                section = "coords"
                continue
            elif s.upper().startswith("DEMAND_SECTION"):
                section = "demand"
                continue
            elif s.upper().startswith("SERVICE_TIME_SECTION"):
                section = "service"
                continue
            elif s.upper().startswith("TIME_WINDOW_SECTION"):
                section = "tw"
                continue
            elif s.upper().startswith("DEPOT_SECTION"):
                section = "depot"
                continue
            elif s.upper().startswith("VEHICLES_DEPOT_SECTION"):
                section = "vehicles"
                continue
            elif s.upper().startswith("EOF"):
                break

            parts = s.split()
            if not parts:
                continue

            if section == "coords":
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
            elif section == "demand":
                if len(parts) >= 2:
                    demands.append(float(parts[1]))
            elif section == "service":
                if len(parts) >= 2:
                    service_times.append(float(parts[1]))
            elif section == "tw":
                if len(parts) >= 3:
                    time_windows.append((float(parts[1]), float(parts[2])))
            elif section == "depot":
                if len(parts) >= 1:
                    d = int(parts[0])
                    if d > 0:
                        depots.append(d)
            elif section == "vehicles":
                if len(parts) >= 2:
                    vehicles_per_depot.append(int(parts[1]))
            elif parts[0].isdigit():
                if len(parts) >= 3 and section == "coords":
                    coords.append((float(parts[1]), float(parts[2])))
                elif len(parts) >= 2 and section == "demand":
                    demands.append(float(parts[1]))

    # Get DIMENSION, VEHICLES, CAPACITY from header
    for line_text in open(path):
        ls = line_text.strip()
        if ls.upper().startswith("DIMENSION"):
            try:
                dim = int(ls.split(":")[-1].strip())
            except:
                dim = len(coords)
        elif ls.upper().startswith("VEHICLES"):
            try:
                num_vehicles = int(ls.split(":")[-1].strip())
            except:
                pass
        elif ls.upper().startswith("CAPACITY"):
            try:
                capacity = float(ls.split(":")[-1].strip())
            except:
                pass

    if len(depots) == 0:
        depots = [1]
    # depot IDs are 1-indexed; convert to 0-index
    depot_ids = [d - 1 for d in depots]

    return {
        "coords": np.array(coords, dtype=np.float32),
        "demands": np.array(demands[: len(coords)], dtype=np.float32),
        "service_times": np.array(service_times[: len(coords)], dtype=np.float32)
        if service_times
        else None,
        "time_windows": np.array(time_windows[: len(coords)], dtype=np.float32)
        if time_windows
        else None,
        "depot_indices": depot_ids,
        "num_depots": len(depots),
        "vehicles_per_depot": vehicles_per_depot
        or [num_vehicles // len(depots)] * len(depots),
        "num_vehicles": num_vehicles or sum(vehicles_per_depot),
        "capacity": capacity or 0,
        "num_clients": len(coords) - len(depots),
    }


def convert_instance_to_routefinder(vrp_path):
    """Convert Cordeau .vrp to routefinder .npz format."""
    parsed = parse_cordeau_vrp(vrp_path)

    coords = parsed["coords"]
    depot_ids = parsed["depot_indices"]
    num_depots = parsed["num_depots"]
    num_clients = parsed["num_clients"]
    capacity = parsed["capacity"]
    demands = parsed["demands"]

    # Separate depot coords and client coords
    client_mask = np.ones(len(coords), dtype=bool)
    client_mask[depot_ids] = False
    client_indices = np.where(client_mask)[0]

    depot_coords = coords[depot_ids]
    client_coords = coords[client_indices]

    # Normalize coordinates to [0, 1]
    all_pts = np.vstack([depot_coords, client_coords])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1.0

    # Store original coords for later cost computation
    orig_depot_coords = depot_coords.copy()
    orig_client_coords = client_coords.copy()

    depot_coords = (depot_coords - min_xy) / range_xy
    client_coords = (client_coords - min_xy) / range_xy

    locs = np.vstack([depot_coords, client_coords]).astype(np.float32)

    # Demands (only for clients, exclude depots)
    client_demands = demands[client_indices].copy()
    if capacity > 0:
        client_demands = client_demands / capacity

    # Time windows
    tw = parsed["time_windows"]
    if tw is not None:
        client_tws = tw[client_indices]
        # Normalize time windows by max time
        max_tw = max(client_tws.max(), 1.0)
        depot_tws = np.zeros((num_depots, 2), dtype=np.float32)
        depot_tws[:, 1] = max_tw
        time_windows = np.vstack([depot_tws, client_tws / max_tw]).astype(np.float32)
    else:
        time_windows = np.zeros((num_depots + num_clients, 2), dtype=np.float32)
        time_windows[:, 1] = 1e8

    # Service times
    st = parsed["service_times"]
    if st is not None:
        client_st = st[client_indices]
        max_st = max(client_st.max(), 1.0)
        depot_st = np.zeros(num_depots, dtype=np.float32)
        service_time = np.concatenate([depot_st, client_st / max_st]).astype(np.float32)
    else:
        service_time = np.zeros(num_depots + num_clients, dtype=np.float32)

    # Convert to tensors with batch dimension
    td = {
        "num_depots": np.array([[num_depots]], dtype=np.int32),
        "locs": locs[np.newaxis, :, :],
        "demand_linehaul": client_demands[np.newaxis, :],
        "demand_backhaul": np.zeros((1, num_clients), dtype=np.float32),
        "backhaul_class": np.array([[1]], dtype=np.float32),
        "distance_limit": np.array([[1e8]], dtype=np.float32),
        "time_windows": time_windows[np.newaxis, :, :],
        "service_time": service_time[np.newaxis, :],
        "vehicle_capacity": np.array([[1.0]], dtype=np.float32),
        "capacity_original": np.array([[float(capacity)]], dtype=np.float32),
        "open_route": np.array([[False]], dtype=bool),
        "speed": np.array([[1.0]], dtype=np.float32),
    }

    return (
        td,
        orig_depot_coords,
        orig_client_coords,
        depot_ids,
        client_indices,
        min_xy,
        range_xy,
    )


def compute_cost_from_actions(actions, locs_orig, num_depots, close_routes=True):
    """Compute raw Euclidean cost from routefinder action sequence using original coordinates."""
    actions = (
        actions.squeeze().cpu().numpy()
        if torch.is_tensor(actions)
        else np.asarray(actions).squeeze()
    )

    def loc_dist(i, j):
        dx = locs_orig[i, 0] - locs_orig[j, 0]
        dy = locs_orig[i, 1] - locs_orig[j, 1]
        return math.sqrt(dx * dx + dy * dy)

    routes = []
    current = []
    for a in actions:
        if a < num_depots:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(int(a))
    if current:
        routes.append(current)
    if not routes:
        routes = [list(a for a in actions if a >= num_depots)]

    total = 0.0
    for route in routes:
        if not route:
            continue
        best_d = min(
            range(num_depots),
            key=lambda d: loc_dist(d, route[0]) + loc_dist(route[-1], d),
        )
        total += loc_dist(best_d, route[0])
        for i in range(len(route) - 1):
            total += loc_dist(route[i], route[i + 1])
        if close_routes:
            total += loc_dist(route[-1], best_d)

    return total


if __name__ == "__main__":
    import argparse
    from routefinder.models import RouteFinderBase, RouteFinderMoE
    from routefinder.models.baselines.mtpomo import MTPOMO
    from routefinder.models.baselines.mvmoe import MVMoE
    from routefinder.envs.mtdvrp.env import MTVRPEnv as MTDVRPEnv
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: auto-select 100/rf-transformer.ckpt)",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=None,
        help="Instance names to run (default: all PR11A-PR24B)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_augment", type=int, default=4)
    parser.add_argument(
        "--num_starts",
        type=int,
        default=None,
        help="Number of start nodes (default: min(env_n_starts, 16))",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=False,
        help="Save converted .npz files for debugging",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        ckpt_path = os.path.join(RF_DIR, "checkpoints", "100", "rf-transformer.ckpt")
    else:
        ckpt_path = args.checkpoint

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    if args.instances:
        instance_names = args.instances
    else:
        instance_names = [f"PR{i}{s}" for i in range(11, 25) for s in ("A", "B")]

    # Load model
    print(f"Loading model from {ckpt_path}...")
    if "mvmoe" in ckpt_path:
        BaseLitModule = MVMoE
    elif "mtpomo" in ckpt_path:
        BaseLitModule = MTPOMO
    elif "moe" in ckpt_path:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    # Compatibility patches for checkpoint loading (torchrl API migration 0.13.x)
    import torchrl.data.tensor_specs as _ts

    _old_spec_aliases = {
        "CompositeSpec": "Composite",
        "BoundedTensorSpec": "Bounded",
        "UnboundedContinuousTensorSpec": "UnboundedContinuous",
        "UnboundedDiscreteTensorSpec": "UnboundedDiscrete",
        "DiscreteTensorSpec": "Bounded",  # Discrete specs now use Bounded
        "BinaryTensorSpec": "Binary",
    }
    for _old, _new in _old_spec_aliases.items():
        if not hasattr(_ts, _old) and hasattr(_ts, _new):
            setattr(_ts, _old, getattr(_ts, _new))

    _orig_torch_load = torch.load

    def _patch_load(f, *a, **kw):
        kw.pop("weights_only", None)
        return _orig_torch_load(f, *a, weights_only=False, **kw)

    torch.load = _patch_load
    try:
        model = BaseLitModule.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            strict=False,
        )
    finally:
        torch.load = _orig_torch_load
    env = MTDVRPEnv(load_solutions=False)
    policy = model.policy.to(device).eval()

    results = {}
    for inst_name in tqdm(instance_names):
        vrp_path = os.path.join(INSTANCE_DIR, f"{inst_name}.vrp")
        if not os.path.exists(vrp_path):
            print(f"  SKIP {inst_name}: file not found")
            continue

        # Convert
        td_dict, orig_depot, orig_client, depot_ids, client_ids, min_xy, range_xy = (
            convert_instance_to_routefinder(vrp_path)
        )

        # Build original locs array: depots + clients in the same order as td_dict
        orig_locs = np.vstack([orig_depot, orig_client]).astype(np.float32)

        # Save npz if requested
        if args.save_npz:
            npz_dir = os.path.join(OUT_DIR, "routefinder_npz")
            os.makedirs(npz_dir, exist_ok=True)
            npz_path = os.path.join(npz_dir, f"{inst_name}.npz")
            np.savez_compressed(npz_path, **td_dict)

        try:
            from rl4co.data.transforms import StateAugmentation
            from rl4co.utils.ops import gather_by_index, unbatchify
            from tensordict.tensordict import TensorDict

            td = TensorDict(
                {k: torch.tensor(v) for k, v in td_dict.items()},
                batch_size=[1],
            )
            td = env.reset(td)  # add env state

            with torch.inference_mode():
                if args.num_augment > 1:
                    td_aug = StateAugmentation(num_augment=args.num_augment)(td)
                else:
                    td_aug = td

                # Limit num_starts for CPU speed (use all depots as starts)
                n_starts = args.num_starts or min(env.get_num_starts(td_aug), 16)
                out = policy(
                    td_aug,
                    env,
                    phase="test",
                    num_starts=n_starts,
                    return_actions=True,
                )

                reward = unbatchify(
                    out["reward"],
                    (args.num_augment, n_starts),
                )

                if n_starts > 1:
                    max_reward, _ = reward.max(dim=-1)
                else:
                    max_reward = reward.squeeze(-1)

                if args.num_augment > 1:
                    max_aug_reward, max_idxs = max_reward.max(dim=1)
                else:
                    max_aug_reward = max_reward.squeeze(-1)

                # Get best actions
                actions_np = out.get("actions")
                if actions_np is not None:
                    actions_np = actions_np.cpu().numpy()

                model_cost = -max_aug_reward.item()

                # Compute actual cost from actions in original coords
                if actions_np is not None:
                    actual_cost = compute_cost_from_actions(
                        actions_np[0], orig_locs, len(orig_depot), close_routes=True
                    )
                else:
                    actual_cost = None

            results[inst_name] = {
                "model_cost": model_cost,
                "actual_cost": actual_cost,
                "status": "success",
            }
            print(
                f"\n  {inst_name}: model(unnorm)={model_cost:.2f}, actual(orig)={actual_cost if actual_cost else 'N/A'}"
            )

        except Exception as e:
            import traceback

            results[inst_name] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"\n  {inst_name}: ERROR - {e}")

    # Summary
    print(f"\n=== RouteFinder Results ===")
    costs = [r["actual_cost"] for r in results.values() if r.get("actual_cost")]
    if costs:
        print(f"  Mean actual cost: {np.mean(costs):.2f}")
        for inst_name, r in results.items():
            if r.get("actual_cost"):
                print(f"  {inst_name}: {r['actual_cost']:.2f}")

    # Save
    summary_path = os.path.join(OUT_DIR, "routefinder_results.json")
    serializable = {}
    for inst_name, r in results.items():
        serializable[inst_name] = {
            k: v
            for k, v in r.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {summary_path}")
