import subprocess
import os
import tempfile
import re
from typing import Dict, List, Optional


class BaselineRunner:
    def __init__(self, time_limit: int = 3600, num_runs: int = 10):
        self.time_limit = time_limit
        self.num_runs = num_runs
        self.results_dir = "reports"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_pyvrp(
        self, instance_file: str, problem_type: str = "mdvrptw", seed: int = 42
    ) -> Dict:
        result = {"solver": "pyvrp", "instance": instance_file, "problem": problem_type}
        try:
            from pyvrp import read, Model
            from pyvrp.stop import MaxRuntime
            import math

            data = read(instance_file)
            model = Model.from_data(data)
            result_obj = model.solve(
                stop=MaxRuntime(self.time_limit), seed=seed, display=False
            )
            result["cost_pyvrp"] = result_obj.cost()
            result["feasible"] = result_obj.is_feasible()
            result["time"] = result_obj.runtime
            result["status"] = "success"

            # Extract routes as 0-based customer indices
            best = result_obj.best
            num_depots = data.num_depots
            routes = []
            for route in best.routes():
                visits = list(route.visits())
                if visits:
                    routes.append(visits)
            result["routes"] = routes

            # Compute raw Euclidean cost from solution routes
            depots = data.depots()
            clients = data.clients()
            loc_coords = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in clients]

            def loc_dist(i, j):
                dx = loc_coords[i][0] - loc_coords[j][0]
                dy = loc_coords[i][1] - loc_coords[j][1]
                return math.sqrt(dx * dx + dy * dy)

            raw_cost = 0.0
            for route in best.routes():
                visits = list(route.visits())
                if not visits:
                    continue
                start_depot = int(route.start_depot())
                end_depot = int(route.end_depot())
                raw_cost += loc_dist(start_depot, visits[0])
                for i in range(len(visits) - 1):
                    raw_cost += loc_dist(visits[i], visits[i + 1])
                raw_cost += loc_dist(visits[-1], end_depot)
            result["cost"] = raw_cost
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def run_hgs(self, instance_file: str, seed: int = 42) -> Dict:
        result = {"solver": "hgs", "instance": instance_file, "problem": "cvrp"}
        hgs_bin = self._find_binary("HGS-CVRP", ["HGS-CVRP", "hgs"])
        if hgs_bin is None:
            result["status"] = "not_found"
            result["error"] = (
                "HGS-CVRP binary not found. Compile from https://github.com/vidalt/HGS-CVRP"
            )
            return result
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sol", delete=False
            ) as sol_f:
                sol_path = sol_f.name
            cmd = [hgs_bin, instance_file, str(seed), str(self.time_limit), sol_path]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.time_limit + 60
            )
            cost = None
            for line in proc.stdout.split("\n"):
                m = re.search(r"Cost[:\s]+([0-9.]+)", line)
                if m:
                    cost = float(m.group(1))
                    break
            result["cost"] = cost
            result["time"] = self.time_limit
            result["status"] = "success" if cost else "no_solution"

            # Parse .sol file (CVRPLib format) for routes
            routes = []
            if os.path.isfile(sol_path) and os.path.getsize(sol_path) > 0:
                try:
                    with open(sol_path) as f:
                        lines = f.readlines()
                    # Line 0: <num_vehicles> <cost> — skip
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        nodes = [int(x) for x in line.split()]
                        # CVRPLib format: 1-indexed customer IDs, 0 = route end marker
                        customers = [n - 1 for n in nodes if n > 0]
                        if customers:
                            routes.append(customers)
                except Exception:
                    pass
            result["routes"] = routes
            os.unlink(sol_path)
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def run_lkh3(
        self, instance_file: str, problem_type: str = "vrptw", seed: int = 42
    ) -> Dict:
        result = {"solver": "lkh3", "instance": instance_file, "problem": problem_type}
        lkh_bin = self._find_binary("LKH-3", ["LKH-3", "lkh", "lkh3"])
        if lkh_bin is None:
            result["status"] = "not_found"
            result["error"] = (
                "LKH-3 binary not found. Download from http://webhotel4.ruc.dk/~keld/research/LKH-3/"
            )
            return result
        tour_file = "/tmp/lkh_tour.txt"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".par", delete=False
            ) as par_f:
                par_f.write(f"PROBLEM_FILE = {instance_file}\n")
                par_f.write(f"RUNS = {self.num_runs}\n")
                par_f.write(f"SEED = {seed}\n")
                par_f.write(f"TIME_LIMIT = {self.time_limit}\n")
                par_f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
                par_path = par_f.name
            # Clean up any previous tour file
            if os.path.exists(tour_file):
                os.unlink(tour_file)
            cmd = [lkh_bin, par_path]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.time_limit + 60
            )
            cost = None
            for line in proc.stdout.split("\n"):
                m = re.search(r"Cost\s*=\s*([0-9.]+)", line)
                if m:
                    cost = float(m.group(1))
                    break
            result["cost"] = cost
            result["status"] = "success" if cost else "no_solution"

            # Parse OUTPUT_TOUR_FILE for routes
            # LKH-3 outputs a single tour with dummy depot nodes (value 1)
            # marking route boundaries. Nodes > dimension are also depots.
            routes = []
            if os.path.isfile(tour_file) and os.path.getsize(tour_file) > 0:
                try:
                    with open(tour_file) as f:
                        lines = f.readlines()
                    # Read the tour as a sequence of 1-indexed node IDs
                    tour = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            tour.append(int(line))
                        except ValueError:
                            continue
                    # Split at depot visits (node == 1) to get routes
                    current_route = []
                    for node in tour:
                        if node == 1:
                            if current_route:
                                routes.append(current_route)
                                current_route = []
                        else:
                            # Customer node: convert 1-indexed to 0-based
                            # (node 2 = customer 0, node 3 = customer 1, etc.)
                            current_route.append(node - 2)
                    if current_route:
                        routes.append(current_route)
                except Exception:
                    pass
            result["routes"] = routes
            os.unlink(par_path)
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def run_ortools(
        self, instance_file: str, problem_type: str = "mdvrptw", seed: int = 42
    ) -> Dict:
        result = {
            "solver": "ortools",
            "instance": instance_file,
            "problem": problem_type,
        }
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2

            data = self._load_ortools_data(instance_file, problem_type)
            num_nodes = len(data["locations"])
            num_vehicles = data["num_vehicles"]
            depot = data["depot"]
            manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
            routing = pywrapcp.RoutingModel(manager)

            def distance_callback(from_idx, to_idx):
                return int(
                    data["distance_matrix"][manager.IndexToNode(from_idx)][
                        manager.IndexToNode(to_idx)
                    ]
                )

            transit_cb = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)
            search_params = pywrapcp.DefaultRoutingSearchParameters()
            search_params.time_limit.seconds = self.time_limit
            search_params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            solution = routing.SolveWithParameters(search_params)
            if solution:
                cost = solution.ObjectiveValue()
                result["cost"] = cost
                result["status"] = "success"

                # Extract routes as 0-based customer index lists
                routes = []
                for v in range(num_vehicles):
                    index = routing.Start(v)
                    route = []
                    while not routing.IsEnd(index):
                        node = manager.IndexToNode(index)
                        if node != depot:
                            route.append(node)
                        index = solution.Value(routing.NextVar(index))
                    if route:
                        routes.append(route)
                result["routes"] = routes
            else:
                result["status"] = "no_solution"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def _find_binary(self, name: str, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            try:
                proc = subprocess.run(
                    ["which", candidate], capture_output=True, text=True
                )
                if proc.returncode == 0:
                    return proc.stdout.strip()
            except FileNotFoundError:
                pass
        for candidate in candidates:
            path = os.path.expanduser(f"~/bin/{candidate}")
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
            path = f"/usr/local/bin/{candidate}"
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        return None

    def _load_ortools_data(self, instance_file: str, problem_type: str) -> Dict:
        data = {"locations": [], "distance_matrix": [], "num_vehicles": 1, "depot": 0}
        with open(instance_file) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y = float(parts[-2]), float(parts[-1])
                    data["locations"].append((x, y))
                except ValueError:
                    continue
        n = len(data["locations"])
        if n == 0:
            data["locations"] = [(0, 0), (1, 1)]
            n = 2
        dist_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dx = data["locations"][i][0] - data["locations"][j][0]
                dy = data["locations"][i][1] - data["locations"][j][1]
                dist_mat[i][j] = int((dx**2 + dy**2) ** 0.5)
        data["distance_matrix"] = dist_mat
        return data

    def run_all(self, instance: str, problem_type: str = "mdvrptw") -> Dict[str, Dict]:
        results = {}
        results["pyvrp"] = self.run_pyvrp(instance, problem_type)
        results["hgs"] = self.run_hgs(instance)
        results["lkh3"] = self.run_lkh3(instance, problem_type)
        results["ortools"] = self.run_ortools(instance, problem_type)
        return results
