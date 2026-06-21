import subprocess
import os
import tempfile
import re
from typing import Dict, List, Optional, Tuple


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
            import pyvrp
            from pyvrp import Model, read

            model = read(
                instance_file,
                instance_format="solomon" if problem_type == "vrptw" else "cordeau",
            )
            model.add_depot(0)
            result_obj = model.solve(time_limit=self.time_limit, seed=seed)
            result["cost"] = result_obj.cost()
            result["feasible"] = result_obj.is_feasible()
            result["time"] = result_obj.runtime
            result["status"] = "success"
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
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".par", delete=False
            ) as par_f:
                par_f.write(f"PROBLEM_FILE = {instance_file}\n")
                par_f.write(f"RUNS = {self.num_runs}\n")
                par_f.write(f"SEED = {seed}\n")
                par_f.write(f"TIME_LIMIT = {self.time_limit}\n")
                par_f.write("TOUR_FILE = /tmp/lkh_tour.txt\n")
                par_path = par_f.name
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
            manager = pywrapcp.RoutingIndexManager(
                len(data["locations"]), data["num_vehicles"], data["depot"]
            )
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
                dist_mat[i][j] = int((dx**2 + dy**2) ** 0.5 * 1000)
        data["distance_matrix"] = dist_mat
        return data

    def run_all(self, instance: str, problem_type: str = "mdvrptw") -> Dict[str, Dict]:
        results = {}
        results["pyvrp"] = self.run_pyvrp(instance, problem_type)
        results["hgs"] = self.run_hgs(instance)
        results["lkh3"] = self.run_lkh3(instance, problem_type)
        results["ortools"] = self.run_ortools(instance, problem_type)
        return results
