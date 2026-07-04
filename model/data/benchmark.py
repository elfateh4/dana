import json
import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")


INSTANCE_SOURCES = {
    "cordeau": {
        "path": "cordeau",
        "url": "https://github.com/PyVRP/Instances/tree/main/MDVRPTW",
        "files": [f"PR{i}{s}.vrp" for i in range(11, 25) for s in ("A", "B")],
    },
    "solomon": {
        "path": "solomon",
        "url": "https://github.com/PyVRP/Instances/tree/main/VRPTW/Solomon",
        "files": [
            "C101.vrp",
            "C201.vrp",
            "R101.vrp",
            "R201.vrp",
            "RC101.vrp",
            "RC201.vrp",
        ],
    },
    "gehring": {
        "path": "gehring",
        "url": "https://github.com/PyVRP/Instances/tree/main/VRPTW",
        "files": [f"{t}{n}_10_1.vrp" for t in ("C", "R", "RC") for n in (1, 2)],
    },
    "x_instances": {
        "path": "x_instances",
        "url": "https://github.com/PyVRP/Instances/tree/main/CVRP",
        "files": ["X-n101-k25.vrp", "X-n106-k14.vrp", "X-n110-k13.vrp"],
    },
}


def parse_vrplib_instance(filepath: str) -> Dict:
    import vrplib

    data = vrplib.read_instance(filepath)
    coords = data.get("node_coords", [])
    if len(coords) == 0:
        coords = np.zeros((data["dimension"], 2), dtype=float)
    demand = data.get("demand", np.zeros(data["dimension"], dtype=float))
    tw_start = data.get("time_window", np.zeros((data["dimension"], 2), dtype=float))[
        :, 0
    ]
    tw_end = data.get("time_window", np.zeros((data["dimension"], 2), dtype=float))[
        :, 1
    ]
    service_time = data.get("service_time", np.zeros(data["dimension"], dtype=float))
    depot = data.get("depot", [0])
    if isinstance(depot, int):
        depot = [depot]
    depot_indices = sorted(set(d for d in depot if d < len(coords)))
    if not depot_indices:
        depot_indices = [0]
    num_vehicles = data.get("vehicles", 25)
    capacity = data.get("capacity", 200)
    return {
        "num_vehicles": num_vehicles,
        "num_customers": data["dimension"] - len(depot_indices),
        "num_depots": len(depot_indices),
        "coords": np.array(coords, dtype=float),
        "demand": np.array(demand, dtype=float),
        "tw_start": np.array(tw_start, dtype=float),
        "tw_end": np.array(tw_end, dtype=float),
        "service_time": np.array(service_time, dtype=float),
        "depot_indices": depot_indices,
        "vehicle_capacity": int(capacity),
        "instance_name": data.get("name", os.path.basename(filepath)),
    }


def parse_cordeau_instance(filepath: str) -> Dict:
    with open(filepath) as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    num_vehicles = int(header[0])
    num_customers = int(header[1])
    num_depots = int(header[2])
    coords = []
    demand = []
    tw_start = []
    tw_end = []
    service_time = []
    depot_indices = list(range(num_depots))
    for i in range(1, 1 + num_depots + num_customers):
        parts = lines[i].strip().split()
        coords.append((float(parts[1]), float(parts[2])))
        demand.append(float(parts[3]))
        tw_start.append(float(parts[4]))
        tw_end.append(float(parts[5]))
        service_time.append(float(parts[6]))
    return {
        "num_vehicles": num_vehicles,
        "num_customers": num_customers,
        "num_depots": num_depots,
        "coords": np.array(coords),
        "demand": np.array(demand),
        "tw_start": np.array(tw_start),
        "tw_end": np.array(tw_end),
        "service_time": np.array(service_time),
        "depot_indices": depot_indices,
    }


def parse_solomon_instance(filepath: str) -> Dict:
    with open(filepath) as f:
        lines = f.readlines()
    header_found = False
    data_start = False
    coords, demand, tw_start, tw_end, service_time = [], [], [], [], []
    vehicle_capacity = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "VEHICLE" in line.upper():
            header_found = True
            continue
        if header_found and not data_start:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    vehicle_capacity = float(parts[1])
                    data_start = True
                except ValueError:
                    continue
            continue
        if "CUST" in line.upper() or "NODE" in line.upper():
            continue
        if data_start:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    coords.append((float(parts[1]), float(parts[2])))
                    demand.append(float(parts[3]))
                    tw_start.append(float(parts[4]))
                    tw_end.append(float(parts[5]))
                    service_time.append(float(parts[6]))
                except ValueError:
                    continue
    return {
        "num_vehicles": 25,
        "num_customers": len(coords) - 1,
        "num_depots": 1,
        "coords": np.array(coords),
        "demand": np.array(demand),
        "tw_start": np.array(tw_start),
        "tw_end": np.array(tw_end),
        "service_time": np.array(service_time),
        "depot_indices": [0],
        "vehicle_capacity": vehicle_capacity,
    }


def _add_distance_matrices(data: Dict) -> Dict:
    coords = data["coords"]
    n = len(coords)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
    data["distance_matrix"] = dist
    data["duration_matrix"] = dist.copy()
    return data


def load_benchmark_instance(set_name: str, filename: str) -> Dict:
    source = INSTANCE_SOURCES[set_name]
    path = os.path.join(INSTANCES_DIR, source["path"], filename)
    if filename.endswith(".vrp"):
        data = parse_vrplib_instance(path)
    elif set_name == "cordeau":
        data = parse_cordeau_instance(path)
    elif set_name == "solomon":
        data = parse_solomon_instance(path)
    else:
        data = parse_vrplib_instance(path)
    return _add_distance_matrices(data)


class DisasterBenchmark:
    def __init__(
        self,
        city_name: str = "cairo",
        num_locations: int = 100,
        disaster_prob: float = 0.05,
        seed: int = 42,
    ):
        self.city_name = city_name
        self.num_locations = num_locations
        self.disaster_prob = disaster_prob
        self.rng = np.random.default_rng(seed)

    def generate_instance(self, num_depots: int = 3) -> Dict:
        from .osm_loader import load_city_data

        try:
            data = load_city_data(self.city_name)
        except FileNotFoundError:
            data = self._synthetic_city()

        points = data["points"]
        n_total = len(points)
        indices = self.rng.choice(n_total, size=self.num_locations, replace=False)
        sorted_idx = np.sort(indices)
        coords = points[sorted_idx]
        dist_mat = data["distance"][sorted_idx][:, sorted_idx]
        dur_mat = data.get("duration", data["distance"])[sorted_idx][:, sorted_idx]

        depot_indices = list(range(num_depots))
        demand = self.rng.integers(1, 10, size=self.num_locations).astype(float)
        tw_start = self.rng.uniform(0, 400, size=self.num_locations).astype(float)
        tw_end = tw_start + self.rng.uniform(30, 120, size=self.num_locations).astype(
            float
        )
        tw_end[:num_depots] = 480.0
        tw_start[:num_depots] = 0.0

        return {
            "coords": coords.astype(np.float32),
            "distance_matrix": dist_mat.astype(np.float32),
            "duration_matrix": dur_mat.astype(np.float32),
            "demand": demand,
            "tw_start": tw_start,
            "tw_end": tw_end,
            "depot_indices": depot_indices,
            "num_depots": num_depots,
            "num_locations": self.num_locations,
            "city": self.city_name,
        }

    def _synthetic_city(self) -> Dict:
        n = 1000
        rng = self.rng
        points = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
        dx = points[:, None, 0] - points[None, :, 0]
        dy = points[:, None, 1] - points[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        noise = rng.uniform(0.8, 1.2, size=(n, n)).astype(np.float32)
        np.fill_diagonal(noise, 1.0)
        dist_asym = dist * noise
        return {"points": points, "distance": dist_asym, "duration": dist_asym}


def generate_disaster_event(
    instance: Dict, event_type: str = None, rng: Optional[np.random.Generator] = None
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()
    if event_type is None:
        event_type = rng.choice(["road_closure", "new_demand", "depot_damage"])
    event = {"type": event_type}
    if event_type == "road_closure":
        i = rng.integers(0, instance["num_locations"])
        j = rng.integers(0, instance["num_locations"])
        while j == i:
            j = rng.integers(0, instance["num_locations"])
        event["from"] = int(i)
        event["to"] = int(j)
        event["original_distance"] = float(instance["distance_matrix"][i, j])
        event["original_duration"] = float(instance["duration_matrix"][i, j])
    elif event_type == "new_demand":
        event["location_idx"] = int(
            rng.integers(instance["num_depots"], instance["num_locations"])
        )
        event["additional_demand"] = float(rng.integers(1, 5))
    elif event_type == "depot_damage":
        depot = int(rng.integers(0, instance["num_depots"]))
        event["depot_idx"] = depot
    return event
