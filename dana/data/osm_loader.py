import json
import os
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict

CITIES_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../external/real-routing-nco/data/dataset/splited_cities_list.json",
)
RRNCO_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../external/real-routing-nco/data/dataset"
)
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(__file__), "osm_cache")


def get_city_lists() -> Tuple[List[str], List[str]]:
    with open(CITIES_JSON_PATH) as f:
        cities = json.load(f)
    return cities["train"], cities["test"]


def load_city_data(
    city_name: str, data_root: str = DEFAULT_DATA_ROOT
) -> Dict[str, np.ndarray]:
    name = city_name.replace(" ", "_").replace("-", "_")
    search_paths = [
        os.path.join(RRNCO_DATA_PATH, name, f"{name}_data.npz"),
        os.path.join(RRNCO_DATA_PATH, name, "data.npz"),
        os.path.join(data_root, name, f"{name}_data.npz"),
        os.path.join(data_root, name, "data.npz"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.keys()}
    raise FileNotFoundError(
        f"City data for '{city_name}' not found. "
        f"Download from HuggingFace: https://huggingface.co/datasets/ai4co/real-routing-nco "
        f"and place in {data_root}/{name}/"
    )


def subsample_city_data(
    data: Dict[str, np.ndarray],
    num_locations: int,
    loc_dist: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    points = data["points"]
    distance = data["distance"]
    duration = data.get("duration", None)
    n_total = len(points)

    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(n_total, size=num_locations, replace=False)
    sorted_idx = np.sort(indices)

    sampled = {
        "indices": sorted_idx,
        "points": points[sorted_idx],
        "distance": distance[sorted_idx][:, sorted_idx],
    }
    if duration is not None:
        sampled["duration"] = duration[sorted_idx][:, sorted_idx]
    return sampled


def city_to_tensor_dict(
    city_name: str,
    num_locations: int,
    num_depots: int = 1,
    loc_dist: str = "uniform",
    data_root: str = DEFAULT_DATA_ROOT,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    data = load_city_data(city_name, data_root)
    sampled = subsample_city_data(data, num_locations, loc_dist, rng)

    points = torch.tensor(sampled["points"], dtype=torch.float)
    dist_mat = torch.tensor(sampled["distance"], dtype=torch.float)
    dur_mat = torch.tensor(
        sampled.get("duration", sampled["distance"]), dtype=torch.float
    )
    n = num_locations

    depot_mask = torch.zeros(n, dtype=torch.bool)
    depot_mask[:num_depots] = True

    demand = torch.randint(1, 10, (n,), dtype=torch.float)
    tw_start = torch.zeros(n, dtype=torch.float)
    tw_end = torch.full((n,), 480.0, dtype=torch.float)
    tw_end[:num_depots] = 480.0
    tw_start[tw_start > tw_end] = 0

    return {
        "coords": points,
        "distance_matrix": dist_mat,
        "duration_matrix": dur_mat,
        "depot_mask": depot_mask,
        "demand": demand,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "num_depots": torch.tensor(num_depots),
        "num_locations": torch.tensor(n),
        "city_name": city_name,
    }


def sample_city_batch(
    cities: List[str],
    batch_size: int,
    num_locations: int,
    num_depots: int = 1,
    data_root: str = DEFAULT_DATA_ROOT,
) -> List[Dict[str, torch.Tensor]]:
    batch = []
    for _ in range(batch_size):
        city = np.random.choice(cities)
        tensordict = city_to_tensor_dict(
            city, num_locations, num_depots, data_root=data_root
        )
        batch.append(tensordict)
    return batch
