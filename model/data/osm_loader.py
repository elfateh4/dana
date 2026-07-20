"""OpenStreetMap city data loader for DANA training.

Provides:
    - Real-world road-network coordinates and distance matrices
    - On-demand generation via OSMnx (with caching)
    - City rotation across training epochs
    - Automatic fallback to synthetic data when OSM unavailable
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).parent
DEFAULT_DATA_ROOT = PACKAGE_DIR / "osm_cache"
RRNCO_DATA_PATH = (
    PACKAGE_DIR / "../../external/real-routing-nco/data/dataset"
).resolve()

# Attempt to load city list from bundled JSON; fallback to hardcoded list
CITIES_JSON_PATH = RRNCO_DATA_PATH / "splited_cities_list.json"

HARDCODED_TRAIN_CITIES = [
    "cairo",
    "alexandria",
    "giza",
    "lagos",
    "nairobi",
    "mumbai",
    "jakarta",
    "manila",
    "mexico_city",
    "sao_paulo",
    "istanbul",
    "tehran",
    "baghdad",
    "dhaka",
    "karachi",
    "addis_ababa",
    "kinshasa",
    "luanda",
    "accra",
    "dakar",
]
HARDCODED_TEST_CITIES = [
    "cape_town",
    "casablanca",
    "riyadh",
    "algiers",
    "tripoli",
    "rome",
    "madrid",
    "tokyo",
    "new_york",
    "london",
    "paris",
    "beijing",
    "berlin",
    "moscow",
    "sydney",
    "toronto",
    "seoul",
    "chicago",
    "los_angeles",
    "singapore",
    "bangkok",
    "buenos_aires",
    "shanghai",
    "delhi",
    "amsterdam",
]


def _normalize(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_").lower()


def get_city_lists() -> Tuple[List[str], List[str]]:
    """Return (train_cities, test_cities) from available data."""
    if CITIES_JSON_PATH.exists():
        with open(CITIES_JSON_PATH) as f:
            cities = json.load(f)
        return cities.get("train", HARDCODED_TRAIN_CITIES), cities.get(
            "test", HARDCODED_TEST_CITIES
        )
    return HARDCODED_TRAIN_CITIES, HARDCODED_TEST_CITIES


# ---------------------------------------------------------------------------
# City data loading
# ---------------------------------------------------------------------------
def _find_city_npz(city_name: str, data_root: str) -> Optional[str]:
    """Search multiple locations for a city's .npz data file."""
    name = _normalize(city_name)

    # Search paths in priority order (fastest first)
    search_paths = [
        os.path.join(data_root, name, f"{name}_data.npz"),
        os.path.join(data_root, name, "data.npz"),
        os.path.join(data_root, name, f"{name}.npz"),
        # Legacy: external real-routing-nco data directory
        os.path.join(RRNCO_DATA_PATH, name, f"{name}_data.npz"),
        os.path.join(RRNCO_DATA_PATH, name, "data.npz"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def load_city_data(
    city_name: str,
    data_root: str = str(DEFAULT_DATA_ROOT),
    generate_if_missing: bool = False,
) -> Dict[str, np.ndarray]:
    """Load city data from cache, optionally generating via OSMnx.

    Returns dict with keys: points (N,2), distance (N,N), duration (N,N).
    """
    npz_path = _find_city_npz(city_name, data_root)

    if npz_path is None and generate_if_missing:
        # Attempt on-the-fly generation via OSMnx
        try:
            from .download_osm import generate_city_osmnx

            print(f"  [OSM] Generating {city_name} on demand...")
            return generate_city_osmnx(city_name, cache_dir=Path(data_root))
        except Exception as e:
            raise FileNotFoundError(
                f"City data for '{city_name}' not found in {data_root} "
                f"and OSMnx generation failed: {e}. "
                f"Run: python -m dana.data.download_osm"
            )

    if npz_path is None:
        raise FileNotFoundError(
            f"City data for '{city_name}' not found in {data_root}. "
            f"Run: python -m dana.data.download_osm"
        )

    loaded = np.load(npz_path, allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}
    return data


# ---------------------------------------------------------------------------
# Subsampling & tensor conversion
# ---------------------------------------------------------------------------
def subsample_city_data(
    data: Dict[str, np.ndarray],
    num_locations: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Subsample a city's road network to num_locations nodes."""
    points = data["points"]
    distance = data["distance"]
    duration = data.get("duration", data["distance"])

    n_total = len(points)
    if n_total <= num_locations:
        return {
            "indices": np.arange(n_total),
            "points": points,
            "distance": distance,
            "duration": duration,
        }

    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(n_total, size=num_locations, replace=False)
    sorted_idx = np.sort(indices)

    return {
        "indices": sorted_idx,
        "points": points[sorted_idx],
        "distance": distance[sorted_idx][:, sorted_idx],
        "duration": duration[sorted_idx][:, sorted_idx],
    }


def city_to_tensor_dict(
    city_name: str,
    num_locations: int,
    num_depots: int = 1,
    data_root: str = str(DEFAULT_DATA_ROOT),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Load OSM city data and convert to training-ready tensors.

    Returns dict with:
        coords, distance_matrix, duration_matrix, depot_mask,
        demand, tw_start, tw_end, num_depots, num_locations, city_name
    """
    data = load_city_data(city_name, data_root, generate_if_missing=True)
    sampled = subsample_city_data(data, num_locations, rng)

    points = torch.tensor(sampled["points"], dtype=torch.float)
    dist_mat = torch.tensor(sampled["distance"], dtype=torch.float)
    dur_mat = torch.tensor(sampled["duration"], dtype=torch.float)

    n = num_locations
    depot_mask = torch.zeros(n, dtype=torch.bool)
    depot_mask[:num_depots] = True

    # Random demand and time windows (since real demand data is unavailable)
    demand = torch.randint(1, 10, (n,), dtype=torch.float)
    tw_start = torch.zeros(n, dtype=torch.float)
    tw_end = torch.full((n,), 480.0, dtype=torch.float)
    if num_depots > 0:
        tw_end[:num_depots] = 480.0
        tw_start[:num_depots] = 0.0

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


# ---------------------------------------------------------------------------
# CityRotation: deterministic city schedule across epochs
# ---------------------------------------------------------------------------
class CityRotation:
    """Rotates through available cities so each epoch sees a different city.

    Ensures uniform coverage of all cities over the course of training.
    """

    def __init__(
        self,
        cities: Optional[List[str]] = None,
        seed: int = 42,
        data_root: str = str(DEFAULT_DATA_ROOT),
    ):
        self.data_root = data_root
        self.rng = np.random.default_rng(seed)

        if cities is not None:
            self.cities = cities
        else:
            all_cities, _ = get_city_lists()
            # Filter to only available cities
            available = []
            for c in all_cities:
                try:
                    load_city_data(c, data_root, generate_if_missing=False)
                    available.append(c)
                except FileNotFoundError:
                    pass
            self.cities = available if available else HARDCODED_TRAIN_CITIES

        self.epoch = 0
        self._perm = self.rng.permutation(len(self.cities))

    def get_city(self, epoch: Optional[int] = None) -> str:
        """Get city for given epoch (deterministic rotation)."""
        e = epoch if epoch is not None else self.epoch
        idx = self._perm[e % len(self._perm)]
        return self.cities[idx]

    def next_epoch(self):
        """Advance to next epoch (new permutation)."""
        self.epoch += 1
        if self.epoch % len(self.cities) == 0:
            self._perm = self.rng.permutation(len(self.cities))

    def __len__(self) -> int:
        return len(self.cities)

    def available_count(self) -> int:
        """Number of cities with data available on disk."""
        return len(self.cities)
