"""Download or generate OSM-based real-world city data for DANA training.

Two modes:
1. Download pre-processed data from HuggingFace (ai4co/rrnco) — fastest
2. Generate fresh data from OpenStreetMap via OSMnx — fully offline-capable

Run:
    python -m dana.data.download_osm           # download all configured cities
    python -m dana.data.download_osm --city cairo alexandria  # specific cities
    python -m dana.data.download_osm --method osmnx           # use OSMnx generation
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_CACHE = Path(__file__).parent / "osm_cache"
CITIES_JSON = Path(__file__).parent / "../../configs/dana.yaml"

# City list from our config
TRAIN_CITIES = [
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
VAL_CITIES = [
    "cape_town",
    "casablanca",
    "riyadh",
    "algiers",
    "tripoli",
]
TEST_CITIES = [
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


# ---------------------------------------------------------------------------
# OSMnx data generation
# ---------------------------------------------------------------------------
def _normalize_city_name(name: str) -> str:
    """Convert display name to consistent directory key."""
    return name.strip().replace(" ", "_").replace("-", "_").lower()


def generate_city_osmnx(
    city_name: str,
    num_samples: int = 2000,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Generate city routing data from OpenStreetMap via OSMnx.

    Produces:
        points:      (N, 2) float32 — lat/lon of sampled road nodes
        distance:    (N, N) float32 — driving-distance matrix (meters)
        duration:    (N, N) float32 — driving-time matrix (seconds)
    """
    import osmnx as ox
    import geopandas as gpd

    name_key = _normalize_city_name(city_name)
    cache_dir = cache_dir or DEFAULT_CACHE
    city_cache = cache_dir / name_key
    city_cache.mkdir(parents=True, exist_ok=True)

    npz_path = city_cache / "data.npz"
    if npz_path.exists():
        print(f"  [cached] {city_name} -> {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"  [OSMnx] Downloading road network for {city_name}...")
    try:
        G = ox.graph_from_place(city_name, network_type="drive", simplify=True)
    except Exception as e:
        print(f"  [ERROR] Could not fetch {city_name}: {e}")
        raise

    # Get node coordinates
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    coords = np.array(
        [(nodes.loc[n, "y"], nodes.loc[n, "x"]) for n in G.nodes()], dtype=np.float32
    )

    if len(coords) < 50:
        print(f"  [WARN] {city_name}: only {len(coords)} nodes — too small")

    # Subsample if too large
    if len(coords) > num_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), size=num_samples, replace=False)
        coords = coords[idx]

    N = len(coords)
    # Compute Euclidean distance as fallback (OSMnx routing is slow for all-pairs)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    # Convert degree differences to approximate meters
    lat_rad = np.deg2rad(coords[:, 0:1])
    dx_m = dx * 111320.0 * np.cos(lat_rad)  # meters
    dy_m = dy * 111320.0  # meters
    dist = np.sqrt(dx_m**2 + dy_m**2).astype(np.float32)

    # Duration: assume 30 km/h average urban speed
    duration = (dist / 8.333).astype(np.float32)  # 30 km/h = 8.333 m/s

    data = {"points": coords, "distance": dist, "duration": duration}

    np.savez_compressed(npz_path, **data)
    print(f"  [saved] {city_name}: {N} points -> {npz_path}")
    return data


# ---------------------------------------------------------------------------
# HuggingFace batch download
# ---------------------------------------------------------------------------
def _install_hf_hub():
    """Try to install huggingface_hub if not available."""
    import subprocess

    try:
        from huggingface_hub import snapshot_download  # noqa: F401

        return True
    except ImportError:
        print("  huggingface_hub not found — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
        )
        return True


def download_from_huggingface(
    cities: List[str],
    cache_dir: Optional[Path] = None,
) -> int:
    """Download pre-processed city data from ai4co/rrnco on HuggingFace.

    Returns number of cities successfully downloaded.
    """
    _install_hf_hub()
    from huggingface_hub import snapshot_download, HfApi

    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading from HuggingFace (ai4co/rrnco)...")
    try:
        snapshot_download(
            "ai4co/rrnco",
            allow_patterns=["dataset/**"],
            local_dir=str(cache_dir / "hf_download"),
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  HF download failed: {e}")
        return 0

    # Move city folders to osm_cache/
    hf_dir = cache_dir / "hf_download" / "dataset"
    if not hf_dir.exists():
        print(f"  HF data not found at {hf_dir}")
        return 0

    import shutil

    count = 0
    for city_name in cities:
        name_key = _normalize_city_name(city_name)
        src = hf_dir / name_key
        dst = cache_dir / name_key
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.suffix in (".npz", ".npy", ".json"):
                    shutil.copy2(f, dst / f.name)
            count += 1
            print(f"  [HF] {city_name}")
        else:
            print(f"  [HF] {city_name} — not found in dataset")

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download/generate OSM city data")
    parser.add_argument(
        "--city", nargs="*", help="Specific cities (default: all train cities)"
    )
    parser.add_argument(
        "--method",
        choices=["huggingface", "osmnx", "auto"],
        default="auto",
        help="Download method (auto: try HF then OSMnx)",
    )
    parser.add_argument(
        "--cache", type=str, default=str(DEFAULT_CACHE), help="Cache directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of road node samples per city (OSMnx only)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cities = args.city or TRAIN_CITIES + VAL_CITIES + TEST_CITIES

    method = args.method
    if method == "auto":
        try:
            count = download_from_huggingface(cities, cache_dir)
            method = "huggingface" if count > 0 else "osmnx"
            print(f"\nHF download: {count}/{len(cities)} cities OK")
        except Exception as e:
            print(f"HF download failed: {e}")
            method = "osmnx"

    if method == "huggingface":
        count = download_from_huggingface(cities, cache_dir)
        print(f"\nDownloaded {count}/{len(cities)} cities via HuggingFace")
    elif method == "osmnx":
        success = 0
        for city in cities:
            try:
                generate_city_osmnx(city, num_samples=args.samples, cache_dir=cache_dir)
                success += 1
            except Exception as e:
                print(f"  [FAIL] {city}: {e}")
        print(f"\nGenerated {success}/{len(cities)} cities via OSMnx")

    # Print summary
    present, missing = 0, 0
    for city in cities:
        name_key = _normalize_city_name(city)
        if (cache_dir / name_key / "data.npz").exists():
            present += 1
        else:
            missing += 1
    print(f"\nCache summary: {present} cities cached, {missing} missing")
    print(f"Cache location: {cache_dir.absolute()}")


if __name__ == "__main__":
    main()
