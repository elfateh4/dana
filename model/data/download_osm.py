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


EARTH_RADIUS_M = 6_371_000.0
REGION_HALF_SIZE_M = 1500.0  # 3x3 km standardized area, per RRNCO Sec. 4.2
EXCLUDED_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link"}
WATER_BUFFER_DEG = 0.0003  # ~30 m buffer around water features


def _haversine_bbox(lat: float, lon: float, half_size_m: float):
    """(north, south, east, west) bounds of a square centered on (lat, lon),
    sized via the Haversine/spherical relation (Eq. 3) so the physical extent
    is identical at every latitude."""
    dlat = np.rad2deg(half_size_m / EARTH_RADIUS_M)
    dlon = np.rad2deg(half_size_m / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat))))
    return lat + dlat, lat - dlat, lon + dlon, lon - dlon


def _filter_edges(edges):
    """Drop bridges, tunnels, and highways (paper: focus on accessible
    street-level locations)."""

    def is_excluded(row) -> bool:
        hw = row.get("highway")
        hws = hw if isinstance(hw, list) else [hw]
        if any(h in EXCLUDED_HIGHWAYS for h in hws):
            return True
        for tag in ("bridge", "tunnel"):
            val = row.get(tag)
            vals = val if isinstance(val, list) else [val]
            if any(v not in (None, "no", False) and v == v for v in vals):
                return True
        return False

    keep = ~edges.apply(is_excluded, axis=1)
    return edges[keep]


def _water_union(north, south, east, west):
    """Buffered union of water features inside the bbox (or None)."""
    import osmnx as ox

    try:
        tags = {"natural": "water", "waterway": True}
        try:  # osmnx >= 2.0
            water = ox.features_from_bbox(
                bbox=(west, south, east, north), tags=tags
            )
        except TypeError:  # osmnx 1.x positional order
            water = ox.features_from_bbox(north, south, east, west, tags=tags)
        if len(water) == 0:
            return None
        return water.geometry.buffer(WATER_BUFFER_DEG).union_all()
    except Exception:
        return None


def _sample_points_on_edges(edges, num_samples: int, water, rng) -> np.ndarray:
    """Length-weighted random sampling of points along road segments
    (paper: segments weighted by length for uniform spatial distribution),
    rejecting points inside water buffer zones. Returns (N, 2) lat/lon."""
    from shapely.geometry import Point

    lengths = edges["length"].to_numpy(dtype=np.float64)
    weights = lengths / lengths.sum()
    geoms = edges.geometry.values
    points = []
    max_tries = num_samples * 10
    tries = 0
    while len(points) < num_samples and tries < max_tries:
        tries += 1
        geom = geoms[rng.choice(len(geoms), p=weights)]
        pt = geom.interpolate(rng.uniform(0, 1), normalized=True)
        if water is not None and water.contains(Point(pt.x, pt.y)):
            continue
        points.append((pt.y, pt.x))  # (lat, lon)
    return np.array(points, dtype=np.float32)


def _osrm_table(
    points: np.ndarray, base_url: str, chunk: int = 250
) -> Optional[tuple]:
    """Query a (local) OSRM table service for real asymmetric distance and
    duration matrices (paper Sec. 4.2). Returns (distance, duration) in
    meters/seconds, or None on failure."""
    import requests

    N = len(points)
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    dist = np.zeros((N, N), dtype=np.float32)
    dur = np.zeros((N, N), dtype=np.float32)
    try:
        for i0 in range(0, N, chunk):
            src = list(range(i0, min(i0 + chunk, N)))
            for j0 in range(0, N, chunk):
                dst = list(range(j0, min(j0 + chunk, N)))
                url = (
                    f"{base_url.rstrip('/')}/table/v1/driving/{coord_str}"
                    f"?sources={';'.join(map(str, src))}"
                    f"&destinations={';'.join(map(str, dst))}"
                    f"&annotations=distance,duration"
                )
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                body = resp.json()
                if body.get("code") != "Ok":
                    raise RuntimeError(f"OSRM error: {body.get('code')}")
                d = np.array(body["distances"], dtype=np.float32)
                t = np.array(body["durations"], dtype=np.float32)
                dist[i0 : i0 + len(src), j0 : j0 + len(dst)] = np.nan_to_num(d)
                dur[i0 : i0 + len(src), j0 : j0 + len(dst)] = np.nan_to_num(t)
        return dist, dur
    except Exception as e:
        print(f"  [OSRM] table query failed ({e}) — falling back to haversine")
        return None


def generate_city_osmnx(
    city_name: str,
    num_samples: int = 2000,
    cache_dir: Optional[Path] = None,
    osrm_url: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Generate city routing data from OpenStreetMap via OSMnx.

    Follows RRNCO Sec. 4.2: a Haversine-standardized 3x3 km region around the
    city center; bridges/tunnels/highways excluded; water features buffered
    out; points sampled along road segments weighted by length. If an OSRM
    server is available (``osrm_url`` or the OSRM_URL env var), real
    asymmetric distance/duration matrices are fetched from its table service;
    otherwise a haversine approximation is used as fallback.

    Produces:
        points:      (N, 2) float32 — lat/lon of sampled road points
        distance:    (N, N) float32 — driving-distance matrix (meters)
        duration:    (N, N) float32 — driving-time matrix (seconds)
    """
    import osmnx as ox

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
        center_lat, center_lon = ox.geocode(city_name)
        north, south, east, west = _haversine_bbox(
            center_lat, center_lon, REGION_HALF_SIZE_M
        )
        try:  # osmnx >= 2.0
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north), network_type="drive", simplify=True
            )
        except TypeError:  # osmnx 1.x positional order
            G = ox.graph_from_bbox(
                north, south, east, west, network_type="drive", simplify=True
            )
    except Exception as e:
        print(f"  [ERROR] Could not fetch {city_name}: {e}")
        raise

    # Street-level road segments only (no bridges/tunnels/highways)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    filtered = _filter_edges(edges)
    if len(filtered) < 10:
        print(f"  [WARN] {city_name}: filtering left {len(filtered)} edges — keeping all")
        filtered = edges

    water = _water_union(north, south, east, west)
    rng = np.random.default_rng(42)
    coords = _sample_points_on_edges(filtered, num_samples, water, rng)

    if len(coords) < 50:
        print(f"  [WARN] {city_name}: only {len(coords)} points — too small")

    N = len(coords)
    # Real routing matrices via OSRM table service, if available
    osrm_url = osrm_url or os.environ.get("OSRM_URL")
    matrices = _osrm_table(coords, osrm_url) if osrm_url else None
    if matrices is not None:
        dist, duration = matrices
    else:
        # Haversine approximation fallback (symmetric)
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        lat_rad = np.deg2rad(coords[:, 0:1])
        dx_m = dx * 111320.0  # lat degrees -> meters
        dy_m = dy * 111320.0 * np.cos(lat_rad)  # lon degrees -> meters
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
    parser.add_argument(
        "--osrm",
        type=str,
        default=None,
        help="OSRM server URL for real distance/duration matrices "
        "(e.g. http://localhost:5000; also read from OSRM_URL env var)",
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
                generate_city_osmnx(
                    city,
                    num_samples=args.samples,
                    cache_dir=cache_dir,
                    osrm_url=args.osrm,
                )
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
