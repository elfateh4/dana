import argparse
import requests
from pathlib import Path

INSTANCES_DIR = Path(__file__).parent / "instances"
BASE_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main"

INSTANCE_SETS = {
    "cordeau": {
        "remote_dir": "MDVRPTW",
        "files": [f"PR{i}{s}" for i in range(11, 25) for s in ("A", "B")],
        "ext": "vrp",
    },
    "solomon": {
        "remote_dir": "VRPTW/Solomon",
        "files": [
            "C101",
            "C102",
            "C103",
            "C104",
            "C105",
            "C106",
            "C107",
            "C108",
            "C109",
            "C201",
            "C202",
            "C203",
            "C204",
            "C205",
            "C206",
            "C207",
            "C208",
            "R101",
            "R102",
            "R103",
            "R104",
            "R105",
            "R106",
            "R107",
            "R108",
            "R109",
            "R110",
            "R111",
            "R112",
            "R201",
            "R202",
            "R203",
            "R204",
            "R205",
            "R206",
            "R207",
            "R208",
            "R209",
            "R210",
            "R211",
            "RC101",
            "RC102",
            "RC103",
            "RC104",
            "RC105",
            "RC106",
            "RC107",
            "RC108",
            "RC201",
            "RC202",
            "RC203",
            "RC204",
            "RC205",
            "RC206",
            "RC207",
            "RC208",
        ],
        "ext": "vrp",
    },
    "gehring": {
        "remote_dir": "VRPTW",
        "files": [
            f"{t}{n}_10_{i}"
            for t in ("C", "R", "RC")
            for n in (1, 2)
            for i in range(1, 11)
        ],
        "ext": "vrp",
    },
    "x_instances": {
        "remote_dir": "CVRP",
        "files": [
            "X-n101-k25",
            "X-n106-k14",
            "X-n110-k13",
            "X-n115-k10",
            "X-n120-k6",
            "X-n125-k30",
            "X-n129-k18",
            "X-n134-k13",
            "X-n139-k10",
            "X-n143-k7",
            "X-n153-k22",
            "X-n157-k13",
            "X-n162-k11",
            "X-n167-k10",
            "X-n176-k26",
            "X-n181-k23",
            "X-n186-k15",
            "X-n190-k8",
            "X-n195-k51",
            "X-n200-k36",
            "X-n204-k19",
            "X-n209-k16",
            "X-n214-k11",
            "X-n219-k73",
            "X-n223-k34",
            "X-n228-k23",
            "X-n233-k16",
            "X-n237-k14",
            "X-n242-k48",
            "X-n247-k50",
        ],
        "ext": "vrp",
    },
}


def download_set(name, config):
    dest_dir = INSTANCES_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = config["remote_dir"]
    ext = config["ext"]
    success = 0
    failed = 0
    for fname in config["files"]:
        remote_file = f"{fname}.{ext}"
        url = f"{BASE_URL}/{remote_dir}/{remote_file}"
        dest_path = dest_dir / remote_file
        if dest_path.exists():
            success += 1
            continue
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                dest_path.write_text(r.text)
                success += 1
                print(f"  OK  {remote_file}")
            else:
                print(f"  MISS {remote_file} (HTTP {r.status_code})")
                failed += 1
        except Exception as e:
            print(f"  FAIL {remote_file}: {e}")
            failed += 1
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Download VRP benchmark instances")
    parser.add_argument(
        "--set", choices=list(INSTANCE_SETS.keys()) + ["all"], default="all"
    )
    args = parser.parse_args()

    sets = list(INSTANCE_SETS.keys()) if args.set == "all" else [args.set]
    total_s, total_f = 0, 0
    for name in sets:
        config = INSTANCE_SETS[name]
        print(f"\nDownloading {name} ({len(config['files'])} files)...")
        s, f = download_set(name, config)
        total_s += s
        total_f += f
        print(f"  -> {s} ok, {f} failed")
    print(f"\nTotal: {total_s} downloaded, {total_f} failed")


if __name__ == "__main__":
    main()
