# DANA — Architecture

DANA (Disaster-Aware Neural Algorithm) is a research codebase for an MSc thesis. It implements an
attention-based policy network, trained with REINFORCE/POMO, that solves the Multi-Depot Vehicle
Routing Problem with Time Windows (MDVRPTW) under simulated disaster conditions (road closures,
demand surges, depot damage) for humanitarian logistics.

It is a **research training + benchmarking pipeline**, not a general-purpose library: a policy
model, a config-driven trainer, and a statistical evaluation harness that compares the trained
policy against classical OR solvers and other neural baselines.

> Note: the README refers to the package as `dana/`; on disk it is actually `model/`. Imports
> throughout `model/` are flat (`from models.encoder import ...`), so the code only resolves
> correctly when `model/` itself is placed on `sys.path` — which is what `kaggle/*/main.py` and
> `dana_benchmark.ipynb` do by cloning the repo and inserting `model/` into `sys.path` at runtime.
> `python -m dana.train` as written in the README does not work as-is.

## Directory layout

| Path | Role |
|---|---|
| `model/` | Core Python package: environment, neural model, training loop, data loading, evaluation. |
| `configs/` | Plain YAML configs (`dana.yaml` full-scale, `test.yaml` smoke test), loaded via `yaml.safe_load` — no Hydra/OmegaConf. |
| `external/` | Four empty placeholder dirs (`CCL-MTLVRP`, `real-routing-nco`, `rl4co`, `routefinder`) — no `.gitmodules`; reserved vendor slots for the papers/codebases the architecture is derived from. Only `osm_loader.py` references one path here, with a fallback if absent. |
| `kaggle/` | Kaggle-notebook-as-script deployment: `train/main.py`, `eval/main.py`, kernel metadata, `setup.sh`. The actual runnable entry points. |
| `dana_benchmark.ipynb` | 48-cell notebook — the most complete entry point; runs DANA plus OR solvers and six external neural baselines, with statistical significance testing. |
| `docs/` | `MSc-proposal.md` — design rationale and thesis context. |
| `papers/`, `latex/` | Literature library and thesis LaTeX source (not application code). |
| `.goopspec/` | AI coding-assistant workflow state — not application code. |

## Core modules (`model/`)

### `envs/mdvrptw_env.py`
- `DisasterMDVRPTWEnv` — stateful, non-batched Gym-like env. `reset()`, `step()`, disaster
  injection (`_maybe_trigger_disaster`: road closure / demand surge / depot damage), and the
  humanitarian reward (response time + satisfaction + equity − TW-violation penalty) via
  `_compute_reward`/`get_total_reward`/`get_metrics`.
- Instantiated by `POMOTrainer` but **not actually stepped** during training — `train.py`
  implements its own batched autoregressive rollout instead.

### `models/encoder.py` — `GraphEncoder` (RRNCO, arXiv:2503.16159)
Faithful RRNCO encoder: `InverseDistanceEmbedding` (k=25 selective sampling of distances, Eq. 4-5)
+ coordinate projection fused via `ContextualGating` (Eq. 7-8) → dual row/column embeddings
(MatNet-style, Eq. 9-10, handles asymmetric matrices) → `NeuralAdaptiveBias` (distance + duration
+ angle matrices, gated fusion → scalar edge bias A) → N × `AAFMBlock` (attention-free operator,
Eq. 11, with instance normalization). Returns `(h_row, h_col)`, each `[B, N, D]`. Duration
matrices are a first-class input (DANA extends the paper's 2-channel NAB gate to 3 channels).

### `models/context.py` — `DynamicContext` (CCL-derived)
`RGCR` (gated multi-head attention over node embeddings, optional mask) → `TSNR` (GRUCell-based
running context updated from pooled node embeddings, broadcast back into embeddings). Returns
`(updated_node_embeddings, context)`.

### `models/policy.py` — `DisasterPolicy` (RRNCO Sec. 5.2 decoder: ReLD + MatNet)
`RouteDecoder`: context vector `[h_row(last node); GRU context; vehicle state]` → single-query
MHA over column embeddings (Eq. 12) → ReLD identity residual `IDT` (Eq. 13) → MLP residual
(Eq. 14) → compatibility with learnable logit clipping/temperature and a `−log(dist)`
nearest-neighbor heuristic term (Eq. 15). `DisasterPolicy` wires `encoder → context → decoder`,
tracks the last visited node, and builds per-node attribute features (demand, time windows,
depot flag); `forward()` returns logits, a cross-entropy loss, or sampled actions.

### `data/`
- `osm_loader.py` — `CityRotation` (deterministic per-epoch city permutation, falls back to
  hardcoded city lists), cached `.npz` loading, subsampling, tensor conversion.
- `download_osm.py` — CLI: pulls OSM city data from HuggingFace (`ai4co/rrnco`) or generates it
  live via OSMnx.
- `download_instances.py` — CLI: fetches standard academic VRP benchmark files (Cordeau, Solomon,
  Gehring & Homberger, X-instances) from `PyVRP/Instances`.
- `benchmark.py` — format-specific parsers (`parse_vrplib_instance`, `parse_cordeau_instance`,
  `parse_solomon_instance`) → common instance dict; `DisasterBenchmark` generates
  disaster-augmented instances from real or synthetic cities.

### `eval/`
- `baselines.py` — `BaselineRunner`: wraps PyVRP, HGS-CVRP, LKH-3, OR-Tools as solver backends.
- `metrics.py` — PassMark-normalized runtime comparison, `compute_gap`, `wilcoxon_signed_rank`,
  `bonferroni_correction`, `evaluate_solver_set`, `compute_summary`.
- `plots.py` — matplotlib reports: performance/gap-vs-time charts, convergence profile, gap
  distribution, `generate_report`.

### `train.py`
`load_config` (YAML) → `build_policy(cfg)` (assembles encoder + context + decoder + policy) →
`POMOTrainer` (AdamW + LR scheduler [cosine w/ warmup or the paper's MultiStepLR], `CityRotation`
with asymmetric-synthetic fallback, encode-once-per-dihedral-augmentation then expand to
POMO multi-start streams, last-node tracking through the rollout, REINFORCE with shared
baseline, gradient clipping, checkpointing every 10 epochs).

## Config & data flow

- **Config**: `configs/dana.yaml` (production) / `configs/test.yaml` (smoke test) → `load_config()`
  → `build_policy(cfg["model"])` + `POMOTrainer(cfg["training"|"pomo"|"environment"|"data"|"paths"])`.
  Kaggle scripts load `dana.yaml` then mutate it in place for their hardware (smaller model, fewer
  instances) — override, not composition.
- **Training data**: `CityRotation`/`osm_loader` (real OSM city, cached) or
  `synthetic_city_to_tensor_dict` (fallback) → tensor dict (coords, distance/duration matrices,
  depot mask, demand, time windows) → batch × POMO-starts → `GraphEncoder` → `DynamicContext` →
  autoregressive decode → REINFORCE loss.
- **Eval data**: `.vrp` files in `model/data/instances/**` → `benchmark.py` parsers → instance
  dicts → `BaselineRunner` (OR solvers) + trained policy (greedy decode) → `eval.metrics` (gap,
  Wilcoxon, Bonferroni) → `eval.plots.generate_report`.

## Entry points

1. `pip install -e .` — packages `model*` per `pyproject.toml`.
2. `kaggle/train/main.py` — self-contained kernel: reinstalls torch/CUDA, clones the repo,
   overrides config for T4×2, runs `POMOTrainer` for 50 epochs, pushes a `dana-checkpoints`
   Kaggle dataset.
3. `kaggle/eval/main.py` — downloads benchmark instances + latest checkpoint, compiles HGS/LKH-3,
   runs all baselines + DANA, computes stats, generates plots, pushes a `dana-results` dataset.
4. `dana_benchmark.ipynb` — most complete entry point: DANA vs. PyVRP/HGS/LKH-3/OR-Tools plus six
   externally-installed neural baselines (POMO, AM, BQ-NCO, GOAL, DeepACO, PARCO, RRNCO), with
   Wilcoxon + Bonferroni significance testing and full report export.
5. `python train.py` directly — no CLI args; expects `configs/dana.yaml` reachable from CWD.

## Known gaps / discrepancies

- README package name (`dana`) vs. actual on-disk package (`model`).
- `python -m dana.train` in README does not work given flat imports + actual package name.
- `external/*` submodule directories are empty, no `.gitmodules`.
- `DisasterMDVRPTWEnv` is defined and reward-complete but not exercised step-by-step in training.
- No automated test suite.
- Without an OSRM server (`OSRM_URL` / `--osrm`), OSM city matrices fall back to a symmetric
  haversine approximation; only the synthetic generator produces asymmetric matrices then.
