# DANA: Disaster-Aware Neural Algorithm for Humanitarian Logistics Routing

## MSc Thesis Proposal (Updated)

**Ahmed AbdelBadie Ahmed** — Information Systems Department, Benha University

**Supervisors:** Prof. Dr. Karam Gouda, Prof. Dr. Tarek Elsheshtwy, Dr. Shaimaa Talaat

---

### Problem Statement

Vehicle Routing Problems (VRPs) are the core of logistics distribution, requiring delivery of goods to people in minimal time with maximum quality of service. In spatial disaster scenarios, the problem becomes critically dynamic — roads close, new demands appear, depots get damaged. Existing solutions fail because:

1. **Pre-NCO literature** uses heuristic/metaheuristic methods (GA, ACO, PSO) that cannot adapt to real-time disaster events
2. **Modern NCO methods** (POMO, AM, RouteFinder) are trained on synthetic Euclidean data and fail on real-world road networks
3. **No existing work** combines GIS road networks with neural dynamic optimization for humanitarian logistics

### DANA: Proposed Architecture

DANA is a pure neural model (no OR solvers, no ACO) trained with Reinforcement Learning:

```
┌─────────────────────────────────────────────────────────────┐
│                    DANA Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  OSM Road Network ──► RRNCO Encoder ──► Node Embeddings     │
│                        (ANE+NAB+AAFM)                        │
│                                                              │
│  Disaster Events ──► CCL Context ──► Updated Embeddings     │
│   (road closure,     (RGCR+TSNR)                             │
│    new demand,                                                │
│    depot damage)                                              │
│                                                              │
│  Vehicle State ──► RouteFinder Decoder ──► Route Actions    │
│                                                              │
│  Environment ──► Ch.14 Humanitarian Reward                   │
│   (MDVRPTW +        (response_time + satisfaction + equity   │
│    disasters)         - violation_penalty)                   │
│                                                              │
│  Training: POMO + REINFORCE (pure PyTorch, no RL4CO)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure PyTorch, no RL4CO | Full control, no framework lock-in |
| OSM real-road data (RRNCO) | Real-world asymmetric distances, not Euclidean |
| Humanitarian reward (Ch.14) | Response time + equity, not commercial tour cost |
| Dynamic disaster events | Road closures, new demand, depot damage |
| Generic file naming | `encoder.py`, `context.py`, `policy.py` — not tied to source papers |

### Contributions

1. **DANA**: First pure neural model for dynamic multi-depot VRPTW with disaster events
2. **Humanitarian reward formulation**: Operationalizes Ch.14 of the VRP book for RL training
3. **GIS-NCO bridge**: First integration of RRNCO's OSM road network encoder with dynamic disaster context
4. **Disaster benchmark**: Cairo/Nile Delta instances with synthetic disaster events
5. **Rigorous evaluation**: Following Guidelines paper — HGS, LKH-3, PyVRP, OR-Tools baselines; Wilcoxon + Bonferroni; PassMark normalization; performance + convergence charts

### Evaluation Plan

| Benchmark | Problem | Baselines |
|-----------|---------|-----------|
| Cordeau MDVRPTW | MDVRPTW | PyVRP, OR-Tools |
| Solomon VRPTW | VRPTW | PyVRP, LKH-3, OR-Tools |
| Gehring & Homberger | VRPTW (large) | PyVRP, LKH-3, OR-Tools |
| X instances | CVRP | PyVRP, HGS, LKH-3, OR-Tools |
| RRNCO 100-city OSM | Real-world asym. | PyVRP, OR-Tools |
| Cairo/Nile Delta + disasters | Disaster MDVRPTW | PyVRP, OR-Tools |

**Statistical methodology** (per Guidelines 2021):
- Wilcoxon signed-rank test (one-tailed)
- Bonferroni correction for multiple comparisons
- PassMark single-thread time normalization
- Performance charts + convergence profiles
- 10+ runs per algorithm

### Research Plan (Updated)

1. ✅ Literature review (135 papers collected, key papers read)
2. ✅ DANA architecture design
3. 🔄 Implementation (scaffold complete, modules in development)
4. ⬜ Kaggle GPU training (T4 x2)
5. ⬜ Full evaluation on all 6 benchmarks
6. ⬜ Thesis writing

### References

See `Papers/__papers_index.md` for full 135-paper index. Key references:
- RRNCO — ICLR 2026 (GIS encoder + OSM 100-city dataset)
- RouteFinder — ICML 2024 (MDVRPTW foundation model)
- CCL — ICLR 2026 (dynamic context understanding)
- Guidelines — arxiv 2021 (evaluation methodology)
- FrontierCO — ICLR 2026 (ML vs OR reality check)
- VRP Book Ch.14 — humanitarian logistics objectives
- NCOSurvey — arxiv 2024 (4 inadequacies taxonomy)
- Disaster Survey — arxiv 2025 (relief distribution §6)
