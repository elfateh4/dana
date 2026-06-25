# DANA

**D**isaster-**A**ware **N**eural **A**lgorithm for humanitarian logistics routing.

A neural combinatorial optimization framework for solving the Multi-Depot Vehicle Routing Problem with Time Windows (MDVRPTW) in disaster response scenarios. Uses reinforcement learning (REINFORCE) with attention-based policy networks to produce feasible routes that balance response time, satisfaction, equity, and constraint adherence.

## Project structure

```
dana/              # Python package
├── envs/          # DisasterMDVRPTW environment
├── models/        # Neural network architectures (encoder, policy, context)
├── eval/          # Evaluation (baselines, metrics, plots)
└── train.py       # Training entry point
configs/           # YAML configuration files
papers/            # Related academic literature
kaggle/            # Kaggle competition submission assets
checkpoints/       # Trained model checkpoints
latex/             # LaTeX sources
docs/              # Documentation
```

## Setup

```bash
python -m venv venv
pip install -e .
```

## Usage

```bash
python -m dana.train --config configs/dana.yaml
```

## License

MIT
