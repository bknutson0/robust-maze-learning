# Robust Maze Learning

This code accompanies the paper  
**“On Logical Extrapolation for Mazes with Recurrent and Implicit Networks.”**  


---

## 1 Overview

We revisit **logical extrapolation**—the ability to solve harder problem instances by running longer—in the context of maze solving.  
Our study compares four models:

| Model            | Type & Depth | Test-time Scaling | Key Finding |
|------------------|-------------|-------------------|-------------|
| **DT-Net**       | weight-tied ResNet-based recurrent network | ✓ | Emulates deadend-filling; converges to 2-point / 2-loop cycles |
| **IT-Net**       | implicit variation of DT-Net  | ✓ | Converges to fixed points; robust to cycles |
| **PI-Net**       | path-independent implicit network | ✓ | Near-algorithmic but less precise than DT-Net |
| **FF-Net**       | 30-layer ResNet | ✗ | Strong in-distribution, poor extrapolation |

We probe three *orthogonal* difficulty axes (maze size, percolation, dead-end start) and analyse convergence with **Topological Data Analysis (TDA)**.  
Results show:

1. DT-Net copies deadend-filling on 99 % of 143 k mazes but inherits its failure modes.  
2. Increasing training diversity (percolation > 0) fixes some errors yet *reduces* size extrapolation.  
3. Successful solvers need not reach a unique fixed point; limit cycles are common and benign.

---

## 2 Installation

We use the [`uv`](https://docs.astral.sh/uv/) for package management.
Install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Then, in the root directory of this repository, simply run:
```
uv sync
```

Alternatively, you can install via `pip`:
```
pip install -r requirements.txt
```

---


## 3 Repository Structure

```
robust-maze-learning
├── data                                  # Maze datasets
│   │   ├── easy-to-hard-data
│   │   └── maze-dataset
├── models                                # Saved models
│   ├── dt_net
│   ├── ff_net
│   ├── it_net
│   └── pi_net
├── notebooks                             # Jupyter notebooks
│   ├── explore_predictions.ipynb
│   └── explore_tda.ipynb
├── outputs                               # Output files  
│   ├── tda
|   ├── tests
│   └── visuals
├── src                                   # Source code
│   ├── models/                           # Model definitions
│   ├── utils/                            # Utility functions
│   ├── train.py                          # Training script
│   ├── test.py                           # Testing script
│   ├── analyze.py                        # Analysis & plotting script
│   ├── explore_mazes.py                  # Maze exploration
│   └── tda.py                            # TDA script
├── README.md                             # This file
└── pyproject.toml                        # uv configuration
```

## 4 Quick Start

| Task                                  | Command                       |
| ------------------------------------- | ----------------------------- |
| Train IT-Net on four percolations     | `uv run -m src.train`         |
| Evaluate all checkpoints              | `uv run -m src.test`          |
| Reproduce TDA of latent iterates      | `uv run -m src.tda`           |
| Plot accuracy vs. percolation / size and heatmaps  | `uv run -m src.analyze`       |

All outputs (metrics, PDFs) are written to `outputs/`.

---
