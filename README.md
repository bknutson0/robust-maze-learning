# Robust Maze Learning

This code accompanies the paper  
**“On Logical Extrapolation for Mazes with Recurrent and Implicit Networks.”**  


---

## 1 Overview

We revisit **logical extrapolation**—the ability to solve harder problem instances by running longer—in the context of maze solving.  
Our study compares four models:

| Model            | Type & Depth |
|------------------|-------------|
| **DT-Net**       | weight-tied ResNet-based recurrent network |
| **IT-Net**       | implicit variation of DT-Net  |
| **PI-Net**       | path-independent implicit network |
| **FF-Net**       | 30-layer ResNet |

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
| Train a model on a dataset     | `uv run -m src.train`         |
| Evaluate evaluate a model on selected datasets              | `uv run -m src.test`          |
| Perform Toplogical Data Analysis (on latent iterates)     | `uv run -m src.tda`           |
| Plot accuracy vs. percolation / size and heatmaps  | `uv run -m src.analyze`       |

All outputs (metrics, PDFs) are written to `outputs/`.

---
