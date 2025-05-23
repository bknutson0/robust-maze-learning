import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import DTypeLike

# Device configuration
# Prioritize PREFERRED_CUDA > CUDA > MPS > CPU
PREFERRED_CUDA = 0
DEVICE = None
if torch.cuda.is_available():
    if torch.cuda.device_count() > PREFERRED_CUDA:
        DEVICE = torch.device(f'cuda:{PREFERRED_CUDA}')
    else:
        DEVICE = torch.device('cuda')  # Default to first available CUDA device
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# Logging configuration
LOGGING_LEVEL = 'INFO'  # Set logging level globally

# Parameters for training and testing the model
TOLERANCE = 1e-1  # Tolerance for convergence in it_net


# Training hyperparameters
@dataclass
class Hyperparameters:
    """Hyperparameters for training the model."""

    # General parameters
    seed: int = 1

    # Dataset parameters
    dataset_name: str = 'maze-dataset'
    generation_method: str = 'dfs_perc'
    maze_size: int = 9
    deadend_start: bool = True
    percolation: float = 0.0
    num_mazes: int = 100000

    # Model hyperparameters
    model_name: str = 'dt_net'
    iters: int = 30

    # Training parameters
    pretrained: str | None = None  # Path to pretrained model, if any
    weight_init: str | None = None  # Weight initialization scheme, if any
    validation_size: float = 0.1
    train_size: float = 1.0 - validation_size
    batch_size: int = 32
    epochs: int = 100
    checkpoint_freq: int | None = None
    learning_rate: float = 0.0001
    grad_clip: float | None = 1.0
    optimizer_name: str = 'AdamW'
    learning_rate_scheduler_name: str = 'ReduceLROnPlateau'
    patience: int = 10
    reduction_factor: float = 0.1

    # dt_net specific
    alpha: float = 0.01  # Progressive loss factor, originally 0.01 in "End-to-end algorithm synthesis"

    # it_net specific
    tolerance: float = TOLERANCE  # Tolerance for convergence
    p: float = torch.inf  # Specify p of p-norm used in convergence condition
    random_iters: bool = False  # Randomly sample number of iterations from [1, iters] for each batch
    contraction: float | None = 1.0  # Contraction factor to weakly enforce at every training step
    train_jfb: bool = False  # Train using Jacobian-free backpropagation (JFB) TODO: change to backrop_iters
    # Implicit network can benefit from warmup training, before implicit layer generally converges to fixed point
    warmup_epochs: int = 10  # Epochs to train without JFB initially
    warmup_iters: int | None = None  # Number of iterations to train during warmup

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary with JSON-serializable values."""

        def sanitize(value: Any) -> int | float | str | dict[str, Any] | list[Any]:  # noqa: ANN401
            """Recursively sanitize values to ensure JSON-serializability."""
            if isinstance(value, float) and math.isinf(value):
                return 'Infinity'  # or another representation if preferred
            elif isinstance(value, dict):
                return {k: sanitize(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize(v) for v in value]
            return value  # type: ignore

        return {k: sanitize(v) for k, v in asdict(self).items()}


# Testing parameters
@dataclass
class TestParameters:
    """Parameters for testing the model."""

    # General parameters
    seed: int = 4242  # Different seed from training

    # Dataset parameters
    dataset_name: list[str] | str = field(default_factory=lambda: ['maze-dataset'])
    generation_method: list[str] | str = field(default_factory=lambda: ['dfs_perc'])
    maze_size: list[int] | int = field(default_factory=lambda: [9])
    deadend_start: list[bool] | bool = field(default_factory=lambda: [True])
    percolation: list[float] | float = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    )
    num_mazes: int = 10000

    # Model hyperparameters
    model_name: list[str] | str = field(default_factory=lambda: ['dt_net'])
    iters: list[int] | int = field(default_factory=lambda: [30])

    # Testing parameters
    batch_size: int = 256
    compare_deadend_fill: bool = False  # Compare with deadend-filling algorithm

    # dt_net specific

    # it_net specific
    tolerance: float = TOLERANCE  # Tolerance for convergence

    # pi_net specific
    threshold: int | str = 'default'

    def are_mostly_single_valued(self) -> bool:
        """Check if all parameters are single-valued, except model_name and iters."""
        return all(
            (isinstance(value, bool | int | float | str) or key in ['model_name', 'iters'])
            for key, value in asdict(self).items()
        )

    def to_json(self, path: str) -> None:
        """Save test parameters to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4, default=str)


# TDA parameters, inherits from TestParameters
@dataclass
class TDAParameters(TestParameters):
    """Parameters for running TDA."""

    iters: list[int] = field(default_factory=lambda: list(range(3001, 3401)))
    dtype: DTypeLike = np.float64
    embed_dim: int = 0
    delay: int = 1
    max_homo: int = 1


# Analysis plotting configuration
@dataclass
class PlotConfig:
    """Configuration for plotting analysis results."""

    font_size: int = 30
    axes_title_size: int = 30
    axes_title_weight: str = 'bold'
    axes_label_size: int = 30
    axes_label_weight: str = 'bold'
    legend_title_fontsize: int = 30
    legend_title_weight: str = 'bold'
    subplot_title_size: int = 30

    def apply(self) -> None:
        """Apply these settings to Matplotlib's rcParams."""
        plt.rcParams.update(
            {
                'font.size': self.font_size,
                'axes.titlesize': self.axes_title_size,
                'axes.titleweight': self.axes_title_weight,
                'axes.labelsize': self.axes_label_size,
                'axes.labelweight': self.axes_label_weight,
                'legend.title_fontsize': self.legend_title_fontsize,
            }
        )
