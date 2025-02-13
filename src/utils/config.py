import json
import os
from dataclasses import asdict, dataclass, field

import torch

# Device configuration
# Prioritize PREFERRED_CUDA > CUDA > MPS > CPU
PREFERRED_CUDA = 2
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


# Training hyperparameters
@dataclass
class Hyperparameters:
    """Hyperparameters for training the model."""

    # General parameters
    seed: int = 0

    # Dataset parameters
    dataset_name: str = 'maze-dataset'
    generation_method: str = 'dfs_perc'
    maze_size: int = 9
    deadend_start: bool = True
    percolation: float = 0.0
    num_mazes: int = 50000

    # Model hyperparameters
    model_name: str = 'dt_net'
    iters: int = 30

    # Training parameters
    validation_size: float = 0.1
    train_size: float = 1.0 - validation_size
    batch_size: int = 32
    epochs: int = 100
    checkpoint_freq: int = 10
    alpha: float = 0.01  # Progressive loss factor, originally 0.01 in "End-to-end algorithm synthesis"
    learning_rate: float = 0.0001
    grad_clip: float | None = 1.0
    optimizer_name: str = 'AdamW'
    learning_rate_scheduler_name: str = 'ReduceLROnPlateau'
    patience: int = 10
    reduction_factor: float = 0.1

    def to_json(self, path: str) -> None:
        """Save hyperparameters to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)


# Testing parameters
@dataclass
class TestParameters:
    """Parameters for testing the model."""

    # General parameters
    seed: int = 1  # Different seed from training

    # Dataset parameters
    dataset_name: list[str] | str = field(default_factory=lambda: ['maze-dataset'])
    generation_method: list[str] | str = field(default_factory=lambda: ['dfs_perc'])
    maze_size: list[int] | int = field(default_factory=lambda: [9])
    deadend_start: list[bool] | bool = field(default_factory=lambda: [True])
    percolation: list[float] | float = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    )
    num_mazes: list[int] | int = field(default_factory=lambda: [10000])

    # Model hyperparameters
    model_name: list[str] | str = field(default_factory=lambda: ['dt_net'])
    iters: list[int] | int = field(default_factory=lambda: [30])

    # Testing parameters
    batch_size: int = 256

    def to_json(self, path: str) -> None:
        """Save test parameters to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
