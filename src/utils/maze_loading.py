import logging
import warnings
from pathlib import Path
from typing import Any

import torch
from maze_dataset import set_serialize_minimal_threshold
from maze_dataset.dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.dataset.rasterized import RasterizedMazeDataset
from maze_dataset.generation import LatticeMazeGenerators
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.config import DEVICE, LOGGING_LEVEL, Hyperparameters, TestParameters

# Ignore warnings from muutils.json_serialize TODO: remove this line
warnings.filterwarnings('ignore', module='muutils.json_serialize')

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# set this threshold to prevent crashing on large datasets. Will be fixed soon.
set_serialize_minimal_threshold(int(10**7))


def _load_mazes(
    dataset_name: str = 'maze-dataset',
    seed: int = 42,
    maze_size: int = 9,
    num_mazes: int = 1,
    gen: str = 'dfs_perc',  #
    percolation: float = 0.0,
    deadend_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate at least num mazes mazes of the given size and return inputs and solutions. Recursively tries again."""
    if dataset_name != 'maze-dataset':
        raise NotImplementedError('Only maze-dataset is currently implemented.')
    if maze_size % 2 == 0:
        raise ValueError(f'Expected odd maze size, got {maze_size}.')
    grid_n = (maze_size + 1) // 2

    # Generate base maze dataset
    if gen == 'dfs':
        maze_ctor = LatticeMazeGenerators.gen_dfs
        maze_ctor_kwargs = {}
    elif gen == 'dfs_perc':
        maze_ctor = LatticeMazeGenerators.gen_dfs_percolation  # type: ignore
        maze_ctor_kwargs = {'p': percolation}
    elif gen == 'percolation':
        maze_ctor = LatticeMazeGenerators.gen_percolation  # type: ignore
        maze_ctor_kwargs = {'p': percolation}
    endpoint_kwargs = {
        'deadend_start': deadend_start,
        'endpoints_not_equal': True,
        'except_on_no_valid_endpoint': False,
    }

    base_dataset: MazeDataset = MazeDataset.from_config(  # type: ignore
        MazeDatasetConfig(  # type: ignore
            name=f'test-{seed}',
            grid_n=grid_n,
            n_mazes=num_mazes,
            seed=seed,
            maze_ctor=maze_ctor,
            maze_ctor_kwargs=maze_ctor_kwargs,
            endpoint_kwargs=endpoint_kwargs,
        ),
        local_base_path=Path('data/maze-dataset/'),
    )

    # Generate rasterized maze dataset
    dataset: Dataset[Any] = RasterizedMazeDataset.from_base_MazeDataset(
        base_dataset=base_dataset,
        added_params={
            'remove_isolated_cells': True,
            'extend_pixels': True,  # maps from 1x1 to 2x2 pixels and adds 3 padding
        },
    )

    # If no mazes were generated, try again with double the number of mazes, unless num_mazes is too large
    if len(base_dataset) == 0:
        if num_mazes > 1000000:
            raise ValueError(f'Failed to generate any mazes with {num_mazes = }')
        else:
            return _load_mazes(dataset_name, seed, maze_size, num_mazes * 2, gen, percolation, deadend_start)

    # Convert to tensor
    dataset = dataset.get_batch(idxs=None)  # type: ignore

    # Get inputs
    inputs = dataset[0, :, :, :]
    inputs = inputs / 255.0
    inputs = inputs.permute(0, 3, 1, 2)
    inputs = inputs.float().detach().to(DEVICE, dtype=torch.float32)

    # Get solutions
    solutions = dataset[1, :, :, :]
    solutions = solutions / 255.0
    solutions = solutions.permute(0, 3, 1, 2)
    solutions, _ = torch.max(solutions, dim=1)
    solutions = solutions.float().detach().to(DEVICE, dtype=torch.float32)

    return inputs, solutions


def load_mazes(
    dataset: str = 'maze-dataset',
    seed: int = 42,
    maze_size: int = 9,
    num_mazes: int = 1,
    gen: str = 'dfs_perc',
    percolation: float = 0.0,
    deadend_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate mazes of the given size and number, from the given dataset, and load to device."""
    if dataset == 'maze-dataset':
        inputs, solutions = _load_mazes(dataset, seed, maze_size, num_mazes, gen, percolation, deadend_start)

        # Reduce number of mazes if necessary
        inputs = inputs[:num_mazes]
        solutions = solutions[:num_mazes]

    elif dataset == 'easy-to-hard-data':
        raise NotImplementedError('Easy-to-hard-data not implemented yet.')
        # from easy_to_hard_data import MazeDataset as EasyToHardMazeDataset TODO: Fix import
        # """ https://github.com/aks2203/easy-to-hard-data """
        # # 50,000 training mazes for maze_size [9]
        # # 10,000 testing mazes for each smaller maze_size in [9, 11, 13, 15, 17]
        # # 1,000 testing mazes for each larger maze_size in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59]

        # if maze_size not in [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59]:
        #     raise ValueError(f'Invalid maze size {maze_size} for easy-to-hard-data.')

        # maze_dataset = EasyToHardMazeDataset(root='data/easy-to-hard-data/', train=False, size=maze_size)
        # inputs = maze_dataset.inputs[:num_mazes].float().detach().to(DEVICE, dtype=torch.float32)
        # solutions = maze_dataset.targets[:num_mazes].float().detach().to(DEVICE, dtype=torch.float32)

    logger.info(
        f'Loaded {num_mazes} mazes with size: {maze_size}, percolation: {percolation}, '
        f'and deadend_start: {deadend_start}'
    )

    return inputs, solutions


def maze_loaders(
    params: Hyperparameters | TestParameters,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]] | DataLoader[Any]:
    """Load the maze dataset and return data loaders for training and validation, or testing."""

    # Load the maze dataset
    def ensure_single_value(value: Any, name: str) -> Any:  # noqa: ANN401
        """Ensure the parameter is a single value, not an empty or multi-value list."""
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            else:
                raise ValueError(f'Expected {name} to have exactly one value, but got: {value}')
        return value

    # Now, apply this to all parameters before passing them
    inputs, solutions = load_mazes(
        dataset=ensure_single_value(params.dataset_name, 'dataset_name'),
        maze_size=ensure_single_value(params.maze_size, 'maze_size'),
        num_mazes=ensure_single_value(params.num_mazes, 'num_mazes'),
        gen=ensure_single_value(params.generation_method, 'generation_method'),
        percolation=ensure_single_value(params.percolation, 'percolation'),
        deadend_start=ensure_single_value(params.deadend_start, 'deadend_start'),
    )

    solutions = solutions.long()  # Convert to long for CrossEntropyLoss
    dataset: TensorDataset = TensorDataset(inputs, solutions)

    # If training and validating, return train, train_subset, and validation loaders
    if isinstance(params, Hyperparameters):
        # Get train and validation sizes
        num_train = int(params.train_size * len(dataset))
        num_validation = len(dataset) - num_train

        # Split the dataset into train and validation sets
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [num_train, num_validation])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

        train_subset, _ = torch.utils.data.random_split(train_dataset, [num_validation, num_train - num_validation])
        train_subset_loader = DataLoader(train_subset, batch_size=params.batch_size, shuffle=False)

        validation_loader = DataLoader(validation_dataset, batch_size=params.batch_size, shuffle=False)

        return train_loader, train_subset_loader, validation_loader

    # If testing, return test loader
    else:
        test_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

        return test_loader
