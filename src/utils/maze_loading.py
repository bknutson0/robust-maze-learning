import logging
from pathlib import Path
from typing import Any

import muutils
import torch
from maze_dataset import set_serialize_minimal_threshold
from maze_dataset.dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.dataset.rasterized import RasterizedMazeDataset
from maze_dataset.generation import LatticeMazeGenerators
from muutils.mlutils import set_reproducibility
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.config import DEVICE, LOGGING_LEVEL, Hyperparameters, TestParameters

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# set this threshold to prevent crashing on large datasets. Will be fixed soon.
set_serialize_minimal_threshold(int(10**7))


def _load_mazes(
    params: Hyperparameters | TestParameters, num_mazes_attempt: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate at least num_mazes mazes, recursively doubling the number of mazes attempted if none are generated."""
    # Initially, attempt to generate exactly the desired number of mazes
    if num_mazes_attempt is None:
        num_mazes_attempt = params.num_mazes  # type: ignore

    logger.info(
        f'Attempting {num_mazes_attempt} mazes to generate {params.num_mazes} mazes with size: {params.maze_size}, '
        f'percolation: {params.percolation}, '
        f'and deadend_start: {params.deadend_start}'
    )

    # Check parameters
    if params.dataset_name != 'maze-dataset':
        raise NotImplementedError('Only maze-dataset is currently implemented.')
    if params.maze_size % 2 == 0:  # type: ignore
        raise ValueError(f'Expected odd maze size, got {params.maze_size}.')
    grid_n = (params.maze_size + 1) // 2  # type: ignore

    # Create maze dataset configuration
    if params.generation_method == 'dfs':
        maze_ctor = LatticeMazeGenerators.gen_dfs
        maze_ctor_kwargs = {}
    elif params.generation_method == 'dfs_perc':
        maze_ctor = LatticeMazeGenerators.gen_dfs_percolation  # type: ignore
        maze_ctor_kwargs = {'p': params.percolation}
    elif params.generation_method == 'percolation':
        maze_ctor = LatticeMazeGenerators.gen_percolation  # type: ignore
        maze_ctor_kwargs = {'p': params.percolation}
    endpoint_kwargs = {
        'deadend_start': params.deadend_start,
        'endpoints_not_equal': True,
        'except_on_no_valid_endpoint': False,
    }
    cfg_original = MazeDatasetConfig(  # type: ignore
        name=f'test-{params.seed}',
        grid_n=grid_n,
        n_mazes=num_mazes_attempt,
        seed=params.seed,
        maze_ctor=maze_ctor,
        maze_ctor_kwargs=maze_ctor_kwargs,
        endpoint_kwargs=endpoint_kwargs,
    )

    # Create base maze dataset from configuration
    # cfg_new = cfg_original.success_fraction_compensate(safety_margin=1.2, epsilon=0.00001)
    cfg_new = cfg_original

    base_dataset: MazeDataset = MazeDataset.from_config(
        cfg_new,
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

    # If not enough mazes were generated, attempt with between 2 and num_mazes_attempt times as many mazes
    if len(base_dataset) < params.num_mazes:  # type: ignore
        if params.num_mazes > 10**9:  # type: ignore
            raise ValueError(f'Failed to generate any mazes after attempting with {params.num_mazes = }')
        else:
            num_mazes_result = len(base_dataset)
            new_num_mazes_attempt_1 = num_mazes_attempt // max(num_mazes_result, 1) * params.num_mazes  # type: ignore
            new_num_mazes_attempt_2 = 2 * num_mazes_attempt  # type: ignore
            new_num_mazes_attempt = max(new_num_mazes_attempt_1, new_num_mazes_attempt_2)
            return _load_mazes(params, new_num_mazes_attempt)  # type: ignore

    # Otherwise return the desired number of mazes
    else:
        # Convert to tensor
        dataset = dataset.get_batch(idxs=None)  # type: ignore

        # Get inputs
        inputs = dataset[0, 0 : params.num_mazes, :, :]  # type: ignore
        inputs = inputs / 255.0
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.float().detach().to(DEVICE, dtype=torch.float32)

        # Get solutions
        solutions = dataset[1, 0 : params.num_mazes, :, :]  # type: ignore
        solutions = solutions / 255.0
        solutions = solutions.permute(0, 3, 1, 2)
        solutions, _ = torch.max(solutions, dim=1)
        solutions = solutions.float().detach().to(DEVICE, dtype=torch.float32)

        return inputs, solutions


def load_mazes(params: Hyperparameters | TestParameters) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate mazes of the given size and number, from the given dataset, and load to device."""
    if params.dataset_name == 'maze-dataset':
        # Set maze-dataset seed
        muutils.mlutils.DEFAULT_SEED = params.seed
        set_reproducibility(params.seed)

        inputs, solutions = _load_mazes(params)

        logger.info(
            f'Loaded {inputs.shape[0]} mazes with size: {params.maze_size}, percolation: {params.percolation}, '
            f'and deadend_start: {params.deadend_start}'
        )

        return inputs, solutions

    elif params.dataset_name == 'easy-to-hard-data':
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
    return torch.tensor([]), torch.tensor([])


def maze_loaders(
    params: Hyperparameters | TestParameters,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]] | DataLoader[Any]:
    """Load the maze dataset and return data loaders for training and validation, or testing."""
    if isinstance(params, TestParameters) and not params.are_single_valued():
        raise ValueError('TestParameters must have a single value for each parameter.')

    inputs, solutions = load_mazes(params)

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
