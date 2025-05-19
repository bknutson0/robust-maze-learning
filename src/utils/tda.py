# ruff: noqa: N803, N806
"""Based on https://arxiv.org/abs/1704.08382."""

import datetime
import logging
import os
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
from numpy.typing import DTypeLike
from pandas import DataFrame
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.models.base_net import BaseNet
from src.utils.config import DEVICE, LOGGING_LEVEL, TDAParameters
from src.utils.maze_loading import load_mazes
from src.utils.model_loading import load_model
from src.utils.testing import is_minimal_path, is_valid_path

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_diagram(
    X: torch.Tensor,
    dtype: DTypeLike = np.float32,
    embed_dim: int = 0,
    delay: int = 1,
    max_homo: int = 1,
) -> tuple[list[np.ndarray], float]:  # type: ignore
    """Get the persistence diagram for data X."""
    # Convert from torch tensor to cpu numpy array
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # type: ignore

    # Ensure desired precision
    X = X.astype(dtype)  # type: ignore

    logger.info(f'Performing TDA with {embed_dim = }, {delay = }, and {X.shape = } ({X.nbytes / 1e9:.3f}GB)')

    # Flatten dimensions beyond first
    X = X.reshape(X.shape[0], -1)
    logger.info(f'Flattened {X.shape = }')

    # Reduce dimensionality of X using SVD, if first dimension is smaller
    F, P = X.shape
    if F < P:
        X = reduce(X)  # type: ignore
        logger.info(f'Reduced {X.shape = } ({X.nbytes / 1e9:.3f}GB)')

    # Compute distance matrix of sliding window embedding of X
    distance_matrix = get_distance_matrix(X)  # type: ignore
    max_distance = np.max(distance_matrix)
    logger.info(f'Computed {distance_matrix.shape = } with {max_distance = :.3f}')

    # Use ripser to compute persistence diagram
    diagram = ripser(distance_matrix, maxdim=max_homo, coeff=2, distance_matrix=True)['dgms']

    return diagram, max_distance


def get_distance_matrix(X: np.ndarray, embed_dim: int = 0, delay: int = 1, threshold: float = 1e-10) -> np.ndarray:  # type: ignore
    """Get distance matrix for rows of sliding window embedding of flattened data X."""
    # Ensure dimensions beyond first have been flattened
    if len(X.shape) != 2:
        raise ValueError('Input array X must have exactly 2 dimensions.')

    # Window size must be less than total number of iterations
    if (embed_dim + 1) * delay >= X.shape[0]:
        raise ValueError('The window size (embed_dim + 1) * delay must be less than the number of rows in X.')

    if delay == 1:
        # Compute squared distance matrix for X
        squared_distance_matriX_X = np.square(squareform(pdist(X), checks=False))

        # Compute distance matrix for sliding window embedding of X
        d = embed_dim
        F = X.shape[0]  # frames
        W = F - d  # windows
        squared_distance_matriX_SW = np.zeros(shape=(W, W))
        for i in range(W):
            for j in range(i + 1, W):
                for k in range(d + 1):
                    squared_distance_matriX_SW[i, j] += squared_distance_matriX_X[i + k, j + k]
                squared_distance_matriX_SW[j, i] = squared_distance_matriX_SW[i, j]
        if not np.all(squared_distance_matriX_SW == squared_distance_matriX_SW.T):
            raise ValueError('The squared distance matrix is not symmetric.')
        if not np.all(squared_distance_matriX_SW >= 0):
            raise ValueError('All elements in the squared distance matrix must be non-negative.')
        distance_matriX_SW = np.sqrt(squared_distance_matriX_SW)

    else:
        raise NotImplementedError

    # Zero distances that are smaller than threshold
    distance_matriX_SW[distance_matriX_SW < threshold] = 0

    return distance_matriX_SW


def get_sw(X: np.ndarray, embed_dim: int, delay: int, verbose: bool = False) -> np.ndarray:  # type: ignore
    """Get normalized sliding window embedding of data.

    Following (Quasi)Periodicity Quantification in Video Data by Tralie & Perea.
        embed_dim: d in their paper
        delay: tau in their paper
    """
    X = X.reshape(X.shape[0], -1)
    f, d = X.shape

    # Create sliding window embedding tensor
    window_size = embed_dim * delay
    sw = np.zeros((f - window_size, (embed_dim + 1) * d))
    for i in range(f - window_size):
        sw[i] = X[i : i + window_size + 1 : delay].flatten()

    if verbose:
        print(f'{sw.shape = }')

    return sw


def reduce(X: torch.Tensor) -> np.ndarray:  # type: ignore
    """Reduce dimensionality of X using SVD for memory efficiency."""
    X = torch.tensor(X, device=DEVICE)
    with torch.no_grad():
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        X_reduced = U * S
    torch.cuda.empty_cache()
    return X_reduced.cpu().numpy()  # type: ignore


def get_betti_nums(diagram: list[np.ndarray], threshold: float) -> np.ndarray:  # type: ignore
    """Calculate Betti numbers for persistence diagram D with given threshold."""
    betti_nums = np.zeros(len(diagram), dtype=int)

    # If diagram corresponds to single point, return [1, 0]
    if (diagram[0] == [[0, np.inf]]).all() and (diagram[1] == np.zeros((0, 2))).all():
        betti_nums[0] = 1
        betti_nums[1] = 0

    # Otherwise, calculate persistent homologies above threshold
    else:
        max_death = get_max_death(diagram)
        for i in range(len(diagram)):
            for j in range(len(diagram[i])):
                lifetime = diagram[i][j, 1] - diagram[i][j, 0]
                if lifetime > threshold * max_death:
                    betti_nums[i] += 1

    return betti_nums


def get_max_death(diagram: list[np.ndarray]) -> float:  # type: ignore
    """Get maximum death in D, ignoring infinity."""
    max_death = 0
    for i in range(len(diagram)):
        for j in range(len(diagram[i])):
            if diagram[i][j, 1] != np.inf:
                max_death = max(max_death, diagram[i][j, 1])
    return max_death


@torch.no_grad()
def specific_tda(params: TDAParameters) -> DataFrame:
    """Run TDA for one combination of parameters and return a DataFrame of results."""
    # Validate that all fields except 'iters' and 'model_name' are scalars
    for key, val in vars(params).items():
        if key not in ('iters', 'model_name') and isinstance(val, list):
            raise ValueError(f'Invalid type for {key}: expected scalar, got list')

    # Load mazes
    inputs, solutions = load_mazes(params)

    # Load and prepare model
    model = load_model(params.model_name)  # type: ignore
    if not isinstance(model, BaseNet):
        raise ValueError(f'Invalid model type: {type(model)}. Need BaseNet or subclass for TDA.')
    model.eval()

    rows = []
    for j, (input, solution) in enumerate(zip(inputs, solutions, strict=False)):
        start = time.time()

        # forward pass to get latent series
        latent = model.input_to_latent(input.unsqueeze(0))
        latent_series = model.latent_forward(latent, input.unsqueeze(0), params.iters)
        output = model.latent_to_output(latent_series[0])
        prediction = model.output_to_prediction(output, input)
        torch.cuda.empty_cache()

        # compute persistence diagram
        diag, max_dist = get_diagram(latent_series, params.dtype, params.embed_dim, params.delay, params.max_homo)  # type: ignore

        # record one row
        rows.append(
            {
                'model_name': params.model_name,
                'maze_size': params.maze_size,
                'percolation': params.percolation,
                'maze_index': j,
                'max_distance': max_dist,
                'h0': diag[0],
                'h1': diag[1],
                'time_s': time.time() - start,
                'matches_solution': prediction == solution,
                'is_valid': is_valid_path(input, prediction),
                'is_minimal': is_minimal_path(input, prediction, solution),
            }
        )

    return pd.DataFrame(rows)


def tda(params: TDAParameters) -> DataFrame:
    """Run TDA over all combinations of list-valued parameters, saving and returning results."""
    # prepare output directory
    os.makedirs('outputs/tda', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join('outputs/tda', timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # save parameters
    params.to_json(os.path.join(run_dir, 'tda_parameters.json'))

    # build list of specific parameter objects
    param_dict = {k: (v if isinstance(v, list) else [v]) for k, v in vars(params).items()}
    for drop in ('iters', 'model_name'):
        param_dict.pop(drop, None)

    specific_list = []
    for combo in product(*param_dict.values()):
        p = TDAParameters()
        p.model_name = params.model_name
        p.iters = sorted(params.iters) if isinstance(params.iters, list) else [params.iters]
        for key, val in zip(param_dict.keys(), combo, strict=False):
            setattr(p, key, val)
        p.num_mazes = params.num_mazes
        p.dtype = params.dtype
        p.embed_dim = params.embed_dim
        p.delay = params.delay
        p.max_homo = params.max_homo
        specific_list.append(p)

    # run and save
    results = pd.DataFrame()
    results_file = os.path.join(run_dir, 'results.csv')
    for spec in specific_list:
        df = specific_tda(spec)
        results = pd.concat([results, df], ignore_index=True)
        results.to_csv(results_file, index=False)

    return results
