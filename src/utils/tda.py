# ruff: noqa: N803, N806
"""Based on https://arxiv.org/abs/1704.08382."""

import datetime
import logging
import os
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.figure import Figure
from numpy.typing import DTypeLike, NDArray
from pandas import DataFrame
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.models.base_net import BaseNet
from src.utils.config import DEVICE, LOGGING_LEVEL, TDAParameters
from src.utils.maze_loading import load_mazes
from src.utils.model_loading import get_model_hyperparameters, load_model
from src.utils.seeding import set_seed
from src.utils.testing import is_minimal_path, is_valid_path

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_diagram(
    latent_series: list[torch.Tensor],
    dtype: DTypeLike = np.float32,
    embed_dim: int = 0,
    delay: int = 1,
    max_homo: int = 1,
) -> tuple[list[np.ndarray], float]:  # type: ignore
    """Get the persistence diagram for data X."""
    # Flatten list of tensors into tensor

    X = torch.cat(latent_series, dim=0)

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
    print(f'{max_homo = }')
    diagram = ripser(distance_matrix, maxdim=max_homo, coeff=2, distance_matrix=True)['dgms']
    print(f'{diagram[1] = }')

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

    # Otherwise, calculate persistent homologies above threshold
    else:
        max_death = get_max_finite_death(diagram)
        for i in range(len(diagram)):
            for j in range(len(diagram[i])):
                lifetime = diagram[i][j, 1] - diagram[i][j, 0]
                if lifetime > threshold * max_death:
                    betti_nums[i] += 1

    return betti_nums


def get_max_finite_death(diagram: list[np.ndarray]) -> float | None:  # type: ignore
    """Get maximum death in D, ignoring infinity."""
    finite_deaths = []
    for d in diagram:
        finite_deaths.extend(d[np.isfinite(d[:, 1]), 1])

    max_finite_death = np.max(finite_deaths) if finite_deaths else None

    return max_finite_death


@torch.no_grad()
def specific_tda(params: TDAParameters) -> DataFrame:
    """Run TDA for one combination of maze parameters across multiple models."""
    # Validate that all fields except 'iters' and 'model_name' are scalars or list for model_name
    for key, val in vars(params).items():
        if key not in ('iters', 'model_name') and isinstance(val, list):
            raise ValueError(f'Invalid type for {key}: expected scalar, got list')

    # Load all mazes once
    inputs, solutions = load_mazes(params)

    rows = []
    for model_name in params.model_name:
        model = load_model(pretrained=model_name)
        if not isinstance(model, BaseNet):
            raise ValueError(f'Invalid model type: {type(model)}. Need BaseNet or subclass for TDA.')
        model.eval()

        train_hp = get_model_hyperparameters(model_name)

        # Process each maze individually
        for j in range(params.num_mazes):
            inp = inputs[j : j + 1]
            sol = solutions[j : j + 1]
            start = time.time()

            # forward to get latent series
            lat0 = model.input_to_latent(inp)
            lat_series = model.latent_forward(lat0, inp, params.iters, params.tolerance)
            torch.cuda.empty_cache()

            # compute persistence diagram
            diag, max_dist = get_diagram(
                lat_series,  # type: ignore
                dtype=params.dtype,
                embed_dim=params.embed_dim,
                delay=params.delay,
                max_homo=params.max_homo,
            )

            # prediction & metrics
            out = model.latent_to_output(lat_series if isinstance(lat_series, torch.Tensor) else lat_series[0])
            pred = model.output_to_prediction(out, inp)

            rows.append(
                {
                    # model info
                    'model_name': model_name,
                    # training hyperparameters
                    'train_percolation': float(train_hp.percolation),
                    # TDA/test parameters
                    'test_maze_size': int(params.maze_size),  # type: ignore
                    'test_percolation': float(params.percolation),  # type: ignore
                    # per‐maze results
                    'maze_index': int(j),
                    'embed_dim': int(params.embed_dim),
                    'delay': int(params.delay),
                    'max_distance': float(max_dist),
                    'diagram': diag,
                    'time': float(time.time() - start),
                    'matches_solution': bool((pred == sol).all().item()),
                    'is_valid': bool(is_valid_path(inp, pred).item()),
                    'is_minimal': bool(is_minimal_path(inp, pred, sol).item()),
                }
            )

    return pd.DataFrame(rows)


def tda(params: TDAParameters) -> DataFrame:
    """Run TDA over all combinations of list-valued parameters, saving and returning results."""
    # Set seed
    set_seed(params.seed)

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
    results_file = os.path.join(run_dir, 'results.pkl')
    for spec in specific_list:
        df = specific_tda(spec)
        results = pd.concat([results, df], ignore_index=True)
        results.to_pickle(os.path.join(run_dir, 'results.pkl'))
        logger.info(f'Partial TDA results saved to {results_file}')
    logger.info(f'Complete TDA results saved to {results_file}')

    return results


def add_betti_column(df: DataFrame, threshold: float) -> DataFrame:
    """Add Betti numbers to the DataFrame based on persistence threshold."""
    betti_list = []
    for diag in df['diagram']:
        betti = get_betti_nums(diag, threshold)
        betti_list.append((int(betti[0]), int(betti[1])))
    df2 = df.copy()
    df2['betti_nums'] = betti_list
    return df2


def make_betti_table(df: DataFrame) -> DataFrame:
    """Create a summary table of Betti number distributions."""
    models = sorted(df['model_name'].unique())
    sizes = sorted(df['test_maze_size'].unique())
    table = pd.DataFrame(index=models, columns=sizes, dtype=object)

    for m in models:
        for s in sizes:
            sub = df[(df['model_name'] == m) & (df['test_maze_size'] == s)]
            counts = Counter(sub['betti_nums'])
            total = sum(counts.values())
            most_common = counts.most_common(3)

            # format entries with HTML <br>
            entries = [f'[{b0},{b1}]-{cnt}' for (b0, b1), cnt in most_common]
            rest = total - sum(cnt for _, cnt in most_common)
            entries.append(f'Other-{rest}')

            table.at[m, s] = '<br>'.join(entries)

    table.index.name = 'model_name'
    table.columns.name = 'test_maze_size'
    return table


def sample_sphere(n: int = 1000, radius: float = 1.0, hollow: bool = False) -> NDArray[Any]:
    """Sample points uniformly in (hollow=False) or on (hollow=True) a sphere."""
    # for solid: radii ~ rand**(1/3); for hollow: radii = 1
    u = np.random.rand(n) ** (1 / 3) if not hollow else np.ones(n)
    dirs = np.random.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs * u[:, None] * radius


def sample_cube(n: int = 1000, half_side: float = 1.0, hollow: bool = False) -> NDArray[Any]:
    """Sample points in (hollow=False) or on (hollow=True) the surface of a cube."""
    if not hollow:
        return np.random.uniform(-half_side, half_side, size=(n, 3))
    # sample surface: pick one of 6 faces, then uniform on that square
    faces = np.random.randint(0, 6, size=n)
    uv = np.random.uniform(-half_side, half_side, size=(n, 2))
    pts = np.zeros((n, 3))
    for i, f in enumerate(faces):
        if f == 0:
            pts[i] = [-half_side, uv[i, 0], uv[i, 1]]
        elif f == 1:
            pts[i] = [half_side, uv[i, 0], uv[i, 1]]
        elif f == 2:
            pts[i] = [uv[i, 0], -half_side, uv[i, 1]]
        elif f == 3:
            pts[i] = [uv[i, 0], half_side, uv[i, 1]]
        elif f == 4:
            pts[i] = [uv[i, 0], uv[i, 1], -half_side]
        else:
            pts[i] = [uv[i, 0], uv[i, 1], half_side]
    return pts


def sample_circle(n: int = 1000, radius: float = 1.0, hollow: bool = True) -> NDArray[Any]:
    """Sample points on (hollow=True) or in (hollow=False) a circle in the z=0 plane."""
    θ = np.random.rand(n) * 2 * np.pi
    r = np.full(n, radius) if hollow else np.sqrt(np.random.rand(n)) * radius
    x, y = r * np.cos(θ), r * np.sin(θ)
    return np.column_stack((x, y, np.zeros(n)))


def sample_square(n: int = 1000, half_side: float = 1.0, hollow: bool = True) -> NDArray[Any]:
    """Sample points on (hollow=True) or in (hollow=False) a square in the z=0 plane."""
    if not hollow:
        xy = np.random.uniform(-half_side, half_side, size=(n, 2))
    else:
        perim = 8 * half_side
        s = np.random.rand(n) * perim
        xy = np.empty((n, 2))
        for i, si in enumerate(s):
            seg = int(si // (2 * half_side))
            t = si % (2 * half_side)
            if seg == 0:
                xy[i] = [-half_side + t, -half_side]
            elif seg == 1:
                xy[i] = [half_side, -half_side + t]
            elif seg == 2:
                xy[i] = [half_side - t, half_side]
            else:
                xy[i] = [-half_side, half_side - t]
    return np.column_stack((xy, np.zeros(n)))


def sample_torus(n: int = 1000, R: float = 2.0, r: float = 0.5, hollow: bool = False) -> NDArray[Any]:
    """Sample points in (hollow=False) or on (hollow=True) a torus."""
    u = np.random.rand(n) * 2 * np.pi
    v = np.random.rand(n) * 2 * np.pi
    if not hollow:
        rad = np.sqrt(np.random.rand(n)) * r
        cx, cy = R * np.cos(u), R * np.sin(u)
        off_x = rad * np.cos(v) * np.cos(u)
        off_y = rad * np.cos(v) * np.sin(u)
        off_z = rad * np.sin(v)
        return np.column_stack((cx + off_x, cy + off_y, off_z))
    else:
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return np.column_stack((x, y, z))


def get_pca(X: NDArray[Any] | torch.Tensor, n: int = 2) -> NDArray[Any]:
    """Project rows of data matrix X onto n principal components using SVD."""
    # Move X to GPU
    X = X.to(DEVICE) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # Flatten data
    X = X.reshape(X.shape[0], -1)

    # Convert to torch on device
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # Center the data
        X = X - torch.mean(X, dim=0)

        # Compute the SVD
        U, S, V = torch.linalg.svd(X, full_matrices=False)

        # Project the data onto the first n principal components
        Y = torch.mm(X, V[:n].T)

    # Convert back to numpy
    Y = Y.cpu().numpy()  # type: ignore

    return Y  # type: ignore


def plot_diagram(
    diagram: list[NDArray[Any]],
    threshold: float | None = None,
    fig_size: tuple[float, float] = (5, 5),
    font_size: int = 8,
    file_path: str | None = None,
) -> Figure:
    """Plot a persistence diagram using given threshold."""
    # filter out infinite deaths
    filtered = [d[np.isfinite(d[:, 1])] for d in diagram]
    if not filtered or all(d.size == 0 for d in filtered):
        raise ValueError('No finite death values found in any diagram.')

    max_death = get_max_finite_death(diagram)
    margin = 1.05 * max_death

    fig, ax = plt.subplots(figsize=fig_size)
    # pick a colormap with enough distinct colors
    cmap = cm.get_cmap('tab10', len(filtered))

    for idx, pts in enumerate(filtered):
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=15,
            label=f'H{idx}',
            color=cmap(idx),
            clip_on=False,
        )

    # diagonal
    ax.plot([0, margin], [0, margin], '--', color='k', linewidth=1)

    # threshold line (parallel to diagonal)
    if threshold is not None:
        offset = threshold * max_death
        ax.plot([0, margin], [offset, margin + offset], '--', color='red', linewidth=1)

    ax.set_xlim(0, margin)
    ax.set_ylim(0, margin)
    ax.set_xlabel('Birth', fontsize=font_size)
    ax.set_ylabel('Death', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size)
    fig.tight_layout()

    if file_path is not None:
        out = Path(file_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')

    return fig
