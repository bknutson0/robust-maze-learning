"""Based on https://arxiv.org/abs/1704.08382."""

import pickle
import time

import numpy as np
import torch
from numpy.typing import DTypeLike
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.utils.config import DEVICE
from src.utils.model_loading import load_model
from src.utils.testing import compare_mazes


def get_diagram(
    x: torch.Tensor,
    dtype: DTypeLike = np.float32,
    embed_dim: int = 0,
    delay: int = 1,
    max_homo: int = 1,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """Get the persistence diagram for data x."""
    # Convert from torch tensor to numpy array if necessary
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # type: ignore

    # Ensure desired precision
    x = x.astype(dtype)

    if verbose:
        print(f'    Performing TDA with {embed_dim = }, {delay = }, and {x.shape = } ({x.nbytes / 1e9:.3f}GB)')

    # Flatten dimensions beyond first
    x = x.reshape(x.shape[0], -1)
    if verbose:
        print(f'    Flattened {x.shape = }')

    # Reduce dimensionality of x using SVD, if first dimension is smaller
    F, P = x.shape
    if F < P:
        x = reduce(x)
        if verbose:
            print(f'    Reduced {x.shape = } ({x.nbytes / 1e9:.3f}GB)')

    # Compute distance matrix of sliding window embedding of x
    distance_matrix = get_distance_matrix(x)
    max_distance = np.max(distance_matrix)
    if verbose:
        print(f'    Computed {distance_matrix.shape = } with {max_distance = :.3f}')

    # Use ripser to compute persistence diagram
    diagram = ripser(distance_matrix, maxdim=max_homo, coeff=2, distance_matrix=True)['dgms']

    return diagram, max_distance


def get_distance_matrix(x: np.ndarray, embed_dim: int = 0, delay: int = 1, threshold: float = 1e-10) -> np.ndarray:
    """Get distance matrix for rows of sliding window embedding of flattened data x."""
    # Ensure dimensions beyond first have been flattened
    assert len(x.shape) == 2

    # Window size must be less than total number of iterations
    assert (embed_dim + 1) * delay < x.shape[0]

    if delay == 1:
        # Compute squared distance matrix for x
        squared_distance_matrix_x = np.square(squareform(pdist(x), checks=False))

        # Compute distance matrix for sliding window embedding of x
        d = embed_dim
        F = x.shape[0]  # frames
        W = F - d  # windows
        squared_distance_matrix_SW = np.zeros(shape=(W, W))
        for i in range(W):
            for j in range(i + 1, W):
                for k in range(d + 1):
                    squared_distance_matrix_SW[i, j] += squared_distance_matrix_x[i + k, j + k]
                squared_distance_matrix_SW[j, i] = squared_distance_matrix_SW[i, j]
        assert np.all(squared_distance_matrix_SW == squared_distance_matrix_SW.T)
        assert np.all(squared_distance_matrix_SW >= 0)
        distance_matrix_SW = np.sqrt(squared_distance_matrix_SW)

    else:
        raise NotImplementedError

    # Zero distances that are smaller than threshold
    distance_matrix_SW[distance_matrix_SW < threshold] = 0

    return distance_matrix_SW


def get_sw(x, embed_dim, delay, verbose=False):
    """Get normalized sliding window embedding of data, following
    (Quasi)Periodicity Quantification in Video Data by Tralie & Perea.
        embed_dim: d in their paper
        delay: tau in their paper
    """
    x = x.reshape(x.shape[0], -1)
    F, D = x.shape

    # Create sliding window embedding tensor
    window_size = embed_dim * delay
    SW = np.zeros((F - window_size, (embed_dim + 1) * D))
    for i in range(F - window_size):
        SW[i] = x[i : i + window_size + 1 : delay].flatten()

    if verbose:
        print(f'{SW.shape = }')

    return SW


def reduce(x):
    """Reduce dimensionality of x using SVD for memory efficiency"""
    x = torch.tensor(x, device=DEVICE)
    with torch.no_grad():
        U, S, V = torch.linalg.svd(x, full_matrices=False)
        x_reduced = U * S
    torch.cuda.empty_cache()
    return x_reduced.cpu().numpy()


def get_betti_nums(diagram, threshold):
    """Calculate Betti numbers for persistence diagram D with given threshold"""
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


def get_max_death(diagram):
    """Get maximum death in D, ignoring infinity"""
    max_death = 0
    for i in range(len(diagram)):
        for j in range(len(diagram[i])):
            if diagram[i][j, 1] != np.inf:
                max_death = max(max_death, diagram[i][j, 1])
    return max_death


class Analysis:
    """Class for TDA analysis and results."""

    def __init__(
        self,
        maze_sizes=[9],
        percolations=[0.0],
        num_mazes=100,
        model_name='dt_net',
        iters=list(range(3001, 3401)),
        dtype=np.float64,
        embed_dim=0,
        delay=1,
        max_homo=1,
        verbose=True,
    ):
        self.maze_sizes = maze_sizes
        self.percolations = percolations
        self.verbose = verbose
        self.extrap_param_name, self.extrap_param = self.get_extrap_param()
        self.num_mazes = num_mazes
        self.model = load_model(model_name)
        self.iters = iters
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.delay = delay
        self.max_homo = max_homo
        self.corrects = np.zeros((len(self.extrap_param), self.num_mazes))
        self.times = np.zeros((len(self.extrap_param), num_mazes), dtype=np.float32)
        self.diagrams = np.zeros((len(self.extrap_param), num_mazes, 2), dtype=object)
        self.max_distances = np.zeros((len(self.extrap_param), num_mazes), dtype=dtype)

    def get_extrap_param(self):
        if len(self.maze_sizes) > 1 and len(self.percolations) == 1:
            self.percolation = self.percolations[0]
            if self.verbose:
                print(f'Performing TDA over maze sizes: {self.maze_sizes}')
            return 'maze_size', self.maze_sizes
        elif len(self.percolations) > 1 and len(self.maze_sizes) == 1:
            self.maze_size = self.maze_sizes[0]
            if self.verbose:
                print(f'Performing TDA over percolations: {self.percolations}')
            return 'percolation', self.percolations
        else:
            raise ValueError(
                'Invalid parameters: Either maze_sizes or percolations should have multiple values, but not both.'
            )

    def get_name(self):
        name = f'{self.model.name()}'
        for attr, value in self.__dict__.items():
            if attr == 'iters':
                name += f'_{attr}-{min(value)},{max(value)}'
            elif attr == 'dtype':
                name += f'_dtype-{self.max_distances.dtype.name}'
            elif attr not in [
                'model',
                'corrects',
                'times',
                'diagrams',
                'max_distances',
                'extrap_param_name',
                'extrap_param',
                'verbose',
            ]:
                name += f'_{attr}-{value}'
        return name

    def save(self):
        """Save analysis object"""
        file_name = f'outputs/tda/analysis/{self.get_name()}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def analyze(self):
        """Perform TDA analysis on latent iterates of model while solving mazes"""
        for i, param in enumerate(self.extrap_param):
            # Print summary
            if self.verbose:
                print(
                    f'Analyzing {self.model.name()} on {self.num_mazes} mazes with {self.extrap_param_name} = {param} ...'
                )

            # Load mazes
            start_time = time.time()
            if self.extrap_param_name == 'maze_size':
                inputs, solutions = get_mazes(maze_size=param, num_mazes=self.num_mazes)
            elif self.extrap_param_name == 'percolation':
                inputs, solutions = get_mazes(percolation=param, num_mazes=self.num_mazes)

            if self.verbose:
                print(f'    Loaded mazes in {time.time() - start_time:.2f}s')

            # Generate each latent series and perform TDA
            for j in range(self.num_mazes):
                # Generate latent series
                start_time = time.time()
                input = inputs[j : j + 1]
                latent = self.model.input_to_latent(input)
                latent_series = self.model.latent_forward(latent, input, self.iters)
                output = self.model.latent_to_output(latent_series[0])
                latent_series = latent_series.cpu().numpy()
                torch.cuda.empty_cache()

                # Evaluate correctness
                prediction = self.model.output_to_prediction(output, input)
                solution = solutions[j : j + 1]
                self.corrects[i, j] = compare_mazes(prediction, solution)[0]

                # Perform TDA
                diagram, max_distance = get_diagram(
                    latent_series.squeeze(), self.dtype, self.embed_dim, self.delay, self.max_homo, verbose=False
                )
                self.diagrams[i, j, 0] = diagram[0]
                self.diagrams[i, j, 1] = diagram[1]
                self.max_distances[i, j] = max_distance

                # Save results
                self.save()

                self.times[i, j] = time.time() - start_time
                if self.verbose:
                    print(
                        f'    Analyzed latent series for {"correct" if self.corrects[i, j] else "incorrect"} maze {j + 1} of {self.num_mazes} in {self.times[i, j]:.2f}s'
                    )

        if self.verbose:
            print(f'Analysis complete after {np.sum(self.times):.2f}s')

    def get_betti_nums(self, threshold):
        """Get Betti numbers for diagrams with given threshold"""
        betti_nums = np.zeros((len(self.extrap_param), self.num_mazes, self.max_homo + 1), dtype=int)
        for i in range(len(self.extrap_param)):
            for j in range(self.num_mazes):
                betti_nums[i, j] = get_betti_nums(self.diagrams[i, j], threshold)
        return betti_nums

    def print_time(self):
        """Print time for analysis"""
        print(f'Time for analysis: {np.sum(self.times) / 60:.2f}min')
