import datetime
import os
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.ndimage import label

from src.utils.config import DEVICE, TestParameters
from src.utils.maze_loading import maze_loaders
from src.utils.model_loading import get_model_hyperparameters, load_model


@torch.no_grad()
def compare_mazes(mazes_1: torch.Tensor, mazes_2: torch.Tensor) -> torch.Tensor:
    """Compare two batches of mazes, and return 1D tensor of boolean values indicating which mazes match."""
    if mazes_1.shape != mazes_2.shape:
        raise ValueError(f'Invalid shapes, must be equal for comparison. Got {mazes_1.shape} and {mazes_2.shape}.')

    return torch.eq(mazes_1, mazes_2).all(dim=tuple(range(1, mazes_1.ndim)))


@torch.no_grad()
def validate_connected_components(predictions: torch.Tensor) -> torch.Tensor:
    """Check path and non-path pixels of predictions each form a single connected component."""
    if predictions.ndim == 2:
        batched = False
        predictions = predictions.unsqueeze(0)  # Add batch dimension if missing
    elif predictions.ndim == 3:
        batched = True
    else:
        raise ValueError(f'Invalid number of dimensions for predictions. Expected 2 or 3, got {predictions.ndim}.')

    predictions_np = predictions.cpu().numpy()

    def check_components(pred: np.ndarray[Any, np.dtype[np.float_]]) -> bool:
        """Check that path and non-path pixels of pred each form a single connected component."""
        _, num_white = label(pred)
        _, num_black = label(1 - pred)
        return bool((num_white == 1) and (num_black == 1))

    result = [check_components(pred) for pred in predictions_np] if batched else [check_components(predictions_np)]

    return torch.tensor(result, dtype=torch.bool, device=predictions.device)


@torch.no_grad()
def is_valid_path(inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Check if predictions are valid paths for inputs mazes."""
    if inputs.dim() != (predictions.dim() + 1):
        raise ValueError(
            f'Invalid number of dimensions for inputs and predictions. '
            f'Expected inputs to have 1 more dimension than predictions, '
            f'got {inputs.dim() = } and {predictions.dim() = }.'
        )

    # If no batch dimension, add
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
        predictions = predictions.unsqueeze(0)
    elif inputs.ndim == 4:
        pass
    else:
        raise ValueError(f'Invalid number of dimensions for inputs. Expected 3 or 4, got {inputs.ndim}.')

    # Compute maze size
    maze_size = (inputs.shape[-1] - 2 * 3) // 2

    # Check that predictions contain start and end positions, indicated by green and red tiles
    red, green, blue = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    green_mask = (red == 0) & (green == 1) & (blue == 0)
    red_mask = (red == 1) & (green == 0) & (blue == 0)

    contains_start = (green_mask * predictions == green_mask).all(dim=(1, 2))
    contains_end = (red_mask * predictions == red_mask).all(dim=(1, 2))

    # Convert non-black pixels to white, and reduce from RGB to black-and-white
    inputs_reduced = torch.any(inputs, dim=1) * 1.0

    # Check that if a prediction pixel does not match the corresponding input pixel, then it is black
    # Predictions can make white pixels black, but not vice versa
    non_matching_pixels_are_black = torch.logical_or(inputs_reduced == predictions, predictions == 0).all(
        dim=tuple(range(1, predictions.ndim))
    )

    # For predictions, check that the pixel at (3+2*n, 3+2*n) matches the pixels at
    # (3+2*n+1,3+2*n+0), (3+2*n+0,3+2*n+1), and (3+2*n+1,3+2*n+1)
    # for all n from 0 to maze_size-1
    i, j = torch.meshgrid(torch.arange(maze_size, device=DEVICE), torch.arange(maze_size, device=DEVICE), indexing='ij')
    base_x, base_y = 3 + 2 * i, 3 + 2 * j
    tiles_are_valid = (
        (predictions[:, base_x, base_y] == predictions[:, base_x + 1, base_y])
        & (predictions[:, base_x, base_y] == predictions[:, base_x, base_y + 1])
        & (predictions[:, base_x, base_y] == predictions[:, base_x + 1, base_y + 1])
    ).all(dim=(1, 2))

    # Check that white pixels have 1 connected component, and black pixels have 1 connected component
    valid_connected_components = validate_connected_components(predictions)

    # Combine all checks and return
    return contains_start & contains_end & non_matching_pixels_are_black & tiles_are_valid & valid_connected_components


@torch.no_grad()
def is_minimal_path(inputs: torch.Tensor, predictions: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
    """Check if predictions are minimal paths for inputs mazes, with same length as solutions."""
    # Check that predictions are valid paths
    are_valid = is_valid_path(inputs, predictions)

    # Check that predictions have same length as solutions
    are_minimal = predictions.sum(dim=(1, 2)) == solutions.sum(dim=(1, 2))

    return are_valid & are_minimal


@torch.no_grad()
def is_correct(inputs: torch.Tensor, predictions: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
    """Determine which predictions exactly match solutions or are also minimal paths."""
    # Mark as correct all predictions that exactly match corresponding solution
    corrects = compare_mazes(predictions, solutions)

    # For predictions that don't exactly match the solution, check if they are alternate solutions
    if not corrects.all():
        corrects[~corrects] = is_minimal_path(inputs[~corrects], predictions[~corrects], solutions[~corrects])

    return corrects


@torch.no_grad()
def specific_test(specific_test_params: TestParameters) -> DataFrame:
    """Run test on models, with specific parameters."""
    result = pd.DataFrame()

    # Check that no test parameter is list, except for iters, model_name, and compare_deadend_model
    for key, value in vars(specific_test_params).items():
        if key not in ('iters', 'model_name') and isinstance(value, list):
            raise ValueError(f'Invalid type for {key}. Expected non-list, got {type(value)}.')

    # Prepare loader
    test_loader = maze_loaders(specific_test_params)

    # Optionally load deadend model
    if getattr(specific_test_params, 'compare_deadend_fill', False):
        deadend_fill = load_model(pretrained='deadend_fill')
    else:
        deadend_fill = None

    # Iterate over models
    for model_name in specific_test_params.model_name:
        model = load_model(pretrained=model_name)
        hyperparameters = get_model_hyperparameters(model_name)
        if isinstance(model, torch.nn.Module):
            model.eval()

        for batch_idx, (inputs, solutions) in enumerate(test_loader):
            current_batch_size = len(inputs)
            maze_indices = list(
                range(
                    batch_idx * specific_test_params.batch_size,
                    batch_idx * specific_test_params.batch_size + current_batch_size,
                )
            )

            # Model predictions for each iter
            predictions = model.predict(inputs, specific_test_params.iters)
            predictions = predictions if isinstance(predictions, list) else [predictions]

            # Deadend model predictions if requested
            if deadend_fill is not None:
                deadend_fill_preds = deadend_fill.predict(inputs)

            # Loop over iteration values
            for iter_index, iter_value in enumerate(specific_test_params.iters):  # type: ignore
                matches_solution = compare_mazes(predictions[iter_index], solutions).cpu().numpy()
                are_valid = is_valid_path(inputs, predictions[iter_index]).cpu().numpy()
                are_minimal = is_minimal_path(inputs, predictions[iter_index], solutions).cpu().numpy()

                # Compare to deadend model if applicable
                if deadend_fill is not None:
                    matches_deadend_fill = compare_mazes(predictions[iter_index], deadend_fill_preds).cpu().numpy()  # type: ignore

                # Build row dict
                row = {
                    'model_name': pd.Series([model_name] * current_batch_size, dtype='string'),
                    'train_maze_size': pd.Series([hyperparameters.maze_size] * current_batch_size, dtype='Int64'),
                    'train_percolation': pd.Series([hyperparameters.percolation] * current_batch_size, dtype='Float64'),
                    'train_deadend_start': pd.Series(
                        [hyperparameters.deadend_start] * current_batch_size, dtype='Int64'
                    ),
                    'train_iter': pd.Series([hyperparameters.iters] * current_batch_size, dtype='Int64'),
                    'maze_index': pd.Series(maze_indices, dtype='Int64'),
                    'test_maze_size': pd.Series([specific_test_params.maze_size] * current_batch_size, dtype='Int64'),
                    'test_percolation': pd.Series(
                        [specific_test_params.percolation] * current_batch_size, dtype='Float64'
                    ),
                    'test_deadend_start': pd.Series(
                        [specific_test_params.deadend_start] * current_batch_size, dtype='Int64'
                    ),
                    'test_iter': pd.Series([iter_value] * current_batch_size, dtype='Int64'),
                    'matches_solution': pd.Series(matches_solution, dtype='bool'),
                    'valid': pd.Series(are_valid, dtype='bool'),
                    'correct': pd.Series(are_minimal, dtype='bool'),
                }
                if deadend_fill is not None:
                    row['matches_deadend_fill'] = pd.Series(matches_deadend_fill, dtype='bool')

                # Append to result
                result = pd.concat([result, pd.DataFrame(row)], ignore_index=True)

    return result


@torch.no_grad()
def test(test_params: TestParameters) -> DataFrame:
    """Run tests with all combinations of parameters on all models, and save results before and during testing."""
    # Create outputs and test directories up front
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(os.path.join('outputs', 'tests'), exist_ok=True)
    test_file = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    test_dir = os.path.join('outputs', 'tests', test_file)
    os.makedirs(test_dir, exist_ok=True)

    # Save test parameters to JSON before testing begins
    test_params_file = os.path.join(test_dir, 'test_parameters.json')
    test_params.to_json(test_params_file)

    # Create a list of specific TestParameters objects for each combination of parameters
    test_params_dict = {k: v if isinstance(v, list) else [v] for k, v in vars(test_params).items()}
    test_params_dict.pop('iters')
    test_params_dict.pop('model_name')
    specific_test_params_list = []
    for values in product(*test_params_dict.values()):
        specific_test_params = TestParameters()
        specific_test_params.model_name = test_params.model_name
        specific_test_params.iters = (
            sorted(test_params.iters) if isinstance(test_params.iters, list) else [test_params.iters]
        )
        for key, value in zip(test_params_dict.keys(), values, strict=False):
            if key not in ['iters', 'model_name']:
                setattr(specific_test_params, key, value)
        specific_test_params_list.append(specific_test_params)

    # File to save partial and final results
    results_file = os.path.join(test_dir, 'results.csv')

    # Run tests and save results after each test iteration
    results = pd.DataFrame()
    for specific_test_params in specific_test_params_list:
        res = specific_test(specific_test_params)
        results = pd.concat([results, res])
        # Save partial results after each test run
        results.to_csv(results_file, index=False)

    return results
