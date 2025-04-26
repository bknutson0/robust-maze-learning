from abc import ABC, abstractmethod

import numpy as np
import torch
from maze_dataset import TargetedLatticeMaze
from maze_dataset.maze.lattice_maze import _remove_isolated_cells


class Model(ABC):
    """Base class for predictors, containing necessary methods common to all models."""

    def __init__(self) -> None:
        """Initialize the predictor."""
        self.name = 'predictor'

    @abstractmethod
    def predict(self, inputs: torch.Tensor, iters: int | list[int] = 1) -> torch.Tensor | list[torch.Tensor]:
        """Make predictions based on the inputs."""
        pass


class DeadendFill(Model):
    """Predictor that solves mazes using the deadend-filling algorithm."""

    def __init__(self) -> None:
        """Initialize the predictor."""
        super().__init__()
        self.name = 'deadend_fill'

    def predict(self, inputs: torch.Tensor, iters: int | list[int] = 1) -> torch.Tensor | list[torch.Tensor]:
        """Solve maze using deadend-filling algorithm."""
        # Check if the input is batched or single
        is_batched = inputs.ndim == 4

        # If single input, add a batch dimension
        if not is_batched:
            inputs = inputs.unsqueeze(0)  # Add batch dimension at axis 0

        # Convert input to pixel grid
        pixel_grid = inputs.cpu().numpy()  # Convert inputs to numpy
        pixel_grid = pixel_grid.transpose(0, 2, 3, 1)  # Move RGB axis to last axis (batch-first format)
        pixel_grid = (pixel_grid * 255).astype(np.uint8)  # Convert values from float (0.0-1.0) to int (0-255)
        pixel_grid = pixel_grid[:, 1:-1, 1:-1]  # Reduce padding from 3 to 2 (for all batches)
        pixel_grid = pixel_grid[:, ::2, ::2]  # Select even pixels (for all batches)

        predictions = []

        for grid in pixel_grid:
            # Convert pixel grid to TargetedLatticeMaze
            targeted_lattice_maze = TargetedLatticeMaze.from_pixels(grid)

            # Backfill the maze
            deadend_filled_maze = self._deadend_fill_maze(targeted_lattice_maze)  # type: ignore
            prediction = _remove_isolated_cells(deadend_filled_maze.as_pixels())

            # Reformat solution
            prediction = prediction.astype(np.float32) / 255.0  # Convert values from int (0-255) to float (0.0-1.0)
            prediction = prediction.max(axis=2)  # Convert from RGB to black and white
            prediction = np.repeat(prediction, 2, axis=0)  # Double pixels in x-axis
            prediction = np.repeat(prediction, 2, axis=1)  # Double pixels in y-axis
            prediction = np.pad(
                prediction, pad_width=1, mode='constant', constant_values=0
            )  # Add 1 pixel of black (value 0) padding

            predictions.append(prediction)

        # Stack predictions along batch dimension
        predictions = np.stack(predictions, axis=0)

        # Convert back to tensor and send to the appropriate device
        predictions = torch.from_numpy(predictions).to(inputs.device)  # type: ignore

        # If the original input was single, squeeze the batch dimension
        if not is_batched:
            predictions = predictions.squeeze(0)  # type: ignore

        return predictions  # type: ignore

    def _deadend_fill_maze(self, maze: TargetedLatticeMaze) -> TargetedLatticeMaze:
        """Iteratively remove single-neighbor nodes not connected to start or end, until no more can be removed."""
        # Create a copy of the connection list to modify
        new_connection_list = maze.connection_list.copy()

        # Create a temporary TargetedLatticeMaze object for using its methods
        temp_maze = TargetedLatticeMaze(
            connection_list=new_connection_list,
            start_pos=maze.start_pos,
            end_pos=maze.end_pos,
        )

        changed = True
        while changed:
            changed = False
            for i in range(maze.grid_shape[0]):
                for j in range(maze.grid_shape[1]):
                    pos = (i, j)
                    if self._should_remove_node(temp_maze, pos):
                        self._remove_node(new_connection_list, pos)
                        changed = True

            # Update the temporary maze with the new connection list
            temp_maze = TargetedLatticeMaze(
                connection_list=new_connection_list,
                start_pos=maze.start_pos,
                end_pos=maze.end_pos,
            )

        return TargetedLatticeMaze(
            connection_list=new_connection_list,
            start_pos=maze.start_pos,
            end_pos=maze.end_pos,
        )

    def _should_remove_node(self, maze: TargetedLatticeMaze, pos: tuple[int, int]) -> bool:
        """Check if a node should be removed."""
        if pos == tuple(maze.start_pos) or pos == tuple(maze.end_pos):
            return False

        neighbors = maze.get_coord_neighbors(np.array(pos))
        return len(neighbors) == 1

    def _remove_node(self, connection_list: np.ndarray[np.bool_, np.dtype[np.bool_]], pos: tuple[int, int]) -> None:
        """Remove a node by disconnecting all its connections."""
        i, j = pos

        # Remove up connection
        if i > 0:
            connection_list[0, i - 1, j] = False
        # Remove down connection
        if i < connection_list.shape[1] - 1:
            connection_list[0, i, j] = False
        # Remove left connection
        if j > 0:
            connection_list[1, i, j - 1] = False
        # Remove right connection
        if j < connection_list.shape[2] - 1:
            connection_list[1, i, j] = False
