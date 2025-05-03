import torch

from src.utils.analysis import plot_mazes
from src.utils.config import Hyperparameters
from src.utils.maze_loading import load_mazes
from src.utils.model_loading import load_model
from src.utils.testing import is_correct


def main() -> None:
    """Main function to explore mazes."""
    # Set parameters
    params = Hyperparameters(
        seed=42,
        maze_size=15,
        num_mazes=30,
        percolation=0.21,
        deadend_start=True,
    )
    # Load mazes
    inputs, solutions = load_mazes(params)

    # Predict
    with torch.no_grad():
        dt_net = load_model(pretrained='models/dt_net/original.pth')
        dt_predictions = dt_net.predict(inputs, iters=200)
        print(
            f'DT-Net is correct predictions: {is_correct(inputs, dt_predictions, solutions).sum()}/{len(dt_predictions)}'
        )

        it_net = load_model(pretrained='models/it_net/2025-03-28_00:07:55/best.pth')
        it_predictions = it_net.predict(inputs, iters=200)
        print(
            f'IT-Net is correct predictions: {is_correct(inputs, it_predictions, solutions).sum()}/{len(it_predictions)}'
        )

    # Plot predictions
    plot_mazes([('Input', inputs), ('Solution', solutions), ('DT-Net', dt_predictions), ('IT-Net', it_predictions)])


if __name__ == '__main__':
    main()
