from src.utils.analysis import plot_mazes
from src.utils.config import Hyperparameters
from src.utils.maze_loading import load_mazes
from src.utils.model_loading import load_model


def main() -> None:
    """Main function to explore mazes."""
    # Set parameters
    params = Hyperparameters(
        seed=2,
        maze_size=19,
        num_mazes=20,
        percolation=0.4,
        deadend_start=True,
    )
    # Load mazes
    inputs, solutions = load_mazes(params)

    # Predict
    dt_net = load_model(pretrained='models/dt_net/original.pth')
    dt_predictions = dt_net.predict(inputs, iters=100)

    it_net = load_model(pretrained='models/it_net/2025-03-28_00:07:55/best.pth')
    it_predictions = it_net.predict(inputs, iters=100)

    # Plot predictions
    plot_mazes(inputs, dt_predictions, it_predictions)


if __name__ == '__main__':
    main()
