from src.utils.config import Hyperparameters
from src.utils.training import train


def main() -> None:
    """Train the model for different percolation values."""
    # Define hyperparameters
    hyperparams = Hyperparameters()
    hyperparams.num_mazes = 100000
    hyperparams.batch_size = 256
    hyperparams.grad_clip = 10.0
    hyperparams.learning_rate = 10 ** (-3)

    # Define percolation values for fine-tuning
    percolations = [0.000, 0.001, 0.003, 0.01, 0.03, 0.1]  # cuda:0
    percolations = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]  # cuda:1

    # Train the model for each percolation value
    for percolation in percolations:
        hyperparams.percolation = percolation
        train(hyperparams)


if __name__ == '__main__':
    main()
