from src.utils.config import Hyperparameters
from src.utils.training import train


def main() -> None:
    """Train the model for different percolation values."""
    # Define hyperparameters
    hyperparams = Hyperparameters()
    hyperparams.model_name = 'it_net'
    hyperparams.train_jfb = False

    # Define percolation values for fine-tuning
    # percolations = [0.000, 0.001, 0.003]  # job:1
    # percolations = [0.01, 0.03, 0.1] # job:2
    # percolations = [0.2, 0.3, 0.4] # job:3
    percolations = [0.5, 0.6, 0.7]  # cuda:0
    # percolations = [0.8, 0.9, 0.99]  # job:4

    # Train the model for each percolation value
    for percolation in percolations:
        hyperparams.percolation = percolation
        train(hyperparams)


if __name__ == '__main__':
    main()
