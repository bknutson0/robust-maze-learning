from src.utils.config import Hyperparameters
from src.utils.training import train


def main() -> None:
    """Train the model for different percolation values."""
    # Define hyperparameters
    hyperparams = Hyperparameters()

    # Modifications for training it_net
    hyperparams.model_name = 'it_net'
    hyperparams.iters = 100
    # hyperparams.num_mazes = 1000
    # hyperparams.warmup = 10
    hyperparams.train_jfb = True
    # hyperparams.learning_rate = 0.00001
    hyperparams.batch_size = 32

    # Define percolation values for fine-tuning
    # percolations = [0.000, 0.001, 0.003, 0.01]  # job:
    # percolations = [0.03, 0.1, 0.2, 0.3] # job:
    percolations = [0.4, 0.5, 0.6, 0.7]  # cuda:2
    # percolations = [0.8, 0.9, 0.99]  # cuda:

    # Train the model for each percolation value
    for percolation in percolations:
        hyperparams.percolation = percolation
        train(hyperparams)


if __name__ == '__main__':
    main()
