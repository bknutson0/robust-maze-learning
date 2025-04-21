from src.utils.config import Hyperparameters
from src.utils.training import train


def main() -> None:
    """Train the model for different percolation values."""
    # Define hyperparameters
    hyperparams = Hyperparameters()

    # Modifications for training it_net
    hyperparams.model_name = 'ff_net'
    hyperparams.pretrained = None  #'models/it_net/2025-03-23_12:27:31/checkpoints/epoch_10.pth'
    hyperparams.iters = 30
    hyperparams.num_mazes = int(1e5)
    hyperparams.random_iters = False
    hyperparams.contraction = None
    hyperparams.train_jfb = False
    hyperparams.learning_rate = 1e-4
    hyperparams.batch_size = 32
    hyperparams.grad_clip = None
    hyperparams.alpha = 0.0  # No progressive loss factor
    hyperparams.tolerance = 1e-1
    hyperparams.warmup_epochs = 0
    hyperparams.warmup_iters = 30
    hyperparams.patience = 10

    # Define percolation values for fine-tuning
    percolations = [0.000, 0.001, 0.003, 0.01]  # job:1
    # percolations = [0.03, 0.1, 0.2, 0.3]  # job:2
    # percolations = [0.4, 0.5, 0.6, 0.7]  # job:3
    # percolations = [0.8, 0.9, 0.99]  # job:4

    # Train the model for each percolation value
    for percolation in percolations:
        hyperparams.percolation = percolation
        train(hyperparams)


if __name__ == '__main__':
    main()
