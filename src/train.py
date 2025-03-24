from src.utils.config import Hyperparameters
from src.utils.training import train


def main() -> None:
    """Train the model for different percolation values."""
    # Define hyperparameters
    hyperparams = Hyperparameters()

    # Modifications for training it_net
    hyperparams.model_name = 'it_net'  # Use as warmup
    # hyperparams.pretrained = 'models/it_net/2025-03-23_12:27:31/checkpoints/epoch_10.pth'
    hyperparams.iters = 100
    hyperparams.num_mazes = int(1e4)
    hyperparams.random_iters = False
    hyperparams.contraction = False  # Try out no weak enforcement of contraction
    hyperparams.train_jfb = True
    hyperparams.learning_rate = 1e-3
    hyperparams.batch_size = 32
    hyperparams.grad_clip = None
    hyperparams.tolerance = 1e-1  # TODO: adjust
    hyperparams.warmup_epochs = 10
    hyperparams.warmup_iters = 10
    hyperparams.patience = 10  # TODO: adjust

    # Define percolation values for fine-tuning
    # percolations = [0.000, 0.001, 0.003, 0.01]  # job:
    # percolations = [0.03, 0.1, 0.2, 0.3] # job:
    # percolations = [0.4, 0.5, 0.6, 0.7]  # cuda:2
    # percolations = [0.8, 0.9, 0.99]  # cuda:
    percolations = [0.4]

    # Train the model for each percolation value
    for percolation in percolations:
        hyperparams.percolation = percolation
        train(hyperparams)


if __name__ == '__main__':
    main()
