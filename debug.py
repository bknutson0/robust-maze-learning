from src.utils.config import Hyperparameters
from src.utils.maze_loading import load_mazes

hyperparams = Hyperparameters()
hyperparams.percolation = 0.99
hyperparams.num_mazes = 100
inputs, solutions = load_mazes(hyperparams)
print(f'{inputs.shape = }')
print(f'{solutions.shape = }')
