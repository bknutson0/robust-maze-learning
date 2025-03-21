from src.utils.config import TestParameters
from src.utils.model_loading import get_all_model_names


def main() -> None:
    """Perform test according to test parameters."""
    model_names = get_all_model_names()
    model_names = [name for name in model_names if ('best' in name) and ('2025-03-18_17:52:53' in name)]
    print(model_names)

    # Define test parameters
    test_params = TestParameters()
    test_params.model_name = model_names
    test_params.maze_size = [9]
    test_params.num_mazes = 10000
    test_params.iters = [30]

    # Perform test
    # test(test_params)


if __name__ == '__main__':
    main()
