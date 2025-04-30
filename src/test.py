from src.utils.config import TestParameters
from src.utils.model_loading import get_all_model_names
from src.utils.testing import test


def main() -> None:
    """Perform test according to test parameters."""
    model_names = get_all_model_names()

    # dt-net
    # Original dt-net model
    # model_names = ['models/dt_net/original.pth']
    # Pre-trained dt-net models (02-26 to 02-28)
    # model_names = [name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-02-2' in name)]
    # Random initial weights dt-net models (03-01 to 03-03)
    # model_names = [name for name in model_names if ('it_net' in name) and ('best' in name) and ('2025-03-0' in name)]
    # Random initial weights and progressive learning dt-net models (03-11 to 03-13)
    # model_names = [name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-03-1' in name)]
    # Random initial weights without gradient clipping or progressive learning dt-net models (04-17 to 04-18)
    # model_names = [name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-04-1' in name)]

    # it-net
    # Random iters, JFB, without contraction it-net models (03-25 to 03-26)
    # model_names = [
    #     name
    #     for name in model_names
    #     if ('it_net' in name) and ('best' in name) and ('2025-03-25' in name or '2025-03-26' in name)
    # ]
    # Random iters, JFB, with contraction it-net models (03-27 to 03-28)
    model_names = [
        name
        for name in model_names
        if ('it_net' in name) and ('best' in name) and ('2025-03-27' in name or '2025-03-28' in name)
    ]

    # ff-net
    # model_names = [
    #     name
    #     for name in model_names
    #     if ('ff_net' in name) and ('best' in name) and ('2025-04-20' in name or '2025-04-21' in name)
    # ]

    # Define test parameters
    test_params = TestParameters()
    test_params.model_name = sorted(model_names)
    test_params.maze_size = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    test_params.num_mazes = 1000
    test_params.iters = [30, 60, 100, 200]
    test_params.compare_deadend_fill = True

    # Perform test
    print('Testing the following models:')
    for name in test_params.model_name:
        print(name)
    test(test_params)


if __name__ == '__main__':
    main()
