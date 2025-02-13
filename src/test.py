from src.utils.config import TestParameters
from src.utils.model_loading import get_all_model_names
from src.utils.testing import test


def main() -> None:
    """Perform test according to test parameters."""
    # Filter model names, keeping only those that contain 'best' from most recent training runs
    model_names = get_all_model_names()
    model_names = [name for name in model_names if ('best' in name) and (('20250211' in name) or ('20250210' in name))]

    # Define test parameters
    test_params = TestParameters()
    test_params.model_name = model_names
    test_params.num_mazes = 10

    # Perform test
    df = test(test_params)

    # Print accuracy of each model
    for model_name in model_names:
        accuracy = df[df['model_name'] == model_name]['correct'].mean()
        print(f'{model_name}: {accuracy:.2f}')


if __name__ == '__main__':
    main()
