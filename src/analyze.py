from src.utils.analys import get_all_model_names


def main() -> None:
    """Analyze test results."""
    model_names = get_all_model_names()

    # Filter model names, keeping only those that contain 'best'
    best_models = [name for name in model_names if 'best' in name]

    # Filter model names, keeping only those that

    print(f'{best_models = }')


if __name__ == '__main__':
    main()
