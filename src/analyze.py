from src.utils.model_loading import get_all_model_names


def main() -> None:
    """Analyze test results."""
    model_names = get_all_model_names()

    # Filter model names, keeping only those that contain 'best'
    best_models = [name for name in model_names if ('best' in name) and (('20250211' in name) or ('20250210' in name))]

    print(len(best_models))


if __name__ == '__main__':
    main()
