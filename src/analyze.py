from src.utils.model_loading import get_all_model_names, get_model_hyperparameters


def main() -> None:
    """Analyze test results."""
    model_names = get_all_model_names()

    # Filter model names, keeping only those that contain 'best'
    best_models = [name for name in model_names if ('best' in name) and (('20250211' in name) or ('20250210' in name))]

    model = best_models[0]
    hyperparameters = get_model_hyperparameters(model)

    print(hyperparameters)


if __name__ == '__main__':
    main()
