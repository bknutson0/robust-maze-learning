import os


def get_all_model_names() -> list[str]:
    """Recursively search a folder for all model files ending in .pth."""
    model_names = []
    for root, _, files in os.walk('models'):
        for file in files:
            if file.endswith('.pth'):
                model_names.append(os.path.join(root, file))

    return model_names
