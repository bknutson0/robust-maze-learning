import json
import logging
import os

import torch

from src.models.base_net import BaseNet
from src.models.dt_net import DTNet
from src.models.it_net import ITNet
from src.utils.config import DEVICE, LOGGING_LEVEL, Hyperparameters

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_model(model_name: str, pretrained: bool = True) -> BaseNet:
    """Initialize model and load weights if pretrained."""
    model: BaseNet
    state_dict = None

    # Initialize model
    if 'dt_net' in model_name:
        model = DTNet()
    elif model_name == 'pi_net':
        raise NotImplementedError('PINet model not implemented yet')
        #     cfg_path = 'models/pi_net/aric/config.yaml'
        #     model_path = 'models/pi_net/aric/model_best_130_100.0.pth'

        #     # Get config dictionary, convert to omega config, and fix attributes
        #     with open(cfg_path) as f:
        #         cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        #     cfg = OmegaConf.create(cfg_dict)
        #     cfg.problem.deq.jacobian_factor = 1.0
        #     cfg.problem.model.model_path = model_path

        #     # Create model and load weights
        #     model = PINet(width=cfg.problem.model.width, in_channels=3, config=cfg)
        #     state_dict = torch.load(model_path, map_location=device, weights_only=True)['net']
    elif 'it_net' in model_name:
        model = ITNet()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # Move model to device and set to eval mode
    model.to(DEVICE)
    model.eval()

    # Load pretrained weights
    if pretrained:
        # Load state dict
        if model_name == 'dt_net':
            state_dict = torch.load('models/dt_net/original.pth', map_location=DEVICE, weights_only=True)['net']
        elif model_name == 'pi_net':
            raise NotImplementedError('PINet model not implemented yet')
        elif model_name == 'it_net':
            state_dict = torch.load('models/it_net/epoch_10.pth', map_location=DEVICE, weights_only=True)
        else:
            state_dict = torch.load(model_name, map_location=DEVICE, weights_only=True)

        # Load state dict into model
        if state_dict is None:
            raise ValueError(f'Failed to load pretrained weights for model: {model_name}')
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=True)

    # Log model loading
    logger.info(f'Loaded {model_name} to {DEVICE}')

    return model


def get_all_model_names() -> list[str]:
    """Recursively search a folder for all model files ending in .pth."""
    model_names = []
    for root, _, files in os.walk('models'):
        for file in files:
            if file.endswith('.pth'):
                model_names.append(os.path.join(root, file))

    return model_names


def get_model_hyperparameters(model_name: str) -> Hyperparameters:
    """Get hyperparameters for a given model."""
    # Load results.JSON file
    results_path = os.path.join(os.path.dirname(model_name), 'results.json')
    with open(results_path) as f:
        results = json.load(f)

    # Load results['hyperparameters'] into Hyperparameters object
    hyperparameters = Hyperparameters(**results['hyperparameters'])

    return hyperparameters
