import inspect
import json
import logging
import os

import torch
from torch import nn

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


def load_model(model_name: str, pretrained: str | None = None, weight_init: str | None = None) -> BaseNet:
    """Initialize model and load weights if pretrained. Optionally perform weight initialization."""
    model: BaseNet
    state_dict = None

    # Initialize model
    if 'dt_net' in model_name:
        model = DTNet()
    elif model_name == 'pi_net':
        raise NotImplementedError('PINet model not implemented yet')
    elif 'it_net' in model_name:
        model = ITNet()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model.to(DEVICE)
    model.eval()

    if pretrained is not None and weight_init is not None:
        raise ValueError('Cannot specify both pretrained and weight_init')
    elif pretrained is not None:
        # Load pretrained weights
        if model_name == 'dt_net':
            state_dict = torch.load(pretrained, map_location=DEVICE, weights_only=True)['net']
        else:
            state_dict = torch.load(pretrained, map_location=DEVICE, weights_only=True)
        if state_dict is None:
            raise ValueError(f'Failed to load pretrained weights for model: {model_name}')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
    else:
        # Explicit weight initialization if no pretrained weights
        if weight_init is not None:
            initialize_weights(model, weight_init)

    logger.info(f'Loaded {model_name} to {DEVICE}')
    return model


def initialize_weights(model: nn.Module, scheme: str) -> None:
    """Initialize model weights according to the specified scheme."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d | nn.Linear):
            if scheme == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif scheme == 'xavier':
                nn.init.xavier_normal_(module.weight)
            else:
                raise ValueError(f'Unsupported scheme: {scheme}')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


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

    # Get only valid keys for Hyperparameters
    hyperparams = results['hyperparameters']
    valid_keys = set(inspect.signature(Hyperparameters.__init__).parameters.keys())
    valid_keys.discard('self')
    filtered_hyperparams = {k: v for k, v in hyperparams.items() if k in valid_keys}

    # Load filtered hyperparameters into Hyperparameters object
    return Hyperparameters(**filtered_hyperparams)
