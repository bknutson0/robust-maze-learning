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
    if pretrained and weight_init:
        raise ValueError('Cannot specify both pretrained and weight_init')

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

    for param in model.parameters():
        param.data.zero_()

    # Initialize model based on its name.
    if 'dt_net' in model_name:
        model = DTNet()
    elif 'it_net' in model_name:
        model = ITNet()
    elif model_name == 'pi_net':
        raise NotImplementedError('PINet model not implemented yet')
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model.to(DEVICE)
    model.eval()

    if pretrained:
        # Load state dict and extract 'net' if available.
        state_dict = torch.load(pretrained, map_location=DEVICE, weights_only=True)
        state_dict = state_dict.get('net', state_dict)
        # Remove potential DataParallel wrapper keys.
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    elif weight_init:
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
