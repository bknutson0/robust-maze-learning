import inspect
import json
import logging
import os

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

from src.models.dt_net import DTNet
from src.models.ff_net import FFNet
from src.models.it_net import ITNet
from src.models.model import DeadendFill, Model
from src.models.pi_net import PINet
from src.utils.config import DEVICE, LOGGING_LEVEL, Hyperparameters

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_model(model_name: str | None = None, pretrained: str | None = None, weight_init: str | None = None) -> Model:
    """Initialize model and load weights from pretrained path or initialize weights."""
    model: Model

    if pretrained and weight_init:
        raise ValueError('Cannot specify both pretrained and weight_init.')

    # Infer model_name from the pretrained path if not explicitly provided.
    if model_name is None:
        if pretrained is None:
            raise ValueError('Either model_name or pretrained path must be provided.')
        # Attempt to infer model type from the pretrained path.
        if 'dt_net' in pretrained:
            model_name = 'dt_net'
        elif 'it_net' in pretrained:
            model_name = 'it_net'
        elif 'ff_net' in pretrained:
            model_name = 'ff_net'
        elif 'deadend_fill' in pretrained:
            model_name = 'deadend_fill'
        elif 'pi_net' in pretrained:
            model_name = 'pi_net'
        else:
            raise ValueError('Could not infer model type from pretrained path.')

    if model_name == 'pi_net':
        if weight_init:
            raise ValueError('weight_init is not supported for pi_net; please use pretrained.')

        if not pretrained:
            raise ValueError('pi_net requires a pretrained .pth path.')

        # derive config path alongside the .pth
        cfg_path = os.path.join(os.path.dirname(pretrained), 'config.yaml')
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f'Config file not found at {cfg_path}')

        # load and patch config
        with open(cfg_path) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg_dict)
        cfg.problem.deq.jacobian_factor = 1.0
        cfg.problem.model.model_path = pretrained

        # instantiate with width & config
        model = PINet(width=cfg.problem.model.width, in_channels=3, config=cfg)
        # load 'net' weights
        sd = torch.load(pretrained, map_location=DEVICE, weights_only=True).get('net', {})
        # strip DataParallel keys
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

        model.to(DEVICE)
        logger.info(f'Loaded pi_net from {pretrained} to device: {DEVICE}')
        return model

    # Initialize model
    if 'dt_net' in model_name:
        model = DTNet()
    elif 'it_net' in model_name:
        model = ITNet()
    elif 'ff_net' in model_name:
        model = FFNet()
    elif 'pi_net' in model_name:
        model = PINet()
    elif 'deadend_fill' in model_name:
        model = DeadendFill()
        return model
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model.to(DEVICE)

    if pretrained:
        # Load state dict and extract 'net' if available.
        state_dict = torch.load(pretrained, map_location=DEVICE, weights_only=True)
        state_dict = state_dict.get('net', state_dict)
        # Remove potential DataParallel wrapper keys.
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    elif weight_init:
        initialize_weights(model, weight_init)

    logger.info(f'Loaded model: {model_name} from {pretrained if pretrained else "scratch"} to device: {DEVICE}')
    return model


def initialize_weights(model: nn.Module, scheme: str) -> None:
    """Initialize model weights according to the specified scheme."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d | nn.Linear):
            if scheme == 'zero':
                nn.init.zeros_(module.weight)
            elif scheme == 'kaiming':
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
    results_path = os.path.join(os.path.dirname(model_name), 'results.json')

    # Algorithm case: no JSON ⇒ override only the four training fields to NaN
    if not os.path.exists(results_path):
        hp = Hyperparameters()
        hp.maze_size = np.nan  # type: ignore
        hp.percolation = np.nan
        hp.deadend_start = np.nan  # type: ignore
        hp.iters = np.nan  # type: ignore
        return hp

    # Model case: load JSON as before
    with open(results_path) as f:
        results = json.load(f)

    hyperparams = results['hyperparameters']
    valid_keys = set(inspect.signature(Hyperparameters.__init__).parameters.keys())
    valid_keys.discard('self')
    filtered = {k: v for k, v in hyperparams.items() if k in valid_keys}

    return Hyperparameters(**filtered)
