import logging

import torch

from src.models.base_net import BaseNet
from src.models.dt_net import DTNet
from src.utils.config import DEVICE, LOGGING_LEVEL

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_model(model_name: str) -> BaseNet:
    """Load the saved model onto device."""
    # Initialize model and load weights
    model: BaseNet
    state_dict = None

    if model_name == 'dt_net':
        model = DTNet()
        state_dict = torch.load('models/dt_net/original.pth', map_location=DEVICE, weights_only=True)['net']

    elif 'dt_net' in model_name:
        model = DTNet()
        state_dict = torch.load(f'models/{model_name}', map_location=DEVICE, weights_only=True)['net']

    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # TODO: add other models
    # Otherwise, if first 6 characters are 'dt_net', load corresponding model
    # elif model_name[:6] == 'dt_net':
    #     model = DTNet()
    #     state_dict = torch.load(f'models/dt_net/{model_name}.pth', map_location=DEVICE, weights_only=True)

    # elif model_name == 'it_net':
    #     model = ITNet()
    #     state_dict = torch.load('models/it_net.pth', map_location=device, weights_only=True)['net']

    # elif model_name == 'pi_net':
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

    # Fix state_dict keys
    if state_dict is not None:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        raise ValueError(f'Failed to load model {model_name}, state_dict is None')

    # Load weights to model
    model.to(DEVICE)
    model.eval()
    model.load_state_dict(new_state_dict, strict=True)

    # Log model loading
    logger.info(f'Loaded {model_name} to {DEVICE}')

    return model
