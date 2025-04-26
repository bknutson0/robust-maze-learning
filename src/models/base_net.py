from abc import abstractmethod

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.models.predictor import Predictor
from src.utils.config import Hyperparameters


class BaseNet(torch.nn.Module, Predictor):
    """Base class for maze networks, containing necessary methods common to all models."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.name = 'base_net'

    @abstractmethod
    def input_to_latent(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        pass

    @abstractmethod
    def latent_forward(
        self, latents: torch.Tensor, inputs: torch.Tensor, iters: int | list[int] = 1, tolerance: float | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """Perform the forward pass in the latent space."""
        pass

    @abstractmethod
    def latent_to_output(self, latents: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        pass

    def output_to_prediction(
        self, outputs: torch.Tensor | list[torch.Tensor], inputs: torch.Tensor, masked: bool = True
    ) -> torch.Tensor:
        """Compute the predictions from the outputs."""
        if isinstance(outputs, list):
            return [self.output_to_prediction(output, inputs, masked) for output in outputs]  # type: ignore
        else:
            predictions: torch.Tensor
            if outputs.dim() == 3:
                unmasked_predictions = torch.argmax(outputs, dim=0)
                if masked:
                    mask, _ = torch.max(inputs, dim=0)
                    predictions = unmasked_predictions * mask
                else:
                    predictions = unmasked_predictions
                return predictions
            if outputs.dim() == 4:
                unmasked_predictions = torch.argmax(outputs, dim=1)
                if masked:
                    mask, _ = torch.max(inputs, dim=1)
                    predictions = unmasked_predictions * mask
                else:
                    predictions = unmasked_predictions
                return predictions
            else:
                raise ValueError(f'Invalid outputs dimension {outputs.dim()}, expected 3 or 4.')

    @abstractmethod
    def train_step(
        self,
        inputs: torch.Tensor,
        solutions: torch.Tensor,
        hyperparams: Hyperparameters,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        frac_epoch: float,
        writer: SummaryWriter | None = None,
    ) -> float:
        """Perform a training step."""
        pass

    def predict(self, inputs: torch.Tensor, iters: int | list[int] = 1) -> torch.Tensor | list[torch.Tensor]:
        """Compute predictions from the inputs, without tracking gradients."""
        latents_initial = self.input_to_latent(inputs)
        latents = self.latent_forward(latents_initial, inputs, iters=iters)
        outputs = self.latent_to_output(latents)
        predictions = self.output_to_prediction(outputs, inputs)

        return predictions
