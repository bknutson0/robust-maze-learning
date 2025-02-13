from abc import ABC, abstractmethod

import torch


class BaseNet(torch.nn.Module, ABC):
    """Base class for maze networks, containing necessary methods common to all models."""

    def __init__(self) -> None:
        """Initialize the model."""
        torch.nn.Module.__init__(self)  # TODO: try changing to super().__init__()
        self.eval()  # Set the model to evaluation mode by default TODO: check if this is necessary
        self.name = 'base_net'

    @abstractmethod
    def input_to_latent(self, inputs: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        pass

    @abstractmethod
    def latent_forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        iters: int | list[int] = 1,
        grad: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Perform the forward pass in the latent space."""
        pass

    @abstractmethod
    def latent_to_output(
        self, latents: torch.Tensor | list[torch.Tensor], grad: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        pass

    @abstractmethod
    def output_to_prediction(
        self, outputs: torch.Tensor | list[torch.Tensor], inputs: torch.Tensor, grad: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compute the predictions from the outputs."""
        pass

    def predict(
        self, inputs: torch.Tensor, iters: int | list[int] = 1, grad: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compute predictions from the inputs."""
        latents_initial = self.input_to_latent(inputs, grad)
        latents = self.latent_forward(latents_initial, inputs, iters, grad)
        outputs = self.latent_to_output(latents, grad)
        predictions = self.output_to_prediction(outputs, inputs, grad)

        return predictions
