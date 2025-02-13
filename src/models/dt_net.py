import torch

from src.models.base_net import BaseNet
from src.models.dt_net_original import DTNetOriginal


class DTNet(BaseNet, DTNetOriginal):
    """Original DeepThinking Network 2D model class, but with modifications for convenience.

    Features:
        - Added methods inherited from BaseNet
        - Modified forward method to not return all_outputs
    """

    def __init__(self) -> None:
        """Initialize the model."""
        BaseNet.__init__(self)
        DTNetOriginal.__init__(self)

    def input_to_latent(self, inputs: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        latents: torch.Tensor
        with torch.no_grad() if not grad else torch.enable_grad():
            latents = self.projection(inputs)
        return latents

    def latent_forward(
        self, latents: torch.Tensor, inputs: torch.Tensor, iters: int | list[int] = 1, grad: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """Perform the forward pass in the latent space."""
        # Ensure iters is always a sorted list
        iters = [iters] if isinstance(iters, int) else sorted(iters)
        latents_list = []

        if 0 in iters:
            index_of_last_0_iter = iters[::-1].index(0)
            for _ in range(index_of_last_0_iter + 1):
                latents_list.append(latents)

        # Perform the forward pass for max iterations, saving at specified iterations
        with torch.no_grad() if not grad else torch.enable_grad():
            for i in range(1, iters[-1] + 1):
                latents = self.recur_block(torch.cat([latents, inputs], dim=1))
                if i in iters:
                    latents_list.append(latents)

        # Return the first element if only one iteration is specified
        if len(iters) == 1:
            return latents_list[0]
        else:
            return latents_list

    def latent_to_output(self, latents: torch.Tensor, grad: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        if isinstance(latents, list):
            return [self.latent_to_output(latent, grad) for latent in latents]
        else:
            outputs: torch.Tensor
            with torch.no_grad() if not grad else torch.enable_grad():
                if latents.dim() in {3, 4}:
                    outputs = self.head(latents)
                else:
                    raise ValueError(f'Invalid latents dimension {latents.dim()}, expected 3 or 4.')
            return outputs

    def output_to_prediction(
        self, outputs: torch.Tensor, inputs: torch.Tensor, grad: bool = False, masked: bool = True
    ) -> torch.Tensor:
        """Compute the predictions from the outputs."""
        if isinstance(outputs, list):
            return [self.output_to_prediction(output, inputs, grad, masked) for output in outputs]
        else:
            predictions: torch.Tensor
            with torch.no_grad() if not grad else torch.enable_grad():
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
