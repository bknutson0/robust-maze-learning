import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.models.base_net import BaseNet
from src.models.dt_net_original import DTNetOriginal
from src.utils.config import Hyperparameters


class DTNet(BaseNet, DTNetOriginal):
    """Original DeepThinking Network 2D model class, but with modifications for convenience.

    Features:
        - Added methods inherited from BaseNet
        - Modified forward method to not return all_outputs
    """

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.name = 'dt_net'

    def input_to_latent(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        latents: torch.Tensor = self.projection(inputs)
        return latents

    def latent_forward(
        self, latents: torch.Tensor, inputs: torch.Tensor, iters: int | list[int] = 1, tolerance: float | None = None
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
        for i in range(1, iters[-1] + 1):
            latents = self.recur_block(torch.cat([latents, inputs], dim=1))
            if i in iters:
                latents_list.append(latents)

        # Return the first element if only one iteration is specified
        if len(iters) == 1:
            return latents_list[0]
        else:
            return latents_list

    def latent_to_output(self, latents: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        if isinstance(latents, list):
            return [self.latent_to_output(latent) for latent in latents]  # type: ignore
        else:
            outputs: torch.Tensor
            if latents.dim() in {3, 4}:
                outputs = self.head(latents)
            else:
                raise ValueError(f'Invalid latents dimension {latents.dim()}, expected 3 or 4.')
            return outputs

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
        """Perform a single training step."""
        self.train()
        optimizer.zero_grad()

        # Compute standard loss "loss_iters"
        latents_initial = self.input_to_latent(inputs)
        latents = self.latent_forward(latents_initial, inputs, iters=hyperparams.iters)
        outputs = self.latent_to_output(latents)

        torch.use_deterministic_algorithms(False)
        loss_iters = criterion(outputs, solutions).mean()
        torch.use_deterministic_algorithms(True)

        # Compute progressive loss "loss_prog"
        loss_prog = 0.0
        if hyperparams.alpha > 0:
            # Sample n ~ U{0, iters - 1}
            n = int(torch.randint(0, hyperparams.iters, (1,)).item())
            # Sample k ~ U{1, iters - n}
            k = int(torch.randint(1, int(hyperparams.iters - n), (1,)).item() if (hyperparams.iters - n) > 1 else 1)

            # Iterate n times w/o gradient tracking
            with torch.no_grad():
                latents = self.latent_forward(latents_initial, inputs, iters=n)

            # Iterate k times with gradient tracking
            latents = (latents - latents_initial).detach() + latents_initial  # Copy latents_initial computation graph
            latents = self.latent_forward(latents, inputs, iters=k)

            # Compute progressive loss
            outputs = self.latent_to_output(latents)
            torch.use_deterministic_algorithms(False)
            loss_prog = criterion(outputs, solutions).mean()
            torch.use_deterministic_algorithms(True)

        # Compute loss gradient
        loss = (1 - hyperparams.alpha) * loss_iters + hyperparams.alpha * loss_prog
        loss.backward()
        if hyperparams.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), hyperparams.grad_clip)

        # Update model parameters
        optimizer.step()

        # Log metrics
        if writer is not None:
            writer.add_scalar('loss/train_batch', loss.item(), int(frac_epoch * 100))
            output_norm = torch.norm(outputs).item()
            writer.add_scalar('output_norm/train_batch', output_norm, int(frac_epoch * 100))
            latent_norm = torch.norm(latents).item()
            writer.add_scalar('latent_norm/train_batch', latent_norm, int(frac_epoch * 100))
            grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])).item()
            writer.add_scalar('grad_norm/train_batch', grad_norm, int(frac_epoch * 100))
            weight_norm = torch.norm(torch.cat([p.view(-1) for p in self.parameters()])).item()
            writer.add_scalar('weight_norm/train_batch', weight_norm, int(frac_epoch * 100))

        return float(loss.item())
