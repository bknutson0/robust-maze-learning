import torch
from torch import Tensor, nn
from torch.utils.tensorboard.writer import SummaryWriter

from src.models.base_net import BaseNet
from src.models.dt_net_original import BasicBlock
from src.utils.config import Hyperparameters


class FFNet(BaseNet):
    """Feedforward Network 2D model class, analog to DTNet without recurrence."""

    def __init__(self, in_channels: int = 3, width: int = 128, group_norm: bool = False) -> None:
        """Initialize the model."""
        super().__init__()
        self.name = 'ff_net'

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU()
        )

        layers = []
        for _ in range(30):
            layers.append(BasicBlock(width, width, stride=1, group_norm=group_norm))
        self.feedforward_blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def input_to_latent(self, inputs: Tensor) -> Tensor:
        """Compute the latent representation from the inputs."""
        return self.projection(inputs)  # type: ignore

    def latent_forward(
        self, latents: Tensor, inputs: Tensor, iters: int | list[int] = 1, tolerance: float | None = None
    ) -> Tensor:
        """Perform the forward pass through the feedforward blocks."""
        return self.feedforward_blocks(latents)  # type: ignore

    def latent_to_output(self, latents: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        """Compute the output from the latent."""
        if isinstance(latents, list):
            return [self.latent_to_output(latent) for latent in latents]  # type: ignore
        else:
            return self.head(latents)  # type: ignore

    def train_step(
        self,
        inputs: Tensor,
        solutions: Tensor,
        hyperparams: Hyperparameters,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        frac_epoch: float,
        writer: SummaryWriter | None = None,
    ) -> float:
        """Perform a single training step."""
        self.train()
        optimizer.zero_grad()

        latents = self.input_to_latent(inputs)
        latents = self.latent_forward(latents, inputs)
        outputs = self.latent_to_output(latents)

        torch.use_deterministic_algorithms(False)
        loss = criterion(outputs, solutions).mean()
        torch.use_deterministic_algorithms(True)
        loss.backward()

        if hyperparams.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), hyperparams.grad_clip)

        optimizer.step()

        if writer is not None:
            writer.add_scalar('loss/train_batch', loss.item(), int(frac_epoch * 100))

        return float(loss.item())
