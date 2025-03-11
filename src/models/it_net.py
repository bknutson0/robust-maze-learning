import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.tensorboard.writer import SummaryWriter

from src.models.base_net import BaseNet
from src.utils.config import Hyperparameters


class BasicBlock(nn.Module):
    """Standard residual block with two convolutions (3x3 kernel) and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut: nn.Module = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        out = f.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)  # Residual connection
        return f.relu(out)


class ITNet(BaseNet):
    """Implicit variation of DeepThinking Network 2D model class."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, num_blocks: int = 4, out_channels: int = 2) -> None:
        """Initialize the model."""
        super().__init__()
        self.name = 'it_net'
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.out_channels = out_channels

        # Define three main layers
        self.input_to_latent_layer = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU()
        )
        self.latent_forward_layer = nn.Sequential(
            nn.Conv2d(latent_dim + in_channels, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            *[BasicBlock(latent_dim, latent_dim) for _ in range(num_blocks)],
        )
        self.latent_to_output_layer = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def input_to_latent(self, inputs: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        latents: torch.Tensor
        with torch.no_grad() if not grad else torch.enable_grad():
            latents = self.input_to_latent_layer(inputs)
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
                latents = self.latent_forward_layer(torch.cat([latents, inputs], dim=1))
                if i in iters:
                    latents_list.append(latents)

        # Return the first element if only one iteration is specified
        if len(iters) == 1:
            return latents_list[0]
        else:
            return latents_list

    def latent_to_output(
        self, latents: torch.Tensor | list[torch.Tensor], grad: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        if isinstance(latents, list):
            return [self.latent_to_output(latent, grad) for latent in latents]  # type: ignore
        else:
            outputs: torch.Tensor
            with torch.no_grad() if not grad else torch.enable_grad():
                outputs = self.latent_to_output_layer(latents)
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

        latents_initial = self.input_to_latent(inputs, grad=True)
        latents = latents_initial.detach()

        # Track normed differences and convergence of each batch element, w/o tracking gradients
        diff_norm = torch.full((latents.size(0),), float('inf'))
        converged = torch.zeros_like(diff_norm, dtype=torch.bool)

        with torch.no_grad():
            for i in range(hyperparams.iters - 2):
                latents_prev = latents.clone()

                # Only update elements that haven't converged
                if not converged.all():
                    latents_new = self.latent_forward(
                        latents_initial[~converged], inputs[~converged], iters=1, grad=False
                    )

                    # Update only unconverged latents
                    latents[~converged] = latents_new  # type: ignore

                    # Compute norm differences for unconverged elements across all dimensions except batch dimension
                    diff_norm[~converged] = torch.norm(
                        latents[~converged] - latents_prev[~converged], dim=tuple(range(1, latents.dim()))
                    )

                    # Update convergence status
                    converged |= diff_norm < hyperparams.tolerance

                else:
                    break

        # Final iteration with gradient tracking
        latents = latents_initial + (latents - latents_initial).detach().requires_grad_()
        latents = self.latent_forward(latents, inputs, iters=1, grad=True)

        outputs = self.latent_to_output(latents, grad=True)

        # # Iterate without tracking gradients (and detach computational graph) until convergence
        # if hyperparams.train_jfb and frac_epoch >= hyperparams.warmup:
        #     # Only track gradient for final iteration
        #     latents = latents_initial.detach()
        #     diff_norm = float('inf')

        #     # Iterate one less than maximum, but stop early if converged within tolerance
        #     for i in range(hyperparams.iters - 2):
        #         latents_prev = latents.detach()
        #         latents = self.latent_forward(latents_initial, inputs, iters=1, grad=False)
        #         diff_norm = torch.norm(latents - latents_prev).item()
        #         if diff_norm < hyperparams.tolerance:
        #             break

        #     # Iterate final time with gradient tracking
        #     latents = latents_initial + (latents - latents_initial).detach().requires_grad_()
        #     latents = self.latent_forward(latents, inputs, iters=1, grad=True)

        # else:
        #     # Track gradient for all iterations
        #     # Iterate maximum times, but stop early if converged within tolerance
        #     for i in range(hyperparams.iters - 2):
        #         latents_prev = latents.detach()
        #         latents = self.latent_forward(latents_initial, inputs, iters=1, grad=True)
        #         diff_norm = torch.norm(latents - latents_prev).item()
        #         if diff_norm < hyperparams.tolerance:
        #             break
        # outputs = self.latent_to_output(latents, grad=True)

        # Compute loss
        torch.use_deterministic_algorithms(False)
        loss = criterion(outputs, solutions).mean()
        torch.use_deterministic_algorithms(True)

        # Compute loss gradient
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
