from contextlib import nullcontext

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

    def input_to_latent(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the latent representation from the inputs."""
        latents: torch.Tensor = self.input_to_latent_layer(inputs)
        return latents

    def latent_forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        iters: int | list[int] = 1,
        tolerance: float | None = None,
        return_extra: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Perform the forward pass in the latent space."""
        tolerance = tolerance if not None else Hyperparameters().tolerance

        # Ensure iters is always a sorted list
        iters = [iters] if isinstance(iters, int) else sorted(iters)

        # Initial latent representation
        latents_initial = latents.clone()
        latents = latents_initial
        latents_list = []

        # Initialize tracking variables for norm differences, convergence, and iterations
        diff_norm = torch.full((latents.size(0),), float('inf'), device=latents.device)
        converged = torch.zeros_like(diff_norm, dtype=torch.bool, device=latents.device)
        iterations = torch.zeros_like(diff_norm, dtype=torch.int32, device=latents.device)

        for i in range(iters[-1]):
            # print(f'Iterating {i + 1}/{iters[-1]} with {converged.sum()} converged')

            latents_prev = latents.clone()

            if not converged.all():
                # Update unconverged latents and iteration counts
                latents[~converged] = self.latent_forward_layer(
                    torch.cat([latents[~converged], inputs[~converged]], dim=1)
                )
                iterations[~converged] = i + 1

                # Compute normed differences for previously unconverged latents
                diff_norm[~converged] = torch.norm(
                    latents[~converged] - latents_prev[~converged], p=2, dim=tuple(range(1, latents.dim()))
                ).detach()

                # Update convergence status
                converged |= diff_norm < tolerance  # type: ignore

            # Save latent representation at specified iterations
            if i + 1 in iters:
                latents_list.append(latents)

        latents_list = latents_list[0] if len(iters) == 1 else latents_list  # type: ignore
        if return_extra:
            return latents_list, iterations, diff_norm  # type: ignore
        else:
            return latents_list

    def latent_to_output(self, latents: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Compute the output from the latent."""
        if isinstance(latents, list):
            return [self.latent_to_output(latent) for latent in latents]  # type: ignore
        else:
            outputs: torch.Tensor = self.latent_to_output_layer(latents)
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

        # Initial latent representation
        latents_initial = self.input_to_latent(inputs)

        # Iterative update loop with optional gradient tracking
        jfb = hyperparams.train_jfb and (frac_epoch >= hyperparams.warmup)
        with torch.no_grad() if jfb else nullcontext():
            latents, iterations, diff_norm = self.latent_forward(
                latents_initial, inputs, iters=hyperparams.iters - 1, tolerance=hyperparams.tolerance, return_extra=True
            )

        # Mildly enforce contraction factor
        if hyperparams.contraction is not None:
            with torch.no_grad():
                # Initialize batch of random pairs of latents
                inputs_noise = torch.randn_like(latents_initial, device=latents_initial.device)
                latents_noise = torch.randn_like(latents_initial, device=latents_initial.device)
                inputs_with_noise = inputs + inputs_noise

                # Compute normed difference before latent forward
                latents_1 = latents_initial.clone()
                latents_2 = latents_initial.clone() + latents_noise
                norm_diff_before = torch.norm(latents_1 , p=2, dim=tuple(range(1, latents_initial.dim())))

                # Compute normed difference after latent forward
                norm_diff_after = torch.norm(
                    self.latent_forward_layer(torch.cat([latents_1, inputs_with_noise], dim=1)) -
                    self.latent_forward_layer(torch.cat([latents_2, inputs_with_noise], dim=1)),
                    p=2, dim=tuple(range(1, latents_initial.dim()))
                )

                # Compute average contraction factor for the batch
                contraction_estimate = (norm_diff_after / norm_diff_before).max().item()
                print(f'{contraction_estimate = }')
                if contraction_estimate > hyperparams.contraction:
                    correction_factor = (hyperparams.contraction / contraction_estimate)**(1 / (2*self.num_blocks + 1)))
                    


        # Final iteration with gradient tracking
        latents = latents_initial + (latents - latents_initial).detach().requires_grad_()
        latents = self.latent_forward_layer(torch.cat([latents, inputs], dim=1))

        # Compute outputs from final latents
        outputs = self.latent_to_output(latents)

        # Compute loss with deterministic algorithms disabled for performance
        torch.use_deterministic_algorithms(False)
        loss = criterion(outputs, solutions).mean()
        torch.use_deterministic_algorithms(True)

        # Backpropagation and gradient clipping (if enabled)
        loss.backward()
        if hyperparams.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), hyperparams.grad_clip)

        # Optimizer step to update model parameters
        optimizer.step()

        # Log metrics to TensorBoard if a writer is provided
        if writer is not None:
            # Log various metrics, starting with lr scheduler learning rate
            writer.add_scalar('lr/train_batch', optimizer.param_groups[0]['lr'], int(frac_epoch * 100))
            writer.add_scalar('jfb/train_batch', int(jfb), int(frac_epoch * 100))
            writer.add_scalar('iterations/mean/train_batch', iterations.float().mean().item(), int(frac_epoch * 100))
            writer.add_scalar('iterations/max/train_batch', iterations.max().item(), int(frac_epoch * 100))
            writer.add_scalar('diff_norm/mean/train_batch', diff_norm.mean().item(), int(frac_epoch * 100))
            writer.add_scalar('diff_norm/max/train_batch', diff_norm.max().item(), int(frac_epoch * 100))
            writer.add_scalar('loss/train_batch', loss.item(), int(frac_epoch * 100))
            writer.add_scalar('output_norm/train_batch', torch.norm(outputs).item(), int(frac_epoch * 100))
            writer.add_scalar('latent_norm/train_batch', torch.norm(latents).item(), int(frac_epoch * 100))
            grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])).item()
            writer.add_scalar('grad_norm/train_batch', grad_norm, int(frac_epoch * 100))
            weight_norm = torch.norm(torch.cat([p.view(-1) for p in self.parameters()])).item()
            writer.add_scalar('weight_norm/train_batch', weight_norm, int(frac_epoch * 100))

        return float(loss.item())
