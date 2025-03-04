import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.base_net import BaseNet


class ResBlock(nn.Module):
    """Standard residual block with two convolutions (3x3 kernel) and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
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
    def __init__(self, in_channels=3, hidden_dim=64, num_layers=4, time_steps=10):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.name = "it_net"

        # Initial Injection Layer
        self.recur_inj = nn.Conv2d(2 * in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)

        # Residual Blocks for Iterative Updates
        self.layers = nn.ModuleList([ResBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Final Output Mapping
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, in_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, u, x):
        """One iterative step in the implicit process"""
        u = self.recur_inj(torch.cat([u, x], dim=1))  # Concatenate inputs
        for layer in self.layers:
            u = layer(u)  # Apply residual blocks
        return self.output_head(u)  # Final mapping

    def input_to_latent(self, inputs: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """Initialize the latent state from inputs."""
        return inputs.detach().requires_grad_(grad) if grad else inputs.clone()

    def latent_forward(self, latents: torch.Tensor, inputs: torch.Tensor, iters=1, grad=False):
        """Iteratively solve for the latent representation."""
        if isinstance(iters, int):
            iters = [iters]

        outputs = []
        u = latents
        with torch.set_grad_enabled(grad):
            for _ in range(max(iters) - 1):
                u = self.forward(u, inputs)  # Iterative update
                if _ + 1 in iters:
                    outputs.append(u.clone())

        u = u.detach().requires_grad_(grad)
        u = self.forward(u, inputs)
        outputs.append(u)

        return outputs if len(outputs) > 1 else outputs[0]

    def latent_to_output(self, latents, grad=False):
        """Directly return the latent representation as output."""
        return latents

    def output_to_prediction(self, outputs, inputs, grad=False):
        """Convert outputs to predictions (identity in this case)."""
        return outputs
