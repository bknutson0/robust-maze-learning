"""Based on https://github.com/aks2203/deep-thinking/blob/main/deepthinking/models/dt_net_2d.py ."""

import torch
import torch.nn.functional as func
from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Basic residual block class 2D."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, group_norm: bool = False) -> None:
        """Initialize the block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the block."""
        out: Tensor = func.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = func.relu(out)
        return out


class DTNetOriginal(nn.Module):
    """Original DeepThinking Network 2D model class, initialized with maze configurations."""

    def __init__(
        self,
        block: type[BasicBlock] = BasicBlock,
        num_blocks: list[int] | None = None,
        width: int = 128,
        in_channels: int = 3,
        recall: bool = True,
        group_norm: bool = False,
    ) -> None:
        """Initialize the model."""
        super().__init__()

        if num_blocks is None:
            num_blocks = [2]

        self.recall: bool = recall
        self.width: int = int(width)
        self.group_norm: bool = group_norm

        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        recur_layers: list[nn.Module] = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.projection: nn.Module = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block: nn.Module = nn.Sequential(*recur_layers)
        self.head: nn.Module = nn.Sequential(head_conv1, nn.ReLU(), head_conv2, nn.ReLU(), head_conv3)

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides: list[int] = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, iters_to_do: int, interim_thought: Tensor | None = None) -> Tensor:
        """Forward pass of the model."""
        _interim_thought: Tensor = interim_thought if interim_thought is not None else self.projection(x)

        all_outputs: Tensor = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3)), device=x.device)

        for i in range(iters_to_do):
            if self.recall:
                _interim_thought = torch.cat([_interim_thought, x], dim=1)
            _interim_thought = self.recur_block(_interim_thought)
            out: Tensor = self.head(_interim_thought)
            all_outputs[:, i] = out

        if self.training:
            raise NotImplementedError('Proper training not implemented for DTNetOriginal.')
            # return out, _interim_thought  # `mypy` may still flag this because return type is ambiguous

        return all_outputs
