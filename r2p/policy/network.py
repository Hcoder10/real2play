"""512-wide residual MLP; training only (torch is not an inference dependency)."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.gelu(x+self.second(self.gelu(self.first(x))))


class StatePolicy(nn.Module):
    def __init__(self, input_dim, width=512, blocks=3):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
        self.output = nn.Linear(width, 15*6)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.gelu(self.input(x))
        for block in self.blocks:
            x = block(x)
        return self.output(x)
