import torch
from condconv import CondConv2D
from torch import nn


class Model(nn.Module):
    def __init__(self, num_experts):
        super(Model, self).__init__()
        self.condconv2d = CondConv2D(10, 128, kernel_size=1, num_experts=num_experts)

    def forward(self, x):
        x = self.condconv2d(x)



