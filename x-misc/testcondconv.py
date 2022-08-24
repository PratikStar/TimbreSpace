import torch
from condconv import CondConv2D
from torch import nn


class Model(nn.Module):
    def __init__(self, num_experts):
        super(Model, self).__init__()
        self.condconv2d = CondConv2D(in_channels=1, out_channels=128, kernel_size=3, num_experts=num_experts, padding=1)

    def forward(self, x):
        x = self.condconv2d(x)
        return x



m = Model(5)



m(torch.randn(1, 1,64,64))