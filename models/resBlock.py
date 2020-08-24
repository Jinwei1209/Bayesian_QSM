import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(
        self,
        input_dim, 
        filter_dim,
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1
    ):

        super(ResBlock, self).__init__()
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_dim = filter_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.basicBlock1 = self._basicBlock(self.input_dim, self.filter_dim)
        self.basicBlock2 = self._basicBlock(self.filter_dim, self.filter_dim)
        # self.basicBlock3 = self._basicBlock(self.filter_dim, self.filter_dim)
        # self.basicBlock4 = self._basicBlock(self.filter_dim, self.filter_dim)
        self.basicBlock5 = self._basicBlock(self.filter_dim, self.output_dim)

    def _basicBlock(self, input_dim, output_dim):
        layers = []
        layers.append(nn.Conv3d(
            input_dim, 
            output_dim, 
            self.kernel_size, 
            self.stride, 
            self.padding)
        )
        layers.append(nn.BatchNorm3d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        basicBlock = nn.Sequential(*layers)
        return basicBlock

    def forward(self, x):
        x = self.basicBlock1(x)
        x = self.basicBlock2(x)
        # x = self.basicBlock3(x) + x
        # x = self.basicBlock4(x) + x
        x = self.basicBlock5(x)
        return x

