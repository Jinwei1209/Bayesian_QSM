import torch.nn.functional as F
import torch.nn as nn

from models.unet_blocks import *
from models.rsaold1 import rsaBlock as rsa1
# from models.rsa1 import rsaBlock as rsa2
from models.fa import faBlockNew as rsa2
from models.multiHead import multiHeadAttention as rsa3
from models.danet1 import dapBlock


class Unet(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters,
        bilateral=0,
        use_deconv=0,
        use_deconv2=0,
        renorm=0,
        r=1e-5,
        flag_r_train=0,
        flag_rsa=0,
        bilateral_infer=0,
        flag_UTFI=0
    ):

        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.bilateral = bilateral
        self.renorm = renorm
        self.flag_rsa = flag_rsa
        self.bilateral_infer = bilateral_infer
        self.flag_UTFI = flag_UTFI
        if self.flag_rsa == 1:
            self.att = rsa1(self.num_filters[-1])
        elif self.flag_rsa == 2:
            self.att = rsa2(self.num_filters[-1])
        elif self.flag_rsa == 3:
            self.att = rsa3(self.num_filters[-1])
        elif self.flag_rsa == 4:
            self.att = dapBlock(self.num_filters[-1])
            
        # self.r = r
        if flag_r_train:
            self.r = nn.Parameter(torch.Tensor(torch.ones(1)*r), requires_grad=True)
        else:
            self.r = nn.Parameter(torch.Tensor(torch.ones(1)*r), requires_grad=False)
        if self.bilateral:
            self.upsampling_path2 = nn.ModuleList()
            self.output_channels = 1
        
        for i in range(len(self.num_filters)):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.downsampling_path.append(DownConvBlock(input_dim, output_dim, pool=pool))

        for i in range(len(self.num_filters)-2, -1, -1):

            input_dim = self.num_filters[i+1]
            output_dim = self.num_filters[i]

            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, use_deconv=use_deconv))

            if self.bilateral:
                self.upsampling_path2.append(UpConvBlock(input_dim, output_dim, use_deconv=use_deconv2))
        
        self.last_layer = nn.Conv3d(output_dim, self.output_channels, kernel_size=1)
        if self.bilateral:
            self.last_layer2 = nn.Conv3d(output_dim, self.output_channels, kernel_size=1)


    def forward(self, x):

        blocks = []

        for idx, down in enumerate(self.downsampling_path):
            x = down(x)

            if idx != len(self.downsampling_path)-1:
                blocks.append(x)
        if self.flag_rsa != 0:
            x = self.att(x)  
        x1 = x

        for idx, up in enumerate(self.upsampling_path):
            x1 = up(x1, blocks[-idx-1])
        x1 = self.last_layer(x1)

        # x1 = torch.exp(x1)
        # if self.renorm:
        #     device = x1.get_device()
        #     xbar = torch.mean(x1)
        #     r = self.r/xbar
        #     beta = (1-self.r) / (1-xbar)
        #     le = (r<=1).to(device, dtype=torch.float32) 
        #     x1 = le * x1 * r + (1-le) * (1 - (1-x1) * beta)

        if self.bilateral:
            if self.bilateral_infer:
                print('Inference Process of PDI')
                return torch.cat([x1, x1], 1)
            else: 
                x2 = x
                for idx, up in enumerate(self.upsampling_path2):
                    x2 = up(x2, blocks[-idx-1])
                x2 = self.last_layer2(x2)
                if not self.flag_UTFI:
                    x2 = torch.exp(x2)

                if self.renorm:
                    device = x2.get_device()
                    xbar = torch.mean(x2)
                    r = self.r/xbar
                    beta = (1-self.r) / (1-xbar)
                    le = (r<=1).to(device, dtype=torch.float32) 
                    x2 = le * x2 * r + (1-le) * (1 - (1-x2) * beta)

        del blocks

        if self.bilateral:
            return torch.cat([x1, x2], 1)
        else:
            return x1