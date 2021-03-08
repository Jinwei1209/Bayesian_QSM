import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.unet_blocks import *

class metaCFT(nn.Module):

    def __init__(self, in_cs):

        super(metaCFT, self).__init__()

        self.generate_Means = nn.Sequential(
            nn.Conv3d(6, 64, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, in_cs, 1, bias=True),
        )

        self.generate_Sigmas = nn.Sequential(
            nn.Conv3d(6, 64, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, in_cs, 1, bias=True),
        )
        

        self.normal_filter = nn.Conv3d(in_cs,1,1,1,0,bias=False)

    def generateCoordsToBoundary(self, x):

        bs, _, h, w, d = x.size()
        ycds, xcds, zcds = np.meshgrid(range(w),range(h),range(d))
        
        d1 = torch.from_numpy(xcds).float().to(x.device) 
        d2 = torch.from_numpy(h-xcds-1.).float().to(x.device)
        d3 = torch.from_numpy(ycds).float().to(x.device)
        d4 = torch.from_numpy(w-ycds-1.).float().to(x.device)
        d5 = torch.from_numpy(zcds).float().to(x.device)
        d6 = torch.from_numpy(d-zcds-1.).float().to(x.device)

        d_all = torch.stack([d1,d2,d3,d4,d5,d6])

        return d_all.expand(bs, 6, h, w, d)

    def forward(self, x):

        d_all = self.generateCoordsToBoundary(x)
        means = self.generate_Means(d_all)
        sigmas = self.generate_Sigmas(d_all)
        
        x = x*sigmas + means
        x = self.normal_filter(x)
        
        return x 


class unetCFTMuSig256(nn.Module):
    def __init__(self,
        input_channels,
        output_channels,
        num_filters,
        use_deconv=0,
        use_bn=2
    ):
        super(unetCFTMuSig256, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()

        # down-sampling
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]
            if i == 0:
                pool = False
            else:
                pool = True
            self.downsampling_path.append(DownConvBlock(input_dim, output_dim, use_bn=use_bn, pool=pool))
        # up-sampling
        for i in range(len(self.num_filters)-2, -1, -1):
            input_dim = self.num_filters[i+1]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, use_bn=use_bn, use_deconv=use_deconv))

        # final conv (without any concat)
        self.final = metaCFT(self.num_filters[0])
        
    def forward(self, x):

        blocks = []
        for idx, down in enumerate(self.downsampling_path):
            x = down(x)
            if idx != len(self.downsampling_path)-1:
                blocks.append(x)

        for idx, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-idx-1])

        x = self.final(x)

        return x
