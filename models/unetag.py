import torch.nn.functional as F
import torch.nn as nn

from models.unet_blocks import *
from models.agnet import GridAttentionBlock3D, UnetGridGatingSignal3, init_weights


class UnetAg(nn.Module):

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
        bilateral_infer=0
    ):

        super(UnetAg, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.conv_path = nn.ModuleList()
        self.down_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.bilateral = bilateral
        self.renorm = renorm
        self.flag_rsa = flag_rsa
        self.bilateral_infer = bilateral_infer
        self.gating = UnetGridGatingSignal3(self.num_filters[4], self.num_filters[3])

        # attention block
        self.attentionblock2 = GridAttentionBlock3D(in_channels=self.num_filters[1], 
                                                    gating_channels=self.num_filters[3],
                                                    inter_channels=self.num_filters[1])
        self.attentionblock3 = GridAttentionBlock3D(in_channels=self.num_filters[2], 
                                                    gating_channels=self.num_filters[3],
                                                    inter_channels=self.num_filters[2])
        self.attentionblock4 = GridAttentionBlock3D(in_channels=self.num_filters[3], 
                                                    gating_channels=self.num_filters[3],
                                                    inter_channels=self.num_filters[3])
            
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

            self.conv_path.append(DownConvBlock(input_dim, output_dim, pool=False))
            if i != len(self.num_filters)-1:
                self.down_path.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))


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
        conv1 = self.conv_path[0](x)
        x = self.down_path[0](conv1)
        conv2 = self.conv_path[1](x)
        x = self.down_path[1](conv2)
        conv3 = self.conv_path[2](x)
        x = self.down_path[2](conv3)
        conv4 = self.conv_path[3](x)
        x = self.down_path[3](conv4)
        x = self.conv_path[4](x)
        gating = self.gating(x)

        # attention
        conv4, _ = self.attentionblock4(conv4, gating)
        conv3, _ = self.attentionblock3(conv3, gating)
        conv2, _ = self.attentionblock2(conv2, gating)

        x = self.upsampling_path[0](x, conv4)
        del conv4
        x = self.upsampling_path[1](x, conv3)
        del conv3
        x = self.upsampling_path[2](x, conv2)
        del conv2
        x = self.upsampling_path[3](x, conv1)
        del conv1

        x = self.last_layer(x)

        return x
