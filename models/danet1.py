import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class PAM_Module_3d(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module_3d, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W X D)
            returns :
                out : attention value + input feature
                attention: B X (HxWXD) X (HxWXD)
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*depth)
        energy = torch.bmm(proj_query, proj_key)
        del proj_query, proj_key
        attention = self.softmax(energy)
        del energy
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*depth)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        del proj_value, attention
        out = out.view(m_batchsize, C, height, width, depth)
        out = self.gamma*out + x

        return out


class CAM_Module_3d(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module_3d, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W X D)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        del proj_query, proj_key
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        del energy, energy_new

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        del proj_value, attention
        out = out.view(m_batchsize, C, height, width, depth)
        out = self.gamma*out + x

        return out


class dapBlock(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm3d):

        super(dapBlock, self).__init__()
        #inter_channels = in_channels // 4
        inter_channels = in_channels
        out_channels=in_channels

        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.sa = PAM_Module_3d(inter_channels)
        self.sc = CAM_Module_3d(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(inter_channels, out_channels, 1))

    def forward(self, x):

        sa_conv = self.conv51(self.sa(self.conv5a(x)))

        sc_conv = self.conv52(self.sc(self.conv5c(x)))

        sasc_output = self.conv8(sa_conv + sc_conv)

        return sasc_output

class dasBlock(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm3d):

        super(dasBlock, self).__init__()
        #inter_channels = in_channels // 4
        inter_channels = in_channels
        out_channels=in_channels

        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.sa = PAM_Module_3d(inter_channels)
        self.sc = CAM_Module_3d(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(inter_channels, out_channels, 1))

    def forward(self, x):

        x = self.conv51(self.sa(self.conv5a(x)))

        x = self.conv52(self.sc(self.conv5c(x)))

        x = self.conv8(x)

        return x