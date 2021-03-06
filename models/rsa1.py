import torch
import random
import itertools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter

class rsaBlockMH1(nn.Module):

    def __init__(self, in_channels, num_heads = 16):

        super(rsaBlockMH1, self).__init__()

        self.num_heads = num_heads
        self.atts = nn.ModuleList()
        for _ in range(num_heads):
            self.atts.append(rsaBlock(in_channels))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        outs = Variable(torch.zeros(x.size())).cuda()
        for idx in range(self.num_heads):
            outs = outs + self.atts[idx](x)
        outs = outs / (1.0 * self.num_heads)

        return outs

class rsaBlockMH2(nn.Module):

    def __init__(self, in_channels, num_heads = 16):

        super(rsaBlockMH2, self).__init__()

        self.num_heads = num_heads
        self.atts = nn.ModuleList()
        self.conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        for _ in range(num_heads):
            self.atts.append(rsaBlock(in_channels))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        outs = Variable(torch.zeros(x.size())).cuda()
        for idx in range(self.num_heads):
            outs = outs + self.atts[idx](x)
        outs = self.conv(outs)
        
        return outs

class rsaBlockEnsemble1(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlockEnsemble1, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        self.theta_conv_c = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv_c   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv_c     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()
        
        self.channel_sa   = saChannelBlock()
        self.channel_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa])

    def forward(self, x):

        idx_permutations = list(itertools.permutations(range(3)))
        x_accu = Variable(torch.zeros(x.size())).cuda()

        for idxs in idx_permutations:
            x_curr = x
            for idx in idxs:
            
                theta_x = self.theta_conv(x_curr)
                phi_x = self.phi_conv(x_curr)
                g_x = self.g_conv(x_curr)

                x_curr = self.saBlocks[idx](theta_x, phi_x, g_x)
                del theta_x, phi_x, g_x

            x_accu = x_accu + x_curr
        
        del x
        theta_x = self.theta_conv_c(x_accu)
        phi_x = self.phi_conv_c(x_accu)
        g_x = self.g_conv_c(x_accu)

        x_accu = x_accu + self.channel_sa(theta_x, phi_x, g_x) * self.channel_beta
        del theta_x, phi_x, g_x

        return x_accu

class rsaBlockEnsemble2(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlockEnsemble2, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        self.theta_conv_c = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv_c   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv_c     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()
        
        self.channel_sa   = saChannelBlock()
        self.channel_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa])

    def forward(self, x):

        idx_permutations = list(itertools.permutations(range(3)))
        x_accu = Variable(torch.zeros(x.size())).cuda()

        for idxs in idx_permutations:
            x_curr = x
            for idx in idxs:
            
                theta_x = self.theta_conv(x_curr)
                phi_x = self.phi_conv(x_curr)
                g_x = self.g_conv(x_curr)

                x_curr = self.saBlocks[idx](theta_x, phi_x, g_x)
                del theta_x, phi_x, g_x
            
            theta_x = self.theta_conv_c(x_curr)
            phi_x = self.phi_conv_c(x_curr)
            g_x = self.g_conv_c(x_curr)
            
            x_curr = x_curr + self.channel_sa(theta_x, phi_x, g_x) * self.channel_beta

            x_accu = x_accu + x_curr
        
        return x_accu

class rsaBlock(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlock, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()
        channel_sa  = saChannelBlock()

        coronal_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        sagittal_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        axial_beta    = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        channel_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa, channel_sa])
        self.betas = nn.ParameterList([coronal_beta, sagittal_beta, axial_beta, channel_beta])

    def forward(self, x):

        for idx, saBlock in enumerate(self.saBlocks):

            theta_x = self.theta_conv(x)
            phi_x = self.phi_conv(x)
            g_x = self.g_conv(x)

            x = x + saBlock(theta_x, phi_x, g_x) * self.betas[idx]
            del theta_x, phi_x, g_x

        return x

class rsaBlock1(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlock1, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()
        channel_sa  = saChannelBlock()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, in_channels),
                                   nn.ReLU(inplace = True))
    
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, in_channels),
                                   nn.ReLU(inplace = True))
        
        self.conv_f = nn.Sequential(nn.Dropout3d(0.1, False), 
                                    nn.Conv3d(in_channels, in_channels, 1, 1, 0))

        coronal_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        sagittal_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        axial_beta    = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        channel_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa, channel_sa])
        self.betas = nn.ParameterList([coronal_beta, sagittal_beta, axial_beta, channel_beta])

    def forward(self, x):

        x = self.conv1(x)

        for idx, saBlock in enumerate(self.saBlocks):

            theta_x = self.theta_conv(x)
            phi_x = self.phi_conv(x)
            g_x = self.g_conv(x)

            x = x + saBlock(theta_x, phi_x, g_x) * self.betas[idx]
            del theta_x, phi_x, g_x
        
        x = self.conv2(x)
        x = self.conv_f(x)
        
        return x

class rsaBlock2(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlock2, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()
        channel_sa  = saChannelBlock()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace = True))
        
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace = True))
        
        self.conv_f = nn.Sequential(nn.Dropout3d(0.1, False), 
                                    nn.Conv3d(in_channels, in_channels, 1, 1, 0))

        coronal_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        sagittal_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        axial_beta    = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        channel_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa, channel_sa])
        self.betas = nn.ParameterList([coronal_beta, sagittal_beta, axial_beta, channel_beta])

    def forward(self, x):

        x = self.conv1(x)

        for idx, saBlock in enumerate(self.saBlocks):

            theta_x = self.theta_conv(x)
            phi_x = self.phi_conv(x)
            g_x = self.g_conv(x)

            x = x + saBlock(theta_x, phi_x, g_x) * self.betas[idx]
            del theta_x, phi_x, g_x

        x = self.conv2(x)
        x = self.conv_f(x)
        
        return x

class saChannelBlock(nn.Module):

    def __init__(self):

        super(saChannelBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        ori_size = x_.size() # original size: n * c * d * h * w
        batch_size = ori_size[0]
        channel_size = ori_size[1]

        x_ = x_.contiguous().view(batch_size, channel_size, -1)
        # n * c * dhw
        g_x = g_x.contiguous().view(batch_size, channel_size, -1)
        # n * c * dhw
        x_t = x_t.contiguous().view(batch_size, channel_size, -1)
        # n * c * dhw
        x_t = x_t.permute(0, 2, 1)
        # n * dhw * c

        # energy = torch.matmul(x_, x_t)
        # attention = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = torch.matmul(x_, x_t)
        # n * c * c
        attention = F.softmax(attention, dim = -1)
        # n * c * c
        del x_, x_t
        # del x_, x_t

        # n * c * c X n * c * dhw
        out = torch.matmul(attention, g_x)
        # n * c * dhw
        del attention, g_x
        out = out.view(batch_size, channel_size, *ori_size[2:])
        # n * c * d * h * w

        return out

class saAxialBlock(nn.Module):

    def __init__(self):

        super(saAxialBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0] # original size: n * c * d * h * w
        depth_size = x_.size()[2]

        x_ = x_.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_ = x_.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw

        g_x = g_x.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw

        x_t = x_t.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_t = x_t.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw
        x_t = x_t.permute(0, 2, 1)
        # n * chw * d

        attention = torch.matmul(x_, x_t)
        # n * d * d
        attention = F.softmax(attention, dim = -1)
        # n * d * d
        del x_, x_t

        # n * d * d X n * d * chw
        out = torch.matmul(attention, g_x)
        # n * d * chw
        del attention, g_x
        out = out.view(batch_size, depth_size, *ori_size[2:])
        # n * d * c * h * w
        out = out.permute(0, 2, 1, 3, 4)
        # n * c * d * h * w

        return out


class saCoronalBlock(nn.Module):

    def __init__(self):

        super(saCoronalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        coronal_size = x_.size()[3]

        # n * c * d * h * w
        x_ = x_.permute(0, 3, 2, 1, 4)
        x_ = x_.contiguous().view(batch_size, coronal_size, -1)

        g_x = g_x.permute(0, 3, 2, 1, 4)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, coronal_size, -1)

        x_t = x_t.permute(0, 3, 2, 1, 4)
        x_t = x_t.contiguous().view(batch_size, coronal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, coronal_size, *ori_size[2:])
        out = out.permute(0, 3, 2, 1, 4)

        return out

class saSagittalBlock(nn.Module):

    def __init__(self):

        super(saSagittalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        sagittal_size = x_.size()[4]

        # n * c * d * h * w
        x_ = x_.permute(0, 4, 2, 3, 1)
        x_ = x_.contiguous().view(batch_size, sagittal_size, -1)

        g_x = g_x.permute(0, 4, 2, 3, 1)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, sagittal_size, -1)

        x_t = x_t.permute(0, 4, 2, 3, 1)
        x_t = x_t.contiguous().view(batch_size, sagittal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, sagittal_size, *ori_size[2:])
        out = out.permute(0, 4, 2, 3, 1)

        return out
