import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.utils import init_weights

class VAE(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        latent_dim=500,
        # num_filters=[32, 64, 128, 256, 256],
        num_filters=[64, 64, 64, 64, 64],
        use_deconv=False,
        renorm=0,
        r=1e-5,
        flag_r_train=0
    ):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_deconv = use_deconv
        self.path_encoder = nn.ModuleList()
        self.path_decoder = nn.ModuleList()
        self.renorm = renorm
        if flag_r_train:
            self.r = nn.Parameter(torch.Tensor(torch.ones(1)*r), requires_grad=True)
        else:
            self.r = nn.Parameter(torch.Tensor(torch.ones(1)*r), requires_grad=False)

        for i in range(len(self.num_filters)):
            pool = False if i == len(self.num_filters)-1 else True
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]
            self.path_encoder.append(self.res_block(input_dim, output_dim, output_dim))
            self.path_encoder.append(self.down_sampling(output_dim, pool))

        # self.fc_z_mean = nn.Linear(4*4*2*self.num_filters[-1], self.latent_dim)
        # self.fc_z_logvar = nn.Linear(4*4*2*self.num_filters[-1], self.latent_dim)
        # self.fc_z_reshape = nn.Linear(self.latent_dim, 4*4*2*self.num_filters[-1])

        self.mu_conv = nn.Conv3d(self.num_filters[-1], self.num_filters[-1], 3, 1, 1)
        self.logvar_conv = nn.Conv3d(self.num_filters[-1], self.num_filters[-1], 3, 1, 1)

        for i in range(len(self.num_filters)-2, -1, -1):
            act_and_bn = False if i == len(self.num_filters)-2 else True
            input_dim, output_dim = self.num_filters[i+1], self.num_filters[i]
            self.path_decoder.append(self.up_sampling(input_dim, act_and_bn))
            self.path_decoder.append(self.res_block(input_dim, output_dim, output_dim, self.use_deconv))
        
        self.last_layer = self.last_block(output_dim, self.output_channels)
        
    def res_block(
        self, 
        input_channels, 
        inter_channels,
        output_channels,
        deconv=False
    ):
        layers = []
        if deconv:
            layers.append(nn.ConvTranspose3d(input_channels, inter_channels, 3, 1, 1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.ConvTranspose3d(inter_channels, output_channels, 3, 1, 1))
        else:
            layers.append(nn.Conv3d(input_channels, inter_channels, 3, 1, 1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv3d(inter_channels, output_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def down_sampling(self, output_channels, pool=True):
        layers = []
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.BatchNorm3d(output_channels))
        if pool:
            # layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            layers.append(nn.Conv3d(output_channels, output_channels, 3, 2, 1))
        # else:
        #     layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def up_sampling(self, input_channels, act_and_bn=True):
        layers = []
        if act_and_bn:
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.BatchNorm3d(input_channels))
        # else:
        #     layers.append(UnFlatten())
        if self.use_deconv:
            layers.append(nn.ConvTranspose3d(input_channels, input_channels, 2, 2))
        else:
            layers.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layers)

    def last_block(self, input_channels, output_channels):
        layers = []
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.BatchNorm3d(input_channels))
        layers.append(nn.Conv3d(input_channels, input_channels, 3, 1, 1))
        layers.append(nn.Conv3d(input_channels, output_channels, 1))
        return nn.Sequential(*layers)

    def encoder(self, x):
        for i in range(len(self.num_filters)):
            x = self.path_encoder[2*i](x) + x
            # x = self.path_encoder[2*i](x)

            x = self.path_encoder[2*i+1](x)

        # return self.fc_z_mean(x), self.fc_z_logvar(x)
        return self.mu_conv(x), self.logvar_conv(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            # std = torch.exp(0.5*logvar)
            # eps = torch.randn_like(std)
            # z = mu + eps*std
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z

    def decoder(self, z):
        # x = self.fc_z_reshape(z)
        # x = x.view(-1, self.num_filters[-1], 4, 4, 2)
        x = z
        for i in range(len(self.num_filters)-1):
            x = self.path_decoder[2*i](x)

            x = self.path_decoder[2*i+1](x) + x
            # x = self.path_decoder[2*i+1](x)
            
        return self.last_layer(x)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decoder(z)
        x_mu, x_logvar = outputs[:, 0:1, ...], outputs[:, 1:2, ...]
        x_var = torch.exp(x_logvar)
        if self.renorm:
            device = x_var.get_device()
            xbar = torch.mean(x_var)
            r = self.r/xbar
            beta = (1-self.r) / (1-xbar)
            le = (r<=1).to(device, dtype=torch.float32) 
            x_var = le * x_var * r + (1-le) * (1 - (1-x_var) * beta)
        return x_mu, x_var, mu, logvar

# class UnFlatten(nn.Module):
#     def forward(self, input, input_channels=256):
#         return input.view(input.size(0), input_channels, 4, 4, 2)

