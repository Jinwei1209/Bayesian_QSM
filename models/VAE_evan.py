class res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        return out
class Encoder(nn.Module):
    def __init__(self, vae, batchnorm= True):
        super(Encoder, self).__init__()
        self.vae = vae
        self.relu = nn.ELU()
        self.batchnorm = batchnorm
        self.bn = nn.BatchNorm3d(16)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        
        self.conv = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        
        self.block = res_block(16,16)
        self.block1 = res_block(32,32)
        self.block2 = res_block(64,64)
        self.block3 = res_block(64,64)
        
        self.down = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.down1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.bottleneck = nn.Conv3d(64,64, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv(x) 
        out = self.block(out)
        if self.batchnorm:
            out = self.bn(out)
        out = self.down(out)
        
        out = self.block1(out)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.down1(out)
        
        out = self.block2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.down2(out)
        
        out = self.block3(out)
        if self.batchnorm:
            out = self.bn3(out)
        
        out = self.bottleneck(out)
        
        return  out    
    
class Decoder(nn.Module):
    def __init__(self, vae, batchnorm= True):
        super(Decoder, self).__init__()
        self.activation = nn.ELU()
        self.vae = vae
        self.batchnorm = batchnorm
        
        self.bottleneck = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block = res_block(64, 64)
        self.block1 = res_block(64,64)
        self.block2 = res_block(32,32)
        self.block3 = res_block(16,16)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=(0,1,1), output_padding=(0,1,1))
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=(1,0,0), output_padding=(1,0,0))
        self.up3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=(1,0,0), output_padding=(1,0,0))
        
        self.bn = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(64)        
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(16)
        self.conv1 = nn.Conv3d(16, 1, kernel_size= 1, stride=1, padding=0)
        
        self.mu_conv = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.logvar_conv = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        
        if self.vae:
            dim = x.size()
            mu = self.mu_conv(x)
            mu = mu.view(mu.size(0), -1)
            log_var = self.logvar_conv(x)
            log_var = log_var.view(log_var.size(0), -1)
            z = self.reparameterize(mu, log_var)
            z = z.view(dim[0], dim[1], dim[2], dim[3], dim[4])
        else: 
            z = x
            mu = []
            log_var = []
        out = self.block(self.bottleneck(z))
        if self.batchnorm:
            out = self.bn(out)
        
        out = self.up1(out)
        out = self.block1(out)
        if self.batchnorm:
            out = self.bn1(out)
        
        out = self.up2(out)
        out = self.block2(out)
        if self.batchnorm:
            out = self.bn2(out)
        
        out = self.up3(out)
        out = self.block3(out)
        if self.batchnorm:
            out = self.bn3(out)
        
        out = self.conv1(out)
        return out, mu, log_var
