import torch.nn.functional as F
import torch.nn as nn

from models.unet_blocks import *

class metaCLF(nn.Module):
      
    def __init__(self, dis_cs, in_cs):

        super(metaCLF, self).__init__()
        
        self.encode_coords = nn.Sequential(
            nn.Conv3d(2, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, 1, bias=False),
        )
        
        self.generate_filters = nn.Sequential(
            nn.Conv3d(27, in_cs, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_cs, in_cs*2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_cs*2, in_cs, 1, bias=False),
        )
    
    def kroneckerProduct(self, x, y):
        
        bs, c1, h, w, d = x.shape   
        bs, c2, h, w, d = y.shape   
        
        x = x.view(bs,c1,h*w*d,1).permute(0,2,1,3)
        y = y.view(bs,c2,h*w*d,1).permute(0,2,3,1)
        
        out = torch.matmul(x,y)
        out = out.permute(0,2,3,1)
        out = out.view(bs,c1*c2,h,w,d)
    
        return out    
    
    def forward(self, x, d_all):
        
        x_x = self.encode_coords(d_all[:,0:2,:,:,:])
        y_y = self.encode_coords(d_all[:,2:4,:,:,:])
        z_z = self.encode_coords(d_all[:,4:6,:,:,:])
        
        enced_coords = self.kroneckerProduct(x_x, y_y)
        enced_coords = self.kroneckerProduct(enced_coords, z_z)
        
        filters = self.generate_filters(enced_coords)
        x = (x * filters).sum(dim=1, keepdim=True)

        return x 

class unetVggBNNAR1CLFEnc(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters,
        use_deconv=0,
        use_bn=2
    ):
        super(unetVggBNNAR1CLFEnc, self).__init__()
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
        self.final = metaCLF(dis_cs=6, in_cs=self.num_filters[0])  

    def generateCoordsToBoundary(self, x):
    
        bs, c, h, w, d = x.size()
        ycds, xcds, zcds = np.meshgrid(range(w),range(h),range(d))
        
        d1 = torch.from_numpy(xcds).float().to(x.device) 
        d2 = torch.from_numpy(h-xcds).float().to(x.device)
        d3 = torch.from_numpy(ycds).float().to(x.device)
        d4 = torch.from_numpy(w-ycds).float().to(x.device)
        d5 = torch.from_numpy(zcds).float().to(x.device)
        d6 = torch.from_numpy(d-zcds).float().to(x.device)    

        d_all = torch.stack([d1,d2,d3,d4,d5,d6])

        return d_all.expand(bs, 6, h, w, d)
    
    def forward(self, x):

        distance_tensor = self.generateCoordsToBoundary(x)

        blocks = []
        for idx, down in enumerate(self.downsampling_path):
            x = down(x)
            if idx != len(self.downsampling_path)-1:
                blocks.append(x)

        for idx, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-idx-1])

        x = self.final(x, distance_tensor)
        return x