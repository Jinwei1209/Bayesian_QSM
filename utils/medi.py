import numpy as np
import torch.fft as fft
import torch
PRECISION = 'float32'
EPS = 1E-8


# dipole kernel in Fourier or image space
def dipole_kernel(matrix_size, voxel_size, B0_dir, Fourier_space=1):

    if Fourier_space:
        x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
        y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
        z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
        Y, X, Z = np.meshgrid(x, y, z)
        
        X = X/(matrix_size[0]*voxel_size[0])
        Y = Y/(matrix_size[1]*voxel_size[1])
        Z = Z/(matrix_size[2]*voxel_size[2])
        
        D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/(X**2 + Y**2 + Z**2)
        D[np.isnan(D)] = 0
        D = np.fft.fftshift(D)

    else:
        x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
        y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
        z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
        Y, X, Z = np.meshgrid(x, y, z)

        X = X*voxel_size[0]
        Y = Y*voxel_size[1]
        Z = Z*voxel_size[2]

        d = (3*( X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2 - X**2-Y**2-Z**2)/(4*math.pi*(X**2+Y**2+Z**2)**2.5)
        d[np.isnan(d)] = 0

    return D

# sphere_kernel to do filtering 
def sphere_kernel(matrix_size,voxel_size, radius):
    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X*voxel_size[0]
    Y = Y*voxel_size[1]
    Z = Z*voxel_size[2]
    
    Sphere_out = (np.maximum(abs(X) - 0.5*voxel_size[0], 0)**2 + np.maximum(abs(Y) - 0.5*voxel_size[1], 0)**2 
                  + np.maximum(abs(Z) - 0.5*voxel_size[2], 0)**2) > radius**2
    
    Sphere_in = ((abs(X) + 0.5*voxel_size[0])**2 + (abs(Y) + 0.5*voxel_size[1])**2 
                  + (abs(Z) + 0.5*voxel_size[2])**2) <= radius**2
    
    Sphere_mid = np.zeros(matrix_size)
    
    split = 10  #such that error is controlled at <1/(2*10)
    
    x_v = np.arange(-split+0.5, split+0.5, 1)
    y_v = np.arange(-split+0.5, split+0.5, 1)
    z_v = np.arange(-split+0.5, split+0.5, 1)
    X_v, Y_v, Z_v = np.meshgrid(x_v, y_v, z_v)
        
    X_v = X_v/(2*split)
    Y_v = Y_v/(2*split)
    Z_v = Z_v/(2*split)
    
    shell = 1-Sphere_in-Sphere_out
    X = X[shell==1]
    Y = Y[shell==1]
    Z = Z[shell==1]
    shell_val = np.zeros(X.shape)
    
    for i in range(X.size):
        xx = X[i]
        yy = Y[i]
        zz = Z[i]
        occupied = ((xx+X_v*voxel_size[0])**2+(yy+Y_v*voxel_size[1])**2+(zz+Z_v*voxel_size[2])**2)<=radius**2
        shell_val[i] = np.sum(occupied)/X_v.size
        
    Sphere_mid[shell==1] = shell_val
    Sphere = Sphere_in + Sphere_mid    
    Sphere = Sphere/np.sum(Sphere)
    y = np.fft.fftn(np.fft.fftshift(Sphere))
    return y

# smv filter
def SMV_kernel(matrix_size,voxel_size, radius):
    return 1-sphere_kernel(matrix_size, voxel_size,radius)


def SMV(iFreq,matrix_size,voxel_size,radius):
    return np.fft.ifftn(np.fft.fftn(iFreq)*sphere_kernel(matrix_size, voxel_size,radius))

# fidelity weighting matrix
def dataterm_mask(N_std, Mask, Normalize=True):
    w = Mask/N_std
    w[np.isnan(w)] = 0
    w[np.isinf(w)] = 0
    w = w*(Mask>0)
    if Normalize:
        w = w/np.mean(w[Mask>0])     
    return w

# directional differences
def dxp(a):
    return torch.cat((a[:,:,1:,:,:], a[:,:,-1:,:,:]), dim=2) - a


def dyp(a):
    return torch.cat((a[:,:,:,1:,:], a[:,:,:,-1:,:]), dim=3) - a


def dzp(a):
    return torch.cat((a[:,:,:,:,1:], a[:,:,:,:,-1:]), dim=4) - a


def fgrad(a, voxel_size):
    Dx = np.concatenate((a[1:,:,:], a[-1:,:,:]), axis=0) - a
    Dy = np.concatenate((a[:,1:,:], a[:,-1:,:]), axis=1) - a
    Dz = np.concatenate((a[:,:,1:], a[:,:,-1:]), axis=2) - a
    
    Dx = Dx/voxel_size[0]
    Dy = Dy/voxel_size[1]
    Dz = Dz/voxel_size[2]
    return np.concatenate((Dx[...,np.newaxis], Dy[...,np.newaxis], Dz[...,np.newaxis]), axis=3)

# weighting mask of total variation term
def gradient_mask(iMag, Mask, voxel_size=[1, 1, 3], percentage=0.9):
    field_noise_level = 0.01*np.max(iMag)
    wG = abs(fgrad(iMag*(Mask>0), voxel_size))
    denominator = np.sum(Mask[:]==1)
    numerator = np.sum(wG[:]>field_noise_level)
    
    if  (numerator/denominator)>percentage:
        while (numerator/denominator)>percentage:
            field_noise_level = field_noise_level*1.05
            numerator = np.sum(wG[:]>field_noise_level)
    else:
        while (numerator/denominator)<percentage:
            field_noise_level = field_noise_level*.95
            numerator = np.sum(wG[:]>field_noise_level)
            
    wG = (wG<=field_noise_level)
    wG = wG.astype(float)
    return wG

# DLL2 step
class DLL2():
    '''
        D: dipole kernel;
        m: fidelity weighting matrix;
        b: local field;
        device: GPU device;
        P: preconditioner;
        alpha: parameter on the fidelity term;
        rho: parameter on the regularization term;
    '''
    def __init__(self, D, m, b, device, P=1, alpha=0.5, rho=10):
        self.D = D
        self.m = m
        self.b = b
        self.device = device
        self.P = P
        self.alpha = alpha
        self.rho = rho
        self.Dconv = lambda x: torch.real(fft.ifftn(self.D * fft.fftn(x, dim=[0, 1, 2])))

    # system matrix in CG iteration
    def AtA(self, x):
        '''
            x: chi;
        '''
        return self.alpha * self.P * self.Dconv(torch.conj(self.m) * self.m * self.Dconv(self.P * x)) + self.rho * self.P**2 * x
    
    # right hand side in CG iteration
    def rhs(self, phi, mu):
        ''' 
            phi: network output;
            mu: dual variable;
        '''
        return self.alpha * self.P * self.Dconv(torch.conj(self.m) * self.m * self.b) + self.rho * self.P * (phi - mu)
    
    def CG_body(self, i, rTr, x, r, p):
        Ap = self.AtA(p)
        alpha = rTr / torch.sum(torch.conj(p) * Ap)

        x = x + p * alpha
        r = r - Ap * alpha
        rTrNew = torch.sum(torch.conj(r) * r)

        beta = rTrNew /  rTr
        p = r + p * beta
        return i+1, rTrNew, x, r, p

    def while_cond(self, i, rTr, max_iter=10):
        return (i<max_iter) and (rTr>1e-10)

    def CG_iter(self, phi, mu, max_iter=10):
        rhs = self.rhs(phi, mu)
        # x = phi / self.P
        x = torch.zeros(phi.shape, device=self.device)
        x[x != x] = 0

        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r) * r)
        while self.while_cond(i, rTr, max_iter):
            i, rTr, x, r, p = self.CG_body(i, rTr, x, r, p)
            if i % 10 == 0:
                print('i = {0}, rTr = {1}'.format(i, rTr))
        return x

     
