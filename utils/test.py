import numpy as np
import math
from utils.data import *
from utils.medi import *

def compute_rmse(chi_recon, chi_true):
    """chi_true is the refernce ground truth"""
    mask = abs(chi_true) > 0
    chi_recon = chi_recon * mask
    return 100 * np.sqrt(np.sum((chi_recon - chi_true)**2)/np.prod(chi_recon.shape))

def compute_fidelity_error(chi, rdf, voxel_size):
    """data consistenty loss, with rdf the measured data"""
    D = dipole_kernel(rdf.shape, voxel_size, [0, 0, 1])
    mask = abs(rdf) > 0
    diff = np.fft.ifftn( np.fft.fftn(chi) * D ) - rdf
    return np.sqrt(np.sum((diff*mask)**2))