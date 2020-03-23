import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *


# dataloader of 3mm patient with smv
class Patient_data_loader(data.Dataset):
    def __init__(
        self,
        dataFolder = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/20190920_MEDI_3mm',
        patientType='ICH'
    ):
        self.patientType = patientType
        self.list_IDs = [1, 4, 6, 8, 9, 14, 16]
        self.dataFolder = dataFolder
        if patientType == 'ICH':
            voxel_size = [0.937500, 0.937500, 2.8]  # hemo cases
            factor = 3.9034  # 3.9034 for hemo cases
        else:
            voxel_size = [0.9376000, 0.9376000, 3.0]  # ms and ms_ cases
            factor = 3.85885  # 3.85885 for ms cases
        radius = 5
        B0_dir = [0, 0, 1]
        self.voxel_size = voxel_size
        self.load_all_volume(voxel_size, radius, B0_dir, factor)

    def load_all_volume(
        self,
        voxel_size,
        radius,
        B0_dir,
        factor
    ):
        self.RDFs, self.MASKs, self.Data_weights, self.wGs = [], [], [], []
        list_IDs = [1, 4, 6, 9, 14]
        for i in range(5):
            print('Loading ID: {0}'.format(self.list_IDs[i]))
            self.patientID = self.patientType + str(self.list_IDs[i])
            self.load_volume(voxel_size, radius, B0_dir, factor)
            self.RDFs.append(self.RDF[0])
            self.MASKs.append(self.Mask[0])
            self.Data_weights.append(self.Data_weight[0])
            self.wGs.append(self.wG[0])

    def load_volume(
        self,
        voxel_size,
        radius,
        B0_dir,
        factor
    ):
        dataDir = self.dataFolder + '/' + self.patientID

        filename = '{0}/Mask.mat'.format(dataDir)
        Mask = np.real(load_mat(filename, varname='Mask'))
        volume_size = Mask.shape
        self.volume_size = volume_size

        filename = '{0}/iMag.mat'.format(dataDir)
        iMag = np.real(load_mat(filename, varname='iMag'))
        
        if self.patientType == 'ICH':
            filename = '{0}/N_std_m.mat'.format(dataDir)
        else:
            filename = '{0}/N_std.mat'.format(dataDir)
        N_std = np.real(load_mat(filename, varname='N_std'))
        tempn = np.double(N_std)

        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        S = SMV_kernel(volume_size, voxel_size, radius)
        Mask = SMV(Mask, volume_size, voxel_size, radius) > 0.999
        D = S*D
        tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

        # gradient Mask
        wG = gradient_mask(iMag, Mask)
        # fidelity term weight
        Data_weight = np.real(dataterm_mask(tempn, Mask, Normalize=False))

        filename = '{0}/RDF.mat'.format(dataDir)
        RDF = np.real(load_mat(filename, varname='RDF'))
        RDF = RDF - SMV(RDF, volume_size, voxel_size, radius)
        RDF = np.real(RDF*Mask)/factor

        self.RDF = RDF[np.newaxis, np.newaxis, ...]
        self.Mask = Mask[np.newaxis, np.newaxis, ...]
        self.Data_weight = Data_weight[np.newaxis, np.newaxis, ...]
        self.wG = wG[np.newaxis, np.newaxis, ...]

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return self.RDFs[idx], self.Masks[idx], self.Data_weights[idx], self.wGs[idx]