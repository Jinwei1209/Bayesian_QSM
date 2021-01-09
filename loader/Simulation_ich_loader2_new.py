import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
dataloader of simulated ICH patient (main_FINE_resnet.py)
'''
class Simulation_ICH_loader(data.Dataset):
    def __init__(
        self,
        dataFolder = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/20200529_new_hemo_cases',
        patientID = [1]
    ):
        self.dataFolder = dataFolder
        print('Loading simulated ICH data')
        self.list_IDs = [str(a) for a in patientID]

        voxel_size = [1, 1, 3]  # hemo cases
        radius = 5
        B0_dir = [0, 0, 1]
        self.num_subs = len(self.list_IDs)
        self.voxel_size = voxel_size
        self.load_all_volume(voxel_size, radius, B0_dir)

    def load_all_volume(
        self,
        voxel_size,
        radius,
        B0_dir
    ):
        self.QSMs, self.RDFs_input, self.RDFs, self.Masks, self.Data_weights, self.wGs, self.Ds = [], [], [], [], [], [], []
        for i in range(self.num_subs):
            print('Loading ID: {0}'.format(self.list_IDs[i]))
            self.patientID = self.list_IDs[i]
            self.load_volume(voxel_size, radius, B0_dir)
            self.QSMs.append(self.QSM[0])
            self.RDFs_input.append(self.RDF_input[0])
            self.RDFs.append(self.RDF[0])
            self.Masks.append(self.Mask[0])
            self.Data_weights.append(self.Data_weight[0])
            self.wGs.append(self.wG[0])
            self.Ds.append(self.D)

    def load_volume(
        self,
        voxel_size,
        radius,
        B0_dir
    ):
        dataDir = self.dataFolder + '/' + self.patientID

        filename = '{0}/mcTFI_removeoffset1.mat'.format(dataDir)
        QSM = np.real(load_mat(filename, varname='QSM'))
        volume_size = QSM.shape
        self.volume_size = volume_size

        Mask = np.real(load_mat(filename, varname='Mask'))
        QSM = QSM * Mask

        filename = '{0}/cmridata.mat'.format(dataDir)
        iMag = np.real(load_mat(filename, varname='iMag'))
        
        N_std = np.real(load_mat(filename, varname='N_std'))
        tempn = np.double(N_std)

        D = np.real(dipole_kernel(volume_size, voxel_size, B0_dir))
        self.D = D
        # tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

        wG = gradient_mask(iMag, Mask)
        Data_weight = np.real(dataterm_mask(tempn, Mask, Normalize=True))
        Data_weight = np.ones(Data_weight.shape) * np.mean(Data_weight[Mask==1])

        sigma = 1
        noise = N_std * np.random.normal(0, sigma)
        noise = 0

        RDF = np.real(np.fft.ifftn(np.fft.fftn(QSM) * D)).astype(np.float32)
        RDF = (np.real(RDF) + noise) * Mask
        RDF_input = RDF

        self.QSM = QSM[np.newaxis, np.newaxis, ...]
        self.RDF_input = RDF_input[np.newaxis, np.newaxis, ...]
        self.RDF = RDF[np.newaxis, np.newaxis, ...]
        self.Mask = Mask[np.newaxis, np.newaxis, ...]
        self.Data_weight = Data_weight[np.newaxis, np.newaxis, ...]
        self.wG = wG[np.newaxis, np.newaxis, ...]

    def __len__(self):
        return self.num_subs

    def __getitem__(self, idx):
        return self.QSMs[idx], self.RDFs_input[idx], self.RDFs[idx], self.Masks[idx], self.Data_weights[idx], self.wGs[idx], self.Ds[idx]