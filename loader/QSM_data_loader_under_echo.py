import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
dataloader of 1*1*2 mm3 data for under-sampled echo-space reconstruction
'''
class QSM_data_loader_under_echo(data.Dataset):
    def __init__(
        self,
        dataFolder = '/data/Jinwei/QSM_raw_CBIC/data_chao_echo_no_csf',
        split = 'train',
        flag_RDF_input = 0, # flag to have non-smv filtered input
    ):
        self.dataFolder = dataFolder
        if split == 'train':
            self.list_IDs = list(range(1, 27))  # remove 13 14 25 26
            self.list_IDs.remove(13), self.list_IDs.remove(14), self.list_IDs.remove(25), self.list_IDs.remove(26)
        elif split == 'val':
            self.list_IDs = list(range(27, 31))
            # self.list_IDs = list(range(1, 5))
        self.flag_RDF_input = flag_RDF_input
        voxel_size = [1, 1, 2]
        factor = 3.1421
        radius = 5
        B0_dir = [0, 0, 1]
        self.num_subs = len(self.list_IDs)
        self.voxel_size = voxel_size
        self.load_all_volume(voxel_size, radius, B0_dir, factor)

    def load_all_volume(
        self,
        voxel_size,
        radius,
        B0_dir,
        factor
    ):
        self.RDFs_input, self.RDFs_inloss, self.QSMs, self.Masks, self.Data_weights, self.wGs, self.Ds = [], [], [], [], [], [], []
        for i in range(self.num_subs):
            print('Loading ID: {0}'.format(self.list_IDs[i]))
            self.patientID = str(self.list_IDs[i])
            self.load_volume(voxel_size, radius, B0_dir, factor)

            self.RDFs_input.append(self.RDF_input[0])
            self.RDFs_inloss.append(self.RDF_inloss[0])
            # self.QSMs.append(self.QSM[0])
            self.Masks.append(self.Mask[0])
            self.Data_weights.append(self.Data_weight[0])
            self.wGs.append(self.wG[0])
            self.Ds.append(self.D)

    def load_volume(
        self,
        voxel_size,
        radius,
        B0_dir,
        factor
    ):
        filename = '{0}/QSM{1}_10.mat'.format(self.dataFolder, self.patientID)
        QSM = np.real(load_mat(filename, varname='QSM'))
        Mask = abs(QSM) < 30
        QSM = QSM * Mask
        volume_size = Mask.shape
        self.volume_size = volume_size
        iMag = np.real(load_mat(filename, varname='iMag'))
        N_std = np.real(load_mat(filename, varname='N_std'))
        tempn = np.double(N_std)

        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        S = SMV_kernel(volume_size, voxel_size, radius)
        D = np.real(S*D)
        self.D = D
        tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

        # gradient Mask
        wG = gradient_mask(iMag, Mask)
        # fidelity term weight
        Data_weight = np.real(dataterm_mask(tempn, Mask, Normalize=True)) * np.sqrt(1000)

        RDF = np.real(load_mat(filename, varname='RDF'))
        RDF = RDF - SMV(RDF, volume_size, voxel_size, radius)
        RDF = np.real(RDF*Mask)/factor

        filename = '{0}/QSM{1}_6.mat'.format(self.dataFolder, self.patientID)
        RDF_6 = np.real(load_mat(filename, varname='RDF'))
        if self.flag_RDF_input == 0:
            print('SMV Input')
            RDF_6 = RDF_6 - SMV(RDF_6, volume_size, voxel_size, radius)
        RDF_6 = np.real(RDF_6*Mask)/factor

        self.RDF_input = RDF_6[np.newaxis, np.newaxis, ...]
        self.RDF_inloss = RDF[np.newaxis, np.newaxis, ...]
        self.QSM = QSM[np.newaxis, np.newaxis, ...]
        self.Mask = Mask[np.newaxis, np.newaxis, ...]
        self.Data_weight = Data_weight[np.newaxis, np.newaxis, ...]
        self.wG = wG[np.newaxis, np.newaxis, ...]

    def __len__(self):
        return self.num_subs

    def __getitem__(self, idx):
        return self.RDFs_input[idx], self.RDFs_inloss[idx], self.Masks[idx], self.Data_weights[idx], self.wGs[idx], self.Ds[idx]