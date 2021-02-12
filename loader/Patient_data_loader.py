import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
dataloader of 3mm patient with smv
'''
class Patient_data_loader(data.Dataset):
    def __init__(
        self,
        dataFolder = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/20190920_MEDI_3mm',
        patientType='ICH',
        patientID=1,
        flag_input=0,  # flag for rdf input, 0: smv input, 1: non-smv input
        flag_simu=0  # flag to simulate rdf from FINE
    ):
        self.patientType = patientType
        self.patientID = patientType + str(patientID)
        self.dataFolder = dataFolder
        self.flag_input = flag_input
        self.flag_simu = flag_simu
        if self.flag_simu == 1:
            self.flag_input = 1
        if patientType == 'ICH':
            print('Loading ICH data: {0}'.format(patientID))
            voxel_size = [0.937500, 0.937500, 2.8]  # hemo cases
            factor = 3.9034  # 3.9034 for hemo cases
        elif patientType == 'MS_old':
            print('Loading old MS data: {0}'.format(patientID))
            voxel_size = [0.9376000, 0.9376000, 3.0]  # ms and ms_ cases
            factor = 3.85885  # 3.85885 for ms cases
            self.dataFolder += '/MS_train'
        elif patientType == 'MS_new':
            print('Loading new MS data: {0}'.format(patientID))
            voxel_size = [0.9376000, 0.9376000, 3.0]  # ms and ms_ cases
            factor = 3.85885  # 3.85885 for ms cases
        radius = 5
        B0_dir = [0, 0, 1]
        self.voxel_size = voxel_size
        self.load_volume(voxel_size, radius, B0_dir, factor)

    def load_volume(
        self,
        voxel_size,
        radius,
        B0_dir,
        factor
    ):
        dataDir = self.dataFolder + '/' + self.patientID

        filename = '{0}/QSM_refine_100.mat'.format(dataDir)
        QSM = np.real(load_mat(filename, varname='QSM_refine'))

        filename = '{0}/Mask.mat'.format(dataDir)
        Mask = np.real(load_mat(filename, varname='Mask'))
        volume_size = Mask.shape
        self.volume_size = volume_size

        filename = '{0}/Mask_CSF.mat'.format(dataDir)
        Mask_CSF = np.real(load_mat(filename, varname='Mask_CSF'))

        filename = '{0}/QSM.mat'.format(dataDir)
        QSM = np.real(load_mat(filename, varname='QSM'))

        filename = '{0}/iMag.mat'.format(dataDir)
        iMag = np.real(load_mat(filename, varname='iMag'))
        
        if self.patientType == 'ICH':
            filename = '{0}/N_std_m.mat'.format(dataDir)
        else:
            filename = '{0}/N_std.mat'.format(dataDir)
        N_std = np.real(load_mat(filename, varname='N_std'))
        tempn = np.double(N_std)

        if self.flag_simu == 1:
            D = np.real(dipole_kernel(volume_size, voxel_size, B0_dir))
            self.D = D
        else:
            D = dipole_kernel(volume_size, voxel_size, B0_dir)
            S = SMV_kernel(volume_size, voxel_size, radius)
            Mask = SMV(Mask, volume_size, voxel_size, radius) > 0.999
            D = np.real(S*D)
            self.D = D
            tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

        # gradient Mask
        wG = gradient_mask(iMag, Mask)
        # fidelity term weight
        Data_weights = np.real(dataterm_mask(tempn, Mask, Normalize=False))

        if self.flag_simu == 1:
            sigma = 1
            np.random.seed(0)
            noise = N_std * np.random.normal(0, sigma) * 1
            RDF = np.real(np.fft.ifftn(np.fft.fftn(QSM) * D)).astype(np.float32)
            RDF = (np.real(RDF) + noise) * Mask
            RDF_input = RDF
        else:
            filename = '{0}/RDF.mat'.format(dataDir)
            RDF = np.real(load_mat(filename, varname='RDF'))
            RDF_input = np.real(RDF*Mask)/factor
            RDF = RDF - SMV(RDF, volume_size, voxel_size, radius)
            RDF = np.real(RDF*Mask)/factor

        self.RDF = RDF[np.newaxis, np.newaxis, ...]
        self.RDF_input = RDF_input[np.newaxis, np.newaxis, ...]
        self.Mask = Mask[np.newaxis, np.newaxis, ...]
        self.Mask_CSF = Mask_CSF[np.newaxis, np.newaxis, ...]
        self.QSM = QSM[np.newaxis, np.newaxis, ...]
        self.Data_weights = Data_weights[np.newaxis, np.newaxis, ...]
        self.wG = wG[np.newaxis, np.newaxis, ...]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.flag_input == 1:
            return self.RDF_input[idx], self.RDF[idx], self.Mask[idx], self.Data_weights[idx], self.wG[idx], self.D
        elif self.flag_input == 0:
            return self.RDF[idx], self.Mask[idx], self.Data_weights[idx], self.wG[idx]
        else:
            return self.QSM[idx], self.RDF_input[idx], self.RDF[idx], self.wG[idx], self.Data_weights[idx], self.Mask_CSF[idx], self.D



    

