import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
dataloader of 0.75*0.75*3mm MS patient for unsupervised total field inversion
'''
class UTFI_data_loader(data.Dataset):
    def __init__(
        self,
        dataFolder = '/data/Jinwei/Unsupervised_total_field_inversion/data/train',
        flag_train = 1
    ):
        self.list_IDs = listFolders(dataFolder)
        self.dataFolder = dataFolder
        self.factor = 3.1421
        self.radius = 5
        self.B0_dir = [0, 0, 1]  # approximated B0_dir
        self.voxel_size = [0.75, 0.75, 3.0]
        self.volume_size = [128, 128, 32]
        self.iFreqs, self.Masks, self.Data_weights, self.wGs = [], [], [], []
        self.flag_train = flag_train
        for i in range(len(self.list_IDs)):
            self.load_volume(self.list_IDs[i])
            print('Loading ID: {0}'.format(self.list_IDs[i]))
            print('Matrix size: {0}'.format(self.Mask[0].shape))
            self.iFreqs.append(self.iFreq[0])
            self.Masks.append(self.Mask[0])
            self.Data_weights.append(self.Data_weight[0])
            self.wGs.append(self.wG[0])
        print('{0} cases in total'.format(len(self.list_IDs)))

    def load_volume(
        self,
        patient_ID
    ):
        dataDir = patient_ID

        filename = '{0}/Mask.mat'.format(dataDir)
        Mask = np.real(load_mat(filename, varname='Mask')).astype(float)

        filename = '{0}/iFreq.mat'.format(dataDir)
        iFreq = np.real(load_mat(filename, varname='iFreq')) * Mask

        filename = '{0}/iMag.mat'.format(dataDir)
        iMag = np.real(load_mat(filename, varname='iMag')) * Mask
        
        filename = '{0}/N_std.mat'.format(dataDir)
        N_std = np.real(load_mat(filename, varname='N_std'))

        width, length, height = Mask.shape
        if length != 320:
            raise ValueError('Second dimension is not 320: {0}'.format(patient_ID))
        if width < 260:
            Mask_new, iFreq_new, iMag_new, N_std_new = np.zeros((260, 320, height)), np.zeros((260, 320, height)), np.zeros((260, 320, height)), np.zeros((260, 320, height))
            # Mask_new[(260-width)//2:width+(260-width)//2, (320-length)//2:length+(320-length)//2, :] = Mask
            Mask_new[(260-width)//2:width+(260-width)//2, :, :] = Mask
            iFreq_new[(260-width)//2:width+(260-width)//2, :, :] = iFreq
            iMag_new[(260-width)//2:width+(260-width)//2, :, :] = iMag
            N_std_new[(260-width)//2:width+(260-width)//2, :, :] = N_std
        else:
            Mask_new = Mask[(width-260)//2:260+(width-260)//2, :, :]
            iFreq_new = iFreq[(width-260)//2:260+(width-260)//2, :, :]
            iMag_new = iMag[(width-260)//2:260+(width-260)//2, :, :]
            N_std_new = N_std[(width-260)//2:260+(width-260)//2, :, :]

        Mask, iFreq, iMag, N_std = Mask_new, iFreq_new, iMag_new, N_std_new


        tempn = np.double(N_std)
        # # SMV on N_std
        # D = dipole_kernel(volume_size, voxel_size, B0_dir)
        # S = SMV_kernel(volume_size, voxel_size, radius)
        # Mask = SMV(Mask, volume_size, voxel_size, radius) > 0.999
        # D = S*D
        # tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

        # gradient Mask
        wG = gradient_mask(iMag, Mask)
        # fidelity term weight
        Data_weight = np.real(dataterm_mask(tempn, Mask, Normalize=True))

        self.iFreq = iFreq[np.newaxis, np.newaxis, ...]
        self.Mask = Mask[np.newaxis, np.newaxis, ...]
        self.Data_weight = Data_weight[np.newaxis, np.newaxis, ...]
        self.wG = wG[np.newaxis, np.newaxis, ...]

    def __len__(self):
        if self.flag_train:
            return len(self.list_IDs) * 3 * 4 * 2  # * L * W * H
        else:
            return len(self.list_IDs)

    def __getitem__(self, idx):
        if self.flag_train:
            w_step, l_step = 64, 66
            sub_idx, rmder = idx // (3 * 4 * 2), idx % (3 * 4 * 2)
            h_idx, rmder2 = rmder // (4 * 3), rmder % (4 * 3)
            l_idx, w_idx = rmder2 // 4, rmder2 % 4

            if h_idx == 0:
                ifreq = self.iFreqs[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, 0:32]
                mask = self.Masks[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, 0:32]
                data_weight = self.Data_weights[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, 0:32]
                wg = self.wGs[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, 0:32]
            else:
                ifreq = self.iFreqs[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, -32:]
                mask = self.Masks[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, -32:]
                data_weight = self.Data_weights[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, -32:]
                wg = self.wGs[sub_idx][:, l_step*l_idx:128+l_step*l_idx, w_step*w_idx:128+w_step*w_idx, -32:]
            return ifreq, mask, data_weight, wg
        else:
            return self.iFreqs[idx], self.Masks[idx], self.Data_weights[idx], self.wGs[idx]