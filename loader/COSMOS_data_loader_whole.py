import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
COSMOS dataloader of 3mm slice thickness data with SMV filtering 
or 1mm slice thickness data without SMV filtering
'''
class COSMOS_data_loader_whole(data.Dataset):

    def __init__(
        self,
        batchSize = 2,
        augmentations = [None],
        voxel_size = (1, 1, 1),
        case_validation = 6,
        case_test = 7,
        test_dir = 0,
        split = 'Train',
        flag_smv = 1,
        flag_gen = 1,  # generate rdf from COSMOS data
        flag_crop = 0
    ):
        self.flag_smv = flag_smv
        self.flag_gen = flag_gen
        self.flag_crop = flag_crop
        if self.flag_smv:
            self.dataFolder = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/20190323_COSMOS_smv_3mm'
            print('Using SMV Filtering')
        else:
            self.dataFolder = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/1&3mm'
            print('Without SMV Filtering')
        if self.flag_gen:
            print('Generate RDF from COSMOS data')
        else:
            print('Using measured RDF')
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.voxel_size = voxel_size
        if split == 'Train':
            self.cases_train = list(range(2, 8))
            self.cases_train.remove(case_validation)
            self.cases_train.remove(case_test)
        self.cases_validation = list(range(case_validation, case_validation+1))
        self.cases_test = list(range(case_test, case_test+1))
        self.split = split
        if split == 'Train':
            self.casesRange = self.cases_train
            self.load_all_volumes()
        elif split == 'Val':
            self.casesRange = self.cases_validation
            self.load_all_volumes()

    def load_all_volumes(self):
        self.RDFs, self.Masks, self.Data_weights, self.wGs, self.Ds = [], [], [], [], []

        for i_case in self.casesRange:
            print('Processing case: {}'.format(i_case))
            if self.flag_smv:  
                filename = '{0}/{1}/RDF_3mm.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF_new'))
                filename = '{0}/{1}/Mask_smv_3mm.mat'.format(self.dataFolder, i_case)
                Mask = np.real(load_mat(filename, varname='Mask_new'))
                filename = '{0}/{1}/N_std_3mm.mat'.format(self.dataFolder, i_case)
                N_std = np.real(load_mat(filename, varname='N_std_new'))
                filename = '{0}/{1}/iMag_smv_3mm.mat'.format(self.dataFolder, i_case)
                iMag = np.real(load_mat(filename, varname='COSMOS_new'))
                filename = '{0}/{1}/COSMOS_smv_3mm.mat'.format(self.dataFolder, i_case)
                QSM = np.real(load_mat(filename, varname='COSMOS_new'))
            else:
                filename = '{0}/{1}/RDF.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF'))
                Mask = np.real(load_mat(filename, varname='Mask'))
                N_std = np.real(load_mat(filename, varname='N_std'))
                iMag = np.real(load_mat(filename, varname='iMag'))
                filename = '{0}/{1}/COSMOS.mat'.format(self.dataFolder, i_case)
                QSM = np.real(load_mat(filename, varname='COSMOS'))
            
            ndir = RDF.shape[-1]
            for i_dir in range(ndir):
                print('    Direction: {}'.format(i_dir))
                RDF_dir = RDF[..., i_dir]
                Mask_dir = Mask[..., i_dir]
                N_std_dir = N_std[..., i_dir]
                iMag_dir = iMag[..., i_dir]
                QSM_dir = QSM[..., i_dir]

                matrix_size = RDF_dir.shape
                voxel_size = self.voxel_size
                B0_dir = (0, 0, 1)
                radius = 5

                if self.flag_smv:
                    RDF_dir = RDF_dir - SMV(RDF_dir, matrix_size, voxel_size, radius)
                    RDF_dir = np.real(RDF_dir * Mask_dir)
                    N_std_dir = np.sqrt(SMV(N_std_dir**2, matrix_size, voxel_size, radius)+N_std_dir**2)
                    N_std_dir = np.real(N_std_dir * Mask_dir)

                # gradient Mask
                wG_dir = gradient_mask(iMag_dir, Mask_dir)
                # fidelity term weight
                Data_weight_dir = np.real(dataterm_mask(N_std_dir, Mask_dir, Normalize=False))

                RDF_aug = augment_data(RDF_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])
                Mask_aug = augment_data(Mask_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])
                Data_weight_aug = augment_data(Data_weight_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])
                wG_aug = augment_data(wG_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])
                QSM_aug = augment_data(QSM_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])

                if self.flag_gen:
                    D = dipole_kernel(matrix_size, voxel_size, B0_dir)
                    # if self.flag_smv:
                    #     S = SMV_kernel(matrix_size, voxel_size, radius)
                    #     D = S * D
                    RDF_dir = np.real(np.fft.ifftn(np.fft.fftn(QSM_dir) * D)).astype(np.float32)
                    RDF_aug = augment_data(RDF_dir, voxel_size=voxel_size, flip='', thetas=[-15, 15])

                self.RDFs.append(RDF_aug)
                self.Masks.append(Mask_aug)
                self.Data_weights.append(Data_weight_aug)
                self.wGs.append(wG_aug)

            self.Ds.append(D)

        self.RDFs = np.concatenate(self.RDFs, axis=0)[:, np.newaxis, ...]
        self.Masks = np.concatenate(self.Masks, axis=0)[:, np.newaxis, ...]
        self.Data_weights = np.concatenate(self.Data_weights, axis=0)[:, np.newaxis, ...]
        self.wGs = np.concatenate(self.wGs, axis=0)[:, np.newaxis, ...]
        self.num_samples = len(self.RDFs)
        print('{0} {1} cases in total'.format(self.num_samples, self.split))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.RDFs[idx], self.Masks[idx], self.Data_weights[idx], self.wGs[idx], self.Ds[idx%15]

        

            


