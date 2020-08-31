import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *
from utils.data import *

'''
COSMOS dataloader for QSMnet+ augmentation strategy
COSMOS dataloader of 3mm slice thickness data with SMV filtering 
or 1mm slice thickness data without SMV filtering
'''
class COSMOS_data_loader(data.Dataset):

    def __init__(
        self,
        batchSize = 2,
        augmentations = [None],
        patchSize = (64, 64, 64),
        extraction_step = (21, 21, 21),
        voxel_size = (1, 1, 1),
        case_validation = 6,
        case_test = 7,
        test_dir = 0,
        split = 'Train',
        linear_factor = 4,  # linear augmentation factor for QSMnet+
        rotation_degree = 30,  # rotation degree for augmentation
        flag_smv = 1,
        flag_gen = 1,  # generate rdf from COSMOS data using linear_factor
        flag_invert = 1 # invert rdf for augmentation
    ):
        self.linear_factor = linear_factor
        self.rotation_degree = rotation_degree
        self.flag_smv = flag_smv
        self.flag_gen = flag_gen
        self.flag_invert = flag_invert
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
        self.patchSize = patchSize
        self.extraction_step = extraction_step
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
        else:
            self.casesRange = self.cases_test
            self.load_one_dir(test_dir)

    def load_all_volumes(self):
        patches_RDFs, patches_MASKs, patches_QSMs = [], [], []

        for i_case in self.casesRange:
            print('Processing case: {}'.format(i_case))
            if self.flag_smv:  
                filename = '{0}/{1}/RDF_smv_3mm.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF_new'))
                filename = '{0}/{1}/Mask_smv_3mm.mat'.format(self.dataFolder, i_case)
                Mask = np.real(load_mat(filename, varname='Mask_new'))
                filename = '{0}/{1}/N_std_3mm.mat'.format(self.dataFolder, i_case)
                N_std = np.real(load_mat(filename, varname='N_std_new'))
                filename = '{0}/{1}/COSMOS_smv_3mm.mat'.format(self.dataFolder, i_case)
                QSM = np.real(load_mat(filename, varname='COSMOS_new'))
            else:
                filename = '{0}/{1}/RDF.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF'))
                Mask = np.real(load_mat(filename, varname='Mask'))
                N_std = np.real(load_mat(filename, varname='N_std'))
                filename = '{0}/{1}/COSMOS.mat'.format(self.dataFolder, i_case)
                QSM = np.real(load_mat(filename, varname='COSMOS'))
            
            ndir = RDF.shape[-1]
            for i_dir in range(ndir):
                print('    Direction: {}'.format(i_dir))
                RDF_dir = RDF[..., i_dir]
                Mask_dir = Mask[..., i_dir]
                QSM_dir = QSM[..., i_dir]

                matrix_size = RDF_dir.shape
                voxel_size = self.voxel_size
                B0_dir = (0, 0, 1)

                RDF_aug = augment_data(RDF_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])
                Mask_aug = augment_data(Mask_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])
                QSM_aug = augment_data(QSM_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])
                for i_aug in range(3):
                    patches_RDF = extract_patches(RDF_aug[i_aug, ...], self.patchSize, self.extraction_step)
                    patches_MASK = extract_patches(Mask_aug[i_aug, ...], self.patchSize, self.extraction_step)
                    patches_QSM = extract_patches(QSM_aug[i_aug, ...], self.patchSize, self.extraction_step)

                    # filter out background patch
                    idxs_valid = patches_MASK.mean(axis=(1,2,3)) > 0.1
                    patches_RDF = patches_RDF[idxs_valid, ...]
                    patches_MASK = patches_MASK[idxs_valid, ...]
                    patches_QSM = patches_QSM[idxs_valid, ...]

                    patches_RDFs.append(patches_RDF)
                    patches_MASKs.append(patches_MASK)
                    patches_QSMs.append(patches_QSM)

                    if self.flag_invert:
                        patches_RDFs.append(-patches_RDF)
                        patches_MASKs.append(patches_MASK)
                        patches_QSMs.append(-patches_QSM)

                if self.flag_gen and self.linear_factor > 1:
                    print('Augment data using linearity')
                    D = dipole_kernel(matrix_size, voxel_size, B0_dir)
                    QSM_dir[Mask_dir > 0] = QSM_dir[Mask_dir > 0] * self.linear_factor
                    RDF_dir = np.real(np.fft.ifftn(np.fft.fftn(QSM_dir) * D)).astype(np.float32)
                    RDF_aug = augment_data(RDF_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])
                    Mask_aug = augment_data(Mask_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])
                    QSM_aug = augment_data(QSM_dir, voxel_size=voxel_size, flip='', thetas=[-self.rotation_degree, self.rotation_degree])

                    for i_aug in range(3):
                        patches_RDF = extract_patches(RDF_aug[i_aug, ...], self.patchSize, self.extraction_step)
                        patches_MASK = extract_patches(Mask_aug[i_aug, ...], self.patchSize, self.extraction_step)
                        patches_QSM = extract_patches(QSM_aug[i_aug, ...], self.patchSize, self.extraction_step)

                        # filter out background patch
                        idxs_valid = patches_MASK.mean(axis=(1,2,3)) > 0.1
                        patches_RDF = patches_RDF[idxs_valid, ...]
                        patches_MASK = patches_MASK[idxs_valid, ...]
                        patches_QSM = patches_QSM[idxs_valid, ...]

                        patches_RDFs.append(patches_RDF)
                        patches_MASKs.append(patches_MASK)
                        patches_QSMs.append(patches_QSM)

                        if self.flag_invert:
                            patches_RDFs.append(-patches_RDF)
                            patches_MASKs.append(patches_MASK)
                            patches_QSMs.append(-patches_QSM)

        self.patches_RDFs = np.concatenate(patches_RDFs, axis=0)[:, np.newaxis, ...]
        del patches_RDFs
        self.patches_MASKs = np.concatenate(patches_MASKs, axis=0)[:, np.newaxis, ...]
        del patches_MASKs
        self.patches_QSMs = np.concatenate(patches_QSMs, axis=0)[:, np.newaxis, ...]
        del patches_QSMs
        self.num_samples = len(self.patches_RDFs)
        print('{0} {1} cases in total'.format(self.num_samples, self.split))

    def load_one_dir(self, test_dir):
        print('Loading test case {0} with direciton {1}'.format(self.cases_test, test_dir))

        for i_case in self.casesRange:
            if self.flag_smv:
                filename = '{0}/{1}/RDF_smv_3mm.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF_new'))
                RDF_dir = RDF[..., test_dir]
                filename = '{0}/{1}/Mask_smv_3mm.mat'.format(self.dataFolder, i_case)
                Mask = np.real(load_mat(filename, varname='Mask_new'))
                Mask_dir = Mask[..., test_dir]

                matrix_size = RDF_dir.shape
                voxel_size = self.voxel_size
                B0_dir = (0, 0, 1)
                radius = 5

                if self.flag_gen:
                    filename = '{0}/{1}/COSMOS_smv_3mm.mat'.format(self.dataFolder, i_case)
                    QSM = np.real(load_mat(filename, varname='COSMOS_new'))
                    QSM_dir = QSM[..., test_dir]

                    D = dipole_kernel(matrix_size, voxel_size, B0_dir)
                    RDF_dir = np.real(np.fft.ifftn(np.fft.fftn(QSM_dir) * D)).astype(np.float32)
                    RDF_dir = RDF_dir * Mask_dir
            else:
                filename = '{0}/{1}/RDF.mat'.format(self.dataFolder, i_case)
                RDF = np.real(load_mat(filename, varname='RDF'))
                RDF_dir = RDF[..., test_dir]

                if self.flag_gen:
                    filename = '{0}/{1}/COSMOS.mat'.format(self.dataFolder, i_case)
                    QSM = np.real(load_mat(filename, varname='COSMOS'))
                    QSM_dir = QSM[..., test_dir]

                    D = dipole_kernel(matrix_size, voxel_size, B0_dir)
                    RDF_dir = np.real(np.fft.ifftn(np.fft.fftn(QSM_dir) * D)).astype(np.float32)

            patches_RDF, patches_MASK, patches_QSM = RDF_dir[np.newaxis, ...], Mask_dir[np.newaxis, ...], QSM_dir[np.newaxis, ...]
            self.patches_RDFs = patches_RDF[:, np.newaxis, ...]
            self.patches_MASKs = patches_MASK[:, np.newaxis, ...]
            self.patches_QSMs = patches_QSM[:, np.newaxis, ...]
            self.num_samples = len(self.patches_RDFs)
            self.volSize = patches_RDF.shape[2:]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.patches_RDFs[idx], self.patches_MASKs[idx], self.patches_QSMs[idx]

        

            


