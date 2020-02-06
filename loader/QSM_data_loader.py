import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *

class QSM_data_loader(data.Dataset):

    folderMatcher = {
        'COSMOS': ['2', '3', '4', '5', '6'],
        'MS': ['ms1', 'ms2', 'ms3', 'ms4', 'ms5', 'ms6', 'ms7', 'ms8'],
        'ICH': ['hemo1', 'hemo4', 'hemo6', 'hemo16']
    }

    def __init__(
        self,
        rootDir = '/home/sdc/Jinwei/QSM/1*1*3recon/',
        batchSize = 2,
        augmentations = [None],
        patchSize = (128, 128, 32),
        voxel_size = (0.9375, 0.9375, 3),
        flag_SMV = 1,
        radius = 5 
    ):

        self.rootDir = rootDir
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.patchSize = patchSize
        self.voxel_size = voxel_size
        self.flag_SMV = flag_SMV
        self.radius = radius

        self.num_Orient = 5
        self.num_COSMOS = len(self.folderMatcher['COSMOS'])*self.num_Orient
        self.replicate_MS = 2
        self.num_MS = len(self.folderMatcher['MS'])*self.replicate_MS
        self.replicate_ICH = 4
        self.num_ICH = len(self.folderMatcher['ICH'])*self.replicate_ICH
        self.num_patients = self.num_MS + self.num_ICH
        self.num_samples = (self.num_COSMOS + self.num_patients) // self.batchSize


    def __len__(self):

        return self.num_samples*self.augSize


    def __getitem__(self, idx):

        input_RDFs, in_loss_RDFs, QSMs, Masks, fidelity_Ws, gradient_Ws = [], [], [], [], [], []
        Patches = [input_RDFs, in_loss_RDFs, QSMs, Masks, fidelity_Ws, gradient_Ws]

        u = random.uniform(0, 1)
        if u < 0.5:
            self.flag_COSMOS = 1
        else:
            self.flag_COSMOS = 0

        # COSMOS data
        if self.flag_COSMOS:
            idxs =  random.sample(range(0, self.num_COSMOS), self.batchSize)
            for idx_ in idxs:
                case = self.idx_to_case(idx_)
                folderName = self.rootDir + self.folderMatcher['COSMOS'][case[0]] + '/'

                input_rdf = np.real(load_mat(folderName+'RDF_smv_3mm.mat', 'RDF_new')[..., case[1]])
                rdf = np.real(load_mat(folderName+'RDF_3mm.mat', 'RDF_new')[..., case[1]])
                qsm = np.real(load_mat(folderName+'COSMOS_smv_3mm.mat', 'COSMOS_new')[..., case[1]])
                mask = np.real(load_mat(folderName+'Mask_smv_3mm.mat', 'Mask_new')[..., case[1]])
                N_std = np.real(load_mat(folderName+'N_std_3mm.mat', 'N_std_new')[..., case[1]])
                iMag = np.real(load_mat(folderName+'iMag_3mm.mat', 'iMag_new')[..., case[1]])

                self.vol_size = iMag.shape
                tempn = np.double(N_std)
                in_loss_rdf = rdf
                if self.flag_SMV:
                    tempn = np.sqrt(SMV(tempn**2, self.vol_size, self.voxel_size, self.radius) + tempn**2)
                    rdf = rdf - SMV(rdf, self.vol_size, self.voxel_size, self.radius)
                    in_loss_rdf = np.real(rdf*mask)

                gradient_W = gradient_mask(iMag, mask)
                fidelity_W = np.real(dataterm_mask(tempn, mask))
                patches = self.patch_extraction([input_rdf, in_loss_rdf, qsm, mask, fidelity_W, gradient_W])
                # aggregate data
                for i in range(0, len(patches)):
                    Patches[i].append(patches[i])

        # patient data
        else:
            idxs = random.sample(range(0, self.num_patients), self.batchSize)
            for idx_ in idxs:
                case = self.idx_to_case(idx_)
                if idx_ < self.num_MS:
                    self.factor = 3.8588
                    folderName = self.rootDir + self.folderMatcher['MS'][case] + '/'
                    N_std = np.real(load_mat(folderName+'N_std.mat', 'N_std'))
                    qsm = np.real(load_mat(folderName+'QSM.mat', 'QSM'))
                else:
                    self.factor = 3.9034
                    folderName = self.rootDir + self.folderMatcher['ICH'][case] + '/'
                    N_std = np.real(load_mat(folderName+'N_std_m.mat', 'N_std'))
                    qsm = np.real(load_mat(folderName+'MEDI0.mat', 'MEDI0'))
                
                rdf = np.real(load_mat(folderName+'RDF.mat', 'RDF'))
                mask = np.real(load_mat(folderName+'Mask.mat', 'Mask'))
                iMag = np.real(load_mat(folderName+'iMag.mat', 'iMag'))

                self.vol_size = iMag.shape
                tempn = np.double(N_std)
                if self.flag_SMV:
                    mask = SMV(mask, self.vol_size, self.voxel_size, self.radius) > 0.999
                    mask = mask.astype(float)
                    tempn = np.sqrt(SMV(tempn**2, self.vol_size, self.voxel_size, self.radius) + tempn**2)
                    rdf_tmp = rdf - SMV(rdf, self.vol_size, self.voxel_size, self.radius)
                    in_loss_rdf = np.real(rdf_tmp*mask)/self.factor
                    input_rdf = rdf/self.factor*mask
                else:
                    mask = SMV(mask, self.vol_size, self.voxel_size, self.radius) > 0.999
                    mask = mask.astype(float)
                    input_rdf = rdf*mask/self.factor
                    in_loss_rdf = input_rdf/self.factor

                gradient_W = gradient_mask(iMag, mask)
                fidelity_W = np.real(dataterm_mask(tempn, mask))
                patches = self.patch_extraction([input_rdf, in_loss_rdf, qsm, mask, fidelity_W, gradient_W])
                # aggregate data
                for i in range(0, len(patches)):
                    Patches[i].append(patches[i])

        input_RDFs = np.concatenate(Patches[0], axis=0)[:, np.newaxis, ...]
        in_loss_RDFs = np.concatenate(Patches[1], axis=0)[:, np.newaxis, ...]
        QSMs = np.concatenate(Patches[2], axis=0)[:, np.newaxis, ...]
        Masks = np.concatenate(Patches[3], axis=0)[:, np.newaxis, ...]
        fidelity_Ws = np.concatenate(Patches[4], axis=0)[:, np.newaxis, ...]
        gradient_Ws = np.concatenate(Patches[5], axis=0)[:, np.newaxis, ...]
        return input_RDFs, in_loss_RDFs, QSMs, Masks, fidelity_Ws, gradient_Ws, self.flag_COSMOS



    def idx_to_case(self, idx_):

        if self.flag_COSMOS:
            case = (idx_ // self.num_Orient, idx_ % self.num_Orient)
        else:
            if idx_ < self.num_MS:
                case = idx_ // self.replicate_MS
            else:
                case = (idx_ - self.num_MS) // self.replicate_ICH
        return case


    def patch_extraction(self, inputs):

        l = len(inputs)
        [x, y, z] = inputs[0].shape
        x0 = random.sample(range(0, x - self.patchSize[0] + 1), 1)[0]
        y0 = random.sample(range(0, y - self.patchSize[1] + 1), 1)[0]
        z0 = random.sample(range(0, z - self.patchSize[2] + 1), 1)[0]
        outputs = []
        for i in range(0, l):
            outputs.append(inputs[i][x0:x0+self.patchSize[0], y0:y0+self.patchSize[1], \
                                     z0:z0+self.patchSize[2], ...][np.newaxis, ...])   
        return outputs
        

            


