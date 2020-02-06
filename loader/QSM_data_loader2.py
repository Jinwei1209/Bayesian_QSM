import os
import numpy as np
import random
from torch.utils import data
from utils.files import *
from utils.medi import *

class QSM_data_loader2(data.Dataset):

    folderMatcher = {
        'COSMOS': ['2', '3', '4', '5', '6'],
        'MS': ['ms1', 'ms2', 'ms3', 'ms4', 'ms5', 'ms6', 'ms7', 'ms8'],
        'ICH': ['hemo1', 'hemo4', 'hemo6', 'hemo16']
    }

    # folderMatcher = {
    #     'COSMOS': ['2'],
    #     'MS': ['ms1'],
    #     'ICH': ['hemo1']
    # }

    def __init__(
        self,
        GPU = 1,
        batchSize = 2,
        patches_per_volume = 8,
        augmentations = [None],
        patchSize = (64, 64, 32),
        voxel_size = (0.9375, 0.9375, 3),
        flag_SMV = 1,
        radius = 5 
    ):
        if GPU == 1:   
            self.rootDir = '/home/sdc/Jinwei/QSM/1*1*3recon/'
        elif GPU == 2:
            self.rootDir = '/data/Jinwei/Bayesian_QSM/20190523_BayesianQSM_dataset/1_1_3recon/'
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.patches_per_volume = patches_per_volume
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

        self.factor_MS = 3.8588
        self.factor_ICH = 3.9034

        self.load_all_volumes()
        self.random_idxs = random.sample(range(0, self.num_COSMOS), self.num_COSMOS)

    def load_all_volumes(self):

        # COSMOS data
        self.COSMOS_input_RDFs = []
        self.COSMOS_in_loss_RDFs = []
        self.COSMOS_QSMs = []
        self.COSMOS_Masks = []
        self.COSMOS_fidelity_Ws = []
        self.COSMOS_gradient_Ws = []


        for idx, element in enumerate(self.folderMatcher['COSMOS']):

            print('Loading COSMOS data {}'.format(idx))

            folderName = self.rootDir + element + '/'

            input_RDF = np.real(load_mat(folderName+'RDF_smv_3mm.mat', 'RDF_new'))
            RDF = np.real(load_mat(folderName+'RDF_3mm.mat', 'RDF_new'))
            QSM = np.real(load_mat(folderName+'COSMOS_smv_3mm.mat', 'COSMOS_new'))
            Mask = np.real(load_mat(folderName+'Mask_smv_3mm.mat', 'Mask_new'))
            N_std = np.real(load_mat(folderName+'N_std_3mm.mat', 'N_std_new'))
            iMAG = np.real(load_mat(folderName+'iMag_3mm.mat', 'iMag_new'))

            fidelity_Ws = np.zeros(iMAG.shape)
            gradient_Ws = np.zeros(iMAG.shape[0:3] + (3, self.num_Orient))
            in_loss_RDFs = np.zeros(iMAG.shape)
            for i in range(0, self.num_Orient):

                print('Orientation {}'.format(i))

                vol_size = iMAG.shape[0:3]
                tempn = np.double(N_std[..., i])
                iMag = iMAG[..., i]
                rdf = RDF[..., i]
                mask = Mask[..., i]
                in_loss_rdf = rdf
                if self.flag_SMV:
                    tempn = np.sqrt(SMV(tempn**2, vol_size, self.voxel_size, self.radius) + tempn**2)
                    rdf = rdf - SMV(rdf, vol_size, self.voxel_size, self.radius)
                    in_loss_rdf = np.real(rdf*mask)
                    
                fidelity_Ws[..., i] = np.real(dataterm_mask(tempn, mask, Normalize=False))
                gradient_Ws[..., i] = gradient_mask(iMag, mask)
                in_loss_RDFs[..., i] = in_loss_rdf

            self.COSMOS_input_RDFs.append(input_RDF)
            self.COSMOS_in_loss_RDFs.append(in_loss_RDFs)
            self.COSMOS_QSMs.append(QSM)
            self.COSMOS_Masks.append(Mask)
            self.COSMOS_fidelity_Ws.append(fidelity_Ws)
            self.COSMOS_gradient_Ws.append(gradient_Ws)

        # MS data
        self.MS_input_RDFs = []
        self.MS_in_loss_RDFs = [] 
        self.MS_QSMs = []
        self.MS_Masks = []
        self.MS_fidelity_Ws = []
        self.MS_gradient_Ws = []

        for idx, element in enumerate(self.folderMatcher['MS']):

            print('Loading MS data {}'.format(idx))

            folderName = self.rootDir + element + '/'

            rdf = np.real(load_mat(folderName+'RDF.mat', 'RDF'))
            qsm = np.real(load_mat(folderName+'QSM.mat', 'QSM'))
            mask = np.real(load_mat(folderName+'Mask.mat', 'Mask'))
            N_std = np.real(load_mat(folderName+'N_std.mat', 'N_std'))
            iMag = np.real(load_mat(folderName+'iMag.mat', 'iMag'))

            vol_size = iMag.shape
            tempn = np.double(N_std)
            if self.flag_SMV:
                mask = SMV(mask, vol_size, self.voxel_size, self.radius) > 0.999
                mask = mask.astype(float)
                tempn = np.sqrt(SMV(tempn**2, vol_size, self.voxel_size, self.radius) + tempn**2)
                rdf_tmp = rdf - SMV(rdf, vol_size, self.voxel_size, self.radius)
                in_loss_rdf = np.real(rdf_tmp*mask)/self.factor_MS
                input_rdf = rdf*mask/self.factor_MS
            else:
                mask = SMV(mask, vol_size, self.voxel_size, self.radius) > 0.999
                mask = mask.astype(float)
                input_rdf = rdf*mask/self.factor_MS
                in_loss_rdf = input_rdf
            gradient_W = gradient_mask(iMag, mask)
            fidelity_W = np.real(dataterm_mask(tempn, mask, Normalize=False))

            self.MS_input_RDFs.append(input_rdf)
            self.MS_in_loss_RDFs.append(in_loss_rdf)
            self.MS_QSMs.append(qsm)
            self.MS_Masks.append(mask)
            self.MS_fidelity_Ws.append(fidelity_W)
            self.MS_gradient_Ws.append(gradient_W)

        # ICH data
        self.ICH_input_RDFs = []
        self.ICH_in_loss_RDFs = [] 
        self.ICH_QSMs = []
        self.ICH_Masks = []
        self.ICH_fidelity_Ws = []
        self.ICH_gradient_Ws = []

        for idx, element in enumerate(self.folderMatcher['ICH']):

            print('Loading ICH data {}'.format(idx))

            folderName = self.rootDir + element + '/'

            rdf = np.real(load_mat(folderName+'RDF.mat', 'RDF'))
            qsm = np.real(load_mat(folderName+'MEDI0.mat', 'MEDI0'))
            mask = np.real(load_mat(folderName+'Mask.mat', 'Mask'))
            N_std = np.real(load_mat(folderName+'N_std_m.mat', 'N_std'))
            iMag = np.real(load_mat(folderName+'iMag.mat', 'iMag'))

            vol_size = iMag.shape
            tempn = np.double(N_std)
            if self.flag_SMV:
                mask = SMV(mask, vol_size, self.voxel_size, self.radius) > 0.999
                mask = mask.astype(float)
                tempn = np.sqrt(SMV(tempn**2, vol_size, self.voxel_size, self.radius) + tempn**2)
                rdf_tmp = rdf - SMV(rdf, vol_size, self.voxel_size, self.radius)
                in_loss_rdf = np.real(rdf_tmp*mask)/self.factor_ICH
                input_rdf = rdf*mask/self.factor_ICH
            else:
                mask = SMV(mask, vol_size, self.voxel_size, self.radius) > 0.999
                mask = mask.astype(float)
                input_rdf = rdf*mask/self.factor_ICH
                in_loss_rdf = input_rdf
            gradient_W = gradient_mask(iMag, mask)
            fidelity_W = np.real(dataterm_mask(tempn, mask, Normalize=False))

            self.ICH_input_RDFs.append(input_rdf)
            self.ICH_in_loss_RDFs.append(in_loss_rdf)
            self.ICH_QSMs.append(qsm)
            self.ICH_Masks.append(mask)
            self.ICH_fidelity_Ws.append(fidelity_W)
            self.ICH_gradient_Ws.append(gradient_W)


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
                input_rdf = self.COSMOS_input_RDFs[case[0]][..., case[1]]
                in_loss_rdf = self.COSMOS_in_loss_RDFs[case[0]][..., case[1]]
                qsm = self.COSMOS_QSMs[case[0]][..., case[1]]
                mask = self.COSMOS_Masks[case[0]][..., case[1]]
                fidelity_W = self.COSMOS_fidelity_Ws[case[0]][..., case[1]]
                gradient_W = self.COSMOS_gradient_Ws[case[0]][..., case[1]]

                patches = self.patch_extraction([input_rdf, in_loss_rdf, qsm, mask, fidelity_W, gradient_W])
                # aggregate data
                for i in range(0, len(patches)):
                    Patches[i%len(Patches)].append(patches[i])

        # patient data
        else:
            idxs = random.sample(range(0, self.num_patients), self.batchSize)
            for idx_ in idxs:
                case = self.idx_to_case(idx_)
                if idx_ < self.num_MS:
                    input_rdf = self.MS_input_RDFs[case]
                    in_loss_rdf = self.MS_in_loss_RDFs[case]
                    qsm = self.MS_QSMs[case]
                    mask = self.MS_Masks[case]
                    fidelity_W = self.MS_fidelity_Ws[case]
                    gradient_W = self.MS_gradient_Ws[case]
                else:
                    input_rdf = self.ICH_input_RDFs[case]
                    in_loss_rdf = self.ICH_in_loss_RDFs[case]
                    qsm = self.ICH_QSMs[case]
                    mask = self.ICH_Masks[case]
                    fidelity_W = self.ICH_fidelity_Ws[case]
                    gradient_W = self.ICH_gradient_Ws[case]

                patches = self.patch_extraction([input_rdf, in_loss_rdf, qsm, mask, fidelity_W, gradient_W])
                # aggregate data
                for i in range(0, len(patches)):
                    Patches[i%len(Patches)].append(patches[i])

        input_RDFs = np.concatenate(Patches[0], axis=0)[:, np.newaxis, ...]
        in_loss_RDFs = np.concatenate(Patches[1], axis=0)[:, np.newaxis, ...]
        QSMs = np.concatenate(Patches[2], axis=0)[:, np.newaxis, ...]
        Masks = np.concatenate(Patches[3], axis=0)[:, np.newaxis, ...]
        fidelity_Ws = np.concatenate(Patches[4], axis=0)[:, np.newaxis, ...]
        gradient_Ws = np.concatenate(Patches[5], axis=0)[:, np.newaxis, ...]

        return input_RDFs, in_loss_RDFs, QSMs, Masks, fidelity_Ws, gradient_Ws, self.flag_COSMOS
        # return input_RDFs, in_loss_RDFs, QSMs, Masks, fidelity_Ws, gradient_Ws, 1

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
        x0s = random.sample(range(0, x - self.patchSize[0] + 1), self.patches_per_volume)
        y0s = random.sample(range(0, y - self.patchSize[1] + 1), self.patches_per_volume)
        z0s = random.sample(range(0, z - self.patchSize[2] + 1), self.patches_per_volume)
        outputs = []

        for j in range(0, self.patches_per_volume):
            x0, y0, z0 = x0s[j], y0s[j], z0s[j]
            for i in range(0, l):
                if inputs[3][x0:x0+self.patchSize[0], y0:y0+self.patchSize[1], \
                    z0:z0+self.patchSize[2], ...].mean(axis=(0,1,2)) > 0.1:
                    outputs.append(inputs[i][x0:x0+self.patchSize[0], y0:y0+self.patchSize[1], \
                        z0:z0+self.patchSize[2], ...][np.newaxis, ...])   
        return outputs
        

            


