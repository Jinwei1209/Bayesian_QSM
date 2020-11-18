import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import math
import argparse

from torch.utils import data
from loader.COSMOS_data_loader import COSMOS_data_loader
from loader.Patient_data_loader import Patient_data_loader
from models.unet import Unet
from models.unetVggBNNAR1CLF import unetVggBNNAR1CLF
from models.unetVggBNNAR1CLFRes import unetVggBNNAR1CLFRes
from models.unetVggBNNAR1CLFEnc import unetVggBNNAR1CLFEnc
from models.utils import count_parameters
from utils.train import *
from utils.medi import *
from utils.data import *
from utils.files import *
from utils.test import *

if __name__ == '__main__':

    # default parameters
    flag_smv = 1
    flag_gen = 1
    trans = 0
    scale = 1
    if flag_smv:
        voxel_size = (1, 1, 3)
    else:
        voxel_size = (1, 1, 1)

    patchSize = (128, 128, 128)
    extraction_step = (42, 42, 42)

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_cfl', type=int, default=0)
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--weight_dir', type=str, default='weight_cv')  # or 'weights_VI'
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    # total, used = os.popen(
    #     '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    #         ).read().split('\n')[int(opt['gpu_id'])].split(',')
    
    # total = int(total)
    # used = int(used)

    # print('Total memory is {0} MB'.format(total))
    # print('Used memory is {0} MB'.format(used))

    # max_mem = int(total*0.8)
    # block_mem = max_mem - used
    
    # x = torch.rand((256, 1024, block_mem)).cuda()
    # x = torch.rand((2, 2)).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM/'

    cfl = opt['flag_cfl']
    val = opt['case_validation']
    test = opt['case_test']

    # network
    if opt['flag_cfl'] == 0:
        unet3d = Unet(
            input_channels=1, 
            output_channels=1, 
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )

    elif opt['flag_cfl'] == 1:
        unet3d = unetVggBNNAR1CLF(
            input_channels=1,
            output_channels=1,
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )
    elif opt['flag_cfl'] == 2:
        unet3d = unetVggBNNAR1CLFRes(
            input_channels=1,
            output_channels=1,
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )
    elif opt['flag_cfl'] == 3:
        unet3d = unetVggBNNAR1CLFEnc(
            input_channels=1,
            output_channels=1,
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )

    print('{0} trainable parameters in total'.format(count_parameters(unet3d)))
    unet3d.to(device)
    unet3d.load_state_dict(torch.load(rootDir+opt['weight_dir']+'/cfl={0}_validation={1}_test={2}'.format(cfl, val, test)+'.pt'))
    # unet3d.load_state_dict(torch.load(rootDir+opt['weight_dir']+'/weights_vi_cosmos.pt'))
    unet3d.eval()

    QSMs, STDs, RDFs = [], [], []
    RMSEs, Fidelities = [], []
    for test_dir in range(0, 5):
        dataLoader = COSMOS_data_loader(
            split='Test',
            case_validation=val,
            case_test=test,
            test_dir=test_dir,
            patchSize=patchSize, 
            extraction_step=extraction_step,
            voxel_size=voxel_size,
            flag_smv=flag_smv,
            flag_gen=flag_gen
        )
        # dataLoader = Patient_data_loader(patientType='MS_new', patientID=3)
        testLoader = data.DataLoader(dataLoader, batch_size=1, shuffle=False)

        patches_means, patches_stds = [], []
        for idx, (rdfs, masks, weights, qsms) in enumerate(testLoader):
        
            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
            # count time of PDI
            t0 = time.time()
            
            means = unet3d(rdfs)[:, 0, ...]
            means = np.asarray(means.cpu().detach())
            patches_means.append(means)

            time_PDI = time.time() - t0
            print('GPU time = {0}'.format(time_PDI))
            
        patches_means = np.concatenate(patches_means, axis=0)

        chi_true = np.squeeze(np.asarray(qsms.cpu().detach()))
        chi_recon = np.squeeze(patches_means)
        rdf_measured = np.squeeze(np.asarray(rdfs.cpu().detach()))

        RMSEs.append(compute_rmse(chi_recon, chi_true))
        Fidelities.append(compute_fidelity_error(chi_recon, rdf_measured, voxel_size=voxel_size))

        QSM = patches_means
        STD = patches_stds

        QSMs.append(QSM)
        STDs.append(STD)
        RDFs.append(rdf_measured)

    QSMs, STDs = np.asarray(QSMs), np.asarray(STDs)
    print('RMSE = {}'.format(sum(RMSEs)/len(RMSEs)))
    print('Fidelity Loss = {}'.format(sum(Fidelities)/len(Fidelities)))

    adict = {}
    adict['QSMs'] = np.squeeze(np.moveaxis(QSMs, 0, -1))
    sio.savemat(rootDir+'/result_cv/QSMs_{0}{1}{2}'.format(math.floor(cfl), math.floor(val), math.floor(test))+'.mat', adict)

    adict = {}
    adict['STDs'] = np.squeeze(np.moveaxis(STDs, 0, -1))
    sio.savemat(rootDir+'/result_cv/STDs_{0}{1}{2}'.format(math.floor(cfl), math.floor(val), math.floor(test))+'.mat', adict)



