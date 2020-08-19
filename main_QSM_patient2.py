import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.utils import data
from loader.Patient_data_loader import Patient_data_loader
from loader.Patient_data_loader_all import Patient_data_loader_all
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

'''
    Patient test for meta-FINE
'''
if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_init', type=int, default=0)  # 0 for linear_factor=1, 1 for linear_factor=4
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}

    # python main_QSM_patient.py --flag_r_train=1 --patientID=8 (or 16)

    patient_type = opt['patient_type']
    patientID = opt['patientID']
    flag_init = opt['flag_init']

    if patient_type == 'ICH':
        folder_weights_VI = '/weights_VI'
    elif patient_type == 'MS_old' or 'MS_new':
        folder_weights_VI = '/weights_VI2'

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # dataloader
    dataLoader_train = Patient_data_loader(
        patientType=patient_type, 
        patientID=patientID,
        flag_input=1
        )

    # parameters
    niter = 1
    # lr = 1e-3
    lr = 5e-4
    batch_size = 1
    B0_dir = (0, 0, 1)

    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size
    S = SMV_kernel(volume_size, voxel_size, radius=5)
    D = dipole_kernel(volume_size, voxel_size, B0_dir)
    D = np.real(S * D)

    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    # # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
        use_deconv=1,
        flag_rsa=0
    )
    unet3d.to(device)
    if flag_init == 0:
        weights_dict = torch.load(rootDir+'/weight_qsmnet_p/linear_factor=1_validation=6_test=7.pt')
    else:
        weights_dict = torch.load(rootDir+'/weight_qsmnet_p/linear_factor=4_validation=6_test=7.pt')
    unet3d.load_state_dict(weights_dict)

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    loss_iters = np.zeros(niter)
    while epoch < niter:
        epoch += 1

        # training phase
        for idx, (rdf_inputs, rdfs, masks, weights, wGs) in enumerate(trainLoader):
            
            rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
            rdfs = rdfs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            if epoch == 1:
                unet3d.eval()

                QSMnet = unet3d(rdf_inputs)[:, 0, ...]
                QSMnet = np.squeeze(np.asarray(QSMnet.cpu().detach()))

                print('Saving initial results')
                adict = {}
                adict['QSMnet'] = QSMnet
                sio.savemat(rootDir+'/QSMnet.mat', adict)

    #         loss_kl,  loss_tv, loss_expectation = BayesianQSM_train(
    #             model=unet3d,
    #             input_RDFs=rdf_inputs,
    #             in_loss_RDFs=rdfs,
    #             QSMs=0,
    #             Masks=masks,
    #             fidelity_Ws=weights,
    #             gradient_Ws=wGs,
    #             D=D,
    #             flag_COSMOS=0,
    #             optimizer=optimizer,
    #             sigma_sq=0,
    #             Lambda_tv=Lambda_tv,
    #             voxel_size=voxel_size,
    #             K=K
    #         )

    #         loss_total = loss_kl + loss_tv + loss_expectation

    #         print('epochs: [%d/%d], time: %ds, Lambda_tv: %f, Entropy loss: %2f, TV_loss: %2f, Expectation_loss: %2f, r: %f, Total loss: %2f'
    #             % (epoch, niter, time.time()-t0, Lambda_tv, loss_kl, loss_tv, loss_expectation, unet3d.r, loss_total))

    #         loss_iters[epoch-1] = loss_total

    #         if epoch % 10 == 0:
    #             unet3d.eval()

    #             stds = unet3d(rdfs)[:, 1, ...]
    #             STD = np.squeeze(np.asarray(stds.cpu().detach()))
    #             adict = {}
    #             adict['STD'] = STD
    #             sio.savemat(rootDir+'/STD_f_{0}.mat'.format(epoch), adict)

    # unet3d.eval()

    # means = unet3d(rdfs)[:, 0, ...]
    # stds = unet3d(rdfs)[:, 1, ...]
    # QSM = np.squeeze(np.asarray(means.cpu().detach()))
    # STD = np.squeeze(np.asarray(stds.cpu().detach()))

    # adict = {}
    # adict['QSM'] = QSM
    # if flag_init == 0:
    #     sio.savemat(rootDir+'/QSM_f.mat', adict)
    # else:
    #     sio.savemat(rootDir+'/QSM_VI_f.mat', adict)