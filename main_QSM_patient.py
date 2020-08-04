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

if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=20)
    parser.add_argument('--flag_r_train', type=int, default=0)
    parser.add_argument('--flag_init', type=int, default=0)  # 0 for COSMOS pre-train, 1 for MEDI VI
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}

    # python main_QSM_patient.py --flag_r_train=1 --patientID=8 (or 16)

    Lambda_tv = opt['lambda_tv']
    patient_type = opt['patient_type']
    patientID = opt['patientID']
    flag_r_train = opt['flag_r_train']
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
    dataLoader_train = Patient_data_loader(patientType=patient_type, patientID=patientID)

    # parameters
    niter = 100
    sigma = 0
    # lr = 1e-3
    lr = 5e-4
    batch_size = 1
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 5  # 5 default
    r = 3e-3

    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size
    S = SMV_kernel(volume_size, voxel_size, radius=5)
    D = dipole_kernel(volume_size, voxel_size, B0_dir)
    D = np.real(S * D)

    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    # # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(3, 8)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=flag_r_train
    )
    # unet3d = Unet(
    #     input_channels=1, 
    #     output_channels=2, 
    #     num_filters=[2**i for i in range(3, 8)],
    #     flag_rsa=2
    # )
    unet3d.to(device)
    if flag_init == 0:
        weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt')
    else:
        weights_dict = torch.load(rootDir+folder_weights_VI+'/weights_lambda_tv={0}.pt'.format(Lambda_tv))
    weights_dict['r'] = (torch.ones(1)*r).to(device)
    unet3d.load_state_dict(weights_dict)

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    loss_iters = np.zeros(niter)
    while epoch < niter:
        epoch += 1

        # training phase
        for idx, (rdfs, masks, weights, wGs) in enumerate(trainLoader):

            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            if epoch == 1:
                unet3d.eval()

                means = unet3d(rdfs)[:, 0, ...]
                stds = unet3d(rdfs)[:, 1, ...]
                QSM = np.squeeze(np.asarray(means.cpu().detach()))
                STD = np.squeeze(np.asarray(stds.cpu().detach()))

                print('Saving initial results')
                adict = {}
                adict['QSM'] = QSM
                sio.savemat(rootDir+'/QSM_0.mat', adict)

                adict = {}
                adict['STD'] = STD
                sio.savemat(rootDir+'/STD_0.mat', adict)

                # unet3d.train()

            loss_kl,  loss_tv, loss_expectation = BayesianQSM_train(
                model=unet3d,
                input_RDFs=rdfs,
                in_loss_RDFs=rdfs-trans*scale,
                QSMs=0,
                Masks=masks,
                fidelity_Ws=weights,
                gradient_Ws=wGs,
                D=D,
                flag_COSMOS=0,
                optimizer=optimizer,
                sigma_sq=0,
                Lambda_tv=Lambda_tv,
                voxel_size=voxel_size,
                K=K
            )

            loss_total = loss_kl + loss_tv + loss_expectation

            print('epochs: [%d/%d], time: %ds, Lambda_tv: %f, Entropy loss: %2f, TV_loss: %2f, Expectation_loss: %2f, r: %f, Total loss: %2f'
                % (epoch, niter, time.time()-t0, Lambda_tv, loss_kl, loss_tv, loss_expectation, unet3d.r, loss_total))

            loss_iters[epoch-1] = loss_total

            if epoch % 10 == 0:
                unet3d.eval()

                stds = unet3d(rdfs)[:, 1, ...]
                STD = np.squeeze(np.asarray(stds.cpu().detach()))
                adict = {}
                adict['STD'] = STD
                sio.savemat(rootDir+'/STD_f_{0}.mat'.format(epoch), adict)
                
                # unet3d.train()

    unet3d.eval()

    means = unet3d(rdfs)[:, 0, ...]
    stds = unet3d(rdfs)[:, 1, ...]
    QSM = np.squeeze(np.asarray(means.cpu().detach()))
    STD = np.squeeze(np.asarray(stds.cpu().detach()))

    adict = {}
    adict['QSM'] = QSM
    if flag_init == 0:
        sio.savemat(rootDir+'/QSM_f.mat', adict)
    else:
        sio.savemat(rootDir+'/QSM_VI_f.mat', adict)
    
    adict = {}
    adict['STD'] = STD
    if flag_init == 0:
        sio.savemat(rootDir+'/STD_f.mat', adict)
    else:
        sio.savemat(rootDir+'/STD_VI_f.mat', adict)

    np.save(rootDir+'/loss_{0}_{1}'.format(patient_type, patientID), loss_iters)