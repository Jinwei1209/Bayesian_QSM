import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
import random
import argparse

from torch.utils import data
from loader.COSMOS_data_loader_whole import COSMOS_data_loader_whole
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':
 
    t0 = time.time()
    rootDir = '/data/Jinwei/Bayesian_QSM'
    folder_weights_VI = '/weights_VI'

    # parameters
    niter = 1000
    sigma = 0
    lr = 1e-3
    batch_size = 1
    flag_smv = 1
    flag_gen = 1
    trans = 0.15
    scale = 3
    K = 5  # 5 default
    r = 3e-3

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=float, default=20)
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--flag_r_train', type=int, default=0)  # fixed r in COSMOS
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Lambda_tv = opt['lambda_tv']
    flag_r_train = opt['flag_r_train']

    B0_dir = (0, 0, 1)
    if flag_smv:
        voxel_size = (1, 1, 3)
    else:
        voxel_size = (1, 1, 1)

    # # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(3, 8)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=flag_r_train,
        r=r
    )
    unet3d.to(device)

    # dataloader
    dataLoader_train = COSMOS_data_loader_whole(
        split='Train',
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_val = COSMOS_data_loader_whole(
        split='Val',
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, pin_memory=True)

    weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt')
    weights_dict['r'] = (torch.ones(1)*r).to(device)
    unet3d.load_state_dict(weights_dict)

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

    val_loss = []
    epoch = 0
    while epoch < niter:
        epoch += 1

        unet3d.train()
        for idx, (rdfs, qsms, masks, weights, wGs, D) in enumerate(trainLoader):
            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            loss_kl,  loss_tv, loss_expectation = BayesianQSM_train(
                model=unet3d,
                input_RDFs=rdfs,
                in_loss_RDFs=rdfs-trans*scale,
                QSMs=0,
                Masks=masks,
                fidelity_Ws=weights,
                gradient_Ws=wGs,
                D=np.asarray(D[0, ...]),
                flag_COSMOS=0,
                optimizer=optimizer,
                sigma_sq=0,
                Lambda_tv=Lambda_tv,
                voxel_size=voxel_size,
                K=K
            )

            print('epochs: [%d/%d], time: %ds, Lambda_tv: %f, KL_loss: %f, Expectation_loss: %f, r: %f'
                % (epoch, niter, time.time()-t0, Lambda_tv, loss_kl+loss_tv, loss_expectation, unet3d.r))

        unet3d.eval()
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdfs, qsms, masks, weights, wGs, D) in enumerate(valLoader):

                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)

                # calculate KLD
                outputs = unet3d(rdfs)
                loss_kl = loss_KL(outputs=outputs, QSMs=0, flag_COSMOS=0, sigma_sq=0)
                loss_expectation, loss_tv = loss_Expectation(
                    outputs=outputs, QSMs=0, in_loss_RDFs=rdfs-trans*scale, fidelity_Ws=weights, 
                    gradient_Ws=wGs, D=np.asarray(D[0, ...]), flag_COSMOS=0, Lambda_tv=Lambda_tv, voxel_size=voxel_size, K=K)
                loss_total = (loss_kl + loss_expectation + loss_tv).item()
                print('KL Divergence on validation set = {0}'.format(loss_total))
        
        val_loss.append(loss_total)
        if val_loss[-1] == min(val_loss):
            if Lambda_tv:
                torch.save(unet3d.state_dict(), rootDir+folder_weights_VI+'/weights_vi_cosmos_{}.pt'.format(Lambda_tv))