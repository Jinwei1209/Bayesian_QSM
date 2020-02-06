import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
from torch.utils import data
from loader.QSM_data_loader2 import QSM_data_loader2
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parameters
    niter = 1000
    lr = 2e-4
    patchSize = (64, 64, 32)
    voxel_size = (0.9375, 0.9375, 3)
    B0_dir = (0, 0, 1)
    radius = 5
    sigma_sq = (3*10**(-5))**2
    Lambda_tv = 10
    D = dipole_kernel(patchSize, voxel_size, B0_dir)
    S = SMV_kernel(patchSize, voxel_size, radius)
    D = S*D

    GPU = 2

    if GPU == 1:
        rootDir = '/home/sdc/Jinwei/BayesianQSM'
    elif GPU == 2:
        rootDir = '/data/Jinwei/Bayesian_QSM/weight'

    # network
    unet3d = Unet(input_channels=1, output_channels=2, num_filters=[2**i for i in range(5, 10)])
    unet3d.to(device)
    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))
    # dataloader
    dataLoader = QSM_data_loader2(GPU=GPU, patchSize=patchSize)
    trainLoader = data.DataLoader(dataLoader, batch_size=1, shuffle=False)

    epoch = 0
    while epoch < niter:

        epoch += 1
        for idx, (input_RDFs, in_loss_RDFs, QSMs, Masks, \
            fidelity_Ws, gradient_Ws, flag_COSMOS) in enumerate(trainLoader):

            input_RDFs = input_RDFs[0, ...].to(device, dtype=torch.float)
            in_loss_RDFs = in_loss_RDFs[0, ...].to(device, dtype=torch.float)
            QSMs = QSMs[0, ...].to(device, dtype=torch.float)
            Masks = Masks[0, ...].to(device, dtype=torch.float)
            fidelity_Ws = fidelity_Ws[0, ...].to(device, dtype=torch.float)
            gradient_Ws = gradient_Ws[0, ...].to(device, dtype=torch.float)

            # loss_KL, loss_expectation = BayesianQSM_train(
            #     unet3d=unet3d,
            #     input_RDFs=input_RDFs,
            #     in_loss_RDFs=in_loss_RDFs,
            #     QSMs=QSMs,
            #     Masks=Masks,
            #     fidelity_Ws=fidelity_Ws,
            #     gradient_Ws=gradient_Ws,
            #     D=D,
            #     flag_COSMOS=flag_COSMOS,
            #     optimizer=optimizer,
            #     sigma_sq=sigma_sq,
            #     Lambda_tv=Lambda_tv,
            #     voxel_size=voxel_size
            # )

            errl1 = BayesianQSM_train(
                unet3d=unet3d,
                input_RDFs=input_RDFs,
                in_loss_RDFs=in_loss_RDFs,
                QSMs=QSMs,
                Masks=Masks,
                fidelity_Ws=fidelity_Ws,
                gradient_Ws=gradient_Ws,
                D=D,
                flag_COSMOS=flag_COSMOS,
                optimizer=optimizer,
                sigma_sq=sigma_sq,
                Lambda_tv=Lambda_tv,
                voxel_size=voxel_size
            )

            print(errl1)
            print('\n')

        print('Finish current epoch')
        torch.save(unet3d.state_dict(), rootDir+'/weights.pt')
            

