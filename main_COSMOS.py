import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
from torch.utils import data
from loader.COSMOS_data_loader import COSMOS_data_loader
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM/weight'

    # parameters
    niter = 100
    lr = 1e-3
    batch_size = 8
    sigma = 1*10**(-5)
    sigma_sq = sigma**2
    flag_smv = 1
    flag_gen = 1

    if flag_smv:
        B0_dir = (0, 0, 1)
        patchSize = (64, 64, 21)
        # patchSize_padding = (64, 64, 128)
        patchSize_padding = patchSize
        extraction_step = (21, 21, 7)
        voxel_size = (1, 1, 3)
        S = SMV_kernel(patchSize, voxel_size, radius=5)
        D = dipole_kernel(patchSize_padding, voxel_size, B0_dir)
        D = np.real(S * D)
    else:
        B0_dir = (0, 0, 1)
        patchSize = (64, 64, 64)
        # patchSize_padding = (64, 64, 128)
        patchSize_padding = patchSize
        extraction_step = (21, 21, 21)
        voxel_size = (1, 1, 1)
        D = dipole_kernel(patchSize_padding, voxel_size, B0_dir)

    # network
    unet3d = Unet(input_channels=1, output_channels=2, num_filters=[2**i for i in range(3, 8)])
    unet3d.to(device)

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))
    # logger
    logger = Logger('logs', rootDir, sigma)

    # dataloader
    dataLoader_train = COSMOS_data_loader(
        split='Train',
        patchSize=patchSize,
        extraction_step=extraction_step,
        voxel_size=voxel_size,
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    dataLoader_val = COSMOS_data_loader(
        split='Val',
        patchSize=patchSize,
        extraction_step=extraction_step,
        voxel_size=voxel_size,
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True)

    # dataLoader_train = dataLoader_val
    # trainLoader = valLoader

    epoch = 0
    gen_iterations = 1
    display_iters = 5
    loss_kl_sum = 0
    loss_expectation_sum = 0
    Validation_loss = []

    while epoch < niter:
        epoch += 1

        # training phase
        for idx, (rdfs, masks, weights, qsms) in enumerate(trainLoader):
            if gen_iterations%display_iters == 0:
                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, sigma: %f'
                    % (epoch, niter, idx, dataLoader_train.num_samples//batch_size+1, time.time()-t0, sigma))
                print('KL_loss: %f, Expectation_loss: %f' % (loss_kl_sum/display_iters, loss_expectation_sum/display_iters))
                if epoch > 1:
                    print('Validation loss of last epoch: %.2f' % (Validation_loss[-1]))

                loss_kl_sum = 0
                loss_expectation_sum = 0

            rdfs = rdfs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            qsms = qsms.to(device, dtype=torch.float)

            loss_kl,  loss_expectation = BayesianQSM_train(
                unet3d=unet3d,
                input_RDFs=rdfs,
                in_loss_RDFs=rdfs,
                QSMs=qsms,
                Masks=masks,
                fidelity_Ws=weights,
                gradient_Ws=0,
                D=D,
                flag_COSMOS=1,
                optimizer=optimizer,
                sigma_sq=sigma_sq,
                Lambda_tv=0,
                voxel_size=voxel_size,
                K=100
            )

            loss_kl_sum += loss_kl
            loss_expectation_sum += loss_expectation
            gen_iterations += 1

        # validation phase
        loss_total = 0
        idx = 0
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdfs, masks, weights, qsms) in enumerate(valLoader):
                idx += 1
                rdfs = rdfs.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                qsms = qsms.to(device, dtype=torch.float)
                outputs = unet3d(rdfs)

                loss_kl = loss_KL(outputs, qsms, 1, sigma_sq)
                loss_expectation = loss_Expectation(outputs, qsms, rdfs, weights, 0, D, 1, 0, voxel_size)
                loss_total += loss_kl + loss_expectation

            print('\n Validation loss: %f \n' % (loss_total / idx))
            Validation_loss.append(loss_total / idx)

        logger.print_and_save('Epoch: [%d/%d], Loss in Validation: %.2f' 
        % (epoch, niter, Validation_loss[-1]))

        if Validation_loss[-1] == min(Validation_loss):
            torch.save(unet3d.state_dict(), rootDir+'/weights_sigma={0}_smv={1}'.format(sigma, flag_smv)+'.pt')
