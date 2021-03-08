import os
import time
import numpy as np
import torch
import torch.optim as optim
import random
import argparse

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
# from loader.COSMOS_data_loader import COSMOS_data_loader
from loader.COSMOS_data_loader_QSMnet_p import COSMOS_data_loader
from loader.COSMOS_data_loader_whole import COSMOS_data_loader_whole
from models.unet import Unet
from models.unetVggBNNAR1CLF import unetVggBNNAR1CLF
from models.unetVggBNNAR1CLFRes import unetVggBNNAR1CLFRes
from models.unetVggBNNAR1CLFEnc import unetVggBNNAR1CLFEnc
from models.unetCFT import unetCFT
from models.unetCFTMuSig256 import unetCFTMuSig256
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    # default parameters
    niter = 1000
    lr = 1e-3
    batch_size = 1  
    flag_smv = 1
    flag_gen = 1
    trans = 0  # 0.15
    scale = 1  # 3

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_cfl', type=int, default=0)
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--linear_factor', type=int, default=1)
    parser.add_argument('--weight_dir', type=str, default='weight_cv')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    t0 = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM/' + opt['weight_dir']

    B0_dir = (0, 0, 1)
    if flag_smv:
        voxel_size = (1, 1, 3)
    else:
        voxel_size = (1, 1, 1)

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
    elif opt['flag_cfl'] == 4:
        unet3d = unetCFT(
            input_channels=1,
            output_channels=1,
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )
    elif opt['flag_cfl'] == 5:
        unet3d = unetCFTMuSig256(
            input_channels=1,
            output_channels=1,
            num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
            use_deconv=1
        )

    print(unet3d)
    unet3d.to(device)

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))
    ms = [0.3, 0.5, 0.7, 0.9]
    ms = [np.floor(m * niter).astype(int) for m in ms]
    scheduler = MultiStepLR(optimizer, milestones = ms, gamma = 0.5)

    # logger
    logger = Logger('logs', rootDir, opt['flag_cfl'], opt['case_validation'], opt['case_test'])

    # dataloader
    # dataLoader_train = COSMOS_data_loader_whole(
    #     split='Train',
    #     voxel_size=voxel_size,
    #     case_validation=opt['case_validation'],
    #     case_test=opt['case_test'],
    #     flag_smv=flag_smv,
    #     flag_gen=flag_gen)
    # trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_val = COSMOS_data_loader_whole(
        split='Val',
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_train = dataLoader_val
    trainLoader = valLoader

    epoch = 0
    gen_iterations = 1
    display_iters = 5
    loss_l1_sum = 0
    Validation_loss = []
    loss_L1 = lossL1()

    while epoch < niter:
        epoch += 1

        # training phase
        unet3d.train()
        for idx, (rdfs, qsms, masks, weights, wGs, D) in enumerate(trainLoader):
            if gen_iterations%display_iters == 0:
                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, case_validation: %f'
                    % (epoch, niter, idx, dataLoader_train.num_samples//batch_size+1, time.time()-t0, opt['case_validation']))
                print('L1_loss: %f' % (loss_l1_sum/display_iters))
                if epoch > 1:
                    print('Validation loss of last epoch: %f' % (Validation_loss[-1]))

                loss_l1_sum = 0

            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)

            loss_l1 = BayesianQSM_train(
                model=unet3d,
                input_RDFs=rdfs,
                in_loss_RDFs=rdfs,
                QSMs=qsms,
                Masks=masks,
                fidelity_Ws=0,
                gradient_Ws=0,
                D=D,
                flag_COSMOS=1,
                optimizer=optimizer,
                sigma_sq=0,
                Lambda_tv=0,
                voxel_size=voxel_size,
                flag_l1=1
            )

            loss_l1_sum += loss_l1
            gen_iterations += 1

        scheduler.step(epoch)

        # validation phase
        unet3d.eval()
        loss_total = 0
        idx = 0
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdfs, qsms, masks, weights, wGs, D) in enumerate(valLoader):
                idx += 1
                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                outputs = unet3d(rdfs)

                # loss_total += loss_L1(outputs[:, 0:1, ...], qsms)

                loss_l1 = BayesianQSM_train(
                    model=unet3d,
                    input_RDFs=rdfs,
                    in_loss_RDFs=rdfs,
                    QSMs=qsms,
                    Masks=masks,
                    fidelity_Ws=0,
                    gradient_Ws=0,
                    D=D,
                    flag_COSMOS=1,
                    optimizer=optimizer,
                    sigma_sq=0,
                    Lambda_tv=0,
                    voxel_size=voxel_size,
                    flag_l1=1,
                    flag_test=1
                )
                loss_total += loss_l1

            print('\n Validation loss: %f \n' % (loss_total / idx))
            Validation_loss.append(loss_total / idx)

        logger.print_and_save('Epoch: [%d/%d], Loss in Validation: %f'
        % (epoch, niter, Validation_loss[-1]))

        if Validation_loss[-1] == min(Validation_loss):
            torch.save(unet3d.state_dict(), rootDir+'/cfl={0}_validation={1}_test={2}'.format(opt['flag_cfl'], opt['case_validation'], opt['case_test'])+'.pt')