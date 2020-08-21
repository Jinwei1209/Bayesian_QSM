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
from models.unet import Unet
from models.resBlock import ResBlock
from models.unetag import UnetAg
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    # default parameters
    niter = 100
    lr = 1e-3
    batch_size = 32
    flag_smv = 1
    flag_gen = 1
    trans = 0  # 0.15
    scale = 1  # 3

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_rsa', type=int, default=-1)
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--linear_factor', type=int, default=1)
    parser.add_argument('--weight_dir', type=str, default='weight_cv')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()

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
    rootDir = '/data/Jinwei/Bayesian_QSM/' + opt['weight_dir']

    if flag_smv:
        B0_dir = (0, 0, 1)
        patchSize = (64, 64, 32)  # (64, 64, 21)
        # patchSize_padding = (64, 64, 128)
        patchSize_padding = patchSize
        extraction_step = (21, 21, 6)  # (21, 21, 7)
        voxel_size = (1, 1, 3)
        D = dipole_kernel(patchSize_padding, voxel_size, B0_dir)
        # S = SMV_kernel(patchSize, voxel_size, radius=5)
        # D = np.real(S * D)
    else:
        B0_dir = (0, 0, 1)
        patchSize = (64, 64, 64)
        # patchSize_padding = (64, 64, 128)
        patchSize_padding = patchSize
        extraction_step = (21, 21, 21)
        voxel_size = (1, 1, 1)
        D = dipole_kernel(patchSize_padding, voxel_size, B0_dir)

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
        use_deconv=1,
        flag_rsa=opt['flag_rsa']
    )

    resnet = ResBlock(
        input_dim=1, 
        filter_dim=32,
        output_dim=1, 
    )

    unet3d.to(device)
    resnet.to(device)
    
    # initialize Pre-trained Unet
    weights_dict = torch.load(rootDir+'/weight_qsmnet_p/linear_factor=1_validation=6_test=7.pt')
    unet3d.load_state_dict(weights_dict)
    unet3d.eval()

    # optimizer
    optimizer = optim.Adam(resnet.parameters(), lr = lr, betas=(0.5, 0.999))
    ms = [0.3, 0.5, 0.7, 0.9]
    ms = [np.floor(m * niter).astype(int) for m in ms]
    scheduler = MultiStepLR(optimizer, milestones = ms, gamma = 0.5)

    # logger
    logger = Logger('logs', rootDir, opt['linear_factor'], opt['case_validation'], opt['case_test'])

    # dataloader
    dataLoader_train = COSMOS_data_loader(
        split='Train',
        patchSize=patchSize,
        extraction_step=extraction_step,
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen,
        linear_factor=opt['linear_factor'])
    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_val = COSMOS_data_loader(
        split='Val',
        patchSize=patchSize,
        extraction_step=extraction_step,
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen,
        linear_factor=opt['linear_factor'])
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, pin_memory=True)

    # dataLoader_train = dataLoader_val
    # trainLoader = valLoader

    epoch = 0
    gen_iterations = 1
    display_iters = 5
    loss_l1_sum = 0
    Validation_loss = []
    loss_L1 = lossL1()

    while epoch < niter:
        epoch += 1

        # training phase
        resnet.train()
        for idx, (rdfs, masks, qsms) in enumerate(trainLoader):
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

            qsm_unet3d = unet3d(rdfs)

            loss_l1 = BayesianQSM_train(
                model=resnet,
                input_RDFs=qsm_unet3d,
                in_loss_RDFs=qsm_unet3d,
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
        resnet.eval()
        loss_total = 0
        idx = 0
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdfs, masks, qsms) in enumerate(valLoader):
                idx += 1
                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)

                qsm_unet3d = unet3d(rdfs)

                loss_l1 = BayesianQSM_train(
                    model=resnet,
                    input_RDFs=qsm_unet3d,
                    in_loss_RDFs=qsm_unet3d,
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
            torch.save(resnet.state_dict(), rootDir+'/linear_factor={0}_validation={1}_test={2}_resnet'.format(opt['linear_factor'], opt['case_validation'], opt['case_test'])+'.pt')
