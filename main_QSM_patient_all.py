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
 
    t0 = time.time()
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # parameters
    niter = 2000
    sigma = 0
    lr = 1e-3
    batch_size = 1
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 5  # 5 default

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=10)
    parser.add_argument('--flag_test', type=int, default=0)
    parser.add_argument('--epoch_test', type=int, default=10)
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}
    # run: (700 the best)
    # python main_QSM_patient_all.py --lambda_tv=10 --flag_test=1 --epoch_test=700 --patientID=16 (or 8)
    # python main_QSM_patient_all.py --lambda_tv=10 --flag_test=1 --epoch_test=400 (300 too good, no shadow at all) --patientID=8

    flag_test = opt['flag_test']
    epoch_test = opt['epoch_test']
    Lambda_tv = opt['lambda_tv']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(3, 8)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=1
    )
    # unet3d = Unet(
    #     input_channels=1, 
    #     output_channels=2, 
    #     num_filters=[2**i for i in range(3, 8)],
    #     flag_rsa=2
    # )
    unet3d.to(device)
    val_loss = []

    # training phase
    if not flag_test:

        # dataloader
        dataLoader_train = Patient_data_loader_all(patientType='ICH')
        trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

        dataLoader_val = Patient_data_loader(patientType='ICH', patientID=14)
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True)

        voxel_size = dataLoader_train.voxel_size
        volume_size = dataLoader_train.volume_size
        S = SMV_kernel(volume_size, voxel_size, radius=5)
        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        D = np.real(S * D)

        unet3d.load_state_dict(torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt'))
        # optimizer
        optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

        epoch = 0
        while epoch < niter:
            epoch += 1

            unet3d.train()
            for idx, (rdfs, masks, weights, wGs) in enumerate(trainLoader):

                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)

                loss_kl,  loss_expectation = BayesianQSM_train(
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

                print('epochs: [%d/%d], time: %ds, Lambda_tv: %f, KL_loss: %f, Expectation_loss: %f, r: %f'
                    % (epoch, niter, time.time()-t0, Lambda_tv, loss_kl, loss_expectation, unet3d.r))

            unet3d.eval()
            for idx, (rdfs, masks, weights, wGs) in enumerate(valLoader):

                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)

                # calculate KLD
                outputs = unet3d(rdfs)
                loss_kl = loss_KL(outputs=outputs, QSMs=0, flag_COSMOS=0, sigma_sq=0)
                loss_expectation, loss_tv = loss_Expectation(
                    outputs=outputs, QSMs=0, in_loss_RDFs=rdfs-trans*scale, fidelity_Ws=weights, 
                    gradient_Ws=wGs, D=D, flag_COSMOS=0, Lambda_tv=Lambda_tv, voxel_size=voxel_size, K=K)
                loss_total = (loss_kl + loss_expectation + loss_tv).item()
                print('KL Divergence on validation set = {0}'.format(loss_total))
            
            val_loss.append(loss_total)
            if val_loss[-1] == min(val_loss):
                if Lambda_tv:
                    torch.save(unet3d.state_dict(), rootDir+'/weights_VI/weights_lambda_tv={0}.pt'.format(Lambda_tv))
                else:
                    torch.save(unet3d.state_dict(), rootDir+'/weights_VI/weights_no_prior.pt')

    # test phase
    else:
        # dataloader
        dataLoader_test = Patient_data_loader(patientType='ICH', patientID=opt['patientID'])
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=True)

        voxel_size = dataLoader_test.voxel_size
        volume_size = dataLoader_test.volume_size
        S = SMV_kernel(volume_size, voxel_size, radius=5)
        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        D = np.real(S * D)

        if Lambda_tv:
            # unet3d.load_state_dict(torch.load(rootDir+'/weights_VI/weights_lambda_tv={0}_{1}.pt'.format(Lambda_tv, epoch_test)))
            # unet3d.load_state_dict(torch.load(rootDir+'/weights_VI/lambda=10/weights_{0}.pt'.format(epoch_test)))
            unet3d.load_state_dict(torch.load(rootDir+'/weights_VI/weights_lambda_tv={0}.pt'.format(Lambda_tv)))
        else:
            unet3d.load_state_dict(torch.load(rootDir+'/weights_VI/weights_no_prior_{0}.pt'.format(epoch_test)))
        unet3d.eval()

        for idx, (rdfs, masks, weights, wGs) in enumerate(testLoader):

            print('Saving test data')

            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            outputs = unet3d(rdfs)
            means = outputs[:, 0, ...]
            stds = outputs[:, 1, ...]
            QSM = np.squeeze(np.asarray(means.cpu().detach()))
            STD = np.squeeze(np.asarray(stds.cpu().detach()))

            # calculate KLD
            loss_kl = loss_KL(outputs=outputs, QSMs=0, flag_COSMOS=0, sigma_sq=0)
            loss_expectation, loss_tv = loss_Expectation(
                outputs=outputs, QSMs=0, in_loss_RDFs=rdfs-trans*scale, fidelity_Ws=weights, 
                gradient_Ws=wGs, D=D, flag_COSMOS=0, Lambda_tv=Lambda_tv, voxel_size=voxel_size, K=K)
            loss_total = (loss_kl + loss_expectation + loss_tv).item()
            print('KL Divergence = {0}'.format(loss_total))

        if Lambda_tv:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/QSM_VI.mat', adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/STD_VI.mat', adict)
        else:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/QSM_VI_no_prior.mat', adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/STD_VI_no_prior.mat', adict)