import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
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
    # niter = 2000
    sigma = 0
    lr = 1e-3
    batch_size = 1
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 5  # 5 default
    r = 3e-3 # 3e-3 for PDI-VI0, 0.001312 for PDI-VI

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=20)
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--flag_test', type=int, default=0)
    parser.add_argument('--flag_r_train', type=int, default=0)
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)  # for test
    opt = {**vars(parser.parse_args())}

    # for test:
    # python main_QSM_patient_all.py --gpu_id=1 --flag_test=1 --lambda_tv=20

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flag_test = opt['flag_test']
    Lambda_tv = opt['lambda_tv']
    niter = opt['niter']
    flag_r_train = opt['flag_r_train']
    patient_type = opt['patient_type']

    if patient_type == 'ICH':
        valID = 14
        folder_weights_VI = '/weights_VI'
    elif patient_type == 'MS_old' or 'MS_new':
        valID = 7
        folder_weights_VI = '/weights_VI2'

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
        dataLoader_train = Patient_data_loader_all(patientType=patient_type)
        trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

        dataLoader_val = Patient_data_loader(patientType=patient_type, patientID=valID)
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True)

        voxel_size = dataLoader_val.voxel_size
        volume_size = dataLoader_val.volume_size
        S = SMV_kernel(volume_size, voxel_size, radius=5)
        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        D_val = np.real(S * D)

        # weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt')
        # weights_dict['r'] = (torch.ones(1)*r).to(device)
        # unet3d.load_state_dict(weights_dict)

        # optimizer
        optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

        epoch = 0
        while epoch < niter:
            epoch += 1

            unet3d.train()
            for idx, (rdfs, masks, weights, wGs, D) in enumerate(trainLoader):

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
                        gradient_Ws=wGs, D=D_val, flag_COSMOS=0, Lambda_tv=Lambda_tv, voxel_size=voxel_size, K=K)
                    loss_total = (loss_kl + loss_expectation + loss_tv).item()
                    print('KL Divergence on validation set = {0}'.format(loss_total))
            
            val_loss.append(loss_total)
            if val_loss[-1] == min(val_loss):
                if Lambda_tv:
                    # torch.save(unet3d.state_dict(), rootDir+folder_weights_VI+'/weights_lambda_tv={0}_epoch={1}.pt'.format(Lambda_tv, niter))
                    torch.save(unet3d.state_dict(), rootDir+folder_weights_VI+'/weights_tv_no_initial.pt')  # no_initial_r: PDI-VI0 with r fixed = 3e-3, 
                                                                                                            # no_initial: PDI-VI0 with r learned
                else:
                    torch.save(unet3d.state_dict(), rootDir+folder_weights_VI+'/weights_no_prior.pt')

    # test phase
    else:
        # dataloader
        dataLoader_test = Patient_data_loader(patientType=patient_type, patientID=opt['patientID'])
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=True)

        voxel_size = dataLoader_test.voxel_size
        volume_size = dataLoader_test.volume_size
        S = SMV_kernel(volume_size, voxel_size, radius=5)
        D = dipole_kernel(volume_size, voxel_size, B0_dir)
        D = np.real(S * D)

        if Lambda_tv:
            # weights_dict = torch.load(rootDir+folder_weights_VI+'/weights_lambda_tv={0}_epoch={1}.pt'.format(Lambda_tv, niter))
            # weights_dict = torch.load(rootDir+folder_weights_VI+'/weights_lambda_tv={0}.pt'.format(Lambda_tv))
            weights_dict = torch.load(rootDir+folder_weights_VI+'/weights_tv_no_initial_r.pt')
            weights_dict['r'] = (torch.ones(1)*r).to(device)
            unet3d.load_state_dict(weights_dict)
        else:
            unet3d.load_state_dict(torch.load(rootDir+folder_weights_VI+'/weights_no_prior.pt'))
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
            loss_total = loss_kl.item() + loss_tv.item() + loss_expectation.item()
            print('r: %f, Entropy loss: %2f, TV_loss: %2f, Expectation_loss: %2f, Total_loss: %2f'
                % (unet3d.r, loss_kl.item(), loss_tv.item(), loss_expectation.item(), loss_total))
        
        if Lambda_tv:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/QSM_VI_ICH{0}.mat'.format(opt['patientID']), adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/STD_VI_ICH{0}.mat'.format(opt['patientID']), adict)
        else:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/QSM_VI_no_prior.mat', adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/STD_VI_no_prior.mat', adict)