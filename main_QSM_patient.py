import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
from torch.utils import data
from loader.Patient_data_loader import Patient_data_loader
from loader.Patient_data_loader_all import Patient_data_loader_all
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # dataloader
    dataLoader_train = Patient_data_loader(patientType='MS', patientID=7)

    # parameters
    niter = 1
    sigma = 0
    Lambda_tv = 1*10**(+1)
    # Lambda_tv = 0
    lr = 1e-3
    batch_size = 1
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 5  # 5 default

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
        flag_r_train=1
    )
    # unet3d = Unet(
    #     input_channels=1, 
    #     output_channels=2, 
    #     num_filters=[2**i for i in range(3, 8)],
    #     flag_rsa=2
    # )
    unet3d.to(device)
    unet3d.load_state_dict(torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt'))
    unet3d.eval()

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    while epoch < niter:
        epoch += 1

        # training phase
        for idx, (rdfs, masks, weights, wGs) in enumerate(trainLoader):

            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            if epoch == 1:
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

    means = unet3d(rdfs)[:, 0, ...]
    stds = unet3d(rdfs)[:, 1, ...]
    QSM = np.squeeze(np.asarray(means.cpu().detach()))
    STD = np.squeeze(np.asarray(stds.cpu().detach()))

    adict = {}
    adict['QSM'] = QSM
    sio.savemat(rootDir+'/QSM_f.mat', adict)
    
    adict = {}
    adict['STD'] = STD
    sio.savemat(rootDir+'/STD_f.mat', adict)