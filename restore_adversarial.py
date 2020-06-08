import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import math
import argparse

from torch.utils import data
from torch.autograd import Variable
from loader.COSMOS_data_loader import COSMOS_data_loader
from models.unet import Unet
from models.unetag import UnetAg
from models.utils import count_parameters
from utils.train import *
from utils.medi import *
from utils.data import *
from utils.files import *
from utils.test import *
from utils.loss import *

if __name__ == '__main__':

    # default params
    niters = 40
    lr = 1e-3
    voxel_size = (1, 1, 3)
    volume_size = (160, 160, 35)
    radius = 5
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 1
    r = 1e-5
    
    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=1e-2)
    parser.add_argument('--flag_r_train', type=int, default=0)
    opt = {**vars(parser.parse_args())}

    Lambda_tv = opt['lambda_tv']
    flag_r_train = opt['flag_r_train']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'
    dataFolder, i_case = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/20190323_COSMOS_smv_3mm', 7

    filename = '{0}/{1}/Mask_smv_3mm.mat'.format(dataFolder, i_case)
    Mask = np.real(load_mat(filename, varname='Mask_new'))[30:190, 30:190, 10:45, 0]

    filename = '{0}/{1}/N_std_smv_3mm.mat'.format(dataFolder, i_case)
    N_std = np.real(load_mat(filename, varname='N_std_new'))[30:190, 30:190, 10:45, 0]
    tempn = np.double(N_std)
    tempn = np.sqrt(SMV(tempn**2, volume_size, voxel_size, radius)+tempn**2)

    filename = '{0}/{1}/iMag_smv_3mm.mat'.format(dataFolder, i_case)
    iMag = np.real(load_mat(filename, varname='iMag_new'))[30:190, 30:190, 10:45, 0]

    # gradient Mask
    wG = gradient_mask(iMag, Mask)
    # fidelity term weight
    Data_weights = np.real(dataterm_mask(tempn, Mask, Normalize=False))

    # adversarial RDF
    RDF = np.real(load_mat('/data/Jinwei/Bayesian_QSM/adv_noise/rdf_r_0.mat', varname='rdf_r'))
    RDF = RDF[:, :, 10:45]

    # to torch tensor
    RDF = torch.from_numpy(RDF[np.newaxis, np.newaxis, ...]).float().to(device)
    wG = torch.from_numpy(wG[np.newaxis, np.newaxis, ...]).float().to(device)
    Data_weights = torch.from_numpy(Data_weights[np.newaxis, np.newaxis, ...]).float().to(device)
    Data_weights = torch.ones(Data_weights.shape).to(device)
    Mask = torch.from_numpy(Mask[np.newaxis, np.newaxis, ...]).float().to(device)

    # dipole kernel
    D = dipole_kernel(volume_size, voxel_size, B0_dir)

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(5, 10)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=flag_r_train
    )

    unet3d.to(device)
    weights_dict = torch.load(rootDir+'/weight_cv/weights_rsa=-1_validation=6_test=7.pt')
    weights_dict['r'] = (torch.ones(1)*r).to(device)
    unet3d.load_state_dict(weights_dict)
    unet3d.train()

    # optimizer
    optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    t0 = time.time()
    while epoch < niters:
        epoch += 1

        if epoch == 1:
            means = unet3d(RDF)[:, 0, ...]
            stds = unet3d(RDF)[:, 1, ...]
            QSM = np.squeeze(np.asarray(means.cpu().detach()))
            STD = np.squeeze(np.asarray(stds.cpu().detach()))

            print('Saving initial results')
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/QSM_0.mat', adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/STD_0.mat', adict)

        loss_kl,  loss_tv, loss_expectation = BayesianQSM_train(
            model=unet3d,
            input_RDFs=RDF,
            in_loss_RDFs=RDF-trans*scale,
            QSMs=0,
            Masks=Mask,
            fidelity_Ws=Data_weights,
            gradient_Ws=wG,
            D=D,
            flag_COSMOS=0,
            optimizer=optimizer,
            sigma_sq=0,
            Lambda_tv=Lambda_tv,
            voxel_size=voxel_size,
            K=K,
            flag_linear=0
        )

        print('epochs: [%d/%d], time: %ds, Lambda_tv: %f, Entropy loss: %2f, TV_loss: %2f, Expectation_loss: %2f, r: %f'
            % (epoch, niters, time.time()-t0, Lambda_tv, loss_kl, loss_tv, loss_expectation, unet3d.r))

    means = unet3d(RDF)[:, 0, ...]
    stds = unet3d(RDF)[:, 1, ...]
    QSM = np.squeeze(np.asarray(means.cpu().detach()))
    STD = np.squeeze(np.asarray(stds.cpu().detach()))

    adict = {}
    adict['QSM'] = QSM
    sio.savemat(rootDir+'/QSM_f.mat', adict)

    adict = {}
    adict['STD'] = STD
    sio.savemat(rootDir+'/STD_f.mat', adict)


