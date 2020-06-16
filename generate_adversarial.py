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

    # default parameters
    flag_smv = 1
    flag_gen = 1
    flag_crop = 1
    trans = 0.15
    scale = 3
    K = 1  # number of samples in MC
    if flag_smv:
        voxel_size = (1, 1, 3)
    else:
        voxel_size = (1, 1, 1)
    B0_dir = (0, 0, 1)
    patchSize = (128, 128, 128)
    extraction_step = (42, 42, 42)

    # parameters for adversarial noise generation
    niter = 1000  # number of iterations to generate adv noise
    Lambda = 1e+2  # 1e+2 best for now
    Lambda_bg = 1e+2
    gamma = 0.9
    eta = 10
    tau = 1e-3 

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--weight_dir', type=str, default='weight_cv')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM/'

    val = opt['case_validation']
    test = opt['case_test']

    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(3, 8)],
        # num_filters=[2**i for i in range(5, 10)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=0
    )

    print('{0} trainable parameters in total'.format(count_parameters(unet3d)))
    unet3d.to(device)
    # unet3d.load_state_dict(torch.load(rootDir+'/weight_cv/weights_rsa=-1_validation=6_test=7.pt'))
    # unet3d.load_state_dict(torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(0, 1)+'.pt'))
    unet3d.load_state_dict(torch.load(rootDir+'/weight_adv/weights_before_adv.pt'))
    unet3d.eval()

    QSMs, STDs, RDFs = [], [], []
    RMSEs, Fidelities = [], []
    for test_dir in range(0, 1):
        dataLoader = COSMOS_data_loader(
            split='Test',
            case_validation=val,
            case_test=test,
            test_dir=test_dir,
            patchSize=patchSize, 
            extraction_step=extraction_step,
            voxel_size=voxel_size,
            flag_smv=flag_smv,
            flag_gen=flag_gen,
            flag_crop=flag_crop
        )
        testLoader = data.DataLoader(dataLoader, batch_size=1, shuffle=False)

        # # dipole kernel
        # D = dipole_kernel(dataLoader.volSize, voxel_size, B0_dir)
        # D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], K, axis=0)
        # D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
        # D_cplx = torch.tensor(D_cplx, device=device).float()

        for idx, (rdfs, masks, weights, qsms) in enumerate(testLoader):

            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
            qsms = tile(qsms, 0, K)
            masks = tile(masks.to(device), 0, K)

            r = Variable(torch.rand(rdfs.size()).to(device), requires_grad=True)
            v = Variable(torch.zeros(rdfs.size()).to(device), requires_grad=False)
            r.data = r.data * tau # intialize r

            print('Start to generate adversarial noise on orientation {0}'.format(test_dir))
            for epoch in range(niter+1):
            
                rdfs_r = rdfs + r * masks
                outputs = unet3d(rdfs)
                # sampling
                mean_Maps = tile(outputs[:, 0:1, ...], 0, K)
                var_Maps = tile(outputs[:, 1:2, ...], 0, K)
                epsilon = torch.normal(mean=torch.zeros(*mean_Maps.size()), std=torch.ones(*var_Maps.size()))
                epsilon = epsilon.to(device, dtype=torch.float)
                samples = mean_Maps + torch.sqrt(var_Maps)*epsilon

                outputs_r = unet3d(rdfs_r)
                # sampling
                mean_Maps_r = tile(outputs_r[:, 0:1, ...], 0, K)
                var_Maps_r = tile(outputs_r[:, 1:2, ...], 0, K)
                epsilon_r = torch.normal(mean=torch.zeros(*mean_Maps_r.size()), std=torch.ones(*var_Maps_r.size()))
                epsilon_r = epsilon_r.to(device, dtype=torch.float)
                samples_r = mean_Maps_r + torch.sqrt(var_Maps_r)*epsilon_r
                
                # objective function
                Q = torch.mean((samples*masks - samples_r*masks)**2)  - Lambda*torch.mean(r**2) 
                # - Lambda_bg*torch.mean((samples*(1-masks) - samples_r*(1-masks))**2)
                # updata
                Q.backward()
                v.data = gamma * v.data + eta * r.grad.data
                r.data += v.data

                Recon = np.squeeze(np.asarray(samples_r.cpu().detach()))
                Truth = np.squeeze(np.asarray(qsms.cpu().detach()))
                Mask = np.squeeze(np.asarray(masks.cpu().detach()))
                print('Iteration: %d/%d, RMSE: %f, L2 norm of r: %f, L2 norm of the difference: %f' % (epoch, niter, 
                    np.mean((Recon*Mask - Truth*Mask)**2), torch.sum(r**2), torch.sum((samples*masks - samples_r*masks)**2)))
                
                if epoch % 10 == 0:
                    adict = {}
                    adict['rdf_r'] = np.squeeze(np.asarray(rdfs_r.cpu().detach()))
                    sio.savemat(rootDir+'adv_noise/rdf_r_{}.mat'.format(epoch), adict)

                    adict = {}
                    adict['mean_r'] = np.squeeze(np.asarray(outputs_r[:, 0:1, ...].cpu().detach()))
                    sio.savemat(rootDir+'adv_noise/mean_r_{}.mat'.format(epoch), adict)

                    adict = {}
                    adict['std_r'] = np.squeeze(np.asarray(outputs_r[:, 1:2, ...].cpu().detach()))
                    sio.savemat(rootDir+'adv_noise/std_r_{}.mat'.format(epoch), adict)
                

