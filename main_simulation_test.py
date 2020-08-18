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
    niters = 25
    lr = 1e-3
    voxel_size = (1, 1, 3)
    radius = 5
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 5  # k = 5 for paper result

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--idx', type=str, default='1')
    parser.add_argument('--snr', type=str, default='50')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'
    dataDir = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/simulation/simulated_data_snr={}'.format(opt['snr'])

    # load input
    RDF = np.real(load_mat(dataDir+'/cmridata{}.mat'.format(opt['idx']), varname='RDF'))
    Mask = np.real(load_mat(dataDir+'/cmridata{}.mat'.format(opt['idx']), varname='Mask'))
    RDF = (RDF + trans) * scale
    RDF = torch.from_numpy(RDF[np.newaxis, np.newaxis, ...]).float().to(device)
    Mask = torch.from_numpy(Mask[np.newaxis, np.newaxis, ...]).float().to(device)

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=2,
        # num_filters=[2**i for i in range(5, 10)],
        num_filters=[2**i for i in range(3, 8)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=1,
        flag_r_train=0,
        bilateral_infer=0
    )

    unet3d.to(device)
    # dir simu: mv8, dir sim2: mv9, dir sim3: rsa=-1_validation=6_test=7, dir sim4: rsa=-1_validation=7_test=8
    # dir simu5: sa=-1_validation=7_test=2,
    # dir simu6: weights_vi_cosmos.pt
    # weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(0, 1)+'.pt')
    # weights_dict = torch.load(rootDir+'/weight_cv/weights_rsa=-1_validation=6_test=7.pt')
    weights_dict = torch.load(rootDir+'/weights_VI/weights_vi_cosmos.pt')
    unet3d.load_state_dict(weights_dict)
    unet3d.train()

    means = unet3d(RDF)[:, 0:1, ...] * Mask
    stds = unet3d(RDF)[:, 1:2, ...] * Mask
    QSM = np.squeeze(np.asarray(means.cpu().detach()))
    STD = np.squeeze(np.asarray(stds.cpu().detach()))

    adict = {}
    adict['QSM'] = QSM
    sio.savemat(rootDir+'/result_simu6/QSM_simu{}.mat'.format(opt['idx']), adict)

    adict = {}
    adict['STD'] = STD
    sio.savemat(rootDir+'/result_simu6/STD_simu{}.mat'.format(opt['idx']), adict)


