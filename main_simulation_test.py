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
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'
    dataDir = '/data/Jinwei/Bayesian_QSM/Data_with_N_std/simulation'

    # load input
    RDF = np.real(load_mat(dataDir+'/cmridata.mat', varname='RDF'))
    Mask = np.real(load_mat(dataDir+'/cmridata.mat', varname='Mask'))
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
        flag_r_train=0
    )

    unet3d.to(device)
    weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(0, 1)+'.pt')
    unet3d.load_state_dict(weights_dict)
    unet3d.train()

    means = unet3d(RDF)[:, 0:1, ...] * Mask
    stds = unet3d(RDF)[:, 1:2, ...] * Mask
    QSM = np.squeeze(np.asarray(means.cpu().detach()))
    STD = np.squeeze(np.asarray(stds.cpu().detach()))

    adict = {}
    adict['QSM'] = QSM
    sio.savemat(rootDir+'/QSM_simu.mat', adict)

    adict = {}
    adict['STD'] = STD
    sio.savemat(rootDir+'/STD_simu.mat', adict)


