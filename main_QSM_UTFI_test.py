import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.utils import data
from loader.UTFI_data_loader import UTFI_data_loader
from models.unet import Unet
from utils.train import utfi_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    t0 = time.time()
    rootDir = '/data/Jinwei/Unsupervised_total_field_inversion'

    # typein parameters
    parser = argparse.ArgumentParser(description='UTFI')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--lambda_tv', type=int, default=1e-3)
    opt = {**vars(parser.parse_args())}

    lambda_tv = opt['lambda_tv']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader
    dataLoader_test = UTFI_data_loader(
        dataFolder='/data/Jinwei/Unsupervised_total_field_inversion/data/test', 
        flag_train=0
    )
    testLoader = data.DataLoader(dataLoader_test, batch_size=1, shuffle=False)

    # network
    model = Unet(
        input_channels=1, 
        output_channels=2,
        num_filters=[2**i for i in range(5, 10)],
        bilateral=1,
        use_deconv=1,
        use_deconv2=1,
        renorm=0,
        flag_r_train=0,
        flag_UTFI=1
    )
    model.to(device)
    model.load_state_dict(torch.load(rootDir+'/weights/weight_tv={0}.pt'.format(lambda_tv)))
    model.eval()

    chi_bs, chi_ls = [], []
    with torch.no_grad():
        for idx, (ifreqs, masks, data_weights, wGs) in enumerate(testLoader):
            ifreqs = ifreqs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            outputs = model(ifreqs)
            chi_b, chi_l = outputs[:, 0:1, ...] * (1-masks), outputs[:, 1:2, ...] * masks
            chi_bs.append(chi_b[0, ...].cpu().detach())
            chi_ls.append(chi_l[0, ...].cpu().detach())

        chi_bs = np.concatenate(chi_bs, axis=0)
        chi_ls = np.concatenate(chi_ls, axis=0)

        chi_bs = np.transpose(chi_bs, [1,2,3,0])
        chi_ls = np.transpose(chi_ls, [1,2,3,0])

    adict = {}
    adict['chi_bs'] = chi_bs
    sio.savemat(rootDir+'/chi_bs.mat', adict)

    adict = {}
    adict['chi_ls'] = chi_ls
    sio.savemat(rootDir+'/chi_ls.mat', adict)

    

