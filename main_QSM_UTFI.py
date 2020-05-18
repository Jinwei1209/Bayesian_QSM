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

    # parameters
    t0 = time.time()
    epoch = 0
    niter = 100
    lr = 1e-3
    batch_size = 5
    B0_dir = (0, 0, 1)

    # typein parameters
    parser = argparse.ArgumentParser(description='UTFI')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=10)
    opt = {**vars(parser.parse_args())}
    
    lambda_tv = opt['lambda_tv']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader
    dataLoader_train = UTFI_data_loader(dataFolder='/data/Jinwei/Unsupervised_total_field_inversion/data/train')
    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    dataLoader_val = UTFI_data_loader(dataFolder='/data/Jinwei/Unsupervised_total_field_inversion/data/validation')
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=False)

    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size
    num_samples = len(dataLoader_train.list_IDs) * 3 * 4 * 2
    S = SMV_kernel(volume_size, voxel_size, radius=5)
    D = dipole_kernel(volume_size, voxel_size, B0_dir)
    D_smv = np.real(S * D)

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

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.5, 0.999))

    Validation_loss = []

    while epoch < niter:
        # training phase
        model.train()
        for idx, (ifreqs, masks, data_weights, wGs) in enumerate(trainLoader):
            ifreqs = ifreqs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            data_weights = data_weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            loss_PDF, loss_fidelity, loss_tv = utfi_train(
                model, optimizer, ifreqs, masks, data_weights, 
                wGs, D, D_smv, lambda_tv, voxel_size, flag_train=1)

            print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                % (epoch, niter, idx, num_samples//batch_size, time.time()-t0))
            print('Loss PDF: {0}, loss fidelity: {1}, loss TV: {2}'.format(
                loss_PDF, loss_fidelity, loss_tv))
            if epoch > 0:
                print('Loss in Validation dataset is %f' % (Validation_loss[-1]))

        model.eval()
        loss_total_list = []
        with torch.no_grad():
            for idx, (ifreqs, masks, data_weights, wGs) in enumerate(valLoader):
                ifreqs = ifreqs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
                data_weights = data_weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)
                loss_PDF, loss_fidelity, loss_tv = utfi_train(
                    model, optimizer, ifreqs, masks, data_weights, 
                    wGs, D, D_smv, lambda_tv, voxel_size, flag_train=0)
                loss_total = loss_PDF + loss_fidelity + loss_tv
                loss_total_list.append(np.asarray(loss_total))
            Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))
        
        if Validation_loss[-1] == min(Validation_loss):
            torch.save(netG_dc.state_dict(), rootDir+'/weights/weight_tv={0}.pt'.format(lambda_tv))
        
        epoch += 1

    

