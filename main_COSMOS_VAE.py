import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from loader.COSMOS_data_loader import COSMOS_data_loader
from models.unet import Unet
from models.unetag import UnetAg
from models.VAE import VAE
from utils.train import vae_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    # default parameters
    niter = 61
    lr = 1e-3
    batch_size = 4
    flag_smv = 1
    flag_gen = 1
    trans = 0.4  # 0.15 for PDI
    scale = 3
    latent_dim = 2000  # 500
    renorm = 1
    use_deconv = 1

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--flag_rsa', type=int, default=-2)  # -1 for PDI, -2 for VAE
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
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
        # patchSize = (64, 64, 32)  # (64, 64, 21)
        patchSize = (128, 128, 32)
        # patchSize_padding = (64, 64, 128)
        patchSize_padding = patchSize
        extraction_step = (42, 42, 6)  # (21, 21, 6)
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
    vae3d = VAE(
        input_channels=1, 
        output_channels=2,
        latent_dim=latent_dim,
        use_deconv=use_deconv,
        renorm=renorm,
        flag_r_train=0
    )

    vae3d.to(device)
    print(vae3d)

    # optimizer
    optimizer = optim.Adam(vae3d.parameters(), lr = lr, betas=(0.5, 0.999))
    ms = [0.3, 0.5, 0.7, 0.9]
    ms = [np.floor(m * niter).astype(int) for m in ms]
    scheduler = MultiStepLR(optimizer, milestones = ms, gamma = 0.2)

    # logger
    logger = Logger('logs', rootDir, opt['flag_rsa'], opt['case_validation'], opt['case_test'])

    # dataloader
    # dataLoader_train = COSMOS_data_loader(
    #     split='Train',
    #     patchSize=patchSize,
    #     extraction_step=extraction_step,
    #     voxel_size=voxel_size,
    #     case_validation=opt['case_validation'],
    #     case_test=opt['case_test'],
    #     flag_smv=flag_smv,
    #     flag_gen=flag_gen)
    # trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_val = COSMOS_data_loader(
        split='Val',
        patchSize=patchSize,
        extraction_step=extraction_step,
        voxel_size=voxel_size,
        case_validation=opt['case_validation'],
        case_test=opt['case_test'],
        flag_smv=flag_smv,
        flag_gen=flag_gen)
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, pin_memory=True)

    dataLoader_train = dataLoader_val
    trainLoader = valLoader

    epoch = 0
    gen_iterations = 1
    display_iters = 5
    recon_loss_sum, kl_loss_sum = 0, 0
    Validation_loss = []
    loss_L1 = lossL1()

    while epoch < niter:
        epoch += 1

        # training phase
        vae3d.train()
        for idx, (rdfs, masks, weights, qsms) in enumerate(trainLoader):
            if gen_iterations%display_iters == 0:
                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, case_validation: %f'
                    % (epoch, niter, idx, dataLoader_train.num_samples//batch_size+1, time.time()-t0, opt['case_validation']))
                print('Recon loss: %f, kl_loss: %f' % (recon_loss_sum/display_iters, kl_loss_sum/display_iters))
                if epoch > 1:
                    print('Validation loss of last epoch: %f' % (Validation_loss[-1]))

                recon_loss_sum, kl_loss_sum = 0, 0

            qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            qsms = qsms * masks

            recon_loss, kl_loss = vae_train(model=vae3d, optimizer=optimizer, x=qsms, mask=masks)
            recon_loss_sum += recon_loss
            kl_loss_sum += kl_loss
            gen_iterations += 1

            time.sleep(1)

        scheduler.step(epoch)

        # validation phase
        vae3d.eval()
        loss_total = 0
        idx = 0
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdfs, masks, weights, qsms) in enumerate(valLoader):
                idx += 1
                qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                qsms = qsms * masks
                
                x_mu, x_var, z_mu, z_logvar = vae3d(qsms)
                x_factor = torch.prod(torch.tensor(x_mu.size()))
                z_factor = torch.prod(torch.tensor(z_mu.size()))    
                # recon_loss = 0.5 * torch.sum((x_mu*masks - qsms*masks)**2 / x_var + torch.log(x_var)*masks)
                recon_loss = 0.5 * torch.sum((x_mu - qsms)**2 / x_var + torch.log(x_var)) / x_factor
                # recon_loss = torch.sum((x_mu - qsms)**2) / x_factor
                kl_loss = -0.5*torch.sum(1 + z_logvar - z_mu**2 - torch.exp(z_logvar)) / z_factor
                loss_total = loss_total + (recon_loss + kl_loss * 0.1)

            print('\n Validation loss: %f \n' % (loss_total / idx))
            Validation_loss.append(loss_total / idx)

        logger.print_and_save('Epoch: [%d/%d], Loss in Validation: %f' 
        % (epoch, niter, Validation_loss[-1]))

        if Validation_loss[-1] == min(Validation_loss):
            torch.save(vae3d.state_dict(), rootDir+'/weights_vae={0}_validation={1}_test={2}'.format(opt['flag_rsa'], opt['case_validation'], opt['case_test'])+'.pt')
