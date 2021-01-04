import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
import random
import argparse

from torch.utils import data
from loader.Simulation_ich_loader import Simulation_ICH_loader
from models.unet import Unet
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':
 
    t0 = time.time()
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # parameters
    sigma = 0
    lr = 1e-3
    batch_size = 1
    B0_dir = (0, 0, 1)
    trans = 0.15
    scale = 3
    K = 10  # 5 default
    r = 3e-3  # 3e-3 for PDI

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=int, default=20)
    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--flag_test', type=int, default=0)
    parser.add_argument('--flag_r_train', type=int, default=0)
    parser.add_argument('--flag_VI', type=int, default=0)  # PDI-VI0 or PDI
    parser.add_argument('--idx', type=int, default=0)  # for test
    opt = {**vars(parser.parse_args())}

    # for test:
    # python main_QSM_patient_all.py --gpu_id=1 --flag_test=1 --lambda_tv=20

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flag_test = opt['flag_test']
    Lambda_tv = opt['lambda_tv']
    niter = opt['niter']
    flag_r_train = opt['flag_r_train']
    flag_VI = opt['flag_VI']

    folder_weights = '/weights_ich_simu'

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
    unet3d.to(device)
    val_loss = []

    # training phase
    if not flag_test:

        # dataloader
        dataLoader_train = Simulation_ICH_loader(split='train')
        trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

        dataLoader_val = Simulation_ICH_loader(split='val')
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True)

        voxel_size = dataLoader_val.voxel_size
        volume_size = dataLoader_val.volume_size
        D_val = dipole_kernel(volume_size, voxel_size, B0_dir)

        if flag_VI:
            weights_dict = torch.load(rootDir+'/weights_VI//weights_lambda_tv=20.pt')
            weights_dict['r'] = (torch.ones(1)*r).to(device)
        else:
            weights_dict = torch.load(rootDir+'/weight/weights_sigma={0}_smv={1}_mv8'.format(sigma, 1)+'.pt')
            weights_dict['r'] = (torch.ones(1)*r).to(device)
        unet3d.load_state_dict(weights_dict)

        # optimizer
        optimizer = optim.Adam(unet3d.parameters(), lr=lr, betas=(0.5, 0.999))

        epoch = 0
        loss_iters = np.zeros(niter)
        while epoch < niter:
            epoch += 1

            unet3d.train()
            for idx, (qsms, rdfs_input, rdfs, masks, weights, wGs, D) in enumerate(trainLoader):
                
                qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
                rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                rdfs_input = (rdfs_input.to(device, dtype=torch.float) + trans) * scale
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)
                # weights = (torch.ones(qsms.size()) * 1 / 5e-3).to(device, dtype=torch.float)
                # wGs = (torch.ones(wGs.size())).to(device, dtype=torch.float)

                if flag_VI:
                    loss_kl,  loss_tv, loss_expectation = BayesianQSM_train(
                        model=unet3d,
                        input_RDFs=rdfs_input,
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

                    print('Epochs: [%d/%d], time: %ds, Lambda_tv: %f, KL_loss: %f, Expectation_loss: %f, r: %f'
                        % (epoch, niter, time.time()-t0, Lambda_tv, loss_kl+loss_tv, loss_expectation, unet3d.r))

                else:
                    optimizer.zero_grad()
                    outputs = unet3d(rdfs_input)
                    mean_Maps = outputs[:, 0:1, ...]
                    var_Maps = outputs[:, 1:2, ...]

                    loss1 = 1/2*torch.sum((mean_Maps*masks - qsms*masks)**2 / var_Maps)
                    loss2 = 1/2*torch.sum(torch.log(var_Maps)*masks)
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()

                    print('Epochs: [%d/%d], batchs: [%d/%d], time: %ds, r: %f'
                    % (epoch, niter, idx, dataLoader_train.num_subs//batch_size, time.time()-t0, unet3d.r))
                    print('Term1_loss: %f, term2_loss: %f' % (loss1.item(), loss2.item()))

            unet3d.eval()
            loss_total = 0
            with torch.no_grad():  # to solve memory exploration issue
                for idx, (qsms, rdfs_input, rdfs, masks, weights, wGs, D) in enumerate(valLoader):
                    
                    qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
                    rdfs_input = (rdfs_input.to(device, dtype=torch.float) + trans) * scale
                    rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
                    masks = masks.to(device, dtype=torch.float)
                    weights = weights.to(device, dtype=torch.float)
                    wGs = wGs.to(device, dtype=torch.float)
                    # weights = (torch.ones(qsms.size()) * 1 / 5e-3).to(device, dtype=torch.float)
                    # wGs = (torch.ones(wGs.size())).to(device, dtype=torch.float)

                    if flag_VI:
                        # calculate KLD
                        outputs = unet3d(rdfs_input)
                        loss_kl = loss_KL(outputs=outputs, QSMs=0, flag_COSMOS=0, sigma_sq=0)
                        loss_expectation, loss_tv = loss_Expectation(
                            outputs=outputs, QSMs=0, in_loss_RDFs=rdfs-trans*scale, fidelity_Ws=weights, 
                            gradient_Ws=wGs, D=D_val, flag_COSMOS=0, Lambda_tv=Lambda_tv, voxel_size=voxel_size, K=K)
                        loss_total += (loss_kl + loss_expectation + loss_tv).item()
                        print('KL Divergence on validation set = {0}'.format(loss_total))
                    
                    else:
                        outputs = unet3d(rdfs_input)
                        mean_Maps = outputs[:, 0:1, ...]
                        var_Maps = outputs[:, 1:2, ...]

                        loss1 = 1/2*torch.sum((mean_Maps*masks - qsms*masks)**2 / var_Maps)
                        loss2 = 1/2*torch.sum(torch.log(var_Maps)*masks)
                        loss = loss1 + loss2
                        loss_total += loss.item()
            
            val_loss.append(loss_total)
            if val_loss[-1] == min(val_loss):
                if flag_VI:
                    torch.save(unet3d.state_dict(), rootDir+folder_weights+'/weights_PDI_VI0.pt'.format(Lambda_tv))
                else:
                    torch.save(unet3d.state_dict(), rootDir+folder_weights+'/weights_PDI.pt')

    # test phase
    else:
        # dataloader
        dataLoader_test = Simulation_ICH_loader(split='test')
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=True)

        if flag_VI:
            # unet3d.load_state_dict(torch.load(rootDir+folder_weights+'/weights_PDI_VI0.pt')) # used for mean prediction
            unet3d.load_state_dict(torch.load(rootDir+'/weights_VI/weights_lambda_tv=20.pt')) # used for std prediction
        else:
            # unet3d.load_state_dict(torch.load(rootDir+folder_weights+'/weights_PDI.pt')) # used for mean prediction
            unet3d.load_state_dict(torch.load(rootDir+'/weight/weights_sigma=0_smv=1_mv8.pt')) # used for std prediction
        unet3d.eval()

        for idx, (qsms, rdfs_input, rdfs, masks, weights, wGs, D) in enumerate(testLoader):

            print('Saving test data')

            qsms = (qsms.to(device, dtype=torch.float) + trans) * scale
            rdfs_input = (rdfs_input.to(device, dtype=torch.float) + trans) * scale
            rdfs = (rdfs.to(device, dtype=torch.float) + trans) * scale
            masks = masks.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            wGs = wGs.to(device, dtype=torch.float)

            outputs = unet3d(rdfs_input)
            means = outputs[:, 0, ...]
            stds = outputs[:, 1, ...]
            QSM = np.squeeze(np.asarray(means.cpu().detach()))
            STD = np.squeeze(np.asarray(stds.cpu().detach()))
        
        if flag_VI:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/result_simu_ich3/QSM_VI0_ICH_simu{}.mat'.format(opt['idx']), adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/result_simu_ich3/STD_VI0_ICH_simu{}.mat'.format(opt['idx']), adict)
        else:
            adict = {}
            adict['QSM'] = QSM
            sio.savemat(rootDir+'/result_simu_ich4/QSM_PDI_ICH_simu{}.mat'.format(opt['idx']), adict)

            adict = {}
            adict['STD'] = STD
            sio.savemat(rootDir+'/result_simu_ich4/STD_PDI_ICH_simu{}.mat'.format(opt['idx']), adict)