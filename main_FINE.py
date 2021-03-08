import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from loader.COSMOS_data_loader_QSMnet_p import COSMOS_data_loader
from loader.Patient_data_loader_all import Patient_data_loader_all
from loader.Patient_data_loader import Patient_data_loader
from models.unet import Unet
from models.resBlock import ResBlock
from models.unetag import UnetAg
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

'''
    fine-tune pre-trained unet3d using the fidelity loss
'''
if __name__ == '__main__':

    # default parameters
    niter = 300
    lr = 5e-4
    batch_size = 1
    Lambda_tv = 20
    trans = 0
    scale = 1

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--loader', type=int, default=0)  # 0 for invivo data, 1 for simulated data
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--linear_factor', type=int, default=1)
    parser.add_argument('--weight_dir', type=str, default='weight_2nets')
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()

    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM/' + opt['weight_dir']

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
        use_deconv=1,
        flag_rsa=0
    )
    unet3d.to(device0)
    # weights_dict = torch.load(rootDir+'/linear_factor=1_validation=6_test=7_unet3d.pt')  # used for visualization on HOBIT
    # weights_dict = torch.load(rootDir+'/rsa=0_validation=6_test=7_.pt')  # good for metrics on HOBIT ICH8
    weights_dict = torch.load('/data/Jinwei/Bayesian_QSM/weight_cv/rsa=0_validation=6_test=7__.pt')
    # weights_dict = torch.load(rootDir+'/unet3d_fine.pt')  # used for metrics on HOBIT ICH8
    unet3d.load_state_dict(weights_dict)

    # optimizer
    optimizer0 = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))

    # dataloaders
    # dataLoader_train_MS = Patient_data_loader(patientType='MS_old', patientID=opt['patientID'], flag_input=1)
    # trainLoader_MS = data.DataLoader(dataLoader_train_MS, batch_size=batch_size, shuffle=True)

    dataLoader_train_ICH = Patient_data_loader(patientType='ICH', patientID=opt['patientID'], flag_input=1, flag_simu=opt['loader'])
    trainLoader_ICH = data.DataLoader(dataLoader_train_ICH, batch_size=batch_size, shuffle=True)

    epoch = 0
    gen_iterations = 1
    display_iters = 5
    loss1_sum, loss2_sum = 0, 0
    Validation_loss = []
    loss_L1 = lossL1()

    while epoch < niter:
        epoch += 1
        # training phase
        unet3d.train()
        for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(trainLoader_ICH):
            # training of unet3d
            rdf_inputs0 = (rdf_inputs.to(device0, dtype=torch.float) + trans) * scale
            rdfs0 = (rdfs.to(device0, dtype=torch.float) + trans) * scale
            masks0 = masks.to(device0, dtype=torch.float)
            weights0 = weights.to(device0, dtype=torch.float)
            wGs0 = wGs.to(device0, dtype=torch.float)

            if epoch == 1:
                QSM = unet3d(rdf_inputs0)[:, 0, ...]
                QSM = np.squeeze(np.asarray(QSM.cpu().detach()))
                adict = {}
                adict['QSM'] = QSM
                sio.savemat('/data/Jinwei/Bayesian_QSM/result_2nets/simu_fine2/QSM_Unet_ICH{0}.mat'.format(opt['patientID']), adict)

            loss_fidelity, loss_tv = BayesianQSM_train(
                model=unet3d,
                input_RDFs=rdf_inputs0,
                in_loss_RDFs=rdfs0-trans*scale,
                QSMs=0,
                Masks=masks0,
                fidelity_Ws=weights0,
                gradient_Ws=wGs0,
                D=np.asarray(D[0, ...]),
                flag_COSMOS=0,
                optimizer=optimizer0,
                sigma_sq=0,
                Lambda_tv=Lambda_tv,
                voxel_size=(1, 1, 3),
                K=1,
                flag_l1=2
            )
            print('epochs: [%d/%d], time: %ds, Fidelity loss: %f, TV loss: %f' % (epoch, niter, time.time()-t0, loss_fidelity, loss_tv))

        if epoch % 10 == 0:
            QSM = unet3d(rdf_inputs0)[:, 0, ...].cpu().detach()
            QSM = np.squeeze(np.asarray(QSM.cpu().detach()))
            adict = {}
            adict['QSM'] = QSM
            sio.savemat('/data/Jinwei/Bayesian_QSM/result_2nets/simu_fine2/QSM_FINE_ICH{0}_{1}.mat'.format(opt['patientID'], epoch), adict)