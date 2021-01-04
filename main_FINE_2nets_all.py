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
from models.unet import Unet
from models.resBlock import ResBlock
from models.unetag import UnetAg
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

'''
    fine-tune resnet and unet3d on COSMOS pre-trained 2nets using fidelity loss
'''
if __name__ == '__main__':

    # default parameters
    niter = 100
    lr = 1e-3
    batch_size = 1

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0, 1')
    parser.add_argument('--case_validation', type=int, default=6)
    parser.add_argument('--case_test', type=int, default=7)
    parser.add_argument('--linear_factor', type=int, default=1)
    parser.add_argument('--weight_dir', type=str, default='weight_2nets')
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()

    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    weights_dict = torch.load(rootDir+'/linear_factor=1_validation=6_test=7_unet3d.pt')
    unet3d.load_state_dict(weights_dict)

    resnet = ResBlock(
        input_dim=2, 
        filter_dim=32,
        output_dim=1, 
    )
    resnet.to(device1)
    weights_dict = torch.load(rootDir+'/linear_factor=1_validation=6_test=7_resnet.pt')
    resnet.load_state_dict(weights_dict)

    # optimizer
    optimizer0 = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))
    optimizer1 = optim.Adam(resnet.parameters(), lr = lr, betas=(0.5, 0.999))

    # dataloaders
    # dataLoader_train_MS = Patient_data_loader_all(patientType='MS_old', flag_RDF_input=1)
    # trainLoader_MS = data.DataLoader(dataLoader_train_MS, batch_size=batch_size, shuffle=True)

    dataLoader_train_ICH = Patient_data_loader_all(patientType='ICH', flag_RDF_input=1)
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
        unet3d.train(), resnet.train()
        for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(trainLoader_ICH):
            # training of unet3d
            rdf_inputs0 = rdf_inputs.to(device0, dtype=torch.float)
            rdfs0 = rdfs.to(device0, dtype=torch.float)
            masks0 = masks.to(device0, dtype=torch.float)
            weights0 = weights.to(device0, dtype=torch.float)
            wGs0 = wGs.to(device0, dtype=torch.float)

            loss_fidelity = BayesianQSM_train(
                model=unet3d,
                input_RDFs=rdf_inputs0,
                in_loss_RDFs=rdfs0,
                QSMs=0,
                Masks=masks0,
                fidelity_Ws=weights0,
                gradient_Ws=wGs0,
                D=np.asarray(D[0, ...]),
                flag_COSMOS=0,
                optimizer=optimizer0,
                sigma_sq=0,
                Lambda_tv=0,
                voxel_size=(1, 1, 3),
                K=1,
                flag_l1=2
            )
            print('epochs: [%d/%d], time: %ds, Fidelity loss of unet3d: %f' % (epoch, niter, time.time()-t0, loss_fidelity))
            qsm_inputs = unet3d(rdf_inputs0).cpu().detach()
            qsm_inputs1 = qsm_inputs.to(device1, dtype=torch.float)

            # training of resnet
            rdf_inputs1 = rdf_inputs.to(device1, dtype=torch.float)
            inputs_cat = torch.cat((rdf_inputs1, qsm_inputs1), dim=1)
            rdfs1 = rdfs.to(device1, dtype=torch.float)
            masks1 = masks.to(device1, dtype=torch.float)
            weights1 = weights.to(device1, dtype=torch.float)
            wGs1 = wGs.to(device1, dtype=torch.float)

            loss_fidelity = BayesianQSM_train(
                model=resnet,
                input_RDFs=inputs_cat,
                in_loss_RDFs=rdfs1,
                QSMs=0,
                Masks=masks1,
                fidelity_Ws=weights1,
                gradient_Ws=wGs1,
                D=np.asarray(D[0, ...]),
                flag_COSMOS=0,
                optimizer=optimizer1,
                sigma_sq=0,
                Lambda_tv=0,
                voxel_size=(1, 1, 3),
                K=1,
                flag_l1=2
            )
            print('epochs: [%d/%d], time: %ds, Fidelity loss of resnet: %f' % (epoch, niter, time.time()-t0, loss_fidelity))
            print(' ')

        torch.save(unet3d.state_dict(), rootDir+'/unet3d_fine.pt')
        torch.save(resnet.state_dict(), rootDir+'/resnet_fine.pt')
