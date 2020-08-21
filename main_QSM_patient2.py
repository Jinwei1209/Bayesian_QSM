import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.utils import data
from loader.Patient_data_loader import Patient_data_loader
from loader.Patient_data_loader_all import Patient_data_loader_all
from models.unet import Unet
from models.resBlock import ResBlock
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

'''
    Patient test for meta-FINE
'''
if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0,1')
    parser.add_argument('--flag_resnet', type=int, default=0)  # 0 for unet, 1 for resnet
    parser.add_argument('--flag_init', type=int, default=0)  # 0 for linear_factor=1, 1 for linear_factor=4
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}

    # python main_QSM_patient.py --flag_r_train=1 --patientID=8 (or 16)

    patient_type = opt['patient_type']
    patientID = opt['patientID']
    flag_init = opt['flag_init']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # dataloader
    dataLoader_train = Patient_data_loader(
        patientType=patient_type, 
        patientID=patientID,
        flag_input=1
        )

    # parameters
    niter = 100
    # lr = 1e-3
    lr = 5e-4
    batch_size = 1
    B0_dir = (0, 0, 1)

    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size
    S = SMV_kernel(volume_size, voxel_size, radius=5)
    D = dipole_kernel(volume_size, voxel_size, B0_dir)
    D = np.real(S * D)

    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    # # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
        use_deconv=1,
        flag_rsa=0
    )
    unet3d.to(device0)
    if flag_init == 0:
        weights_dict = torch.load(rootDir+'/weight_qsmnet_p2/linear_factor=1_validation=6_test=7.pt')
    else:
        weights_dict = torch.load(rootDir+'/weight_qsmnet_p2/linear_factor=4_validation=6_test=7.pt')
    unet3d.load_state_dict(weights_dict)
    model = unet3d

    if opt['flag_resnet']:
        resnet = ResBlock(
            input_dim=1, 
            filter_dim=32,
            output_dim=1, 
        )
        resnet.to(device1)
        weights_dict = torch.load(rootDir+'/weight_qsmnet_p2/linear_factor=1_validation=6_test=7_resnet.pt')
        resnet.load_state_dict(weights_dict)
        resnet.eval()
        model = resnet

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    loss_iters = np.zeros(niter)
    while epoch < niter:
        epoch += 1

        # training phase
        for idx, (rdf_inputs, rdfs, masks, weights, wGs) in enumerate(trainLoader):
            
            rdf_inputs = rdf_inputs.to(device0, dtype=torch.float)
            rdfs = rdfs.to(device0, dtype=torch.float)
            masks = masks.to(device0, dtype=torch.float)
            weights = weights.to(device0, dtype=torch.float)
            wGs = wGs.to(device0, dtype=torch.float)

            if epoch == 1:
                unet3d.eval()
                QSMnet = unet3d(rdf_inputs)
                resnet_input = QSMnet
                QSMnet = np.squeeze(np.asarray(QSMnet.cpu().detach()))
                print('Saving initial results')
                adict = {}
                adict['QSMnet'] = QSMnet
                sio.savemat(rootDir+'/QSMnet{}.mat'.format(flag_init), adict)

                if opt['flag_resnet']:
                    rdf_inputs = resnet_input.to(device1, dtype=torch.float)
                    rdfs = rdfs.to(device1, dtype=torch.float)
                    masks = masks.to(device1, dtype=torch.float)
                    weights = weights.to(device1, dtype=torch.float)
                    wGs = wGs.to(device1, dtype=torch.float)

            print(model.get_device())
            loss_fidelity = BayesianQSM_train(
                model=model,
                input_RDFs=rdf_inputs,
                in_loss_RDFs=rdfs,
                QSMs=0,
                Masks=masks,
                fidelity_Ws=weights,
                gradient_Ws=wGs,
                D=D,
                flag_COSMOS=0,
                optimizer=optimizer,
                sigma_sq=0,
                Lambda_tv=0,
                voxel_size=voxel_size,
                K=1,
                flag_l1=2
            )

            print('epochs: [%d/%d], time: %ds, Fidelity loss: %f' % (epoch, niter, time.time()-t0, loss_fidelity))

    FINE = unet3d(rdf_inputs)[:, 0, ...]
    FINE = np.squeeze(np.asarray(FINE.cpu().detach()))

    adict = {}
    adict['FINE'] = FINE
    sio.savemat(rootDir+'/FINE{}.mat'.format(flag_init), adict)